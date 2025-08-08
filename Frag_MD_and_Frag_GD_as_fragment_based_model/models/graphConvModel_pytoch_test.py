import torch
import torch.nn as nn
import torch.nn.functional as F
from deepchem.models.torch_models import TorchModel
from deepchem.models.torch_models.layers import GraphConv, GraphPool, GraphGather
from deepchem.models.losses import L2Loss
from deepchem.feat.mol_graphs import ConvMol

from typing import List
import numpy as np

class _GraphConvPyTorchModel(nn.Module):
    """
    PyTorch版本的Graph Convolutional Model核心实现
    
    这个模型实现了图卷积神经网络，用于分子属性预测。
    主要组件包括：
    1. 多个GraphConv层用于特征提取
    2. GraphPool层用于池化操作
    3. 可选的Dense层用于进一步的特征变换
    4. GraphGather层用于图级别的表示学习
    5. 最终的回归/分类层
    """
    def __init__(self, 
                n_tasks, 
                graph_conv_layers=[128, 128], 
                dense_layers=[64, 64],
                dropout=0.01, 
                number_atom_features=75, 
                uncertainty=True,
                batch_size=8,
                batch_normalize=True,
                mode="regression",
                **kwargs):
        
        super(_GraphConvPyTorchModel, self).__init__()
        self.uncertainty = uncertainty
        self.n_tasks = n_tasks

        if not isinstance(dropout, (list, tuple)):
            dropout = [dropout] * (len(graph_conv_layers) + len(dense_layers))
        

        if len(dropout) != len(graph_conv_layers) + len(dense_layers):
            raise ValueError("Wrong number of dropout probabilities provided")
        if uncertainty and any(d == 0.0 for d in dropout):
            raise ValueError("Dropout must be included in every layer to predict uncertainty!")


        self.graph_convs = nn.ModuleList([
            GraphConv(
                out_channel=layer_size,
                number_input_features=number_atom_features if i == 0 else graph_conv_layers[i-1],
                activation_fn=F.relu
            )
            for i, layer_size in enumerate(graph_conv_layers)
        ])

        self.graph_pools = nn.ModuleList([
            GraphPool() for _ in graph_conv_layers
        ])

        input_dim = graph_conv_layers[-1] if graph_conv_layers else number_atom_features
        self.dense_layers = nn.ModuleList()

        for i, layer_size in enumerate(dense_layers):
            self.dense_layers.append(
                nn.Linear(input_dim, layer_size)
            )
            input_dim = layer_size  # 下一层的输入维度
        

        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(graph_conv_layers[i] if i < len(graph_conv_layers) 
                        else dense_layers[i - len(graph_conv_layers)])
            if batch_normalize else None
            for i in range(len(graph_conv_layers) + len(dense_layers))
        ])

        self.dropouts = nn.ModuleList([
            nn.Dropout(dropout_rate) if dropout_rate > 0.0 else None 
            for dropout_rate in dropout
        ])

        final_dim = dense_layers[-1] if dense_layers else graph_conv_layers[-1]
        self.graph_gather = GraphGather(batch_size=batch_size, activation=torch.tanh)

        self.trim = TrimGraphOutput()


        self.regression_dense = nn.Linear(final_dim, n_tasks)

        

        if self.uncertainty:
            self.uncertainty_dense = nn.Linear(final_dim, n_tasks)
            self.uncertainty_trim = TrimGraphOutput()
        
    
    def forward(self, inputs):
        """
        前向传播函数
        
        Args:
            inputs: 包含以下元素的列表
                - inputs[0]: atom_features (原子特征)
                - inputs[1]: degree_slice (度数切片)
                - inputs[2]: membership (成员关系)
                - inputs[3]: n_samples (样本数量)
                - inputs[4:]: deg_adjs (度数邻接列表)
        
        Returns:
            输出列表，根据uncertainty设置返回不同内容
        """

        atom_features = inputs[0]
        degree_slice = inputs[1]
        membership = inputs[2]
        n_samples = inputs[3]
        deg_adjs = inputs[4:]

        x = atom_features

        for i, (conv, pool) in enumerate(zip(self.graph_convs, self.graph_pools)):

            conv_input = [x, degree_slice, membership] + deg_adjs
            x = conv(conv_input)

            if self.batch_norms[i] is not None:
                # 重塑数据以适应 BatchNorm1d
                original_shape = x.shape
                x = x.view(-1, original_shape[-1])
                x = self.batch_norms[i](x)
                x = x.view(original_shape)
            
            if self.dropouts[i] is not None and self.training:
                x = self.dropouts[i](x)

            pool_input = [x, degree_slice, membership] + deg_adjs
            x = pool(pool_input)
        
        start_idx = len(self.graph_convs)
        for i, dense in enumerate(self.dense_layers):
            x = dense(x)
            x = F.relu(x)

            bn_idx = start_idx + i

            if self.batch_norms[bn_idx] is not None:
                original_shape = x.shape
                x = x.view(-1, original_shape[-1])
                x = self.batch_norms[bn_idx](x)
                x = x.view(original_shape)
            
            if self.dropouts[bn_idx] is not None and self.training:
                x = self.dropouts[bn_idx](x)
        
        gather_input = [x, degree_slice, membership] + deg_adjs
        neural_fingerprint = self.graph_gather(gather_input)

        output = self.regression_dense(neural_fingerprint)

        output = self.trim([output, n_samples])

        if self.uncertainty:
            log_var = self.uncertainty_dense(neural_fingerprint)
            log_var = self.uncertainty_trim([log_var, n_samples])
            var = torch.exp(log_var)

            return [output, var, output, log_var, neural_fingerprint]

        else:
            return [output, neural_fingerprint]

class GraphConvModel(TorchModel):
    """
    PyTorch版本的GraphConvModel
    
    这是一个用于分子属性预测的图卷积神经网络模型。
    该模型继承自DeepChem的TorchModel，提供了完整的训练、预测和评估功能。
    
    主要特点：
    1. 支持多层图卷积和池化操作
    2. 可配置的Dense层用于特征变换
    3. 支持不确定性预测
    4. 支持批量归一化和Dropout
    5. 灵活的网络架构配置
    """
    def __init__(self,
                n_tasks: int,
                graph_conv_layers: List[int] = [128, 128],
                dense_layers: List[int] = [64, 64], 
                dropout: float = 0.0,
                number_atom_features: int = 75,
                batch_size: int = 100,
                batch_normalize: bool = True,
                uncertainty: bool = False,
                **kwargs):
        """
        初始化GraphConvModel
        
        Args:
            n_tasks: 任务数量（输出维度）
            graph_conv_layers: 图卷积层的隐藏单元数列表
            dense_layers: 密集层的隐藏单元数列表
            dropout: Dropout概率
            number_atom_features: 原子特征维度
            batch_size: 批次大小
            batch_normalize: 是否使用批量归一化
            uncertainty: 是否启用不确定性预测
            **kwargs: 传递给TorchModel的其他参数
        """

        self.n_tasks = n_tasks
        self.batch_size = batch_size
        self.uncertainty = uncertainty

        model = _GraphConvPyTorchModel(
            n_tasks=n_tasks,
            graph_conv_layers=graph_conv_layers,
            dense_layers=dense_layers,
            dropout=dropout,
            number_atom_features=number_atom_features,
            batch_normalize=batch_normalize,
            uncertainty=uncertainty,
            batch_size=batch_size
        )

        if self.uncertainty:
            output_types = ['prediction', 'variance', 'loss', 'loss', 'embedding']

            def loss(outputs, labels, weights):
                pred = outputs[0]
                log_var = outputs[3]
                target = labels[0]

                diff = target - pred
                loss_val = torch.mean(diff * diff / torch.exp(log_var) + log_var)
                return loss_val
        
        else:
            output_types = ['prediction', 'embedding']
            loss = L2Loss()
        
        super(GraphConvModel, self).__init__(
            model=model, 
            loss=loss, 
            output_types=output_types, 
            batch_size=batch_size, 
            **kwargs
        )
    
    def default_generator(self, dataset, epochs=1, mode='fit', deterministic=True, pad_batches=True):
        """
        默认数据生成器，将DeepChem数据集转换为模型可接受的格式
        
        Args:
            dataset: DeepChem数据集
            epochs: 训练轮数
            mode: 模式 ('fit', 'predict', 'uncertainty')
            deterministic: 是否确定性地遍历数据
            pad_batches: 是否填充批次
            
        Yields:
            (inputs, labels, weights): 训练数据批次
        """

        for epoch in range(epochs):
            for (X_b, y_b, w_b, ids_b) in dataset.iterbatches(
                batch_size=self.batch_size,
                deterministic=deterministic,
                pad_batches=pad_batches):

                multiConvMol = ConvMol.agglomerate_mols(X_b)

                n_samples = torch.tensor(X_b.shape[0], dtype=torch.long)


                inputs = [
                    torch.tensor(multiConvMol.get_atom_features(), dtype=torch.float32),
                    torch.tensor(multiConvMol.deg_slice, dtype=torch.long),
                    torch.tensor(multiConvMol.membership, dtype=torch.long),
                    n_samples
                ]

                deg_adjs = multiConvMol.get_deg_adjacency_lists()
                for i in range(1, len(deg_adjs)):
                    inputs.append(torch.tensor(deg_adjs[i], dtype=torch.long))

                labels = [torch.tensor(y_b, dtype=torch.float32)]
                weights = [torch.tensor(w_b, dtype=torch.float32)]
                
                yield (inputs, labels, weights)







class TrimGraphOutput(nn.Module):
    """
    Trim the output to the correct number of samples.
    Since GraphGather always outputs the fixed size batches, this layer trims the output to
    the number of samples that were in the actual input tensors.
    """
    def __init__(self):
        super(TrimGraphOutput, self).__init__()

    def forward(self, inputs):
        output, n_samples = inputs
        n_samples = n_samples.squeeze()
        return output[:n_samples]


        







