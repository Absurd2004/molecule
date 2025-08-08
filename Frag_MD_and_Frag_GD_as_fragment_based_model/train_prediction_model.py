import pandas as pd
import numpy as np
import torch
import deepchem as dc
from sklearn.model_selection import train_test_split
import os
import logging
from typing import List, Tuple
import matplotlib.pyplot as plt
import wandb


from models.graphConvModel_pytorch import GraphConvModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(csv_paths: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """加载多个CSV数据文件"""
    logger.info(f"Loading data from {len(csv_paths)} CSV files")
    
    all_smiles = []
    all_st_gaps = []
    all_hl_gaps = []
    
    for i, csv_path in enumerate(csv_paths):
        logger.info(f"Loading file {i+1}/{len(csv_paths)}: {csv_path}")
        
        try:
            df = pd.read_csv(csv_path)
            logger.info(f"  Loaded {len(df)} samples from {csv_path}")
            
            # 检查必要的列是否存在
            required_columns = ['SMILES', 'ST Gap', 'HL Gap']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logger.warning(f"  Missing columns in {csv_path}: {missing_columns}")
                continue
            
            # 过滤掉包含NaN值的行
            df_clean = df.dropna(subset=required_columns)
            logger.info(f"  Valid samples after cleaning: {len(df_clean)}")
            
            # 收集数据
            all_smiles.extend(df_clean['SMILES'].values)
            all_st_gaps.extend(df_clean['ST Gap'].values)
            all_hl_gaps.extend(df_clean['HL Gap'].values)
            
        except Exception as e:
            logger.error(f"Error loading {csv_path}: {str(e)}")
            continue
    
    # 转换为numpy数组
    smiles_list = np.array(all_smiles)
    targets = np.column_stack([np.array(all_st_gaps), np.array(all_hl_gaps)])
    
    logger.info(f"Total loaded samples: {len(smiles_list)}")
    logger.info(f"ST Gap range: {targets[:, 0].min():.4f} - {targets[:, 0].max():.4f}")
    logger.info(f"HL Gap range: {targets[:, 1].min():.4f} - {targets[:, 1].max():.4f}")
    
    return smiles_list, targets

def create_datasets(smiles_list: np.ndarray, targets: np.ndarray, 
                   test_size: float = 0.2, random_state: int = 42):
    """创建训练、验证和测试数据集"""
    logger.info("Creating datasets...")
    
    # 第一步：使用ConvMolFeaturizer对SMILES进行特征化
    # ConvMolFeaturizer将SMILES字符串转换为ConvMol对象，这是GraphConv模型需要的输入格式
    featurizer = dc.feat.ConvMolFeaturizer()
    
    # 对所有SMILES进行特征化
    X = featurizer.featurize(smiles_list)
    
    # 过滤掉特征化失败的样本（None值）
    valid_indices = [i for i, mol in enumerate(X) if mol is not None]
    X_valid = [X[i] for i in valid_indices]
    y_valid = targets[valid_indices]
    
    logger.info(f"Valid samples after featurization: {len(X_valid)}")
    
    # 分割数据集：70%训练，20%测试，10%验证
    X_train, X_test, y_train, y_test = train_test_split(
        X_valid, y_valid, test_size=test_size, random_state=random_state
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.125, random_state=random_state  # 0.125 * 0.8 = 0.1
    )
    
    # 创建DeepChem数据集对象
    train_dataset = dc.data.NumpyDataset(X_train, y_train)
    val_dataset = dc.data.NumpyDataset(X_val, y_val)
    test_dataset = dc.data.NumpyDataset(X_test, y_test)
    
    logger.info(f"Train set size: {len(train_dataset)}")
    logger.info(f"Validation set size: {len(val_dataset)}")
    logger.info(f"Test set size: {len(test_dataset)}")
    
    return train_dataset, val_dataset, test_dataset


def normalize_targets(train_dataset, val_dataset, test_dataset):
    """标准化目标值"""
    logger.info("Normalizing targets...")
    
    # 使用NormalizationTransformer对目标值进行标准化
    # 这有助于模型训练的稳定性和收敛速度
    transformers = [
        dc.trans.NormalizationTransformer(
            transform_y=True, dataset=train_dataset, move_mean=True
        )
    ]

    original_y = train_dataset.y
    y_mean = np.mean(original_y, axis=0)
    y_std = np.std(original_y, axis=0)

    st_gap_mean = np.mean(original_y[:, 0])
    st_gap_std = np.std(original_y[:, 0])
    hl_gap_mean = np.mean(original_y[:, 1])
    hl_gap_std = np.std(original_y[:, 1])

    logger.info(f"Target normalization parameters:")
    logger.info(f"  ST Gap - Mean: {y_mean[0]:.4f}, Std: {y_std[0]:.4f}")
    #ogger.info(f" ST Gap - Mean: {st_gap_mean:.4f}, Std: {st_gap_std:.4f}")
    logger.info(f"  HL Gap - Mean: {y_mean[1]:.4f}, Std: {y_std[1]:.4f}")
    #logger.info(f" HL Gap - Mean: {hl_gap_mean:.4f}, Std: {hl_gap_std:.4f}")
    
    # 对所有数据集应用相同的标准化参数（基于训练集计算）
    train_dataset = transformers[0].transform(train_dataset)
    val_dataset = transformers[0].transform(val_dataset)
    test_dataset = transformers[0].transform(test_dataset)

    norm_params = {
        'y_mean': y_mean,
        'y_std': y_std
    }

    return train_dataset, val_dataset, test_dataset, transformers, norm_params

def denormalize_predictions(predictions, targets, norm_params):
    """将标准化的预测和目标值转换回真实值"""
    y_mean = norm_params['y_mean']
    y_std = norm_params['y_std']
    
    # 反标准化：y_real = y_normalized * std + mean
    predictions_real = predictions * y_std + y_mean
    targets_real = targets * y_std + y_mean
    return predictions_real, targets_real

def create_model(model_dir: str = "./prediction_models"):
    """创建GraphConv模型"""
    logger.info("Creating GraphConv model...")
    
    # 创建模型目录
    os.makedirs(model_dir, exist_ok=True)
    
    # 根据官方文档，GraphConvModel的关键参数说明：
    model = GraphConvModel(
        n_tasks=2,  # 预测2个任务：ST gap和HL gap
        
        # number_input_features: 每个GraphConv层的输入特征数
        # 第一层输入是原子特征数(75)，后续层输入是前一层的输出
        number_input_features=[75, 64],  
        
        # graph_conv_layers: 每个GraphConv层的输出通道数
        graph_conv_layers=[64, 64],  # 2层GraphConv，每层64个通道
        
        # dense_layer_size: GraphPool后的全连接层大小
        dense_layer_size=128,
        
        dropout=0.01,  # Dropout率，防止过拟合
        mode='regression',  # 回归任务
        
        # number_atom_features: 原子特征数，默认75
        number_atom_features=75,
        
        batch_size=32,  # 批次大小
        batch_normalize=True,  # 使用批标准化
        uncertainty=False,  # 不使用不确定性估计
        learning_rate=0.001,  # 学习率
        model_dir=model_dir  # 模型保存目录
    )


    logger.info("Model created successfully")
    return model

def get_learning_rate(model):
    """获取模型当前学习率"""
    try:

        if hasattr(model, 'optimizer') and model.optimizer is not None:
            #print(f"Optimizer found: model.optimizer")
            return model.optimizer.lr
        else:
            return 0.001  # 返回默认学习率
    except:
        return 0.001  # 如果获取失败，返回默认学习率

def set_learning_rate(model, new_lr):
    """手动设置模型学习率"""
    try:
        if hasattr(model, 'optimizer') and model.optimizer is not None:
            model.optimizer.lr = new_lr
            logger.info(f"Learning rate updated to {new_lr:.6f}")
        else:
            logger.warning("Optimizer not found, cannot update learning rate")
    except Exception as e:
        logger.error(f"Error updating learning rate: {e}")

def evaluate_detailed_metrics(model, dataset, dataset_name="Dataset",norm_params=None):
    """计算详细的评估指标"""
    predictions = model.predict(dataset)
    targets = dataset.y
    
    mae_st_norm = np.mean(np.abs(predictions[:, 0] - targets[:, 0]))
    mae_hl_norm = np.mean(np.abs(predictions[:, 1] - targets[:, 1]))
    
    
    metrics = {
        f'{dataset_name}/MAE_ST_Gap_normalized': mae_st_norm,
        f'{dataset_name}/MAE_HL_Gap_normalized': mae_hl_norm,
    }

    if norm_params is not None:
        predictions_real, targets_real = denormalize_predictions(predictions, targets, norm_params)

        mae_st_real = np.mean(np.abs(predictions_real[:, 0] - targets_real[:, 0]))
        mae_hl_real = np.mean(np.abs(predictions_real[:, 1] - targets_real[:, 1]))

        real_metrics = {
            f'{dataset_name}/MAE_ST_Gap_real': mae_st_real,
            f'{dataset_name}/MAE_HL_Gap_real': mae_hl_real,
        }

        metrics.update(real_metrics)
    
        return metrics, mae_st_real, mae_hl_real
    
    return metrics, mae_st_norm, mae_hl_norm

def train_model(model, train_dataset, val_dataset, nb_epoch=400,checkpoint_interval=50,model_dir="./prediction_models",norm_params=None, use_wandb=False):
    """训练模型"""
    logger.info("Starting model training...")

    best_model_dir = model_dir
    os.makedirs(best_model_dir, exist_ok=True)
   
    # 定义评估指标
    metric = dc.metrics.Metric(dc.metrics.mean_absolute_error, mode="regression")
    
    # 训练模型
    train_losses = []
    val_scores = []
    best_val_mae_st = float('inf')
    best_val_mae_hl = float('inf')
    best_epoch = 0

    # 学习率调度相关变量
    lr_patience = 10  # 10个epoch没有改善就降低学习率
    lr_decay_factor = 0.9
    epochs_without_improvement = 0
    last_best_mae_st = float('inf')
    last_best_mae_hl = float('inf')


    for epoch in range(nb_epoch):
        # 训练一个epoch
        train_loss = model.fit(train_dataset, nb_epoch=1, checkpoint_interval=0)
        train_losses.append(train_loss)

        train_metrics, train_mae_st, train_mae_hl = evaluate_detailed_metrics(
            model, train_dataset, "Train", norm_params
        )
        val_metrics, val_mae_st, val_mae_hl = evaluate_detailed_metrics(
            model, val_dataset, "Validation", norm_params
        )

        current_lr = get_learning_rate(model)

        both_improved = (val_mae_st < last_best_mae_st) and (val_mae_hl < last_best_mae_hl)

        if both_improved:
            # 两个MAE都有改善，重置计数器并更新最佳值
            epochs_without_improvement = 0
            last_best_mae_st = val_mae_st
            last_best_mae_hl = val_mae_hl
            logger.info(f"Both MAE improved! Reset patience counter.")
        
        else:
            # 没有同时改善，增加计数器
            epochs_without_improvement += 1
        
        if epochs_without_improvement >= lr_patience:
            current_lr = get_learning_rate(model)
            new_lr = current_lr * lr_decay_factor
            set_learning_rate(model, new_lr)
            epochs_without_improvement = 0  # 重置计数器
            logger.info(f"Learning rate reduced to {new_lr:.6f} after {lr_patience} epochs without both MAE improving")
        


        if use_wandb:
            wandb_logs = {
                'epoch': epoch,
                'train_loss': train_loss,
                'train_mae_st_real': train_mae_st,
                'train_mae_hl_real': train_mae_hl,
                'val_mae_st_real': val_mae_st,
                'val_mae_hl_real': val_mae_hl,
                'learning_rate': current_lr,  # 获取当前学习率
            }

            wandb_logs.update(train_metrics)
            wandb_logs.update(val_metrics)

            wandb.log(wandb_logs)

        logger.info(f"Epoch {epoch+1}/{nb_epoch}: "
                   f"Train Loss = {train_loss:.4f}, "
                   f"Train MAE ST = {train_mae_st:.4f}, "
                   f"Train MAE HL = {train_mae_hl:.4f}, "
                   f"Val MAE ST = {val_mae_st:.4f}, "
                   f"Val MAE HL = {val_mae_hl:.4f},"
                   f"LR = {current_lr:.6f}")
        
        if val_mae_st < best_val_mae_st and val_mae_hl < best_val_mae_hl:
            best_val_mae_st = val_mae_st
            best_val_mae_hl = val_mae_hl
            best_epoch = epoch
            
            # 保存最佳模型
            best_model_path = os.path.join(model_dir, "best.pt")
            torch.save({
                'model_state_dict': model.model.state_dict(),
                'optimizer_state_dict': model._pytorch_optimizer.state_dict(),
                'global_step': model._global_step,
                'epoch': epoch,
                'best_val_mae_st': best_val_mae_st,
                'best_val_mae_hl': best_val_mae_hl
            }, best_model_path)
            
            logger.info(f"✓ New best model saved! Val MAE: ST = {best_val_mae_st:.4f}, Val MAE: HL = {best_val_mae_hl:.4f} at epoch {epoch+1}")
            
            # 在wandb中标记最佳模型
            if use_wandb:
                wandb.log({
                    'best_val_mae_st': best_val_mae_st,
                    'best_val_mae_hl': best_val_mae_hl,
                    'best_epoch': best_epoch + 1
                })
        
        # 每50个epoch评估一次
        if (epoch + 1) % checkpoint_interval == 0:
            checkpoint_path = os.path.join(model_dir, f"checkpoint_epoch_{epoch+1}.pt")
            torch.save({
                'model_state_dict': model.model.state_dict(),
                'optimizer_state_dict': model._pytorch_optimizer.state_dict(),
                'global_step': model._global_step,
                'epoch': epoch,
                'train_loss': train_loss,
                'val_mae_st': val_mae_st,
                'val_mae_hl': val_mae_hl
            }, checkpoint_path)
            logger.info(f"Checkpoint saved at epoch {epoch+1}")
            
            # 保存检查点
            #model.save_checkpoint()
        
        #if (epoch + 1) % 10 == 0:
            #model.save_checkpoint()
    
    logger.info(f"Training completed!")
    logger.info(f"Best validation MAE: ST = {best_val_mae_st:.4f}, HL = {best_val_mae_hl:.4f} at epoch {best_epoch + 1}")


    return train_losses, val_scores, best_val_mae_st,best_val_mae_hl, best_epoch

def evaluate_model(model, test_dataset, transformers, norm_params=None):
    """评估模型"""
    logger.info("Evaluating model on test set...")
    
    # 预测
    predictions = model.predict(test_dataset)
    
    # 反标准化预测结果和真实值
    test_y = test_dataset.y

    if norm_params is not None:
        predictions_denorm, true_values_denorm = denormalize_predictions(predictions, test_y, norm_params)
    
    else:
        # 方法2：使用DeepChem的transformer（作为备用）
        if transformers:
            temp_pred_dataset = dc.data.NumpyDataset(test_dataset.X, predictions)
            temp_true_dataset = dc.data.NumpyDataset(test_dataset.X, test_y)
            
            temp_pred_dataset = transformers[0].untransform(temp_pred_dataset)
            temp_true_dataset = transformers[0].untransform(temp_true_dataset)
            
            predictions_denorm = temp_pred_dataset.y
            true_values_denorm = temp_true_dataset.y
        else:
            predictions_denorm = predictions
            true_values_denorm = test_y
    
    # 计算评估指标
    mae_st = np.mean(np.abs(predictions_denorm[:, 0] - true_values_denorm[:, 0]))
    mae_hl = np.mean(np.abs(predictions_denorm[:, 1] - true_values_denorm[:, 1]))

    

    
    logger.info(f"Test Results (Real Values):")
    logger.info(f"ST Gap - MAE: {mae_st:.4f}")
    logger.info(f"HL Gap - MAE: {mae_hl:.4f}")
    
    return {
        'predictions': predictions_denorm,
        'true_values': true_values_denorm,
        'mae_st': mae_st,
        'mae_hl': mae_hl,
        

    }

def main():
    """主函数"""
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)

    use_wandb = True  # 是否使用wandb进行实验跟踪

    if use_wandb:
        wandb.init(project="GraphConvModel_Training", name="Photosensitizers_Training")
        wandb.config.update({
            "epochs": 400,
            "batch_size": 32,
            "learning_rate": 0.001,
            "model_dir": "./models"
        })
    
    # 1. 加载数据
    file_list = ["./dftdata/Photosensitizers_DA.csv","./dftdata/Photosensitizers_DAD.csv"]  # 替换为你的数据文件路径
    #file_list = ["./dftdata/Photosensitizers_DA.csv"]  # 替换为你的数据文件路径
    smiles_list, targets = load_data(file_list)
    
    # 2. 创建数据集
    train_dataset, val_dataset, test_dataset = create_datasets(smiles_list, targets)
    
    # 3. 标准化目标值
    train_dataset, val_dataset, test_dataset, transformers, norm_params = normalize_targets(
        train_dataset, val_dataset, test_dataset
    )

    if use_wandb:
        wandb.log({
            'normalization/st_gap_mean': norm_params['y_mean'][0],
            'normalization/st_gap_std': norm_params['y_std'][0],
            'normalization/hl_gap_mean': norm_params['y_mean'][1],
            'normalization/hl_gap_std': norm_params['y_std'][1],
        })
    
    # 4. 创建模型
    model = create_model(model_dir="./prediction_models_lr")
    
    # 5. 训练模型
    train_losses, val_scores, best_val_mae_st,best_val_mae_hl, best_epoch = train_model(model, train_dataset, val_dataset, nb_epoch=400,
                                           checkpoint_interval=50, model_dir="./prediction_models_lr",norm_params=norm_params, use_wandb=use_wandb)

    best_model_path = os.path.join("./prediction_models_lr", "best.pt")
    if os.path.exists(best_model_path):
        logger.info(f"Loading best model (from epoch {best_epoch+1}) for final evaluation")
        checkpoint = torch.load(best_model_path, map_location=model.device)
        model.model.load_state_dict(checkpoint['model_state_dict'])
    else:
        logger.warning("Best model not found, using current model for evaluation")
    
    # 6. 评估模型
    results = evaluate_model(model, test_dataset, transformers,norm_params)


    logger.info("Training pipeline completed successfully!")

    if use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()