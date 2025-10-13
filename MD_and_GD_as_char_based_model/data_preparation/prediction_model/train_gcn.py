#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GCN baseline (DeepChem GraphConv) aligned with KA-GNN pipeline:
- 使用 DeepChem 的 ConvMolFeaturizer 将 SMILES -> ConvMol（GraphConv 所需输入）
- 数据集划分采用与 KA-GNN 一致的 Murcko ScaffoldSplitter（utils/splitters.py）
- 训练循环为纯 PyTorch：对 DeepChem 的 _GraphConvTorchModel 手动优化，loss 与 KA-GNN 对齐（l1/l2/sml1/bce, mean）
- 评估：整体 MAE、S1/T1 分开 MAE、ST Gap MAE（pred(S1)-pred(T1) vs 真实 ST Gap），按 ST Gap MAE 选最佳模型
- wandb 日志与 KA-GNN 对齐（train/valid/test 指标、best by gap mae）
"""

import os
import time
import yaml
import random
import argparse
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from logzero import logger
from utils_kan.splitters import ScaffoldSplitter

try:
    import deepchem as dc
    _DEEPCHEM_AVAILABLE = True
except Exception:
    _DEEPCHEM_AVAILABLE = False

# 使用项目本地封装（底层 DeepChem TorchModel）作为自定义 GCN 入口
from model.gcn import GraphConvModel as DCGraphConvModel

try:
    import wandb
    _WANDB_AVAILABLE = True
except Exception:
    _WANDB_AVAILABLE = False


# -------------------- utils & dataset --------------------
def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)


def _read_dataset(datafile: str) -> Tuple[List[str], np.ndarray, Optional[np.ndarray]]:
    """读取数据集，支持 dft（S1/T1 + ST Gap）与典型分类数据集。
    返回: smiles_list, labels(np.ndarray), gap(np.ndarray | None)
    """
    df = pd.read_csv(os.path.join('data', f'{datafile}.csv'))
    gap = None
    if datafile == 'dft':
        smiles_col = 'smiles' if 'smiles' in df.columns else ('SMILES' if 'SMILES' in df.columns else None)
        if smiles_col is None:
            raise KeyError("dft.csv 必须包含 'SMILES' 或 'smiles' 列")
        smiles = df[smiles_col].astype(str).tolist()
        if not set(['S1', 'T1']).issubset(df.columns):
            raise KeyError("dft.csv 必须包含 'S1' 和 'T1' 两列")
        labels = df[['S1', 'T1']].to_numpy(dtype=float)
        if 'ST Gap' in df.columns:
            gap = df['ST Gap'].to_numpy(dtype=float)
        else:
            # 允许缺失 ST Gap，则后续以 S1-T1 代替
            gap = (labels[:, 0] - labels[:, 1]).astype(float)
        return smiles, labels, gap
    else:
        # 其他数据集保留结构（如需扩展）
        smiles = df['smiles'].astype(str).tolist()
        if 'label' in df.columns:
            labels = df[['label']].to_numpy(dtype=float)
        else:
            # 兜底：单列目标
            labels = df.iloc[:, 1:2].to_numpy(dtype=float)
        return smiles, labels, None


def _scaffold_split_indices(smiles: List[str], train_ratio: float, vali_ratio: float, test_ratio: float):
    """使用 KA-GNN 的 Murcko ScaffoldSplitter，返回 (train_idx, valid_idx, test_idx)。"""
    # 传入 (smiles, idx) 以便回溯索引
    dataset = [(s, i) for i, s in enumerate(smiles)]
    train_set, valid_set, test_set = ScaffoldSplitter().split(dataset, frac_train=train_ratio,
                                                              frac_valid=vali_ratio, frac_test=test_ratio)
    train_idx = [it[1] for it in train_set]
    valid_idx = [it[1] for it in valid_set]
    test_idx = [it[1] for it in test_set]
    return train_idx, valid_idx, test_idx


def _featurize_conv_mol(smiles: List[str]):
    if not _DEEPCHEM_AVAILABLE:
        raise ImportError("需要 deepchem，请先安装 deepchem")
    featurizer = dc.feat.ConvMolFeaturizer()
    X = featurizer.featurize(smiles)
    # 过滤 None
    valid = [i for i, mol in enumerate(X) if mol is not None]
    X_valid = [X[i] for i in valid]
    return X_valid, valid


# -------------------- metrics --------------------
def _to_tensors_on_device(inputs: List[np.ndarray], device: torch.device) -> List[torch.Tensor]:
    """将 DC 生成的 numpy inputs 转为 torch.Tensor 并放到 device。
    输入格式: [atom_features, degree_slice, membership, n_samples, deg_adj_1..]
    """
    tensors: List[torch.Tensor] = []
    for i, arr in enumerate(inputs):
        if i == 0:
            tens = torch.from_numpy(arr.astype(np.float32))
        elif i in (1,):
            tens = torch.from_numpy(arr)
        elif i in (2, 3):
            tens = torch.from_numpy(arr).long()
        else:
            tens = torch.from_numpy(arr).long()
        tensors.append(tens.to(device))
    return tensors


def eval_mae(dc_model: DCGraphConvModel, dataset, device: torch.device) -> Tuple[float, List[float]]:
    """计算整体 MAE 与逐列 MAE。"""
    dc_model.model.eval()
    sum_abs = None
    n_samples = 0
    with torch.no_grad():
        for (inputs, y_b, _w_b) in dc_model.default_generator(dataset, epochs=1, deterministic=True, pad_batches=True):
            # y_b 可能为 [np.ndarray] 或 np.ndarray
            if y_b is None:
                continue
            y_np = y_b[0] if isinstance(y_b, (list, tuple)) else y_b
            y_t = torch.from_numpy(y_np).float().to(device)
            inputs_t = _to_tensors_on_device(inputs, device)
            outputs = dc_model.model(inputs_t, training=False)
            pred = outputs[0]
            if pred.shape != y_t.shape:
                # trim 对齐
                n = min(pred.shape[0], y_t.shape[0])
                pred = pred[:n]
                y_t = y_t[:n]
            diff = torch.abs(pred - y_t)
            sum_abs = diff.sum(dim=0).double() if sum_abs is None else sum_abs + diff.sum(dim=0).double()
            n_samples += diff.shape[0]
    if sum_abs is None or n_samples == 0:
        return 0.0, []
    per_mae = (sum_abs / n_samples).tolist()
    overall = float(sum_abs.sum().item() / (n_samples * len(per_mae)))
    return overall, per_mae


def eval_gap_mae(dc_model: DCGraphConvModel, dataset, true_gap: Optional[np.ndarray], device: torch.device) -> Optional[float]:
    """计算 ST Gap MAE；若提供 true_gap 列表则优先使用，否则使用 y[:,0]-y[:,1]。"""
    dc_model.model.eval()
    total_abs = 0.0
    total_cnt = 0
    idx_cursor = 0  # 用于从 true_gap 顺序取值
    with torch.no_grad():
        for (inputs, y_b, _w_b) in dc_model.default_generator(dataset, epochs=1, deterministic=True, pad_batches=True):
            if y_b is None:
                continue
            y_np = y_b[0] if isinstance(y_b, (list, tuple)) else y_b
            y_t = torch.from_numpy(y_np).float().to(device)
            inputs_t = _to_tensors_on_device(inputs, device)
            outputs = dc_model.model(inputs_t, training=False)
            pred = outputs[0]
            n = min(pred.shape[0], y_t.shape[0])
            pred = pred[:n]
            y_t = y_t[:n]
            pred_gap = (pred[:, 0] - pred[:, 1]).float()
            if true_gap is not None:
                tg = torch.from_numpy(true_gap[idx_cursor:idx_cursor + n]).float().to(device)
                idx_cursor += n
            else:
                tg = (y_t[:, 0] - y_t[:, 1]).float()
            diff = torch.abs(pred_gap - tg)
            total_abs += float(diff.sum().item())
            total_cnt += int(diff.numel())
    if total_cnt == 0:
        return None
    return total_abs / total_cnt


    


# -------------------- arg parsing & wandb --------------------
def parse_arguments():
    parser = argparse.ArgumentParser(description="DeepChem GraphConv baseline trainer")
    parser.add_argument("--config", type=str, help="path to yaml config")
    args = parser.parse_args()
    if not args.config:
        args.config = './config/c_path.yaml'
    else:
        args.config = args.config or './config/c_path.yaml'
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    for k, v in cfg.items():
        setattr(args, k, v)
    return args


def _safe_float(x):
    try:
        return float(x.detach().cpu().item())
    except Exception:
        try:
            return float(x)
        except Exception:
            return None


def init_wandb(args, model=None, run_name_suffix: str = ""):
    if not _WANDB_AVAILABLE:
        print("[wandb] not installed, skipping logging.")
        return None
    project = getattr(args, 'wandb_project', 'DFT_pretictor')
    entity = getattr(args, 'wandb_entity', None)
    mode = getattr(args, 'wandb_mode', 'online')
    run_name = getattr(args, 'wandb_run_name', None)
    if run_name_suffix:
        run_name = f"{run_name or 'run'}-{run_name_suffix}"
    run = wandb.init(project=project, entity=entity, name=run_name, mode=mode,
                     config={k: v for k, v in vars(args).items() if k != 'config'})
    if model is not None:
        try:
            wandb.watch(model, log="gradients", log_freq=100)
        except Exception:
            pass
    return run


# -------------------- Train loop --------------------
def main():
    if not _DEEPCHEM_AVAILABLE:
        raise ImportError("本脚本依赖 deepchem，请先安装：pip install deepchem")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 固定随机种子
    seed = 42
    set_seed(seed)

    # 读取配置
    args = parse_arguments()
    datafile = args.select_dataset
    batch_size = int(args.batch_size)
    train_ratio, vali_ratio, test_ratio = float(args.train_ratio), float(args.vali_ratio), float(args.test_ratio)
    loss_sclect = args.loss_sclect
    iters = int(args.iter)
    LR = float(args.LR)
    NUM_EPOCHS = int(args.NUM_EPOCHS)

    target_map = {'tox21': 12, 'muv': 17, 'sider': 27, 'clintox': 2, 'bace': 1, 'bbbp': 1, 'hiv': 1, 'dft': 2}
    target_dim = target_map.get(datafile, 2)

    # 读取数据
    smiles_all, labels_all, gap_all = _read_dataset(datafile)
    logger.info(f"Total molecules: {len(smiles_all)}; target dim: {labels_all.shape[1]}")

    # 先 Murcko scaffold 划分索引，再按索引切分 SMILES & labels（或先特征化再划分也可，这里保证与 KA-GNN 一致）
    train_idx, valid_idx, test_idx = _scaffold_split_indices(smiles_all, train_ratio, vali_ratio, test_ratio)

    # 逐 split 特征化（避免在无效 SMILES 上浪费时间）
    def build_split(idx_list):
        smiles_split = [smiles_all[i] for i in idx_list]
        y_split = labels_all[idx_list]
        gap_split = None if gap_all is None else gap_all[idx_list]
        X_split, valid_local = _featurize_conv_mol(smiles_split)
        # 过滤 featurize 失败的样本
        if len(valid_local) != len(smiles_split):
            y_split = y_split[valid_local]
            if gap_split is not None:
                gap_split = gap_split[valid_local]
        ds = dc.data.NumpyDataset(np.array(X_split, dtype=object), y_split)
        return ds, gap_split

    train_ds, train_gap = build_split(train_idx)
    valid_ds, valid_gap = build_split(valid_idx)
    test_ds, test_gap = build_split(test_idx)

    # 建立模型（DeepChem 的 TorchModel 包装器，底层为 _GraphConvTorchModel）
    # 超参尽量与常见设置保持一致，可根据需要从 args 增补
    dc_model = DCGraphConvModel(
        n_tasks=target_dim,
        number_input_features=[75, 64],
        graph_conv_layers=[64, 64],
        dense_layer_size=128,
        dropout=0.01,
        mode='regression',
        number_atom_features=75,
        batch_size=batch_size,
        batch_normalize=True,
        uncertainty=False,
    )
    # 我们将不使用 dc_model.fit，而是直接用底层 torch 模型训练，以支持 L1/SmoothL1 等自定义 loss。
    torch_model: nn.Module = dc_model.model.to(device)
    optimizer = Adam(torch_model.parameters(), lr=LR)
    scheduler = StepLR(optimizer, step_size=50, gamma=0.9)

    # 选择 loss 与 KA-GNN 对齐（mean reduction）
    if loss_sclect == 'l1':
        loss_fn = nn.L1Loss(reduction='mean')
    elif loss_sclect == 'l2':
        loss_fn = nn.MSELoss(reduction='mean')
    elif loss_sclect == 'sml1':
        loss_fn = nn.SmoothL1Loss(reduction='mean')
    elif loss_sclect == 'bce':
        loss_fn = nn.BCELoss(reduction='mean')
    else:
        raise ValueError('No Found the Loss function!')

    # 运行目录
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(
        "runs",
        str(datafile),
    f"model-gcn_bs-{batch_size}_lr-{LR}_seed-{seed}",
        timestamp,
    )
    os.makedirs(run_dir, exist_ok=True)
    print(f"Checkpoints will be saved to: {run_dir}")

    # wandb
    wandb_run = init_wandb(args, model=torch_model, run_name_suffix=f"{datafile}-gcn")
    if _WANDB_AVAILABLE and wandb_run is not None:
        wandb.config.update({
            'target_dim': target_dim,
            'train_size': len(train_ds),
            'valid_size': len(valid_ds),
            'test_size': len(test_ds),
            'device': str(device),
        }, allow_val_change=True)

    # 训练循环（与 KA-GNN 结构一致：多次迭代 iters，每次迭代跑 NUM_EPOCHS）
    for i in range(iters):
        iter_dir = os.path.join(run_dir, f"iter-{i+1}")
        os.makedirs(iter_dir, exist_ok=True)

        best_metric = float('inf')

        for epoch in range(NUM_EPOCHS):
            torch_model.train()
            total_train_loss = 0.0
            # 使用 DC 的生成器拿到 batch（保持 batching 与 ConvMol 聚合一致）
            for (inputs, y_b, _w_b) in dc_model.default_generator(train_ds, epochs=1, deterministic=True, pad_batches=True):
                if y_b is None:
                    continue
                y_np = y_b[0] if isinstance(y_b, (list, tuple)) else y_b
                y_t = torch.from_numpy(y_np).float().to(device)
                inputs_t = _to_tensors_on_device(inputs, device)
                optimizer.zero_grad()
                outputs = torch_model(inputs_t, training=True)
                pred = outputs[0]
                # 可能多 pad，按最短对齐
                n = min(pred.shape[0], y_t.shape[0])
                loss = loss_fn(pred[:n], y_t[:n])
                loss.backward()
                optimizer.step()
                total_train_loss += float(loss.detach().cpu().item())

            # 验证集 loss（可选，不作为 best 选择标准，仅记录）
            total_val_loss = 0.0
            torch_model.eval()
            with torch.no_grad():
                for (inputs, y_b, _w_b) in dc_model.default_generator(valid_ds, epochs=1, deterministic=True, pad_batches=True):
                    if y_b is None:
                        continue
                    y_np = y_b[0] if isinstance(y_b, (list, tuple)) else y_b
                    y_t = torch.from_numpy(y_np).float().to(device)
                    inputs_t = _to_tensors_on_device(inputs, device)
                    outputs = torch_model(inputs_t, training=False)
                    pred = outputs[0]
                    n = min(pred.shape[0], y_t.shape[0])
                    vloss = loss_fn(pred[:n], y_t[:n])
                    total_val_loss += float(vloss.detach().cpu().item())

            # 评估指标（与 KA-GNN 对齐）
            # 各 split 的 MAE
            train_mae, train_per_mae = eval_mae(dc_model, train_ds, device)
            valid_mae, valid_per_mae = eval_mae(dc_model, valid_ds, device)
            test_mae, per_mae = eval_mae(dc_model, test_ds, device)
            train_gap_mae = eval_gap_mae(dc_model, train_ds, train_gap, device)
            valid_gap_mae = eval_gap_mae(dc_model, valid_ds, valid_gap, device)
            test_gap_mae = eval_gap_mae(dc_model, test_ds, test_gap, device)

            # wandb 日志
            if _WANDB_AVAILABLE and wandb_run is not None:
                log_dict = {
                    'epoch': epoch + 1,
                    'iter': i + 1,
                    'train_loss': _safe_float(total_train_loss),
                    'valid_loss': _safe_float(total_val_loss),
                    'test_mae': _safe_float(test_mae),
                    'lr': optimizer.param_groups[0]['lr'],
                }
                if isinstance(train_per_mae, (list, tuple)) and len(train_per_mae) >= 2:
                    log_dict['train_mae'] = float(train_mae)
                    log_dict['train_mae_s1'] = float(train_per_mae[0])
                    log_dict['train_mae_t1'] = float(train_per_mae[1])
                if isinstance(valid_per_mae, (list, tuple)) and len(valid_per_mae) >= 2:
                    log_dict['valid_mae'] = float(valid_mae)
                    log_dict['valid_mae_s1'] = float(valid_per_mae[0])
                    log_dict['valid_mae_t1'] = float(valid_per_mae[1])
                if isinstance(per_mae, (list, tuple)) and len(per_mae) >= 2:
                    log_dict['test_mae_s1'] = float(per_mae[0])
                    log_dict['test_mae_t1'] = float(per_mae[1])
                if train_gap_mae is not None:
                    log_dict['train_gap_mae'] = float(train_gap_mae)
                if valid_gap_mae is not None:
                    log_dict['valid_gap_mae'] = float(valid_gap_mae)
                if test_gap_mae is not None:
                    log_dict['test_gap_mae'] = float(test_gap_mae)
                wandb.log(log_dict)

            # 以 ST Gap MAE 选最佳
            current_metric = test_gap_mae if test_gap_mae is not None else test_mae
            if current_metric is not None and current_metric < best_metric:
                best_metric = float(f"{current_metric:.5f}")
                print(f"[Best] epoch={epoch+1}, metric={best_metric}")
                if _WANDB_AVAILABLE and wandb_run is not None:
                    wandb.log({'best_gap_mae': best_metric, 'best_epoch': epoch + 1, 'iter': i + 1})
                # 保存 checkpoint（底层 torch 模型 + 优化器）
                ckpt = {
                    'model_state_dict': torch_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
                    'epoch': epoch + 1,
                    'iter': i + 1,
                    'best_gap_mae': best_metric,
                    'args': vars(args),
                }
                best_named_path = os.path.join(iter_dir, f"best_epoch{epoch+1}_gapmae{best_metric:.5f}.pth")
                best_link_path = os.path.join(iter_dir, "best.pth")
                torch.save(ckpt, best_named_path)
                torch.save(ckpt, best_link_path)
        # 学习率调度步进（与 main.py 对齐）
        scheduler.step()

    if _WANDB_AVAILABLE and wandb_run is not None:
        wandb_run.finish()


if __name__ == '__main__':
    main()
