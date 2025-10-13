#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 15:08:37 2024

@author: longlee
"""

import torch.nn as nn
import torch
import pandas as pd
import numpy as np
import networkx as nx
import os
import argparse
import torch.nn.functional as F
import matplotlib.pyplot as plt
import yaml
import random
import dgl
import statistics
import csv
import time

from logzero import logger
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, confusion_matrix
from sklearn.metrics import cohen_kappa_score, accuracy_score, roc_auc_score, precision_score, recall_score
from sklearn.metrics import balanced_accuracy_score, r2_score, mean_squared_error, mean_absolute_error
from sklearn.metrics import precision_recall_curve, auc

from sklearn import metrics
from model.ka_gnn import KA_GNN,KA_GNN_two
from model.mlp_sage import MLPGNN,MLPGNN_two
from model.kan_sage import KANGNN, KANGNN_two
from torch.optim.lr_scheduler import StepLR
from ruamel.yaml import YAML
from utils_kan.splitters import ScaffoldSplitter
from utils_kan.graph_path import path_complex_mol
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from rdkit import Chem
from rdkit.Chem import AllChem
try:
    import wandb
    _WANDB_AVAILABLE = True
except Exception:
    _WANDB_AVAILABLE = False


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


class CustomDataset(Dataset):
    def __init__(self, label_list, graph_list, gap_list=None):
        self.labels = label_list
        self.graphs = graph_list
        self.gaps = gap_list  # optional: true ST Gap values
        self.device = torch.device('cpu') 

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        label = self.labels[index].to(self.device)
        graph = self.graphs[index].to(self.device)
        if self.gaps is not None:
            gap_val = torch.tensor(self.gaps[index], dtype=torch.float32, device=self.device)
            return label, graph, gap_val
        else:
            return label, graph
    


def collate_fn(batch):
    # Support (label, graph) or (label, graph, gap)
    first = batch[0]
    if len(first) == 3:
        labels, graphs, gaps = zip(*batch)
        labels = torch.stack(labels)
        batched_graph = dgl.batch(graphs)
        gaps = torch.stack([g if torch.is_tensor(g) else torch.tensor(g, dtype=torch.float32) for g in gaps])
        return labels, batched_graph, gaps
    else:
        labels, graphs = zip(*batch)
        labels = torch.stack(labels)
        batched_graph = dgl.batch(graphs)
        return labels, batched_graph



def has_node_with_zero_in_degree(graph):
    if (graph.in_degrees() == 0).any():
                return True
    return False




def is_file_in_directory(directory, target_file):
    file_path = os.path.join(directory, target_file)
    return os.path.isfile(file_path)


#others
def get_label():
    """Get that default sider task names and return the side results for the drug"""
    
    return ['label']


#tox21,12     
def get_tox():
    """Get that default sider task names and return the side results for the drug"""
    
    return ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
           'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']

#clintox,2
def get_clintox():
    
    return ['FDA_APPROVED', 'CT_TOX']

#sider,27
def get_sider():

    return ['Hepatobiliary disorders',
           'Metabolism and nutrition disorders', 'Product issues', 'Eye disorders',
           'Investigations', 'Musculoskeletal and connective tissue disorders',
           'Gastrointestinal disorders', 'Social circumstances',
           'Immune system disorders', 'Reproductive system and breast disorders',
           'Neoplasms benign, malignant and unspecified (incl cysts and polyps)',
           'General disorders and administration site conditions',
           'Endocrine disorders', 'Surgical and medical procedures',
           'Vascular disorders', 'Blood and lymphatic system disorders',
           'Skin and subcutaneous tissue disorders',
           'Congenital, familial and genetic disorders',
           'Infections and infestations',
           'Respiratory, thoracic and mediastinal disorders',
           'Psychiatric disorders', 'Renal and urinary disorders',
           'Pregnancy, puerperium and perinatal conditions',
           'Ear and labyrinth disorders', 'Cardiac disorders',
           'Nervous system disorders',
           'Injury, poisoning and procedural complications']

#muv
def get_muv():
    
    return ['MUV-466','MUV-548','MUV-600','MUV-644','MUV-652','MUV-689','MUV-692',
            'MUV-712','MUV-713','MUV-733','MUV-737','MUV-810','MUV-832','MUV-846',
            'MUV-852',	'MUV-858','MUV-859']




def creat_data(datafile, encoder_atom, encoder_bond, batch_size, train_ratio, vali_ratio, test_ratio, force_field: str = 'mmff'):
    

    datasets = datafile

    directory_path = 'data/processed/'
    target_file_name = datafile +'.pth'

    if is_file_in_directory(directory_path, target_file_name):

        return True
    
    else:

        df = pd.read_csv('data/' + datasets + '.csv')#
        if datasets == 'tox21':
            smiles_list, labels = df['smiles'], df[get_tox()] 
            #labels = labels.replace(0, -1)
            labels = labels.fillna(0)

        if datasets == 'muv':
            smiles_list, labels = df['smiles'], df[get_muv()]  
            labels = labels.fillna(0)

        if datasets == 'sider':
            smiles_list, labels = df['smiles'], df[get_sider()]  

        if datasets == 'clintox':
            smiles_list, labels = df['smiles'], df[get_clintox()] 
        
        if datasets == 'dft':
            smiles_col = 'smiles' if 'smiles' in df.columns else ('SMILES' if 'SMILES' in df.columns else None)
            if smiles_col is None:
                raise KeyError("dft.csv 必须包含 'SMILES' 或 'smiles' 列")
            smiles_list = df[smiles_col].tolist()
            if not set(['S1','T1']).issubset(df.columns):
                raise KeyError("dft.csv 必须包含 'S1' 和 'T1' 两列")
            labels = df[['S1','T1']]
            # ST Gap column (exact name in provided CSV is 'ST Gap')
            gap_col = 'ST Gap' if 'ST Gap' in df.columns else None
            if gap_col is None:
                raise KeyError("dft.csv 必须包含 'ST Gap' 列用于评估")
            gap_series = df[gap_col].astype(float).tolist()
    

        if datasets in ['hiv','bbbp','bace']:
            smiles_list, labels = df['smiles'], df[get_label()] 
        
        print(f"Number of molecules: {len(smiles_list)}")
        print(f"number of labels for each molecule: {labels.shape[1]}")
            
        #labels = labels.replace(0, -1)
        #labels = labels.fillna(0)

        #smiles_list, labels = df['smiles'], df['label']        
        #labels = labels.replace(0, -1)
        
        #labels, min_val, max_val = min_max_normalize(labels)

        data_list = []
        feature_sets = ("atomic_number", "basic", "cfid", "cgcnn")
        for i in range(len(smiles_list)):
            if i % 10000 == 0:
                print(i)

            smiles = smiles_list[i]
            print(f"smiles: {smiles}")
            
            #if has_isolated_hydrogens(smiles) == False and conformers_is_zero(smiles) == True :

            Graph_list = path_complex_mol(smiles, encoder_atom, encoder_bond, force_field=force_field)
            if Graph_list == False:
                continue

            else:
                if has_node_with_zero_in_degree(Graph_list):
                    continue
                
                else:
                    if datasets == 'dft':
                        st_gap = float(gap_series[i])
                        data_list.append([smiles, torch.tensor(labels.iloc[i]).float(), Graph_list, st_gap])
                    else:
                        data_list.append([smiles, torch.tensor(labels.iloc[i]).float(), Graph_list])



        #data_list = [['occr',albel,[c_size, features, edge_indexs],[g,liearn_g]],[],...,[]]

        print('Graph list was done!')

        splitter = ScaffoldSplitter().split(data_list, frac_train=train_ratio, frac_valid=vali_ratio, frac_test=test_ratio)
        
        print('splitter was done!')
        

        
        train_label = []
        train_graph_list = []
        train_gap = []
        for tmp_train_graph in splitter[0]:
            train_label.append(tmp_train_graph[1])
            train_graph_list.append(tmp_train_graph[2])
            if datasets == 'dft' and len(tmp_train_graph) > 3:
                train_gap.append(tmp_train_graph[3])


        valid_label = []
        valid_graph_list = []
        valid_gap = []
        for tmp_valid_graph in splitter[1]:
            valid_label.append(tmp_valid_graph[1])
            
            valid_graph_list.append(tmp_valid_graph[2])
            if datasets == 'dft' and len(tmp_valid_graph) > 3:
                valid_gap.append(tmp_valid_graph[3])

        test_label = []
        test_graph_list = []
        test_gap = []
        for tmp_test_graph in splitter[2]:
            test_label.append(tmp_test_graph[1])
            test_graph_list.append(tmp_test_graph[2])
            if datasets == 'dft' and len(tmp_test_graph) > 3:
                test_gap.append(tmp_test_graph[3])

        #batch_size = 256

        save_state = {
            'train_label': train_label,
            'train_graph_list': train_graph_list,
            'valid_label': valid_label,
            'valid_graph_list': valid_graph_list,
            'test_label': test_label,
            'test_graph_list': test_graph_list,
            'batch_size': batch_size,
            'shuffle': True,  
        }
        if datasets == 'dft':
            save_state['train_gap'] = train_gap
            save_state['valid_gap'] = valid_gap
            save_state['test_gap'] = test_gap
        torch.save(save_state, 'data/processed/'+ datafile +'.pth')



def message_func(edges):
    return {'feat': edges.data['feat']}

def reduce_func(nodes):
    num_edges = nodes.mailbox['feat'].size(1)  
    agg_feats = torch.sum(nodes.mailbox['feat'], dim=1) / num_edges  
    return {'agg_feats': agg_feats}

def update_node_features(g):
    g.send_and_recv(g.edges(), message_func, reduce_func)

    g.ndata['feat'] = torch.cat((g.ndata['feat'], g.ndata['agg_feats']), dim=1)

    return g





def train(model, device, train_loader, valid_loader, optimizer, epoch, label_dim, is_dft):
    model.train()

    total_train_loss = 0.0

    for batch_idx, data in enumerate(train_loader):
        optimizer.zero_grad()
        if len(data) == 3:
            y, graphs, gaps = data
            gaps = gaps.float()
        else:
            y, graphs = data
            gaps = None
        y = y.float()
        graph_list = update_node_features(graphs).to(device)
        node_features = graph_list.ndata['feat'].to(device)
        output = model(graph_list, node_features).cpu()

        pred_core = output[:, :label_dim]
        train_loss = loss_fn(pred_core, y)
        if is_dft:
            coefficient = 0.5
            pred_gap = output[:, label_dim]
            diff_gap = (pred_core[:, 0] - pred_core[:, 1])
            gap_consistency_loss = loss_fn(diff_gap, pred_gap)
            train_loss = train_loss + gap_consistency_loss
        train_loss.backward()
        optimizer.step()
        total_train_loss += float(train_loss.detach().cpu().item())

    model.eval()
    total_loss_val = 0.0
    for batch_idx, valid_data in enumerate(valid_loader):
        if len(valid_data) == 3:
            y, graphs, gaps = valid_data
        else:
            y, graphs = valid_data
        y = y.float()
        graph_list = update_node_features(graphs).to(device)
        node_features = graph_list.ndata['feat'].to(device)
        output = model(graph_list, node_features).cpu()

        pred_core = output[:, :label_dim]
        valid_loss = loss_fn(pred_core, y)
        if is_dft:
            pred_gap = output[:, label_dim]
            diff_gap = (pred_core[:, 0] - pred_core[:, 1])
            gap_consistency_loss = loss_fn(diff_gap, pred_gap)
            valid_loss = valid_loss + gap_consistency_loss
        total_loss_val += float(valid_loss.detach().cpu().item())

    print(f"Epoch {epoch}|Train Loss: {total_train_loss:.4f}| Vali Loss:{total_loss_val:.4f}")

    return total_train_loss, total_loss_val


def predicting(model, device, data_loader, label_dim):
    model.eval()

    total_abs_err = 0.0
    total_count = 0

    with torch.no_grad():
        for batch_idx, data in enumerate(data_loader):
            if len(data) == 3:
                y, graphs, _ = data
            else:
                y, graphs = data
            y = y.float()
            graph_list = update_node_features(graphs).to(device)
            node_features = graph_list.ndata['feat'].to(device)
            output = model(graph_list, node_features).cpu()

            diff = torch.abs(output[:, :label_dim] - y)
            total_abs_err += float(diff.sum().detach().cpu().item())
            total_count += int(y.numel())

    mae = total_abs_err / max(1, total_count)
    return mae


def predicting_gap_mae(model, device, data_loader, label_dim, is_dft):
    """Compute gap-related metrics for DFT dataset.

    Returns dict with keys:
        'diff_vs_true': MAE between (S1-T1) and true gap.
        'pred_vs_true': MAE between predicted gap output and true gap.
        'diff_vs_pred': MAE between (S1-T1) and predicted gap output.
    """
    if (not is_dft) or isinstance(data_loader, list) or data_loader is None:
        return {'diff_vs_true': None, 'pred_vs_true': None, 'diff_vs_pred': None}
    model.eval()
    total_abs_err = 0.0
    total_count = 0
    total_abs_pred_gap = 0.0
    total_abs_consistency = 0.0
    with torch.no_grad():
        for batch in data_loader:
            # Support (labels, graph) or (labels, graph, gaps)
            if len(batch) == 3:
                y, graphs, true_gap = batch
                true_gap = true_gap.float()
            else:
                y, graphs = batch
                true_gap = (y[:, 0] - y[:, 1]).float()
            graphs = update_node_features(graphs).to(device)
            node_features = graphs.ndata['feat'].to(device)
            pred = model(graphs, node_features).cpu()
            diff_gap = (pred[:, 0] - pred[:, 1]).float()
            pred_gap = pred[:, label_dim].float()
            diff = torch.abs(diff_gap - true_gap)
            total_abs_err += float(diff.sum().item())
            total_abs_pred_gap += float(torch.abs(pred_gap - true_gap).sum().item())
            total_abs_consistency += float(torch.abs(diff_gap - pred_gap).sum().item())
            total_count += int(diff.numel())
    if total_count == 0:
        return {'diff_vs_true': None, 'pred_vs_true': None, 'diff_vs_pred': None}
    denom = total_count
    return {
        'diff_vs_true': total_abs_err / denom,
        'pred_vs_true': total_abs_pred_gap / denom,
        'diff_vs_pred': total_abs_consistency / denom,
    }


def predicting_split(model, device, data_loader, label_dim):
    """Compute overall MAE and per-column MAE (e.g., S1/T1) for a loader.
    Returns (overall_mae: float, per_col_mae: List[float]).
    """
    model.eval()

    sum_abs_per_col = None
    n_samples = 0

    with torch.no_grad():
        for _, data in enumerate(data_loader):
            if len(data) == 3:
                y, graphs, _ = data
            else:
                y, graphs = data
            y = y.float()
            graph_list = update_node_features(graphs).to(device)
            node_features = graph_list.ndata['feat'].to(device)
            output = model(graph_list, node_features).cpu()

            diff = torch.abs(output[:, :label_dim] - y)
            if sum_abs_per_col is None:
                sum_abs_per_col = diff.sum(dim=0).detach().cpu().double()
            else:
                sum_abs_per_col += diff.sum(dim=0).detach().cpu().double()
            n_samples += diff.shape[0]

    if sum_abs_per_col is None:
        return 0.0, []

    per_col_mae = (sum_abs_per_col / max(1, n_samples)).tolist()
    overall_mae = float(sum_abs_per_col.sum().item() / max(1, n_samples * len(per_col_mae)))
    return overall_mae, per_col_mae



def parse_arguments():
    parser = argparse.ArgumentParser(description="help")


    parser.add_argument("--config", type=str, help="path")

    args = parser.parse_args()
    args.config = './config/c_path.yaml'
    if args.config:
        with open(args.config, "r") as config_file:
            config = yaml.safe_load(config_file)
        for key, value in config.items():
            setattr(args, key, value)

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
    """Initialize Weights & Biases run using fields from args.
    If wandb is not available, return None.
    """
    if not _WANDB_AVAILABLE:
        print("[wandb] not installed, skipping online logging. pip install wandb to enable.")
        return None

    project = getattr(args, 'wandb_project', 'DFT_pretictor')
    entity = getattr(args, 'wandb_entity', None)
    mode = getattr(args, 'wandb_mode', 'online')  # 'online' | 'offline' | 'disabled'
    run_name = getattr(args, 'wandb_run_name', None)
    if run_name_suffix:
        run_name = f"{run_name or 'run'}-{run_name_suffix}"

    run = wandb.init(project=project,
                     entity=entity,
                     name=run_name,
                     mode=mode,
                     config={k: v for k, v in vars(args).items() if k != 'config'})

    if model is not None:
        try:
            wandb.watch(model, log="gradients", log_freq=100)
        except Exception:
            pass
    return run


if __name__ == '__main__':
    
    #mp.set_start_method('spawn', force=True)
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('The code uses GPU...')
    else:
        device = torch.device('cpu')
        print('The code uses CPU!!!')

    # 设置种子
    seed = 42
    set_seed(seed)

    args = parse_arguments()
    for key, value in vars(args).items():
        if key != 'config':
            print(f"{key}: {value}")
    datafile = args.select_dataset
    batch_size = args.batch_size
    train_ratio = args.train_ratio
    vali_ratio = args.vali_ratio
    test_ratio = args.test_ratio
    target_map = {'tox21':12,'muv':17,'sider':27,'clintox':2,'bace':1,'bbbp':1,'hiv':1, 'dft':2}
    label_dim = target_map[datafile]
    is_dft = (datafile == 'dft')
    target_dim = label_dim + (1 if is_dft else 0)

    

    encoder_atom = args.encoder_atom
    encoder_bond = args.encoder_bond

    encode_dim = [0,0]
    encode_dim[0] = 92
    encode_dim[1] = 21
    

    
    # Prepare processed dataset if not cached
    creat_data(datafile, encoder_atom, encoder_bond, batch_size, train_ratio, vali_ratio, test_ratio, force_field=getattr(args, 'force_field', 'mmff'))
    #assert False,"check data"

    model_select = args.model_select
    loss_sclect = args.loss_sclect

    state = torch.load('data/processed/'+datafile+'.pth')

    loaded_train_dataset = CustomDataset(state['train_label'], state['train_graph_list'], state.get('train_gap'))
    loaded_valid_dataset = CustomDataset(state['valid_label'], state['valid_graph_list'], state.get('valid_gap'))
    loaded_test_dataset = CustomDataset(state['test_label'], state['test_graph_list'], state.get('test_gap'))
    
   

    loaded_train_loader = DataLoader(loaded_train_dataset, batch_size=batch_size, shuffle=state['shuffle'],num_workers=4, pin_memory=True, drop_last=True, collate_fn=collate_fn)
    if vali_ratio == 0.0:
        loaded_valid_loader = []
    else:
        loaded_valid_loader = DataLoader(loaded_valid_dataset, batch_size=batch_size, shuffle=state['shuffle'],num_workers=4, pin_memory=True, drop_last=False, collate_fn=collate_fn)

    loaded_test_loader = DataLoader(loaded_test_dataset, batch_size=batch_size, shuffle=state['shuffle'],num_workers=4, pin_memory=True, drop_last=False, collate_fn=collate_fn)

    #assert False,"check data"


    print('dataset was loaded!')

    print("length of training set:",len(loaded_train_dataset))
    print("length of validation set:",len(loaded_valid_dataset))
    print("length of testing set:",len(loaded_test_dataset))

    # Initialize wandb once per run
    wandb_run = init_wandb(args, model=None, run_name_suffix=f"{datafile}-{args.model_select}")
    if _WANDB_AVAILABLE and wandb_run is not None:
        # Log dataset meta
        wandb.config.update({
            'target_dim': target_dim,
            'train_size': len(loaded_train_dataset),
            'valid_size': len(loaded_valid_dataset),
            'test_size': len(loaded_test_dataset),
            'device': str(device),
        }, allow_val_change=True)
    
    iter = args.iter
    LR = args.LR
    NUM_EPOCHS = args.NUM_EPOCHS
    grid_feat = args.grid_feat
    num_layers = args.num_layers
    pooling = args.pooling

    All_AUC = []

    # Build output directory named by hyperparams for easy lookup
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(
        "runs",
        str(datafile),
        f"model-{args.model_select}_pool-{pooling}_grid-{grid_feat}_layers-{num_layers}_bs-{batch_size}_lr-{LR}_seed-{seed}",
        timestamp,
    )
    os.makedirs(run_dir, exist_ok=True)
    print(f"Checkpoints will be saved to: {run_dir}")

    start_time = time.time()

    for i in range(iter):
        # Create subdir per iter
        iter_dir = os.path.join(run_dir, f"iter-{i+1}")
        os.makedirs(iter_dir, exist_ok=True)
        
        AUC_list = []
        if model_select == 'ka_gnn':
            model = KA_GNN(in_feat=encode_dim[0]+encode_dim[1], hidden_feat=64, out_feat=32, out=target_dim, 
                           grid_feat=grid_feat, num_layers=num_layers, pooling = pooling, use_bias=True)

        elif model_select == 'ka_gnn_two':
            model = KA_GNN_two(in_feat=encode_dim[0]+encode_dim[1], hidden_feat=64, out_feat=32, out=target_dim, 
                               grid_feat=grid_feat, num_layers=num_layers, pooling = pooling, use_bias=True)
        
        elif model_select == 'mlp_sage':
            model = MLPGNN(in_feat=encode_dim[0]+encode_dim[1], hidden_feat=64, out_feat=32, out=target_dim, 
                           grid_feat=grid_feat, num_layers=num_layers, pooling = pooling, use_bias=True)

        elif model_select == 'mlp_sage_two':
            model = MLPGNN_two(in_feat=encode_dim[0]+encode_dim[1], hidden_feat=64, out_feat=32, out=target_dim, 
                               grid_feat=grid_feat, num_layers=num_layers, pooling = pooling, use_bias=True)

        elif model_select == 'kan_sage':
            model = KANGNN(in_feat=encode_dim[0]+encode_dim[1], hidden_feat=64, out_feat=32, out=target_dim, 
                           grid_feat=grid_feat, num_layers=num_layers, pooling = pooling, use_bias=True)

        elif model_select == 'kan_sage_two':
            model = KANGNN_two(in_feat=encode_dim[0]+encode_dim[1], hidden_feat=64, out_feat=32, out=target_dim, 
                           grid_feat=grid_feat, num_layers=num_layers, pooling = pooling, use_bias=True)

        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params}")
        if _WANDB_AVAILABLE and wandb_run is not None:
            wandb.log({
                'total_params': total_params,
                'iter': i + 1,
            })

        train_loss_dic = {}
        vali_loss_dic = {}

        model = model.to(device)
        if _WANDB_AVAILABLE and wandb_run is not None:
            try:
                wandb.watch(model, log="gradients", log_freq=100)
            except Exception:
                pass
        # Choose regression loss
        if loss_sclect == 'l1':
            loss_layer = nn.L1Loss(reduction='mean')
        elif loss_sclect == 'l2':
            loss_layer = nn.MSELoss(reduction='mean')
        elif loss_sclect == 'sml1':
            loss_layer = nn.SmoothL1Loss(reduction='mean')
        elif loss_sclect == 'bce':
            loss_layer = nn.BCELoss(reduction='mean')
        else:
            raise ValueError('No Found the Loss function!')
        # Bind to global name expected by train()/predicting()
        global loss_fn
        loss_fn = loss_layer
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        scheduler = StepLR(optimizer, step_size=50, gamma=0.9)
        best_metric = float('inf')  # lower is better for MAE
        for epoch in range(NUM_EPOCHS):
            train_loss,vali_loss = train(model, device, loaded_train_loader, loaded_valid_loader, optimizer, epoch + 1, label_dim, is_dft)
            #print("start to train")

            
            # Per-target MAE (S1/T1)
            MAE, per_mae = predicting_split(model, device, loaded_test_loader, label_dim)
            # Gap MAE across splits
            train_gap_metrics = predicting_gap_mae(model, device, loaded_train_loader, label_dim, is_dft)
            valid_gap_metrics = predicting_gap_mae(model, device, loaded_valid_loader, label_dim, is_dft) if vali_ratio > 0 else {'diff_vs_true': None, 'pred_vs_true': None, 'diff_vs_pred': None}
            test_gap_metrics = predicting_gap_mae(model, device, loaded_test_loader, label_dim, is_dft)
            # Log metrics every epoch
            if _WANDB_AVAILABLE and wandb_run is not None:
                current_lr = optimizer.param_groups[0]['lr'] if optimizer.param_groups else None
                log_dict = {
                    'epoch': epoch + 1,
                    'iter': i + 1,
                    'train_loss': _safe_float(train_loss),
                    'valid_loss': _safe_float(vali_loss),
                    'test_mae': _safe_float(MAE),
                    'lr': current_lr,
                }
                # Log per-task MAE if available (assume dim 0=S1, 1=T1)
                if isinstance(per_mae, (list, tuple)) and len(per_mae) >= 2:
                    log_dict['test_mae_s1'] = float(per_mae[0])
                    log_dict['test_mae_t1'] = float(per_mae[1])
                # Log ST Gap MAE on train/valid/test
                if train_gap_metrics['diff_vs_true'] is not None:
                    log_dict['train_gap_mae_from_diff'] = float(train_gap_metrics['diff_vs_true'])
                    log_dict['train_gap_mae_direct'] = float(train_gap_metrics['pred_vs_true'])
                    log_dict['train_gap_consistency'] = float(train_gap_metrics['diff_vs_pred'])
                if valid_gap_metrics['diff_vs_true'] is not None:
                    log_dict['valid_gap_mae_from_diff'] = float(valid_gap_metrics['diff_vs_true'])
                    log_dict['valid_gap_mae_direct'] = float(valid_gap_metrics['pred_vs_true'])
                    log_dict['valid_gap_consistency'] = float(valid_gap_metrics['diff_vs_pred'])
                if test_gap_metrics['diff_vs_true'] is not None:
                    log_dict['test_gap_mae_from_diff'] = float(test_gap_metrics['diff_vs_true'])
                    log_dict['test_gap_mae_direct'] = float(test_gap_metrics['pred_vs_true'])
                    log_dict['test_gap_consistency'] = float(test_gap_metrics['diff_vs_pred'])
                wandb.log(log_dict)
            
            if is_dft and test_gap_metrics['pred_vs_true'] is not None:
                print(f"Gap MAE (pred gap vs true): {test_gap_metrics['pred_vs_true']:.5f} | "
                      f"Gap MAE (S1-T1 vs true): {test_gap_metrics['diff_vs_true']:.5f} | "
                      f"Gap Consistency (S1-T1 vs pred gap): {test_gap_metrics['diff_vs_pred']:.5f}")

            # Select best by ST Gap MAE
            current_metric = None
            if test_gap_metrics['pred_vs_true'] is not None:
                current_metric = test_gap_metrics['pred_vs_true']
            elif test_gap_metrics['diff_vs_true'] is not None:
                current_metric = test_gap_metrics['diff_vs_true']
            else:
                current_metric = MAE
            if current_metric < best_metric:
                best_metric = current_metric
                logger.info(f'MAE: {best_metric:.5f}')
                formatted_number = "{:.5f}".format(best_metric)
                best_metric = float(formatted_number)
                AUC_list.append(best_metric)

                print(f"Epoch [{epoch+1}], Learning Rate: {scheduler.get_last_lr()}")
                if _WANDB_AVAILABLE and wandb_run is not None:
                    wandb.log({'best_gap_mae': best_metric, 'best_epoch': epoch + 1, 'iter': i + 1})

                # Save best checkpoint (model + optimizer + scheduler + meta)
                best_ckpt = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                    'epoch': epoch + 1,
                    'iter': i + 1,
                    'best_gap_mae': best_metric,
                    'args': vars(args),
                }
                best_named_path = os.path.join(iter_dir, f"best_epoch{epoch+1}_mae{best_metric:.5f}.pth")
                best_link_path = os.path.join(iter_dir, "best.pth")
                try:
                    torch.save(best_ckpt, best_named_path)
                    torch.save(best_ckpt, best_link_path)
                    print(f"[Checkpoint] Saved best to: {best_named_path}")
                    if _WANDB_AVAILABLE and wandb_run is not None:
                        try:
                            wandb.save(best_named_path)
                        except Exception:
                            pass
                except Exception as e:
                    print(f"[Checkpoint] Failed to save best checkpoint: {e}")
            
            #scheduler.step()
                

                
                
        
            if epoch % 10 == 0:
                #MAE_list.append(best_MAE)
                print("-------------------------------------------------------")
                print("epoch:",epoch)
                print('best_MAE:', best_metric)
            
            if epoch == NUM_EPOCHS-1:
                print(f"the best result up to {i+1}-loop is {best_metric:.4f}.")
                formatted_number = "{:.5f}".format(best_metric)
                All_AUC.append(best_metric)
                if _WANDB_AVAILABLE and wandb_run is not None:
                    # mark the best of this iter
                    wandb.log({
                        'iter_best_mae': best_metric,
                        'iter': i + 1,
                    })
                
    # Removed saving the last model; best checkpoints are already saved per iter

    #mean_value = statistics.mean(All_AUC)

    #std_dev = statistics.stdev(All_AUC)
    
    #print("mean:", mean_value)
    #print("std:", std_dev)
    if _WANDB_AVAILABLE and wandb_run is not None:
        
        wandb_run.finish()

