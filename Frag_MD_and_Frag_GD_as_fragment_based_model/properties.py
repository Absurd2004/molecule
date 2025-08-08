#!/usr/bin/env python
from __future__ import print_function, division
import numpy as np
from rdkit import Chem
from rdkit import rdBase
from rdkit.Chem import AllChem
from rdkit import DataStructs
import rdkit.Chem.QED as QED
import scripts.sascorer as sascorer
import pickle
import pandas as pd
from rdkit.Chem import Descriptors
from rdkit.Chem import rdFingerprintGenerator


#import tensorflow as tf 
import deepchem as dc 
from models.graphConvModel_pytorch import GraphConvModel
import os

import warnings
warnings.filterwarnings('ignore')
import logging
logger = logging.getLogger()
logger.setLevel(logging.ERROR)
import torch


rdBase.DisableLog('rdApp.error')

import gc

class gsk3_model():
    """Scores based on an ECFP classifier for activity."""

    kwargs = ["clf_path"]
    clf_path = 'data/gsk3/gsk3.pkl'

    def __init__(self):
        with open(self.clf_path, "rb") as f:
            self.clf = pickle.load(f)

    def __call__(self, smiles_list):
        fps = []
        mask = []
        for i,smiles in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smiles)
            mask.append( int(mol is not None) )
            fp = gsk3_model.fingerprints_from_mol(mol) if mol else np.zeros((1, 2048))
            fps.append(fp)

        fps = np.concatenate(fps, axis=0)
        scores = self.clf.predict_proba(fps)[:, 1]
        scores = scores * np.array(mask)
        return np.float32(scores)

    @classmethod
    def fingerprints_from_mol(cls, mol):  # use ECFP4
        mfpgen = rdFingerprintGenerator.GetMorganGenerator(
            radius=2,
            fpSize=2048,
            countSimulation=False,
        )

        fingerprint = mfpgen.GetFingerprint(mol)
        #features_vec = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        features = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(fingerprint, features)
        return features.reshape(1, -1)
    

class jnk3_model():
    """Scores based on an ECFP classifier for activity."""

    kwargs = ["clf_path"]
    clf_path = 'data/jnk3/jnk3.pkl'

    def __init__(self):
        with open(self.clf_path, "rb") as f:
            self.clf = pickle.load(f)

    def __call__(self, smiles_list):
        fps = []
        mask = []
        for i,smiles in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smiles)
            mask.append( int(mol is not None) )
            fp = jnk3_model.fingerprints_from_mol(mol) if mol else np.zeros((1, 2048))
            fps.append(fp)

        fps = np.concatenate(fps, axis=0)
        scores = self.clf.predict_proba(fps)[:, 1]
        scores = scores * np.array(mask)
        return np.float32(scores)

    @classmethod
    def fingerprints_from_mol(cls, mol):  # use ECFP4
        mfpgen = rdFingerprintGenerator.GetMorganGenerator(
            radius=2,
            fpSize=2048,
            countSimulation=False,
        )

        fingerprint = mfpgen.GetFingerprint(mol)
        #features_vec = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        features = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(fingerprint, features)
        return features.reshape(1, -1)


class drd2_model():
    """Scores based on an ECFP classifier for activity."""

    kwargs = ["clf_path"]
    # clf_path = '/apdcephfs/private_jikewang/W4_reduce_RL/data/drd2/drd2.pkl'
    clf_path = 'data/drd2/drd2.pkl'


    def __init__(self):
        with open(self.clf_path, "rb") as f:
            self.clf = pickle.load(f)

    def __call__(self, smiles_list):
        fps = []
        mask = []
        for i,smiles in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smiles)
            mask.append( int(mol is not None) )
            fp = drd2_model.fingerprints_from_mol(mol) if mol else np.zeros((1, 2048))
            fps.append(fp)

        fps = np.concatenate(fps, axis=0)
        scores = self.clf.predict_proba(fps)[:, 1]
        scores = scores * np.array(mask)
        return np.float32(scores)

    @classmethod
    def fingerprints_from_mol(cls, mol):  # use ECFP4
        mfpgen = rdFingerprintGenerator.GetMorganGenerator(
            radius=2,
            fpSize=2048,
            countSimulation=False,
        )

        fingerprint = mfpgen.GetFingerprint(mol)
        #features_vec = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        features = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(fingerprint, features)
        return features.reshape(1, -1)

class pre_model():
    """Scores based on an ECFP classifier for ST_energy and absorption wavelength."""
    def __init__(self):
        self.model_dir = "./models/L_model"
        self.model = GraphConvModel(n_tasks = 2,
                            graph_conv_layers = [512, 512, 512, 512], 
                            dense_layers = [128, 128, 128],
                            dropout = 0.01,
                            learning_rate = 0.001,
                            batch_size = 10,
                            model_dir = self.model_dir)
        self.model.restore(self.model.get_checkpoints()[-1])
    def __call__(self,smiles_list):
        # feature SMILES
        score1 = []
        score2 = []
        graph_featurizer = dc.feat.graph_features.ConvMolFeaturizer()
        for i,smiles in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                graphs = graph_featurizer.featurize(mol)
                data = dc.data.NumpyDataset(graphs)
                try:
                    scores = self.model.predict(data)
                    score1.append(scores[0][0])
                    score2.append(scores[0][1]*27.2114)                    
                    tf.keras.backend.clear_session()
                    gc.collect()
                except:
                    score1.append(3)
                    score2.append(5)
            else:
                score1.append(3)
                score2.append(5)
         #预测的是能极差和波长的能级，所以，还并不是score, score打分构建如下
         #score1是能级差，选择小于0.2 ev分子, 0.2 ev的分子的score2是
         #score2是吸收波长的能级，300 nm的吸收是4.13 ev, 800 nm的吸收是1.55 ev, 600 nm的吸收是2.07 eV，差值是2.58 ev, e.g.500 nm的吸收是2.48 ev，对应的score2为0.63
        st_energy = []
        abs_energy=[]
        for x in score1:
            if x < 0.1 and x >= 0:
                st_energy.append(np.array(4.0))
            elif x < 0.2 and x >= 0.1:
                st_energy.append(np.array(2.0))
            elif x < 0.3 and x >=0.2:
                st_energy.append(np.array(1.0))
            else:
                st_energy.append(np.array(0.0)) 
        for x in score2:
            if x < 2.07 and x >= 1.55:
                abs_energy.append(np.array(2.0))
            elif x > 2.07 and x <= 2.48:
                abs_energy.append(np.array(1.0))
            else:
                abs_energy.append(np.array(0.0))
    
        # st_energy = np.array([float(2/(1 + np.exp(-x+0.2))) for x in score1],
        #               dtype=np.float32)

        # abs_energy = np.array([float((4.13-x)/2.58) for x in score2],
        #               dtype=np.float32)
        return np.float32(st_energy),np.float32(abs_energy)     
        
        # mols = [Chem.MolFromSmiles(smi) for smi in smiles_list]
        # graphs = graph_featurizer.featurize(mols)
       
        # data = dc.data.NumpyDataset([graphs])
        # # predict with the mol data 
        # pred = self.model.predict(data)
        # # print(f"{list_names[-1]} ST gap: {pred[0][0]:.4f} eV | HL gap: {pred[0][1]*27.2114:.4f} eV")
        # return pred[0][0], pred[0][1]

class pre_model_f1():
    """Scores based on an ECFP classifier for ST_energy and absorption wavelength."""
    def __init__(self):
        self.model_dir = "./models/f1_model"
        self.model = GraphConvModel(n_tasks = 2,
                        graph_conv_layers = [512, 512, 512,512], 
                        dense_layers = [128, 128, 128],
                        dropout = 0.01,
                        mode = 'regression',
                        learning_rate = 0.001,
                        batch_size = 32,
                        uncertainty = True,
                        model_dir = self.model_dir)
        self.model.restore(self.model.get_checkpoints()[-1])


    def __call__(self,smiles_list):
        # feature SMILES
        score1 = []
        score2 = []
        graph_featurizer = dc.feat.graph_features.ConvMolFeaturizer()
        for i,smiles in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                graphs = graph_featurizer.featurize(mol)
                data = dc.data.NumpyDataset(graphs)
                try:
                    scores = self.model.predict(data)
                    score1.append(scores[0][0]*0.6635+1.1721)
                    score2.append(scores[0][1]*0.9691+3.2070)                    
                    tf.keras.backend.clear_session()
                    gc.collect()
                except:
                    score1.append(3)
                    score2.append(5)
            else:
                score1.append(3)
                score2.append(5)
         #预测的是能极差和波长的能级，所以，还并不是score, score打分构建如下
         #score1是能级差，选择小于0.2 ev分子, 0.2 ev的分子的score2是
         #score2是吸收波长的能级，300 nm的吸收是4.13 ev, 800 nm的吸收是1.55 ev, 600 nm的吸收是2.07 eV，差值是2.58 ev, e.g.500 nm的吸收是2.48 ev，对应的score2为0.63
        st_energy = []
        abs_energy=[]
        for x in score1:
            if x < 0.1 and x >= 0:
                st_energy.append(np.array(4.0))
            elif x < 0.2 and x >= 0.1:
                st_energy.append(np.array(2.0))
            elif x < 0.3 and x >=0.2:
                st_energy.append(np.array(1.0))
            else:
                st_energy.append(np.array(0.0)) 
        for x in score2:
            if x < 2.07 and x >= 1.55:
                abs_energy.append(np.array(2.0))
            elif x > 2.07 and x <= 2.48:
                abs_energy.append(np.array(1.0))
            else:
                abs_energy.append(np.array(0.0))
    
        # st_energy = np.array([float(2/(1 + np.exp(-x+0.2))) for x in score1],
        #               dtype=np.float32)

        # abs_energy = np.array([float((4.13-x)/2.58) for x in score2],
        #               dtype=np.float32)
        return np.float32(st_energy),np.float32(abs_energy)     
        
        # mols = [Chem.MolFromSmiles(smi) for smi in smiles_list]
        # graphs = graph_featurizer.featurize(mols)
       
        # data = dc.data.NumpyDataset([graphs])
        # # predict with the mol data 
        # pred = self.model.predict(data)
        # # print(f"{list_names[-1]} ST gap: {pred[0][0]:.4f} eV | HL gap: {pred[0][1]*27.2114:.4f} eV")
        # return pred[0][0], pred[0][1]
class pre_model_f2():
    """Scores based on an ECFP classifier for ST_energy and absorption wavelength."""
    def __init__(self):
        self.model_dir = "./prediction_models"
        model_path = os.path.join(self.model_dir, "best.pt")
        self.model = GraphConvModel(n_tasks = 2,
                        number_input_features=[75, 64],  
                        graph_conv_layers = [64,64], 
                        dense_layer_size=128,
                        dropout = 0.01,
                        mode = 'regression',
                        number_atom_features=75,
                        learning_rate = 0.001,
                        batch_size = 32,
                        model_dir = self.model_dir)
        #self.model.restore(self.model.get_checkpoints()[-1])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(model_path, map_location=device)
        self.model.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.model.eval()

        self.norm_params = self._load_or_calculate_normalization_params()
    def _load_or_calculate_normalization_params(self):
        """加载或计算标准化参数"""
        import pandas as pd
        import pickle
        
        # 标准化参数保存路径
        norm_params_path = os.path.join(self.model_dir, "norm_params.pkl")
        
        # 尝试加载已保存的标准化参数
        if os.path.exists(norm_params_path):
            try:
                with open(norm_params_path, 'rb') as f:
                    norm_params = pickle.load(f)
                #print(f"Loaded normalization parameters from {norm_params_path}")
                #print(f"  ST Gap - Mean: {norm_params['y_mean'][0]:.4f}, Std: {norm_params['y_std'][0]:.4f}")
                #print(f"  HL Gap - Mean: {norm_params['y_mean'][1]:.4f}, Std: {norm_params['y_std'][1]:.4f}")
                return norm_params
            except Exception as e:
                print(f"Error loading normalization parameters: {e}")
                print("Recalculating normalization parameters...")
        
        # 如果文件不存在或加载失败，重新计算
        print("Calculating normalization parameters from training data...")
        
        # 加载训练数据文件
        file_list = ["./dftdata/Photosensitizers_DA.csv", "./dftdata/Photosensitizers_DAD.csv"]
        
        all_st_gaps = []
        all_hl_gaps = []
        
        for csv_path in file_list:
            try:
                df = pd.read_csv(csv_path)
                # 检查必要的列是否存在
                required_columns = ['SMILES', 'ST Gap', 'HL Gap']
                if all(col in df.columns for col in required_columns):
                    # 过滤掉包含NaN值的行
                    df_clean = df.dropna(subset=required_columns)
                    
                    # 收集数据
                    all_st_gaps.extend(df_clean['ST Gap'].values)
                    all_hl_gaps.extend(df_clean['HL Gap'].values)
            except Exception as e:
                print(f"Error loading {csv_path}: {str(e)}")
                continue
        
        # 转换为numpy数组并计算标准化参数
        targets = np.column_stack([np.array(all_st_gaps), np.array(all_hl_gaps)])
        y_mean = np.mean(targets, axis=0)
        y_std = np.std(targets, axis=0)
        
        norm_params = {
            'y_mean': y_mean,
            'y_std': y_std
        }
        
        # 保存标准化参数
        try:
            os.makedirs(self.model_dir, exist_ok=True)
            with open(norm_params_path, 'wb') as f:
                pickle.dump(norm_params, f)
            print(f"Normalization parameters saved to {norm_params_path}")
        except Exception as e:
            print(f"Warning: Could not save normalization parameters: {e}")
        
        print(f"Calculated normalization parameters:")
        print(f"  ST Gap - Mean: {y_mean[0]:.4f}, Std: {y_std[0]:.4f}")
        print(f"  HL Gap - Mean: {y_mean[1]:.4f}, Std: {y_std[1]:.4f}")
        
        return norm_params
    def _denormalize_predictions(self, predictions):
        """将标准化的预测值转换回原始分布"""
        y_mean = self.norm_params['y_mean']
        y_std = self.norm_params['y_std']
        
        # 反标准化：y_real = y_normalized * std + mean
        predictions_real = predictions * y_std + y_mean
        return predictions_real


    def __call__(self,smiles_list):
        # feature SMILES
        score1 = []
        score2 = []
        graph_featurizer = dc.feat.graph_features.ConvMolFeaturizer()
        for i,smiles in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                graphs = graph_featurizer.featurize([mol])

                data = dc.data.NumpyDataset(graphs)
                try:
                    scores_normalized = self.model.predict(data)
                    scores_real = self._denormalize_predictions(scores_normalized)
                    score1.append(scores_real[0][0])  # ST Gap
                    score2.append(scores_real[0][1])  #              

                    gc.collect()
                except Exception as e:
                    # 预测失败
                    print(f"Prediction failed for {smiles}: {e}")
                    score1.append(3)
                    score2.append(5)
            else:
                # 分子解析失败
                score1.append(3)
                score2.append(5)
         #预测的是能极差和波长的能级，所以，还并不是score, score打分构建如下
         #score1是能级差，选择小于0.2 ev分子, 0.2 ev的分子的score2是
         #score2是吸收波长的能级，300 nm的吸收是4.13 ev, 800 nm的吸收是1.55 ev, 600 nm的吸收是2.07 eV，差值是2.58 ev, e.g.500 nm的吸收是2.48 ev，对应的score2为0.63
        st_energy = []
        HL_energy=[]
        for x in score1:
            if x < 0.1 and x >= 0:
                st_energy.append(np.array(4.0))
            elif x < 0.2 and x >= 0.1:
                st_energy.append(np.array(2.0))
            elif x < 0.3 and x >=0.2:
                st_energy.append(np.array(1.0))
            else:
                st_energy.append(np.array(0.0)) 
        for x in score2:
            if x < 2.51 and x >= 1.33:
                HL_energy.append(np.array(2.0))
            elif 2.70 >=x >= 2.51 or 1.1 <= x < 1.33:
                HL_energy.append(np.array(1.0))
            else:
                HL_energy.append(np.array(0.0))
    
        # st_energy = np.array([float(2/(1 + np.exp(-x+0.2))) for x in score1],
        #               dtype=np.float32)

        # abs_energy = np.array([float((4.13-x)/2.58) for x in score2],
        #               dtype=np.float32)
        return np.float32(st_energy),np.float32(HL_energy),np.float32(score1),np.float32(score2)
        
        # mols = [Chem.MolFromSmiles(smi) for smi in smiles_list]
        # graphs = graph_featurizer.featurize(mols)
       
        # data = dc.data.NumpyDataset([graphs])
        # # predict with the mol data 
        # pred = self.model.predict(data)
        # # print(f"{list_names[-1]} ST gap: {pred[0][0]:.4f} eV | HL gap: {pred[0][1]*27.2114:.4f} eV")
        # return pred[0][0], pred[0][1]

class qed_func():

    def __call__(self, smiles_list):
        scores = []
        for smiles in smiles_list:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    qed =0
                else:
                    try:
                        qed = QED.qed(mol)
                    except:
                        qed = 0
            except:
                qed = 0
            scores.append(qed)
        return np.float32(scores)


class sa_func():

    def __call__(self, smiles_list):
        scores = []
        for smiles in smiles_list:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    scores.append(100)
                else:
                    scores.append(sascorer.calculateScore(mol))
            except:
                scores.append(100)
        return np.float32(scores)

class logp_func():

    def __call__(self, smiles_list):
        scores = []
        for smiles in smiles_list:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    scores.append(0.0)  # 默认惩罚值
                else:
                    try:
                        # 计算 logP
                        log_p = Descriptors.MolLogP(mol)

                        adjusted_score = 1 / (1 + np.exp(-log_p))
                        scores.append(adjusted_score)
                    except Exception as e:
                        print(f"Error processing {smiles}: {e}")
                        scores.append(0.0)  # 计算失败时的默认值
            except:
                scores.append(0.0)
        return np.float32(scores)
class symmetry_func():
    def __call__(self, smiles_list):
        scores = []
        for smiles in smiles_list:
            try:
                mol = Chem.MolFromSmiles(smiles)

                
                if mol is None or mol.GetNumAtoms() == 0:
                    scores.append(0.0)
                    continue

                num_atoms_without_h = mol.GetNumAtoms() 
                
                # 添加氢原子以获得完整的对称性信息
                mol_with_h = Chem.AddHs(mol)

                num_atoms = mol_with_h.GetNumAtoms()
                
                # 处理单原子分子
                if num_atoms == 1:
                    scores.append(0.0)  # 单原子给0分，不鼓励
                    continue
                
                # 获取原子的对称等价类
                ranks = Chem.CanonicalRankAtoms(mol_with_h, breakTies=False)
                
                # 计算对称性指标
                import numpy as np
                counts = np.bincount(ranks)
                
                # 方法1：最大对称类的比例
                max_symmetry_ratio = counts.max() / num_atoms

                # 原子数惩罚：小分子的对称性分数会被降低
                if num_atoms_without_h < 20:
                    # 原子数少于20个时，对称性分数会被大幅降低
                    atom_penalty = num_atoms_without_h / 20.0  # 0.1-0.9的惩罚因子
                elif num_atoms_without_h < 40:
                    # 原子数20-40之间时，轻微惩罚
                    atom_penalty = 0.9 + (num_atoms_without_h - 20) * 0.01  # 0.9-1.0
                else:
                    # 原子数大于20时，不惩罚
                    atom_penalty = 1.0
                
                # 方法2：对称性熵（可选）
                # probs = counts / mol_with_h.GetNumAtoms()
                # probs = probs[probs > 0]  # 移除零概率
                # symmetry_entropy = -np.sum(probs * np.log2(probs))
                # normalized_entropy = symmetry_entropy / np.log2(mol_with_h.GetNumAtoms())

                final_score = max_symmetry_ratio * atom_penalty
                
                scores.append(float(final_score))
                
            except Exception as e:
                # print(f"Error processing {smiles}: {e}")
                scores.append(0.0)
        
        return np.float32(scores)
def get_scoring_function(prop_name):
    """Function that initializes and returns a scoring function by name"""
    if prop_name == 'st_abs':
        return pre_model()
    # elif prop_name == 'gsk3':
    #     return gsk3_model()
    elif prop_name == 'qed':
        return qed_func()
    elif prop_name == 'sa':
        return sa_func()
    elif prop_name == 'st_abs_f1':
        return pre_model_f1()
    elif prop_name == 'st_hl_f2':
        return pre_model_f2()
    elif prop_name == 'logP':
        return logp_func()
    elif prop_name == 'symmetry':
        return symmetry_func()


def multi_scoring_functions_one_hot_drd(data, function_list):
    funcs = [get_scoring_function(prop) for prop in function_list]
    props = np.array([func(data) for func in funcs])

    props = pd.DataFrame(props).T
    props.columns = function_list

    scoring_sum = condition_convert(props).values.sum(1)

    # scoring_sum = props.sum(axis=0)

    return scoring_sum

def multi_scoring_functions_one_hot_dual(data, function_list):
    funcs = [get_scoring_function(prop) for prop in function_list]
    # props = np.array([func(data) for func in funcs])
    props =[]
    score1,score2 = funcs[0](data)
    qed = funcs[1](data)
    sa = funcs[2](data)
    
    score1 = np.array([float(x >= 1) for x in score1],
                      dtype=np.float32) 
    score2 = np.array([float(x >= 1) for x in score2],
                      dtype=np.float32) 
    qed = np.array([float(x > 0.38) for x in qed],
                      dtype=np.float32) 
    sa = np.array([float(x < 4.0) for x in sa],
                      dtype=np.float32) 
    
    
    props.append(score1)
    props.append(score2)
    props.append(qed)
    props.append(sa)
    
    # props = pd.DataFrame(props).T
    # props.columns = ['st','abs', 'qed', 'sa']
    props = np.array([x.tolist() for x in props]) 
    props = props.T
    scoring_sum = props.sum(1)
     # scoring_sum = props.sum(axis=0)

    return scoring_sum

def multi_scoring_functions_one_hot_dual_test(data, function_list):
    funcs = [get_scoring_function(prop) for prop in function_list]
    # props = np.array([func(data) for func in funcs])
    props =[]
    score1,score2 = funcs[0](data)
    logP = funcs[1](data)
    qed = funcs[2](data)
    sa = funcs[3](data)

    score1 = np.array([float(x >= 1) for x in score1],
                      dtype=np.float32) 
    score2 = np.array([float(x >= 1) for x in score2],
                      dtype=np.float32) 
    
    logP = np.array([float(x >= 0.9) for x in logP],
                      dtype=np.float32) 
    
    qed = np.array([float(x > 0.38) for x in qed],
                      dtype=np.float32) 
    
    sa = np.array([float(x < 4.0) for x in sa],
                      dtype=np.float32) 
    
    
    props.append(score1)
    props.append(score2)
    props.append(logP)
    props.append(qed)
    props.append(sa)
    
    # props = pd.DataFrame(props).T
    # props.columns = ['st','abs', 'qed', 'sa']
    props = np.array([x.tolist() for x in props]) 
    props = props.T
    scoring_sum = props.sum(1)
     # scoring_sum = props.sum(axis=0)

    return scoring_sum
def multi_scoring_functions_one_hot_dual_logP(data, function_list):
    funcs = [get_scoring_function(prop) for prop in function_list]
    # props = np.array([func(data) for func in funcs])
    props =[]
    logP = funcs[0](data)
    qed = funcs[1](data)
    sa = funcs[2](data)
    
    logP = np.array([float(x >= 0.9) for x in logP],
                      dtype=np.float32) 

    qed = np.array([float(x > 0.38) for x in qed],
                      dtype=np.float32) 
    sa = np.array([float(x < 4.0) for x in sa],
                      dtype=np.float32) 
    
    
    props.append(logP)
    props.append(qed)
    props.append(sa)
    
    # props = pd.DataFrame(props).T
    # props.columns = ['st','abs', 'qed', 'sa']
    props = np.array([x.tolist() for x in props]) 
    props = props.T
    scoring_sum = props.sum(1)
     # scoring_sum = props.sum(axis=0)

    return scoring_sum
def multi_scoring_functions_one_hot_jnk_gsk(data, function_list):
    funcs = [get_scoring_function(prop) for prop in function_list]
    props = np.array([func(data) for func in funcs])

    props = pd.DataFrame(props).T
    props.columns = function_list

    scoring_sum = condition_convert_jnk_gsk(props).values.sum(1)

    # scoring_sum = props.sum(axis=0)

    return scoring_sum

def multi_scoring_functions_one_hot_jnk_qed_sa(data, function_list):
    funcs = [get_scoring_function(prop) for prop in function_list]
    props = np.array([func(data) for func in funcs])

    props = pd.DataFrame(props).T
    props.columns = function_list

    scoring_sum = condition_convert_jnk_qed_sa(props).values.sum(1)

    # scoring_sum = props.sum(axis=0)

    return scoring_sum

def multi_scoring_functions_one_hot_gsk_qed_sa(data, function_list):
    funcs = [get_scoring_function(prop) for prop in function_list]
    props = np.array([func(data) for func in funcs])

    props = pd.DataFrame(props).T
    props.columns = function_list

    scoring_sum = condition_convert_gsk_qed_sa(props).values.sum(1)

    # scoring_sum = props.sum(axis=0)

    return scoring_sum

def condition_convert(con_df):
    # convert to 0, 1
    con_df['drd2'][con_df['drd2'] >= 0.5] = 1
    con_df['drd2'][con_df['drd2'] < 0.5] = 0
    con_df['qed'][con_df['qed'] >= 0.6] = 1
    con_df['qed'][con_df['qed'] < 0.6] = 0
    con_df['sa'][con_df['sa'] <= 4.0] = 1
    con_df['sa'][con_df['sa'] > 4.0] = 0
    return con_df

def condition_convert_st_abs(con_df):
    # convert to 0, 1
    con_df['st'][con_df['st'] >= 1] = 1
    con_df['st'][con_df['st'] < 1] = 0
    con_df['abs'][con_df['abs'] >= 0.63] = 1
    con_df['abs'][con_df['abs'] < 0.63] = 0
    con_df['qed'][con_df['qed'] >= 0.6] = 1
    con_df['qed'][con_df['qed'] < 0.6] = 0
    con_df['sa'][con_df['sa'] <= 4.0] = 1
    con_df['sa'][con_df['sa'] > 4.0] = 0
    return con_df

def condition_convert_dual(con_df):
    # convert to 0, 1
    con_df['jnk3'][con_df['jnk3'] >= 0.5] = 1
    con_df['jnk3'][con_df['jnk3'] < 0.5] = 0
    con_df['gsk3'][con_df['gsk3'] >= 0.5] = 1
    con_df['gsk3'][con_df['gsk3'] < 0.5] = 0
    con_df['qed'][con_df['qed'] >= 0.6] = 1
    con_df['qed'][con_df['qed'] < 0.6] = 0
    con_df['sa'][con_df['sa'] <= 4.0] = 1
    con_df['sa'][con_df['sa'] > 4.0] = 0
    return con_df

def condition_convert_jnk_gsk(con_df):
    # convert to 0, 1
    con_df['jnk3'][con_df['jnk3'] >= 0.5] = 1
    con_df['jnk3'][con_df['jnk3'] < 0.5] = 0
    con_df['gsk3'][con_df['gsk3'] >= 0.5] = 1
    con_df['gsk3'][con_df['gsk3'] < 0.5] = 0
    #con_df['qed'][con_df['qed'] >= 0.6] = 1
    #con_df['qed'][con_df['qed'] < 0.6] = 0
    #con_df['sa'][con_df['sa'] <= 4.0] = 1
    #con_df['sa'][con_df['sa'] > 4.0] = 0
    return con_df

def condition_convert_jnk_qed_sa(con_df):
    # convert to 0, 1
    con_df['jnk3'][con_df['jnk3'] >= 0.5] = 1
    con_df['jnk3'][con_df['jnk3'] < 0.5] = 0
    #con_df['gsk3'][con_df['gsk3'] >= 0.5] = 1
    #con_df['gsk3'][con_df['gsk3'] < 0.5] = 0
    con_df['qed'][con_df['qed'] >= 0.6] = 1
    con_df['qed'][con_df['qed'] < 0.6] = 0
    con_df['sa'][con_df['sa'] <= 4.0] = 1
    con_df['sa'][con_df['sa'] > 4.0] = 0
    return con_df

def condition_convert_gsk_qed_sa(con_df):
    # convert to 0, 1
    #con_df['jnk3'][con_df['jnk3'] >= 0.5] = 1
    #con_df['jnk3'][con_df['jnk3'] < 0.5] = 0
    con_df['gsk3'][con_df['gsk3'] >= 0.5] = 1
    con_df['gsk3'][con_df['gsk3'] < 0.5] = 0
    con_df['qed'][con_df['qed'] >= 0.6] = 1
    con_df['qed'][con_df['qed'] < 0.6] = 0
    con_df['sa'][con_df['sa'] <= 4.0] = 1
    con_df['sa'][con_df['sa'] > 4.0] = 0
    return con_df

if __name__ == "__main__":
    import sys
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--prop', required=True)

    args = parser.parse_args()
    funcs = [get_scoring_function(prop) for prop in args.prop.split(',')]

    data = [line.split()[:2] for line in sys.stdin]
    all_x, all_y = zip(*data)
    props = [func(all_y) for func in funcs]

    col_list = [all_x, all_y] + props
    for tup in zip(*col_list):
        print(*tup)
