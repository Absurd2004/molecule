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


import tensorflow as tf 
import deepchem as dc 
from models.graphConvModel import GraphConvModel

import warnings
warnings.filterwarnings('ignore')
import logging
logger = logging.getLogger()
logger.setLevel(logging.ERROR)


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
        features_vec = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        features = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(features_vec, features)
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
        features_vec = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        features = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(features_vec, features)
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
        features_vec = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        features = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(features_vec, features)
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
         #score1是能级差，选择小于0.2 ev分子 
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
