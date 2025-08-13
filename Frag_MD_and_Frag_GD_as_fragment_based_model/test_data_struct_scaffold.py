import argparse

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from MCMG_utils.data_structs_scaffold import *
from models.model_MCMG import transformer_RL
from torch.optim import Adam
from MCMG_utils.Optim import ScheduledOptim
from MCMG_utils.early_stop.pytorchtools import EarlyStopping
import time
import wandb
from fragment_utils.mol_utils import split_molecule, join_fragments

def decode_molecule(enc):
    fs = [Chem.MolFromSmiles(x) for x in enc]
    try:
        fs = join_fragments(fs)
    except Exception as e:
        fs = None
    return fs


# 创建词汇表
decorator_voc = DecoratorVocabulary.from_files(
    "./data1/scaffold_vocab.csv",
    "./data1/decoration_vocab.csv"
)

dataset = DecoratorDataset("./data1/train.csv", decorator_voc, smiles_col='SMILES')


scaffold_with_con, decoration = dataset[0]
print("Scaffold shape:", scaffold_with_con.shape)
print("Decoration shape:", decoration.shape)
print("Scaffold with decoration:", scaffold_with_con)
print("Decoration:", decoration)

scaffold_only = scaffold_with_con[2:]
print(f"Scaffold only shape: {scaffold_only.shape}, Scaffold only: {scaffold_only}")
  # 移除填充的0
decoded_scaffold = decorator_voc.scaffold_vocabulary.reversed_vocab[int(scaffold_only.item())]

con = scaffold_with_con[:2]  # 前两个元素是连接符
decoded_con = decorator_voc.scaffold_vocabulary.reversed_vocab[int(con[0].item())] + decorator_voc.scaffold_vocabulary.reversed_vocab[int(con[1].item())]

chars = []
chars.append(decorator_voc.scaffold_vocabulary.reversed_vocab[int(scaffold_only.item())])
for idx in decoration[:-1]:
    chars.append(decorator_voc.decoration_vocabulary.reversed_vocab[int(idx.item())])
decoded_smiles =  Chem.MolToSmiles(decode_molecule(chars))
print("Decoded condition:", decoded_con)
print("Decoded scaffold:", decoded_scaffold)
print("Decoded decoration:", decoded_smiles)