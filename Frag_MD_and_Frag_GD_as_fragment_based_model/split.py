import os
import pandas as pd
from rdkit import Chem
from tqdm import tqdm
from fragment_utils.mol_utils import split_molecule
import rdkit.Chem.QED as QED
import scripts.sascorer as sascorer

input_path = "./data1/SMILES.csv"
df = pd.read_csv(input_path)

# 先过滤无法正常切割的分子
valid_mask = []
for smi in tqdm(df["SMILES"].astype(str).tolist(), desc="校验可切割性"):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        valid_mask.append(False)
        continue
    try:
        _ = split_molecule(mol)
        valid_mask.append(True)
    except Exception:
        valid_mask.append(False)

df_valid = df[valid_mask].reset_index(drop=True)
dropped = len(df) - len(df_valid)
print(f"过滤无效/不可切割分子: 共删除 {dropped} 条，保留 {len(df_valid)} 条")

# 9:1 随机划分（保留所有列）
train = df_valid.sample(frac=0.9, random_state=42)
valid = df_valid.drop(train.index)

# 重排行顺序索引并重新生成 Index 列（从 1 开始）
train = train.reset_index(drop=True)
valid = valid.reset_index(drop=True)
train["Index"] = range(1, len(train) + 1)
valid["Index"] = range(1, len(valid) + 1)

# 计算 qed 与 sa（参考 properties.py 的实现）
def compute_qed_sa(smiles_list):
    qed_list, sa_list = [], []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        # QED
        if mol is None:
            qed_list.append(0.0)
            sa_list.append(100.0)
            continue
        try:
            qed_list.append(float(QED.qed(mol)))
        except Exception:
            qed_list.append(0.0)
        # SA Score
        try:
            sa_list.append(float(sascorer.calculateScore(mol)))
        except Exception:
            sa_list.append(100.0)
    return qed_list, sa_list

train_qed, train_sa = compute_qed_sa(train["SMILES"].astype(str).tolist())
valid_qed, valid_sa = compute_qed_sa(valid["SMILES"].astype(str).tolist())
train["qed"] = train_qed
train["sa"] = train_sa
valid["qed"] = valid_qed
valid["sa"] = valid_sa

# 确保列顺序为原始列 + ['qed','sa']
base_cols = [c for c in ["Index", "SMILES", "ST Gap", "HL Gap", "S1", "T1"] if c in train.columns]
train = train[base_cols + ["qed", "sa"]]
valid = valid[base_cols + ["qed", "sa"]]

out_dir = os.path.dirname(input_path)
os.makedirs(out_dir, exist_ok=True)
train.to_csv(os.path.join(out_dir, "train.csv"), index=False)
valid.to_csv(os.path.join(out_dir, "valid.csv"), index=False)
print(f"train: {len(train)}, valid: {len(valid)}")