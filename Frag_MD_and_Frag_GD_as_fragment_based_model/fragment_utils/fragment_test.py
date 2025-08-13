import os
import pandas as pd
from rdkit import Chem
from tqdm import tqdm
from mol_utils import split_molecule, MOL_SPLIT_START

INPUT_CSV = "../data1/SMILES.csv"          # 需存在且包含列 SMILES
SMILES_COL = "SMILES"
OUT_DIR = "../data1"
SCAFFOLD_VOC_FILE = os.path.join(OUT_DIR, "scaffold_vocab.csv")
DECORATION_VOC_FILE = os.path.join(OUT_DIR, "decoration_vocab.csv")
os.makedirs(OUT_DIR, exist_ok=True)

def count_cut_sites(mol):
    # split_molecule 用原子序号 >= MOL_SPLIT_START 作为切割位点标记
    return sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() >= MOL_SPLIT_START)

def ring_count(mol):
    return mol.GetRingInfo().NumRings()

def choose_scaffold(frag_mols):
    """
    规则：
      1. 优先含环片段 (ring_count > 0)
      2. 切割位点数(标记原子数)最大
      3. 若并列：环数最大
      4. 若并列：重原子数最大
      5. 若并列：SMILES 字典序最小
    若无含环片段，忽略第1条。
    返回 (scaffold_idx, decoration_indices)
    """
    if not frag_mols:
        return None, []
    stats = []
    for idx, m in enumerate(frag_mols):
        cs = count_cut_sites(m)
        rc = ring_count(m)
        hv = m.GetNumHeavyAtoms()
        smi = Chem.MolToSmiles(m, canonical=True)
        stats.append((idx, cs, rc, hv, smi))
    ring_subset = [x for x in stats if x[2] > 0]
    candidates = ring_subset if ring_subset else stats
    candidates.sort(key=lambda x: (-x[1], -x[2], -x[3], x[4]))
    chosen = candidates[0][0]
    others = [i for i, *_ in stats if i != chosen]
    return chosen, others

def build_vocab():
    if not os.path.isfile(INPUT_CSV):
        print(f"未找到输入文件: {INPUT_CSV}")
        return
    df = pd.read_csv(INPUT_CSV)
    if SMILES_COL not in df.columns:
        print(f"输入文件缺少列 {SMILES_COL}")
        return

    scaffold_set = set()
    decoration_set = set()
    total = 0
    failed = 0
    no_frag = 0

    for smi in tqdm(df[SMILES_COL].astype(str).tolist(), desc="处理分子"):
        total += 1
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            failed += 1
            continue
        try:
            frags = split_molecule(mol)
            frags = [f for f in frags if f is not None]
        except Exception:
            failed += 1
            continue
        if not frags:
            no_frag += 1
            continue
        sc_idx, others = choose_scaffold(frags)
        if sc_idx is None:
            no_frag += 1
            continue
        sc_smi = Chem.MolToSmiles(frags[sc_idx], canonical=True)
        scaffold_set.add(sc_smi)
        for oi in others:
            deco_smi = Chem.MolToSmiles(frags[oi], canonical=True)
            decoration_set.add(deco_smi)

    pd.DataFrame(sorted(scaffold_set), columns=["scaffold"]).to_csv(SCAFFOLD_VOC_FILE, index=False, header=False)
    pd.DataFrame(sorted(decoration_set), columns=["decoration"]).to_csv(DECORATION_VOC_FILE, index=False, header=False)

    print("=== 统计 ===")
    print(f"总分子: {total}")
    print(f"SMILES 解析失败: {failed}")
    print(f"无片段/无有效选择: {no_frag}")
    print(f"Scaffold 去重数: {len(scaffold_set)}")
    print(f"Decoration 去重数: {len(decoration_set)}")
    print(f"输出文件: {SCAFFOLD_VOC_FILE}, {DECORATION_VOC_FILE}")

if __name__ == "__main__":
    build_vocab()