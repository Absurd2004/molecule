import pandas as pd
import glob
import os
from rdkit import Chem
from tqdm import tqdm

EXPECTED_COLUMNS = ["Index", "SMILES", "ST Gap", "HL Gap", "S1", "T1"]

def concatenate_csv_files():
    file_pattern = './dftdata/Photosensitizers*.csv'
    csv_files = glob.glob(file_pattern)
    #csv_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    print(f"找到 {len(csv_files)} 个CSV文件:")
    for f in csv_files:
        print(f"  - {f}")

    all_valid_dfs = []
    total_rows = 0
    invalid_smiles_rows = 0

    for f in csv_files:
        try:
            df = pd.read_csv(f)  # 读取表头
            total_rows += len(df)

            # 列名规范化（如果大小写或空格差异可在此做修正）
            df.columns = [c.strip() for c in df.columns]

            # 检查必需列
            missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
            if missing:
                print(f"跳过 {f}，缺少列: {missing}")
                continue

            # 只保留需要的列（若文件有额外列）
            df = df[EXPECTED_COLUMNS].copy()

            # SMILES 校验
            print(f"验证 {f} 中 SMILES 有效性...")
            valid_mask = []
            for smi in tqdm(df["SMILES"], desc=f"验证SMILES-{os.path.basename(f)}", leave=False):
                mol = Chem.MolFromSmiles(str(smi))
                valid_mask.append(mol is not None)
            df_valid = df[valid_mask].copy()
            invalid_count_file = len(df) - len(df_valid)
            invalid_smiles_rows += invalid_count_file

            print(f"{f}: 总行 {len(df)}, 有效 {len(df_valid)}, 无效 {invalid_count_file}")
            all_valid_dfs.append(df_valid)
        except Exception as e:
            print(f"读取文件 {f} 时出错: {e}")

    if not all_valid_dfs:
        print("没有可合并的有效数据")
        return

    combined = pd.concat(all_valid_dfs, ignore_index=True)

    # 去重（按 SMILES）
    before_dup = len(combined)
    combined_unique = combined.drop_duplicates(subset=["SMILES"], keep="first").reset_index(drop=True)
    after_dup = len(combined_unique)
    dup_rows = before_dup - after_dup

    # 重新生成 Index（如果希望连续可重排；若想保留原 Index 可注释掉）
    combined_unique["Index"] = range(1, len(combined_unique) + 1)

    output_file = './data1/SMILES.csv'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    combined_unique.to_csv(output_file, index=False)

    print("\n=== 汇总统计 ===")
    print(f"原始总行数: {total_rows}")
    print(f"合并后有效行数: {before_dup}")
    print(f"无效SMILES行数: {invalid_smiles_rows}")
    print(f"重复SMILES行数: {dup_rows}")
    print(f"最终唯一有效行数: {after_dup}")
    print(f"输出文件: {output_file}")
    print(f"列: {list(combined_unique.columns)}")

if __name__ == "__main__":
    concatenate_csv_files()