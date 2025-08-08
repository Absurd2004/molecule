import pandas as pd
import glob
import os
from rdkit import Chem
from tqdm import tqdm

def concatenate_csv_files():
    # 获取所有匹配的CSV文件
    file_pattern = './data/Transformer/gen_dataset_*.csv'
    csv_files = glob.glob(file_pattern)
    
    # 按文件名排序（确保按数字顺序）
    csv_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    
    print(f"找到 {len(csv_files)} 个CSV文件:")
    for file in csv_files:
        print(f"  - {file}")
    
    # 读取并合并所有CSV文件
    all_smiles = []
    total_rows = 0
    
    for file in csv_files:
        try:
            df = pd.read_csv(file, header=None)
            # 假设SMILES在第一列
            smiles_list = df.iloc[:, 0].tolist()
            all_smiles.extend(smiles_list)
            total_rows += len(df)
            print(f"已读取 {file}: {len(df)} 行")
        except Exception as e:
            print(f"读取文件 {file} 时出错: {e}")
    
    print(f"\n原始数据统计:")
    print(f"总SMILES数量: {len(all_smiles)}")
    
    # 验证SMILES有效性
    valid_smiles = []
    invalid_count = 0
    
    print("\n验证SMILES有效性...")
    for smiles in tqdm(all_smiles, desc="验证SMILES"):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                valid_smiles.append(smiles)
            else:
                invalid_count += 1
        except:
            invalid_count += 1
    
    print(f"\n有效性验证结果:")
    print(f"有效SMILES数量: {len(valid_smiles)}")
    print(f"无效SMILES数量: {invalid_count}")
    print(f"有效率: {len(valid_smiles)/len(all_smiles)*100:.2f}%")
    
    # 去重
    print("\n去重处理...")
    unique_smiles = list(set(valid_smiles))
    duplicate_count = len(valid_smiles) - len(unique_smiles)
    
    print(f"去重结果:")
    print(f"去重前有效SMILES数量: {len(valid_smiles)}")
    print(f"去重后唯一SMILES数量: {len(unique_smiles)}")
    print(f"重复SMILES数量: {duplicate_count}")
    print(f"去重率: {duplicate_count/len(valid_smiles)*100:.2f}%")
    
    # 保存结果
    if unique_smiles:
        output_file = './data/Transformer/01_succeed_smiles.csv'
        
        # 确保目录存在
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # 保存为DataFrame
        result_df = pd.DataFrame(unique_smiles, columns=['SMILES'])
        result_df.to_csv(output_file, header=True, index=False)
        
        print(f"\n处理完成!")
        print(f"最终保存的唯一有效SMILES数量: {len(unique_smiles)}")
        print(f"输出文件: {output_file}")
        
        # 打印统计摘要
        print(f"\n=== 统计摘要 ===")
        print(f"原始总数: {len(all_smiles)}")
        print(f"有效数量: {len(valid_smiles)} ({len(valid_smiles)/len(all_smiles)*100:.2f}%)")
        print(f"最终唯一: {len(unique_smiles)} ({len(unique_smiles)/len(all_smiles)*100:.2f}%)")
        print(f"无效丢弃: {invalid_count} ({invalid_count/len(all_smiles)*100:.2f}%)")
        print(f"重复丢弃: {duplicate_count} ({duplicate_count/len(all_smiles)*100:.2f}%)")
        
    else:
        print("没有找到有效的SMILES进行保存")

if __name__ == "__main__":
    concatenate_csv_files()