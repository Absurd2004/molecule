import pandas as pd

def check_smiles_overlap():
    """检查两个CSV文件中的SMILES重叠情况"""
    
    # 读取两个CSV文件
    train_df = pd.read_csv('./data/train.csv')
    photo_df = pd.read_csv('./dftdata/Photosensitizers_DAD.csv')
    
    print(f"train_1.csv 文件信息:")
    print(f"  行数: {len(train_df)}")
    print(f"  列名: {list(train_df.columns)}")
    print(f"  前几行:")
    print(train_df.head())
    
    print(f"\nPhotosensitizers_DA.csv 文件信息:")
    print(f"  行数: {len(photo_df)}")
    print(f"  列名: {list(photo_df.columns)}")
    print(f"  前几行:")
    print(photo_df.head())
    
    # 获取SMILES集合
    train_smiles = set(train_df['smiles'].dropna())
    photo_smiles = set(photo_df['SMILES'].dropna())
    
    print(f"\n数据统计:")
    print(f"  train_1.csv 中有效SMILES数量: {len(train_smiles)}")
    print(f"  Photosensitizers_DA.csv 中有效SMILES数量: {len(photo_smiles)}")
    
    # 检查重叠
    overlap = photo_smiles.intersection(train_smiles)
    only_in_photo = photo_smiles - train_smiles
    only_in_train = train_smiles - photo_smiles
    
    print(f"\n重叠情况:")
    print(f"  Photosensitizers_DA.csv 中的SMILES在train_1.csv中存在的数量: {len(overlap)}")
    print(f"  重叠比例: {len(overlap)/len(photo_smiles)*100:.2f}%")
    print(f"  仅在Photosensitizers_DA.csv中的SMILES数量: {len(only_in_photo)}")
    print(f"  仅在train_1.csv中的SMILES数量: {len(only_in_train)}")
    
    # 显示一些重叠的SMILES示例
    if len(overlap) > 0:
        print(f"\n重叠SMILES示例 (前10个):")
        for i, smiles in enumerate(list(overlap)[:10]):
            print(f"  {i+1}: {smiles}")
    
    # 显示一些仅在Photosensitizers_DA.csv中的SMILES示例
    if len(only_in_photo) > 0:
        print(f"\n仅在Photosensitizers_DA.csv中的SMILES示例 (前5个):")
        for i, smiles in enumerate(list(only_in_photo)[:5]):
            print(f"  {i+1}: {smiles}")
    
    return {
        'overlap_count': len(overlap),
        'overlap_ratio': len(overlap)/len(photo_smiles),
        'overlap_smiles': overlap,
        'only_in_photo': only_in_photo,
        'only_in_train': only_in_train
    }

if __name__ == "__main__":
    result = check_smiles_overlap()