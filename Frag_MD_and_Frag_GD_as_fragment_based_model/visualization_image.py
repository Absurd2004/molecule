import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Draw import rdMolDraw2D
from properties import get_scoring_function, qed_func, sa_func
import os
import random
from tqdm import tqdm

def visualize_selected_molecules(epoch=1, select_n=10):
    """
    随机选择score1==4.0且score2==2.0的分子，生成2D图片
    """
    smiles_file = f'./data/Frag_ML_agent_function1_little/epoch_{epoch}_smiles.csv'
    try:
        smiles_df = pd.read_csv(smiles_file, header=None, names=['smiles', 'step_index'])
        unique_smiles = smiles_df['smiles'].unique()
    except FileNotFoundError as e:
        print(f"文件未找到: {e}")
        return

    molecule_scores = []
    print("正在筛选分子...")
    for smiles in tqdm(unique_smiles, desc="select molecule"):
        try:
            score1, score2,origin_score1,origin_score2 = get_scoring_function('st_hl_f2')([smiles])
            logP = get_scoring_function('logP')([smiles])
            qed_scores = qed_func()([smiles])
            sa_scores = sa_func()([smiles])
            sa_binary = float(sa_scores[0] < 4.0)
            # 只保留score1==4.0且score2==2.0的分子
            if score1[0] == 4.0 and score2[0] == 2.0:
                molecule_scores.append({
                    'smiles': smiles,
                    'logp': logP[0],
                    'qed': qed_scores[0],
                    'sa': sa_scores[0],
                    'sa_binary': sa_binary,
                    'st_gap': score1[0],
                    'hl_gap': score2[0],
                    'origin_st_gap': origin_score1[0],
                    'origin_hl_gap': origin_score2[0]
                })
        except Exception as e:
            continue

    if len(molecule_scores) == 0:
        print("没有找到满足条件的分子。")
        return

    # 随机选择10个
    selected_molecules = random.sample(molecule_scores, min(select_n, len(molecule_scores)))

    output_dir = f'./images/epoch_{epoch}_selected_molecules'
    os.makedirs(output_dir, exist_ok=True)
    #创建一个txt文件，把所有的smiles和对应的值都输出到这个文件里
    with open(f'{output_dir}/selected_molecules_info.txt', 'w') as f:
        f.write("Index\tSMILES\tST_Gap\tHL_Gap\tLogP\tQED\tSA_Score\tSA_Binary\n")
        for i, mol_data in enumerate(selected_molecules, 1):
            f.write(f"{i}\t{mol_data['smiles']}\t{mol_data['origin_st_gap']:.3f}\t{mol_data['origin_hl_gap']:.3f}\t{mol_data['logp']:.3f}\t{mol_data['qed']:.3f}\t{mol_data['sa']:.2f}\t{mol_data['sa_binary']:.0f}\n")

    print(f"\n开始生成分子结构图片...")
    for i, mol_data in enumerate(selected_molecules, 1):
        smiles = mol_data['smiles']


        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            AllChem.Compute2DCoords(mol)
            drawer = rdMolDraw2D.MolDraw2DCairo(1000, 900)
            drawer.SetFontSize(1.0)
            drawer.DrawMolecule(mol)
            drawer.FinishDrawing()
            from PIL import Image, ImageDraw, ImageFont
            import io
            mol_img_pil = Image.open(io.BytesIO(drawer.GetDrawingText()))
            final_img = Image.new('RGB', (1000, 1000), 'white')
            final_img.paste(mol_img_pil, (0, 200))
            draw = ImageDraw.Draw(final_img)
            try:
                font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 36)
                font_medium = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 28)
            except:
                font_large = ImageFont.load_default()
                font_medium = ImageFont.load_default()
            # 修改后的文字内容
            draw.text((50, 30), f"ST Gap Score: {mol_data['origin_st_gap']:.3f} | HL Gap Score: {mol_data['origin_hl_gap']:.3f}", fill='black', font=font_large)
            draw.text((50, 80), f"LogP: {mol_data['logp']:.3f}", fill='blue', font=font_medium)
            draw.text((50, 115), f"QED: {mol_data['qed']:.3f}", fill='green', font=font_medium)
            draw.text((50, 150), f"SA Score: {mol_data['sa']:.2f} (Binary: {mol_data['sa_binary']:.0f})", fill='red', font=font_medium)
            draw.text((50, 960), f"SMILES: {smiles}", fill='black', font=font_medium)
            final_img.save(f'{output_dir}/selected_{i:02d}_annotated.png')
            print(f"✓ 已保存分子 {i}: {smiles}")
        except Exception as e:
            print(f"✗ 处理分子 {i} 时出错: {e}")
            continue

if __name__ == "__main__":
    visualize_selected_molecules(epoch=1, select_n=10)