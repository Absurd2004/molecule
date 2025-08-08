import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
import os
from PIL import Image, ImageDraw, ImageFont
import re
from rdkit import RDLogger

from models.model_rnn import RNN
from MCMG_utils.data_structs import Vocabulary
from MCMG_utils.utils import Variable, seq_to_smiles

# 禁用RDKit的日志输出
RDLogger.DisableLog('rdApp.*')

def create_molecule_generation_video_simple(num_molecules=3, max_length=150, fps=1):
    """
    生成分子逐步生成过程的视频 - 简化版本，使用run_dqn.py的视频生成方法
    """
    # 加载词汇表和模型
    voc = Vocabulary(init_from_file="./data/voc2.csv")
    Agent = RNN(voc)
    
    # 加载训练好的epoch1模型
    if torch.cuda.is_available():
        Agent.rnn.load_state_dict(torch.load('./data/Agent_ML_models/Agent_RNN_epoch_1.ckpt'))
    else:
        Agent.rnn.load_state_dict(torch.load('./data/Agent_ML_models/Agent_RNN_epoch_1.ckpt', 
                                           map_location=lambda storage, loc: storage))
    
    Agent.rnn.eval()
    
    # 为每个分子创建单独的视频
    output_base_dir = './rnn_molecule_videos'
    os.makedirs(output_base_dir, exist_ok=True)
    
    print(f"开始生成 {num_molecules} 个分子的逐步生成视频...")
    
    for mol_idx in range(num_molecules):
        print(f"正在生成第 {mol_idx + 1} 个分子...")
        
        mol_output_dir = os.path.join(output_base_dir, f'molecule_{mol_idx + 1}_generation')
        os.makedirs(mol_output_dir, exist_ok=True)
        
        success = generate_single_molecule_images(
            Agent, voc, mol_idx + 1, max_length, mol_output_dir
        )
        
        if success:
            # 使用run_dqn.py中的视频生成方法
            create_video_from_images_imageio(mol_output_dir, mol_idx + 1)
            print(f"第 {mol_idx + 1} 个分子视频生成完成")

def generate_single_molecule_images(agent, voc, mol_num, max_length, output_dir):
    """
    生成单个分子的图片序列
    """
    try:
        # 初始化生成过程
        batch_size = 1
        start_token = Variable(torch.zeros(batch_size).long())
        start_token[:] = voc.vocab['GO']
        h = agent.rnn.init_h(batch_size)
        x = start_token
        
        sequence = []
        image_count = 0
        
        print(f"  开始生成分子 {mol_num}...")
        
        for step in range(max_length):
            # 前向传播获取下一个token
            with torch.no_grad():
                logits, h = agent.rnn(x, h)
                prob = torch.softmax(logits, dim=1)
                x = torch.multinomial(prob, 1).view(-1)
                
                token_idx = x.item()
                sequence.append(token_idx)
                
                if token_idx == voc.vocab['EOS']:
                    print(f"  在第 {step + 1} 步遇到EOS")
                    break
                
                # 构建当前SMILES
                current_tokens = [voc.reversed_vocab[idx] for idx in sequence 
                                if idx not in [voc.vocab['GO'], voc.vocab['EOS']]]
                current_smiles = ''.join(current_tokens)
                print(f"  第 {step + 1} 步生成的SMILES: {current_smiles}")
                
                # 尝试解析SMILES
                try:
                    mol = Chem.MolFromSmiles(current_smiles)
                    if mol is not None and mol.GetNumAtoms() > 0:
                        # 生成图片
                        img_path = draw_molecule_step(
                            current_smiles, mol, mol_num, step + 1, image_count, output_dir
                        )
                        
                        if img_path:
                            image_count += 1
                            if image_count % 5 == 0:
                                print(f"    已生成 {image_count} 张图片")
                
                except Exception as e:
                    # SMILES无效，跳过
                    pass
                
                x = Variable(x.data)
        
        # 构建最终的SMILES
        final_tokens = [voc.reversed_vocab[idx] for idx in sequence 
                       if idx not in [voc.vocab['GO'], voc.vocab['EOS']]]
        final_smiles = ''.join(final_tokens)
        print(f"  分子 {mol_num} 最终SMILES: {final_smiles}")
        print(f"  生成了 {image_count} 张有效图片")
        
        return image_count > 0
            
    except Exception as e:
        print(f"  生成分子 {mol_num} 时出错: {e}")
        return False

def draw_molecule_step(smiles, mol, mol_num, step_num, image_count, output_dir):
    """
    绘制分子的一步，参考run_dqn.py的方法
    """
    try:
        # 使用RDKit绘制分子，和run_dqn.py保持一致的方法
        mol_img = Draw.MolToImage(mol, size=(400, 300))
        
        # 创建最终图像，为文字留出空间
        img_size = (500, 400)
        final_img = Image.new('RGB', img_size, 'white')
        
        # 粘贴分子图像
        final_img.paste(mol_img, (50, 50))
        
        # 添加文字标注
        draw = ImageDraw.Draw(final_img)
        
        try:
            font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
            font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        except:
            font_large = ImageFont.load_default()
            font_small = ImageFont.load_default()
        
        # 标题
        draw.text((10, 10), f"Molecule {mol_num} - Step {step_num}", fill='black', font=font_large)
        
        # 分子信息
        atom_count = mol.GetNumAtoms()
        bond_count = mol.GetNumBonds()
        draw.text((10, 360), f"Atoms: {atom_count}, Bonds: {bond_count}", fill='gray', font=font_small)
        
        # SMILES（截断长的SMILES）
        display_smiles = smiles if len(smiles) <= 50 else smiles[:47] + "..."
        draw.text((10, 380), f"SMILES: {display_smiles}", fill='blue', font=font_small)
        
        # 保存图片，使用run_dqn.py的命名格式
        img_path = os.path.join(output_dir, f'step_{image_count:03d}.png')
        final_img.save(img_path)
        
        return img_path
        
    except Exception as e:
        print(f"    绘制步骤失败: {e}")
        return None

def create_video_from_images_imageio(episode_dir, mol_num):
    """
    从图片创建视频 - 直接使用run_dqn.py中的方法
    """
    try:
        import imageio
        
        # 获取所有图片文件
        image_files = sorted([f for f in os.listdir(episode_dir) if f.endswith('.png')])
        if len(image_files) < 2:
            print(f"    分子 {mol_num}: 图片数量不足，无法创建视频")
            return
        
        print(f"    正在为分子 {mol_num} 创建视频，共 {len(image_files)} 张图片...")
        
        # 读取图片并创建视频
        images = []
        for img_file in image_files:
            img_path = os.path.join(episode_dir, img_file)
            img = imageio.imread(img_path)
            # 每张图片显示2秒
            for _ in range(2):
                images.append(img)
        
        # 生成MP4视频
        video_path = os.path.join(episode_dir, f'molecule_{mol_num}_evolution.mp4')
        imageio.mimsave(video_path, images, fps=1)
        print(f"    视频已保存: {video_path}")
        
        # 生成GIF
        gif_path = os.path.join(episode_dir, f'molecule_{mol_num}_evolution.gif')
        imageio.mimsave(gif_path, images[::2], fps=1)  # 降低帧率
        print(f"    GIF已保存: {gif_path}")
        
    except ImportError:
        print("    需要安装imageio: pip install imageio[ffmpeg]")
    except Exception as e:
        print(f"    创建视频时出错: {e}")

def create_overview_summary():
    """
    创建所有分子的概览摘要
    """
    output_base_dir = './rnn_molecule_videos'
    if not os.path.exists(output_base_dir):
        print("没有找到视频输出目录")
        return
    
    print("\n=== 分子生成视频概览 ===")
    
    # 收集所有分子的视频和信息
    for mol_dir in sorted(os.listdir(output_base_dir)):
        mol_path = os.path.join(output_base_dir, mol_dir)
        if os.path.isdir(mol_path):
            # 统计图片数量
            png_files = [f for f in os.listdir(mol_path) if f.endswith('.png')]
            video_files = [f for f in os.listdir(mol_path) if f.endswith('.mp4')]
            gif_files = [f for f in os.listdir(mol_path) if f.endswith('.gif')]
            
            print(f"{mol_dir}:")
            print(f"  - 生成步骤: {len(png_files)} 步")
            print(f"  - 视频文件: {len(video_files)} 个")
            print(f"  - GIF文件: {len(gif_files)} 个")
            print(f"  - 路径: {mol_path}")
            print()

if __name__ == "__main__":
    # 先检查是否安装了imageio
    try:
        import imageio
        print("✓ imageio已安装")
    except ImportError:
        print("✗ 需要安装imageio: pip install imageio[ffmpeg]")
        exit(1)
    
    # 生成分子视频
    create_molecule_generation_video_simple(num_molecules=3, max_length=100, fps=1)
    
    # 创建概览
    create_overview_summary()