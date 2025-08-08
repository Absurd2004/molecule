#!/usr/bin/env python
import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
import os
from PIL import Image, ImageDraw, ImageFont
from rdkit import RDLogger
import imageio
from tqdm import tqdm

from models.model import RNN
from MCMG_utils.data_structs_fragment import Vocabulary
from MCMG_utils.utils import Variable, seq_to_smiles_frag

# 禁用RDKit的日志输出
RDLogger.DisableLog('rdApp.*')

def create_fragment_molecule_video(num_molecules=3, max_length=100):
    """
    生成fragment-based分子逐步生成过程的视频
    """
    # 加载词汇表和模型 - 使用与训练相同的设置
    print("正在加载词汇表...")
    voc = Vocabulary(init_from_file="./data/fragments_Voc2.csv")
    print(f"词汇表大小: {len(voc.vocab)}")
    print(f"GO token: {voc.vocab.get('GO', 'NOT FOUND')}")
    print(f"EOS token: {voc.vocab.get('EOS', 'NOT FOUND')}")
    
    print("正在加载模型...")
    Agent = RNN(voc)
    
    # 加载训练好的epoch1模型
    model_path = './data/Agent_ML_models_new/Agent_RNN_epoch_1.ckpt'
    print(f"正在加载模型: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在: {model_path}")
        # 尝试其他可能的路径
        alternative_paths = [
            './data/Frag_ML/Agent_RNN_epoch_1.ckpt',
            './data/Frag_ML/Prior_RNN_frag_ML.ckpt',
            './data/Agent_RNN_epoch_1.ckpt'
        ]
        for alt_path in alternative_paths:
            if os.path.exists(alt_path):
                print(f"找到替代模型: {alt_path}")
                model_path = alt_path
                break
        else:
            print("无法找到任何模型文件，请检查路径")
            return
    
    try:
        if torch.cuda.is_available():
            Agent.rnn.load_state_dict(torch.load(model_path))
            print("模型已加载到GPU")
        else:
            Agent.rnn.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
            print("模型已加载到CPU")
    except Exception as e:
        print(f"加载模型失败: {e}")
        return
    
    Agent.rnn.eval()
    
    # 创建输出目录
    output_base_dir = './fragment_molecule_videos'
    os.makedirs(output_base_dir, exist_ok=True)
    
    print(f"开始生成 {num_molecules} 个fragment-based分子的逐步生成视频...")
    
    for mol_idx in range(num_molecules):
        print(f"\n正在生成第 {mol_idx + 1} 个分子...")
        
        mol_output_dir = os.path.join(output_base_dir, f'molecule_{mol_idx + 1}_generation_new')
        os.makedirs(mol_output_dir, exist_ok=True)
        
        success = generate_single_fragment_molecule_images(
            Agent, voc, mol_idx + 1, max_length, mol_output_dir
        )
        
        if success:
            create_video_from_images(mol_output_dir, mol_idx + 1)
            print(f"第 {mol_idx + 1} 个分子视频生成完成")
        else:
            print(f"第 {mol_idx + 1} 个分子生成失败")

def generate_single_fragment_molecule_images(agent, voc, mol_num, max_length, output_dir):
    """
    生成单个fragment-based分子的图片序列 - 利用现有的采样函数
    """
    try:
        print(f"  开始逐步生成fragment tokens...")
        
        # 使用与训练相同的采样方式，但批量大小为1
        batch_size = 1
        image_count = 0
        last_valid_info = None  # 保存最后一个有效的分子信息
        
        # 初始化序列生成
        start_token = Variable(torch.zeros(batch_size).long())
        start_token[:] = voc.vocab['GO']
        h = agent.rnn.init_h(batch_size)
        x = start_token
        
        sequence = [voc.vocab['GO']]  # 保存完整序列
        print(f"  初始序列: {sequence}")
        
        with torch.no_grad():
            for step in range(max_length):
                # 前向传播获取下一个token
                logits, h = agent.rnn(x, h)
                prob = torch.softmax(logits, dim=1)
                x = torch.multinomial(prob, 1).view(-1)
                
                token_idx = x.item()
                sequence.append(token_idx)
                
                # 打印当前token信息
                token_str = voc.reversed_vocab.get(token_idx, f"UNKNOWN_{token_idx}")
                print(f"  第 {step + 1} 步: token_idx={token_idx}, token='{token_str}'")
                
                # 检查是否遇到EOS
                if token_idx == voc.vocab['EOS']:
                    print(f"  在第 {step + 1} 步遇到EOS token")
                    # 如果有最后的有效分子信息，生成带评分的最终帧
                    if last_valid_info:
                        print("  生成带评分的最终帧...")
                        img_path = draw_fragment_molecule_step(
                            last_valid_info['smiles'], 
                            last_valid_info['mol'], 
                            last_valid_info['fragments'], 
                            mol_num, step + 1, image_count, output_dir, 
                            is_final=True
                        )
                        if img_path:
                            image_count += 1
                            print(f"    ✓ 生成最终评分图片: {img_path}")
                    break
                
                # 等待足够的fragments再尝试解码
                fragment_count = count_valid_fragments(sequence, voc)
                if fragment_count >= 1:  # 从第一个fragment就开始尝试解码
                    try:
                        # 获取当前fragments（跳过GO/EOS） - 使用character-based的简单方法
                        current_fragments = [voc.reversed_vocab[idx] for idx in sequence 
                                        if idx in voc.reversed_vocab and 
                                        voc.reversed_vocab[idx] not in ['GO', 'EOS']]
                        
                        # 如果只有1个fragment，直接尝试作为SMILES
                        if len(current_fragments) == 1:
                            current_smiles = current_fragments[0]
                        else:
                            # 多个fragments时使用原来的解码方法
                            clean_sequence = create_clean_sequence_for_decoding(sequence, voc)
                            if len(clean_sequence) > 1:
                                current_seq = torch.tensor([clean_sequence]).long()
                                smiles_list = seq_to_smiles_frag(current_seq, voc)
                                current_smiles = smiles_list[0] if smiles_list else None
                            else:
                                current_smiles = None
                        
                        print(f"  解码结果: {current_smiles}")
                        
                        if current_smiles and current_smiles != '=' and current_smiles != '' and current_smiles != 'None':
                            # 验证SMILES是否有效
                            mol = Chem.MolFromSmiles(current_smiles)
                            if mol is not None and mol.GetNumAtoms() > 0:
                                print(f"  SMILES有效，原子数: {mol.GetNumAtoms()}")
                                
                                # 保存最后的有效信息
                                last_valid_info = {
                                    'smiles': current_smiles,
                                    'mol': mol,
                                    'fragments': current_fragments
                                }
                                
                                # 生成图片（普通帧）
                                img_path = draw_fragment_molecule_step(
                                    current_smiles, mol, current_fragments, 
                                    mol_num, step + 1, image_count, output_dir
                                )
                                
                                if img_path:
                                    image_count += 1
                                    print(f"    ✓ 生成图片 {image_count}: {img_path}")
                                else:
                                    print(f"    ✗ 图片生成失败")
                            else:
                                print(f"  SMILES无效: {current_smiles}")
                        else:
                            print(f"  跳过无效SMILES: {current_smiles}")
                        
                    except Exception as e:
                        print(f"  解码第 {step + 1} 步失败: {e}")
                        import traceback
                        traceback.print_exc()
                        continue
                else:
                    print(f"  片段数量不足 ({fragment_count})，跳过解码")
                
                x = Variable(x.data)
        
        print(f"  分子 {mol_num} 生成完成")
        print(f"  最终序列长度: {len(sequence)}")
        print(f"  共生成了 {image_count} 张有效图片")
        
        # 如果没有生成任何图片，尝试最后一次完整解码
        if image_count == 0:
            print("  尝试最后一次完整解码...")
            try:
                clean_sequence = create_clean_sequence_for_decoding(sequence, voc)
                if len(clean_sequence) > 2:
                    final_seq = torch.tensor([clean_sequence]).long()
                    final_smiles_list = seq_to_smiles_frag(final_seq, voc)
                    final_smiles = final_smiles_list[0] if final_smiles_list else None
                    print(f"  最终SMILES: {final_smiles}")
                    
                    if final_smiles and final_smiles != '=' and final_smiles != '':
                        mol = Chem.MolFromSmiles(final_smiles)
                        if mol is not None and mol.GetNumAtoms() > 0:
                            current_fragments = get_fragments_from_sequence(sequence, voc)
                            img_path = draw_fragment_molecule_step(
                                final_smiles, mol, current_fragments, 
                                mol_num, len(sequence), 0, output_dir,
                                is_final=True
                            )
                            if img_path:
                                image_count = 1
                                print(f"  ✓ 最终生成了1张图片")
            except Exception as e:
                print(f"  最终解码也失败: {e}")
        
        return image_count > 0
            
    except Exception as e:
        print(f"  生成分子 {mol_num} 时出错: {e}")
        import traceback
        traceback.print_exc()
        return False

def count_valid_fragments(sequence, voc):
    """计算序列中有效fragment的数量"""
    count = 0
    for token_idx in sequence:
        if token_idx in voc.reversed_vocab:
            token_str = voc.reversed_vocab[token_idx]
            if token_str not in ['GO', 'EOS']:
                count += 1
    return count

def create_clean_sequence_for_decoding(sequence, voc):
    """创建一个用于解码的清理序列，移除GO，确保有EOS"""
    clean_sequence = []
    
    for token_idx in sequence:
        if token_idx in voc.reversed_vocab:
            token_str = voc.reversed_vocab[token_idx]
            if token_str != 'GO':  # 跳过GO token
                clean_sequence.append(token_idx)
    
    # 如果序列末尾没有EOS，添加一个
    if clean_sequence and clean_sequence[-1] != voc.vocab['EOS']:
        clean_sequence.append(voc.vocab['EOS'])
    
    return clean_sequence

def get_fragments_from_sequence(sequence, voc):
    """
    从序列中提取fragment tokens的字符串表示
    """
    fragments = []
    for token_idx in sequence:
        if token_idx in voc.reversed_vocab:
            token_str = voc.reversed_vocab[token_idx]
            if token_str not in ['GO', 'EOS']:
                fragments.append(token_str)
    return fragments

def draw_fragment_molecule_step(smiles, mol, fragments, mol_num, step_num, image_count, output_dir, is_final=False):
    """
    绘制分子的一步，显示当前SMILES和使用的fragments
    """
    try:
        print(f"    正在绘制分子图片...")
        
        # 使用RDKit绘制分子
        mol_img = Draw.MolToImage(mol, size=(400, 300))
        
        # 创建最终图像，为文字留出空间
        img_size = (600, 500)
        final_img = Image.new('RGB', img_size, 'white')
        
        # 粘贴分子图像
        final_img.paste(mol_img, (100, 80))
        
        # 添加文字标注
        draw = ImageDraw.Draw(final_img)
        
        try:
            font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
            font_medium = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
            font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
            font_tiny = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)  # 小字体
        except:
            font_large = ImageFont.load_default()
            font_medium = ImageFont.load_default()
            font_small = ImageFont.load_default()
            font_tiny = ImageFont.load_default()
        
        # 标题
        title_text = f"Fragment-based Molecule {mol_num} - Step {step_num}"
        if is_final:
            title_text += " (FINAL)"
        draw.text((10, 10), title_text, fill='black', font=font_large)
        
        # 分子信息
        atom_count = mol.GetNumAtoms()
        bond_count = mol.GetNumBonds()
        draw.text((10, 40), f"Atoms: {atom_count}, Bonds: {bond_count}", 
                 fill='gray', font=font_medium)
        
        # 当前使用的fragments（限制长度）
        fragments_str = ' + '.join(fragments[-5:]) if fragments else 'None'  # 只显示最后5个
        if len(fragments_str) > 80:
            fragments_str = fragments_str[:77] + "..."
        draw.text((10, 390), f"Recent Fragments: {fragments_str}", 
                 fill='purple', font=font_small)
        
        # SMILES（截断长的SMILES）
        display_smiles = smiles if len(smiles) <= 60 else smiles[:57] + "..."
        draw.text((10, 410), f"SMILES: {display_smiles}", 
                 fill='blue', font=font_small)
        
        # 步骤信息
        draw.text((10, 430), f"Total fragments: {len(fragments)}", 
                 fill='green', font=font_small)
        
        # 如果是最终帧，添加评分信息
        if is_final:
            try:
                from properties import get_scoring_function, qed_func, sa_func
                
                # 计算四个指标
                logP = get_scoring_function('logP')([smiles])
                symmetry = get_scoring_function('symmetry')([smiles])
                qed_scores = qed_func()([smiles])
                sa_scores = sa_func()([smiles])
                
                # 在右上角显示评分（小字体）
                x_start = 420
                y_start = 50
                
                draw.text((x_start, y_start), f"LogP: {logP[0]:.3f}", fill='red', font=font_tiny)
                draw.text((x_start, y_start + 15), f"QED: {qed_scores[0]:.3f}", fill='red', font=font_tiny)
                draw.text((x_start, y_start + 30), f"SA: {sa_scores[0]:.2f}", fill='red', font=font_tiny)
                draw.text((x_start, y_start + 45), f"Sym: {symmetry[0]:.3f}", fill='red', font=font_tiny)
                
                # 计算总分
                sa_binary = float(sa_scores[0] < 4.0)
                total_score = logP[0] + qed_scores[0] + sa_binary + symmetry[0]
                draw.text((x_start, y_start + 65), f"Total: {total_score:.2f}", fill='black', font=font_small)
                
            except Exception as e:
                print(f"计算评分失败: {e}")
        
        # 保存图片
        img_path = os.path.join(output_dir, f'step_{image_count:03d}.png')
        final_img.save(img_path)
        print(f"    图片已保存: {img_path}")
        
        return img_path
        
    except Exception as e:
        print(f"    绘制步骤失败: {e}")
        import traceback
        traceback.print_exc()
        return None
        
    except Exception as e:
        print(f"    绘制步骤失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_video_from_images(episode_dir, mol_num):
    """
    从图片创建视频
    """
    try:
        # 获取所有图片文件
        image_files = sorted([f for f in os.listdir(episode_dir) if f.endswith('.png')])
        if len(image_files) < 1:
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
        video_path = os.path.join(episode_dir, f'fragment_molecule_{mol_num}_evolution.mp4')
        imageio.mimsave(video_path, images, fps=1)
        print(f"    视频已保存: {video_path}")
        
        # 生成GIF
        gif_path = os.path.join(episode_dir, f'fragment_molecule_{mol_num}_evolution.gif')
        imageio.mimsave(gif_path, images[::2], fps=0.8)  # 降低帧率
        print(f"    GIF已保存: {gif_path}")
        
    except Exception as e:
        print(f"    创建视频时出错: {e}")

def create_overview_summary():
    """
    创建所有分子的概览摘要
    """
    output_base_dir = './fragment_molecule_videos'
    if not os.path.exists(output_base_dir):
        print("没有找到视频输出目录")
        return
    
    print("\n=== Fragment-based分子生成视频概览 ===")
    
    # 收集所有分子的视频和信息
    for mol_dir in sorted(os.listdir(output_base_dir)):
        mol_path = os.path.join(output_base_dir, mol_dir)
        if os.path.isdir(mol_path):
            # 统计文件数量
            png_files = [f for f in os.listdir(mol_path) if f.endswith('.png')]
            video_files = [f for f in os.listdir(mol_path) if f.endswith('.mp4')]
            gif_files = [f for f in os.listdir(mol_path) if f.endswith('.gif')]
            
            print(f"{mol_dir}:")
            print(f"  - 生成步骤: {len(png_files)} 步")
            print(f"  - 视频文件: {len(video_files)} 个")
            print(f"  - GIF文件: {len(gif_files)} 个")
            print(f"  - 路径: {mol_path}")
            
            # 列出图片文件（用于调试）
            if png_files:
                print(f"  - 图片文件: {png_files[:5]}{'...' if len(png_files) > 5 else ''}")
            print()

if __name__ == "__main__":
    # 检查依赖
    try:
        import imageio
        print("✓ imageio已安装")
    except ImportError:
        print("✗ 需要安装imageio: pip install imageio[ffmpeg]")
        exit(1)
    
    # 生成fragment-based分子视频
    print("开始生成fragment-based分子演示视频...")
    create_fragment_molecule_video(num_molecules=1, max_length=50)  # 先测试1个分子
    
    # 创建概览
    create_overview_summary()
    
    print("\n视频生成完成！")
    print("每个分子的生成过程都保存在 ./fragment_molecule_videos/ 目录下")
    print("包含MP4视频和GIF动画文件")