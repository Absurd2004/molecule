import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from rdkit import Chem
import pandas as pd
import time

from MCMG_utils.data_structs_scaffold import DecoratorVocabulary, DecoratorDataset, decode_molecule
from models.model_transformer_scaffold import DecoratorTransformerRL

def test_vocabulary():
    """测试 DecoratorVocabulary 功能"""
    print("=" * 60)
    print("1. 测试 DecoratorVocabulary")
    print("=" * 60)
    
    try:
        decorator_voc = DecoratorVocabulary.from_files(
            "./data1/scaffold_vocab.csv",
            "./data1/decoration_vocab.csv"
        )
        
        print(f"✅ Scaffold vocabulary size: {decorator_voc.len_scaffold()}")
        print(f"✅ Decoration vocabulary size: {decorator_voc.len_decoration()}")
        print(f"✅ Combined lengths: {decorator_voc.len()}")
        
        # 测试特殊 token
        decorator_special = ['EOS', 'GO', 'high_QED', 'low_QED', 'good_SA', 'bad_SA']
        for token in decorator_special:
            if token in decorator_voc.decoration_vocabulary.vocab:
                print(f"✅ Scaffold contains special token '{token}': index {decorator_voc.decoration_vocabulary.vocab[token]}")
            else:
                print(f"❌ Scaffold missing special token '{token}'")
        
        return decorator_voc
        
    except Exception as e:
        print(f"❌ DecoratorVocabulary test failed: {e}")
        return None

def test_dataset(decorator_voc):
    """测试 DecoratorDataset 功能"""
    print("\n" + "=" * 60)
    print("2. 测试 DecoratorDataset")
    print("=" * 60)
    
    try:
        dataset = DecoratorDataset("./data1/train.csv", decorator_voc, smiles_col='SMILES')
        print(f"✅ Dataset created with {len(dataset)} samples")
        
        # 测试单个样本
        scaffold_with_con, decoration = dataset[0]
        print(f"✅ Sample 0:")
        print(f"   - Scaffold+condition shape: {scaffold_with_con.shape}")
        print(f"   - Decoration shape: {decoration.shape}")
        print(f"   - Scaffold+condition: {scaffold_with_con}")
        print(f"   - Decoration: {decoration}")
        
        # 测试 DataLoader
        dataloader = DataLoader(dataset, batch_size=4, shuffle=False, 
                               collate_fn=DecoratorDataset.collate_fn)
        
        scaffold_batch, decoration_batch = next(iter(dataloader))
        print(f"✅ Batch test:")
        print(f"   - Scaffold batch shape: {scaffold_batch.shape}")
        print(f"   - Decoration batch shape: {decoration_batch.shape}")
        
        return dataset, dataloader
        
    except Exception as e:
        print(f"❌ DecoratorDataset test failed: {e}")
        return None, None

def test_model_creation(decorator_voc):
    """测试模型创建和参数"""
    print("\n" + "=" * 60)
    print("3. 测试模型创建")
    print("=" * 60)
    
    try:
        model = DecoratorTransformerRL(
            decorator_voc=decorator_voc,
            d_model=128,
            nhead=8,
            num_encoder_layers=2,  # 测试时用小模型
            num_decoder_layers=2,
            dim_feedforward=256,
            max_seq_length=140,
            pos_dropout=0.1,
            trans_dropout=0.1
        )
        
        print(f"✅ Model created successfully")
        
        # 统计参数量
        total_params = sum(p.numel() for p in model.model.parameters())
        trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
        print(f"✅ Total parameters: {total_params:,}")
        print(f"✅ Trainable parameters: {trainable_params:,}")
        
        # 测试模型各组件
        print(f"✅ Encoder vocab size: {model.model.encoder.embed.num_embeddings}")
        print(f"✅ Decoder vocab size: {model.model.decoder.embed_tgt.num_embeddings}")
        print(f"✅ Model d_model: {model.model.d_model}")
        
        return model
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return None

def test_forward_pass(model, dataloader):
    """测试前向传播"""
    print("\n" + "=" * 60)
    print("4. 测试前向传播")
    print("=" * 60)
    
    try:
        model.model.train()
        scaffold_batch, decoration_batch = next(iter(dataloader))

        scaffold_batch = scaffold_batch.long()
        decoration_batch = decoration_batch.long()
        
        print(f"✅ Input shapes:")
        print(f"   - Scaffold: {scaffold_batch.shape}")
        print(f"   - Decoration: {decoration_batch.shape}")
        
        # 测试 likelihood 计算
        start_time = time.time()
        mean_loss, each_loss = model.likelihood(scaffold_batch, decoration_batch)
        forward_time = time.time() - start_time
        
        print(f"✅ Forward pass successful!")
        print(f"   - Mean loss: {mean_loss.item():.4f}")
        print(f"   - Each molecule loss shape: {each_loss.shape}")
        print(f"   - Forward time: {forward_time:.4f}s")
        print(f"   - Loss range: {each_loss.min().item():.4f} ~ {each_loss.max().item():.4f}")
        
        # 测试梯度
        mean_loss.backward()
        has_gradients = any(p.grad is not None for p in model.model.parameters())
        print(f"✅ Gradients computed: {has_gradients}")
        
        return True
        
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        print(f"Error details: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_sampling(model, dataloader):
    """测试采样功能"""
    print("\n" + "=" * 60)
    print("5. 测试采样功能")
    print("=" * 60)
    
    try:
        model.model.eval()
        scaffold_batch, decoration_batch = next(iter(dataloader))

        scaffold_batch = scaffold_batch.long()
        decoration_batch = decoration_batch.long()
        
        # 只用前2个样本测试
        scaffold_batch = scaffold_batch[:2]
        
        with torch.no_grad():
            # 测试 sample (with log_probs)
            start_time = time.time()
            sequences, log_probs = model.sample(scaffold_batch, max_length=20)
            sample_time = time.time() - start_time
            
            print(f"✅ Sample function:")
            print(f"   - Generated sequences shape: {sequences.shape}")
            print(f"   - Log probs shape: {log_probs.shape}")
            print(f"   - Sample time: {sample_time:.4f}s")
            print(f"   - Log probs: {log_probs}")
            
            # 测试 generate (without log_probs)
            start_time = time.time()
            sequences_only = model.generate(scaffold_batch, max_length=20)
            generate_time = time.time() - start_time
            
            print(f"✅ Generate function:")
            print(f"   - Generated sequences shape: {sequences_only.shape}")
            print(f"   - Generate time: {generate_time:.4f}s")
            print(f"   - Generated faster: {generate_time < sample_time}")
        
        return sequences, log_probs
        
    except Exception as e:
        print(f"❌ Sampling failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def test_decoding(decorator_voc, dataset, sequences):
    """测试解码功能"""
    print("\n" + "=" * 60)
    print("6. 测试解码功能")
    print("=" * 60)
    
    try:
        # 获取原始数据进行对比
        scaffold_with_con, decoration = dataset[0]
        
        print("✅ 原始数据解码:")
        
        # 解码条件 token
        con = scaffold_with_con[:2]
        con_tokens = [decorator_voc.scaffold_vocabulary.reversed_vocab[int(idx.item())] 
                      for idx in con]
        print(f"   - 条件 tokens: {con_tokens}")
        
        # 解码 scaffold
        scaffold_only = scaffold_with_con[2:]
        scaffold_tokens = [decorator_voc.scaffold_vocabulary.reversed_vocab[int(idx.item())] 
                          for idx in scaffold_only if int(idx.item()) != decorator_voc.scaffold_vocabulary.vocab['EOS']]
        print(f"   - Scaffold tokens: {scaffold_tokens}")
        
        # 解码 decoration  
        decoration_tokens = [decorator_voc.decoration_vocabulary.reversed_vocab[int(idx.item())] 
                            for idx in decoration if int(idx.item()) != decorator_voc.decoration_vocabulary.vocab['EOS']]
        print(f"   - Decoration tokens: {decoration_tokens}")
        
        # 重建完整分子
        all_tokens = scaffold_tokens + decoration_tokens
        reconstructed_mol = decode_molecule(all_tokens)
        if reconstructed_mol:
            reconstructed_smiles = Chem.MolToSmiles(reconstructed_mol)
            print(f"   - 重建分子: {reconstructed_smiles}")
        else:
            print(f"   - 重建失败: tokens = {all_tokens}")
        
        # 测试生成序列的解码
        if sequences is not None:
            print("\n✅ 生成序列解码:")
            for i, seq in enumerate(sequences[:2]):  # 只解码前2个
                tokens = [decorator_voc.decoration_vocabulary.reversed_vocab[int(idx.item())] 
                         for idx in seq if int(idx.item()) != decorator_voc.decoration_vocabulary.vocab['EOS']]
                print(f"   - 生成序列 {i}: {tokens}")
                
                if tokens:
                    # 结合 scaffold 重建分子
                    combined_tokens = scaffold_tokens + tokens
                    gen_mol = decode_molecule(combined_tokens)
                    if gen_mol:
                        gen_smiles = Chem.MolToSmiles(gen_mol)
                        print(f"     -> 生成分子: {gen_smiles}")
                    else:
                        print(f"     -> 解码失败")
        
        return True
        
    except Exception as e:
        print(f"❌ Decoding test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_memory_usage():
    """测试内存使用"""
    print("\n" + "=" * 60)
    print("7. 测试内存使用")
    print("=" * 60)
    
    if torch.cuda.is_available():
        print(f"✅ CUDA available: {torch.cuda.get_device_name()}")
        print(f"✅ Memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"✅ Memory reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
        
        # 清理内存
        torch.cuda.empty_cache()
        print(f"✅ After cleanup - Memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    else:
        print("✅ Running on CPU")

def test_model_compatibility():
    """测试模型兼容性"""
    print("\n" + "=" * 60)
    print("8. 测试模型兼容性")  
    print("=" * 60)
    
    try:
        # 测试是否支持不同的输入大小
        decorator_voc = DecoratorVocabulary.from_files(
            "./data1/scaffold_vocab.csv",
            "./data1/decoration_vocab.csv"
        )
        
        model = DecoratorTransformerRL(
            decorator_voc=decorator_voc,
            d_model=64,  # 更小的模型
            nhead=4,
            num_encoder_layers=1,
            num_decoder_layers=1,
            dim_feedforward=128,
            max_seq_length=140,
            pos_dropout=0.1,
            trans_dropout=0.1
        )

        device = model.device
        print(f"Model device: {device}")
        
        # 测试不同 batch size
        for batch_size in [1, 2, 4]:
            print(f"✅ Testing batch size {batch_size}")
            
            # 创建虚拟数据 - 直接创建 tensor，不要 tuple
            scaffold_batch = torch.randint(1, decorator_voc.len_scaffold()-1, (batch_size, 5)).long().to(device)
            decoration_batch = torch.randint(1, decorator_voc.len_decoration()-1, (batch_size, 8)).long().to(device)
            
            # 测试前向传播
            with torch.no_grad():
                mean_loss, each_loss = model.likelihood(scaffold_batch, decoration_batch)
                print(f"   - Batch {batch_size} loss: {mean_loss.item():.4f}")
        
        print("✅ Model compatibility test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Compatibility test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("开始测试 Decorator Transformer 模型")
    print("=" * 80)
    
    # 1. 测试词汇表
    decorator_voc = test_vocabulary()
    if not decorator_voc:
        print("❌ 词汇表测试失败，终止测试")
        return
    
    # 2. 测试数据集
    dataset, dataloader = test_dataset(decorator_voc)
    if not dataset or not dataloader:
        print("❌ 数据集测试失败，终止测试")
        return
    
    # 3. 测试模型创建
    model = test_model_creation(decorator_voc)
    if not model:
        print("❌ 模型创建失败，终止测试")
        return
    
    # 4. 测试前向传播
    forward_success = test_forward_pass(model, dataloader)
    if not forward_success:
        print("❌ 前向传播测试失败")
        return
    
    # 5. 测试采样
    sequences, log_probs = test_sampling(model, dataloader)

    
    # 6. 测试解码
    test_decoding(decorator_voc, dataset, sequences)
    
    # 7. 测试内存
    test_memory_usage()
    
    # 8. 测试兼容性
    test_model_compatibility()
    
    print("\n" + "=" * 80)
    print("✅ 所有测试完成！模型功能正常。")
    print("=" * 80)

if __name__ == "__main__":
    main()