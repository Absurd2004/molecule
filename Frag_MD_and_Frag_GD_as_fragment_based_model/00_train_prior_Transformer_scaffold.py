#!/usr/bin/env python
import argparse
import time
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm

from MCMG_utils.data_structs_scaffold import DecoratorVocabulary, DecoratorDataset
from models.model_transformer_scaffold import DecoratorTransformerRL
from MCMG_utils.Optim import ScheduledOptim
from MCMG_utils.early_stop.pytorchtools import EarlyStopping

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    print("Warning: wandb not installed. Logging will be disabled.")


start = time.time()


def train_prior_scaffold(train_data_path, valid_data_path, scaffold_vocab_path, decoration_vocab_path, 
                        save_prior_path, use_wandb=False):
    """训练基于 scaffold-decoration 的 Prior Transformer"""

    print("Loading vocabularies...", flush=True)

    decorator_voc = DecoratorVocabulary.from_files(scaffold_vocab_path, decoration_vocab_path)

    print(f"Scaffold vocabulary size: {decorator_voc.len_scaffold()}", flush=True)
    print(f"Decoration vocabulary size: {decorator_voc.len_decoration()}", flush=True)

    print("Loading datasets...", flush=True)
    # 创建数据集
    train_dataset = DecoratorDataset(train_data_path, decorator_voc, smiles_col='SMILES')
    valid_dataset = DecoratorDataset(valid_data_path, decorator_voc, smiles_col='SMILES')


    print(f"Training samples: {len(train_dataset)}", flush=True)
    print(f"Validation samples: {len(valid_dataset)}", flush=True)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        drop_last=True,
        collate_fn=DecoratorDataset.collate_fn
    )

    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        drop_last=True,
        collate_fn=DecoratorDataset.collate_fn
    )

    print("Creating model...", flush=True)


    model = DecoratorTransformerRL(
        decorator_voc=decorator_voc,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=dim_feedforward,
        max_seq_length=max_seq_length,
        pos_dropout=pos_dropout,
        trans_dropout=trans_dropout
    )

    print(f"Model device: {model.device}", flush=True)
    print(f"Model parameters: {sum(p.numel() for p in model.model.parameters()):,}", flush=True)

    optim = ScheduledOptim(
        Adam(model.model.parameters(), betas=(0.9, 0.98), eps=1e-09),
        d_model * 8, 
        n_warmup_steps
    )


    if use_wandb and HAS_WANDB:
        wandb.init(project="Frag_Scaffold_MD", name="train_prior_scaffold", config={
            "d_model": d_model,
            "num_encoder_layers": num_encoder_layers,
            "num_decoder_layers": num_decoder_layers,
            "dim_feedforward": dim_feedforward,
            "nhead": nhead,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "max_seq_length": max_seq_length,
            "pos_dropout": pos_dropout,
            "trans_dropout": trans_dropout,
            "n_warmup_steps": n_warmup_steps,
            "scaffold_vocab_size": decorator_voc.len_scaffold(),
            "decoration_vocab_size": decorator_voc.len_decoration(),
        })
        print("Wandb initialized", flush=True)
    
    print("Starting training...", flush=True)

    train_losses, val_losses = train(
        train_loader, valid_loader, model, optim, num_epochs, 
        save_prior_path, use_wandb=(use_wandb and HAS_WANDB)
    )

    if use_wandb and HAS_WANDB:
        wandb.finish()
    
    torch.cuda.empty_cache()
    return train_losses, val_losses

def train(train_loader, valid_loader, model, optim, num_epochs, save_prior_path, use_wandb=False):
    """训练循环"""
    model.model.train()
    lowest_val = 1e9
    train_losses = []
    val_losses = []
    total_step = 0

    early_stopping = EarlyStopping(patience=5, verbose=True)

    for epoch in tqdm(range(num_epochs), desc="Training epochs"):
        total_loss = 0

        for step, (scaffold_batch, decoration_batch) in tqdm(
            enumerate(train_loader), 
            total=len(train_loader), 
            desc=f"Epoch {epoch+1}", 
            leave=False
        ):
            scaffold_batch = scaffold_batch.long().to(model.device)
            decoration_batch = decoration_batch.long().to(model.device)

            loss, each_molecule_loss = model.likelihood(scaffold_batch, decoration_batch)

            optim.zero_grad()
            loss.backward()
            optim.step_and_update_lr()

            total_loss += loss.item()

            if use_wandb and HAS_WANDB:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/lr": optim._optimizer.param_groups[0]['lr'],
                    "step": total_step,
                    "epoch": epoch
                })
            total_step += 1

            if step % 200 == 0 and step != 0:
                print("*" * 50, flush=True)
                print(f"Epoch {epoch:3d}   step {step:3d}    loss: {loss.item():.4f}", flush=True)
                
                # 打印一些诊断信息
                print(f"   - Scaffold batch shape: {scaffold_batch.shape}", flush=True)
                print(f"   - Decoration batch shape: {decoration_batch.shape}", flush=True)
                print(f"   - Loss range: {each_molecule_loss.min().item():.4f} ~ {each_molecule_loss.max().item():.4f}", flush=True)
        
        avg_epoch_loss = total_loss / len(train_loader)
        print(f'Average epoch {epoch+1} loss: {avg_epoch_loss:.4f}', flush=True)
        train_losses.append((epoch, avg_epoch_loss))

        if use_wandb and HAS_WANDB:
            wandb.log({
                "train/epoch_loss": avg_epoch_loss,
                "epoch": epoch
            })
        val_loss = validate(valid_loader, model)
        val_losses.append((epoch, val_loss))
        print(f"Validation loss: {val_loss:.4f}", flush=True)

        if use_wandb and HAS_WANDB:
            wandb.log({
                "val/loss": val_loss,
                "epoch": epoch
            })
        

        early_stopping(val_loss, model.model, 'scaffold_prior')
        if early_stopping.early_stop:
            print("Early stopping triggered", flush=True)
            break

        if val_loss < lowest_val:
            lowest_val = val_loss
            torch.save(model.model.state_dict(), save_prior_path)
            print(f"New best model saved with validation loss: {val_loss:.4f}", flush=True)
    
    return train_losses, val_losses


def validate(valid_loader, model):
    """验证函数"""
    model.model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for step, (scaffold_batch, decoration_batch) in tqdm(
            enumerate(valid_loader), 
            total=len(valid_loader), 
            desc="Validating",
            leave=False
        ):
            # 确保数据在正确设备上
            scaffold_batch = scaffold_batch.long().to(model.device)
            decoration_batch = decoration_batch.long().to(model.device)
            
            # 计算损失
            loss, each_molecule_loss = model.likelihood(scaffold_batch, decoration_batch)
            total_loss += loss.item()
    
    model.model.train()  # 切换回训练模式
    return total_loss / len(valid_loader)

def test_single_batch(train_loader, model):
    """测试单个 batch，用于调试"""
    print("\n" + "="*60)
    print("Testing single batch...")
    print("="*60)
    
    model.model.eval()
    
    scaffold_batch, decoration_batch = next(iter(train_loader))
    scaffold_batch = scaffold_batch.long().to(model.device)
    decoration_batch = decoration_batch.long().to(model.device)
    
    print(f"Scaffold batch shape: {scaffold_batch.shape}")
    print(f"Decoration batch shape: {decoration_batch.shape}")
    print(f"Scaffold batch device: {scaffold_batch.device}")
    print(f"Decoration batch device: {decoration_batch.device}")
    
    with torch.no_grad():
        try:
            loss, each_loss = model.likelihood(scaffold_batch, decoration_batch)
            print(f"✅ Forward pass successful!")
            print(f"   - Loss: {loss.item():.4f}")
            print(f"   - Loss per molecule shape: {each_loss.shape}")
            print(f"   - Loss range: {each_loss.min().item():.4f} ~ {each_loss.max().item():.4f}")
        except Exception as e:
            print(f"❌ Forward pass failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    model.model.train()
    return True


if __name__ == "__main__":
    # 模型超参数
    max_seq_length = 140
    d_model = 128
    num_encoder_layers = 6  # 新增：Encoder 层数
    num_decoder_layers = 6  # 保持与原来相近
    dim_feedforward = 512
    nhead = 8
    pos_dropout = 0.1
    trans_dropout = 0.1
    n_warmup_steps = 500
    
    # 训练超参数
    num_epochs = 10
    batch_size = 128  # 可能需要调小，因为是 Encoder-Decoder
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}", flush=True)
    
    parser = argparse.ArgumentParser(description="Train scaffold-decoration Transformer Prior")


    parser.add_argument('--train-data', dest='train_data_path', 
                       default='./data1/train.csv',
                       help='Path to training CSV file with SMILES column')
    parser.add_argument('--valid-data', dest='valid_data_path',
                       default='./data1/valid.csv', 
                       help='Path to validation CSV file with SMILES column')
    
    # 词汇表路径
    parser.add_argument('--scaffold-vocab', dest='scaffold_vocab_path',
                       default='./data1/scaffold_vocab.csv',
                       help='Path to scaffold vocabulary file')
    parser.add_argument('--decoration-vocab', dest='decoration_vocab_path', 
                       default='./data1/decoration_vocab.csv',
                       help='Path to decoration vocabulary file')
    
    # 模型保存路径
    parser.add_argument('--save-prior-path', dest='save_prior_path',
                       default='./data1/models/scaffold_transformer_prior.ckpt',
                       help='Path to save the trained model')
    
    # 其他选项
    parser.add_argument('--use-wandb', action='store_true', dest='use_wandb',
                       help='Use wandb for logging. Default: False')
    parser.add_argument('--test-batch', action='store_true', dest='test_batch',
                       help='Test single batch and exit. Useful for debugging.')
    

    args = parser.parse_args()
    
    # 创建保存目录
    import os
    os.makedirs(os.path.dirname(args.save_prior_path), exist_ok=True)


    if args.test_batch:
        # 只测试单个 batch
        print("\n" + "="*60)
        print("TESTING MODE: Testing single batch only")
        print("="*60)
        
        decorator_voc = DecoratorVocabulary.from_files(
            args.scaffold_vocab_path, args.decoration_vocab_path
        )
        train_dataset = DecoratorDataset(args.train_data_path, decorator_voc, smiles_col='SMILES')
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=False, 
                                 collate_fn=DecoratorDataset.collate_fn)
        
        model = DecoratorTransformerRL(
            decorator_voc=decorator_voc,
            d_model=64,  # 小模型用于测试
            nhead=4,
            num_encoder_layers=1,
            num_decoder_layers=1, 
            dim_feedforward=128,
            max_seq_length=max_seq_length,
            pos_dropout=pos_dropout,
            trans_dropout=trans_dropout
        )
        
        success = test_single_batch(train_loader, model)
        if success:
            print("✅ Single batch test passed! You can now run full training.")
        else:
            print("❌ Single batch test failed! Please fix the issues first.")
    
    else:
        # 正常训练
        train_losses, val_losses = train_prior_scaffold(
            train_data_path=args.train_data_path,
            valid_data_path=args.valid_data_path,
            scaffold_vocab_path=args.scaffold_vocab_path,
            decoration_vocab_path=args.decoration_vocab_path,
            save_prior_path=args.save_prior_path,
            use_wandb=args.use_wandb
        )
        
        finish = time.time()
        print(f"Training completed in {(finish-start)/3600:.2f} hours", flush=True)
        
        # 保存训练历史
        torch.save({
            'train_losses': train_losses,
            'val_losses': val_losses,
            'config': {
                'd_model': d_model,
                'num_encoder_layers': num_encoder_layers,
                'num_decoder_layers': num_decoder_layers,
                'dim_feedforward': dim_feedforward,
                'nhead': nhead,
                'batch_size': batch_size,
                'num_epochs': num_epochs,
            }
        }, args.save_prior_path.replace('.ckpt', '_history.pt'))

    











