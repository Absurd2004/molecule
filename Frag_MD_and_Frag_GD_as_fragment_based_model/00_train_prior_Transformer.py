#!/usr/bin/env python
import argparse

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from MCMG_utils.data_structs import MolData, Vocabulary
from models.model_MCMG import transformer_RL
from torch.optim import Adam
from MCMG_utils.Optim import ScheduledOptim
from MCMG_utils.early_stop.pytorchtools import EarlyStopping
import time
import wandb


start = time.time()

def train_prior(train_data,valid_data,save_prior_path,use_wandb=False):

    """Trains the Prior decodertf"""

    # Read vocabulary from a file
    voc = Vocabulary(init_from_file="./data/fragments_Voc_1.csv")

    # Create a Dataset from a SMILES file
    moldata = MolData(train_data, voc)
    valid = MolData(valid_data, voc)

    train_data = DataLoader(moldata, batch_size=batch_size, shuffle=True, drop_last=True,
                      collate_fn=MolData.collate_fn) # 例如将数据拆成input_data和target两个数据，即两个batch。
                                                     # collate_f就是将smiles长度统一为max_length，不够的地方添加0 
    valid_data = DataLoader(valid, batch_size=batch_size, shuffle=True, drop_last=True,
                      collate_fn=MolData.collate_fn)

    Prior = transformer_RL(voc, d_model, nhead, num_decoder_layers,
                           dim_feedforward, max_seq_length,
                           pos_dropout, trans_dropout)

    optim = ScheduledOptim(
        Adam(Prior.decodertf.parameters(), betas=(0.9, 0.98), eps=1e-09),
        d_model * 8,n_warmup_steps)

    if use_wandb:

        wandb.init(project="Frag_MD", name="train_prior", config={
            "d_model": d_model,
            "num_decoder_layers": num_decoder_layers,
            "dim_feedforward": dim_feedforward,
            "nhead": nhead,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "max_seq_length": max_seq_length,
            "pos_dropout": pos_dropout,
            "trans_dropout": trans_dropout,
            "n_warmup_steps": n_warmup_steps,
        })

    train_losses, val_losses = train(train_data, valid_data, Prior, optim, num_epochs,save_prior_path, use_wandb=use_wandb)

    if use_wandb:

        wandb.finish()

    torch.cuda.empty_cache()



def train(train_data, valid_data, model, optim, num_epochs,save_prior_path,use_wandb=False):
    model.decodertf.to(device)
    model.decodertf.train()
    lowest_val = 1e9
    train_losses = []
    val_losses = []
    total_step = 0

    early_stopping = EarlyStopping(patience=5, verbose=False)

    for epoch in tqdm(range(num_epochs), desc="Training epochs"):
        # When training on a few million compounds, this model converges
        # in a few of epochs or even faster. If model sized is increased
        # its probably a good idea to check loss against an external set of
        # validation SMILES to make sure we dont overfit too much.
        total_loss = 0
        for step, batch in tqdm(enumerate(train_data), total=len(train_data), desc=f"Epoch {epoch+1}", leave=False):
 
            # Sample from DataLoader
            seqs = batch.long()

            # Calculate loss, each_molecule_loss is the loss of  each molecule

            loss, each_molecule_loss = model.likelihood(seqs)
            # loss = - log_p.mean()

            # Calculate gradients and take a step
            optim.zero_grad()
            loss.backward()
            optim.step_and_update_lr()
            # print(loss)

            total_loss += loss.item()
            # train_losses.append((step, loss.item()))

            if use_wandb:


                wandb.log({
                    "train/loss": loss.item(),
                    "train/lr": optim._optimizer.param_groups[0]['lr'],
                    "step": total_step,
                    "epoch": epoch
                })

            total_step += 1

            # if step % print_every == print_every - 1:


            if step % 200 == 0 and step != 0:
                # decrease_learning_rate(optim, decrease_by=0.03)
                print("*" * 50,flush=True)
                print("Epoch {:3d}   step {:3d}    loss: {:5.2f}\n".format(epoch, step, loss.data),flush=True)
            # break
        #print('average epoch loss:', total_loss / len(train_data),flush=True)  # 此处的len(train_data)是一个训练集中的batch的数目，因为train_data是dataloader的结果
        avg_epoch_loss = total_loss / len(train_data)
        print('average epoch loss:', avg_epoch_loss, flush=True)
        if use_wandb:
            wandb.log({"train/avg_epoch_loss": avg_epoch_loss, "epoch": epoch})

        val_loss = validate(valid_data, model)
        val_losses.append((total_step, val_loss))
        if use_wandb:

            wandb.log({"val/loss": val_loss, "epoch": epoch})

        early_stopping(val_loss, model.decodertf, 'qed_sa_Prior')

        if early_stopping.early_stop:
            print("Early stopping",flush=True)
            break

        # Save the Prior
        if val_loss < lowest_val:
            lowest_val = val_loss
            torch.save(model.decodertf.state_dict(), save_prior_path)
        print(f'Val Loss: {val_loss}',flush=True)
        # break
    return train_losses, val_losses


def validate(valid_data, model):
    # pbar = tqdm(total=len(iter(valid_loader)), leave=False)
    model.decodertf.to(device)
    model.decodertf.eval()
    total_loss = 0

    for step, batch in tqdm(enumerate(valid_data), total=len(valid_data)):
        with torch.no_grad():
            # Sample from DataLoader
            seqs = batch.long()

            # Calculate loss, each_molecule_loss is the loss of  each molecule
            loss, each_molecule_loss = model.likelihood(seqs)
            # loss = - log_p.mean()

            total_loss += loss.item()
            # train_losses.append((step, loss.item()))
        # break
    return total_loss / len(valid_data)



if __name__ == "__main__":
    max_seq_length = 140
    # num_tokens=71
    # vocab_size=71
    d_model = 128
    # num_encoder_layers = 6
    num_decoder_layers = 12
    dim_feedforward = 512
    nhead = 8
    pos_dropout = 0.1
    trans_dropout = 0.1
    n_warmup_steps = 500

    num_epochs = 10
    batch_size = 512
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print(device,flush=True)

    parser = argparse.ArgumentParser(description="Main script for running the model")
    parser.add_argument('--train-data', action='store', dest='train_data',default = "./data/train_1.csv")
    parser.add_argument('--valid-data', action='store', dest='valid_data',default = "./data/valid_1.csv")


    parser.add_argument('--save-prior-path', action='store', dest='save_prior_path',
                        default='./data/Transformer/train_prior_Transformer_for_TL.ckpt',
                        help='Path to save an c-Transformer checkpoint.')
    parser.add_argument('--use-wandb', action='store_true', dest='use_wandb',
                        help='Use wandb for logging. Default: False')

    arg_dict = vars(parser.parse_args())

    train_prior(**arg_dict)
    finish = time.time()
    print("{:.2f}".format((finish-start)/3600)+" hours",flush=True)
