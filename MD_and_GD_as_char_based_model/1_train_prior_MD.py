#!/usr/bin/env python
import argparse

import torch
from torch.utils.data import DataLoader
from rdkit import Chem
from rdkit import rdBase
from tqdm import tqdm

from MCMG_utils.data_structs import MolData, Vocabulary
from models.model_rnn import RNN
from MCMG_utils.utils import  decrease_learning_rate
rdBase.DisableLog('rdApp.error')

def train_middle(train_data, save_prior_path):
    """Trains the Prior RNN"""

    # Read vocabulary from a file
    voc = Vocabulary(init_from_file="./data/voc2.csv")

    # Create a Dataset from a SMILES file
    moldata = MolData(train_data, voc)
    data = DataLoader(moldata, batch_size=64, shuffle=True, drop_last=True,
                      collate_fn=MolData.collate_fn)

    Prior = RNN(voc)


    optimizer = torch.optim.Adam(Prior.rnn.parameters(), lr = 0.001)
    for epoch in range(1, 11):

        for step, batch in tqdm(enumerate(data), total=len(data)):

            # Sample from DataLoader
            seqs = batch.long()

            # Calculate loss
            log_p = Prior.likelihood(seqs)
            loss = - log_p.mean()

            # Calculate gradients and take a step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
              
            # Every 500 steps we decrease learning rate and print some information
            if step % 500 == 0 and step != 0:
                decrease_learning_rate(optimizer, decrease_by=0.03)
                print("*" * 50,flush = True)
                print(loss.cpu().data,flush = True)
                print("Epoch {:3d}   step {:3d}    loss: {:5.2f}\n".format(epoch, step, loss.cpu().data),flush = True)
                seqs, likelihood, _ = Prior.sample(128)
                valid = 0
                for i, seq in enumerate(seqs.cpu().numpy()):
                    smile = voc.decode(seq)
                    if Chem.MolFromSmiles(smile):
                        valid += 1
                    if i < 5:
                        print(smile)
                print("\n{:>4.1f}% valid SMILES".format(100 * valid / len(seqs)),flush = True)
                print("*" * 50 + "\n",flush = True)
                torch.save(Prior.rnn.state_dict(), save_prior_path)
                # break

        
        # Save the Prior
        torch.save(Prior.rnn.state_dict(), save_prior_path)
        # break
if __name__ == "__main__":
  
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    parser = argparse.ArgumentParser(description="Main script for running the model")
    parser.add_argument('--train-data', action='store', dest='train_data',default='./data/gen_train.csv')
    # parser.add_argument('--valid-data', action='store', dest='valid_data'
    parser.add_argument('--save-prior-path', action='store', dest='save_prior_path',
                        default='./data/ML/Prior_RNN_ML.ckpt',
                        help='Path to save a checkpoint.')


    arg_dict = vars(parser.parse_args())

    train_middle(**arg_dict)
