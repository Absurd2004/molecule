#!/usr/bin/env python

import torch
from torch.utils.data import DataLoader

from rdkit import Chem
from rdkit import rdBase
from tqdm import tqdm

from MCMG_utils.data_structs_fragment import MolData, Vocabulary
from models.model import RNN
from MCMG_utils.utils import Variable, decrease_learning_rate
rdBase.DisableLog('rdApp.error')

import time

def pretrain(restore_from=None):
    """Trains the Prior RNN"""

    start_time = time.time()
    # Read vocabulary from a file
    voc = Vocabulary(init_from_file="./data/fragments_Voc.csv")

    # Create a Dataset from a SMILES file
    moldata = MolData("./data/train.csv", voc)
    data = DataLoader(moldata, batch_size=128, shuffle=True, drop_last=True,
                      collate_fn=MolData.collate_fn)

    Prior = RNN(voc)
    # if torch.cuda.is_available():
    #     Prior.rnn.load_state_dict(torch.load(restore_from))
    # else:
    #     Prior.rnn.load_state_dict(torch.load(restore_from, map_location=lambda storage,ni loc: storage))

    for param in Prior.rnn.parameters():
        param.requires_grad = True


    # Can restore from a saved RNN
    if restore_from:
        Prior.rnn.load_state_dict(torch.load(restore_from))

    optimizer = torch.optim.Adam(Prior.rnn.parameters(), lr = 0.001)
    for epoch in range(1, 11):
        # When training on a few million compounds, this model converges
        # in a few of epochs or even faster. If model sized is increased
        # its probably a good idea to check loss against an external set of
        # validation SMILES to make sure we dont overfit too much.100.110.33.24
        for step, batch in tqdm(enumerate(data), total=len(data)):

            # Sample from DataLoader
            seqs = batch.long()

            # Calculate loss
            log_p, _ = Prior.likelihood(seqs)
            loss = - log_p.mean()

            # Calculate gradients and take a step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Every 500 steps we decrease learning rate and print some information
            if step % 500 == 0 and step != 0:
            # if step == 0:
                decrease_learning_rate(optimizer, decrease_by=0.03)
                print("*" * 50,flush=True)
                print("Epoch {:3d}   step {:3d}    loss: {:5.2f}\n".format(epoch, step, loss.cpu().data),flush=True)
                seqs, likelihood, _ = Prior.sample(128)
                valid = 0
                for i, seq in enumerate(seqs.cpu().numpy()):
                    mol = voc.decode_frag(seq)
                    if mol is not None:
                        if Chem.MolToSmiles(mol):
                            valid += 1
                        if i < 5:
                            print(Chem.MolToSmiles(mol),flush=True)
                print("\n{:>5.2f}% valid SMILES".format(100 * valid / len(seqs)),flush=True)
                print("*" * 50 + "\n",flush=True)
 
                # with open('output_train_prior.txt', 'a+') as f:
                #     f.write("*" * 50+ "\n")
                #     f.write("Epoch {:3d}   step {:3d}    loss: {:5.2f}\n".format(epoch, step, loss.cpu().data))
                #     f.write("{:>5.2f}% valid SMILES\n".format(100 * valid / len(seqs)))
                # f.close()
                torch.save(Prior.rnn.state_dict(), "./data/Frag_GL/Prior_RNN_frag_GL.ckpt")
                # break
        # Save the Prior
        torch.save(Prior.rnn.state_dict(), "./data/Frag_GL/Prior_RNN_frag_GL_epoch_{}.ckpt".format(str(epoch)))
        print("train time:"+"{:.2f}".format((time.time()-start_time)/3600)+" hours",flush=True)
        # break
if __name__ == "__main__":
    pretrain()
