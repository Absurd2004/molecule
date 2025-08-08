#!/usr/bin/env python

import torch
import numpy as np
import pandas as pd
import time
import os
from shutil import copyfile

from models.model import RNN
from MCMG_utils.data_structs_fragment import Vocabulary, Experience

from properties import multi_scoring_functions_one_hot_dual
from properties import get_scoring_function, qed_func, sa_func
from MCMG_utils.utils import Variable, seq_to_smiles, fraction_valid_smiles, unique, seq_to_smiles_frag
from vizard_logger import VizardLog

import warnings
warnings.filterwarnings('ignore')
import logging
logger = logging.getLogger()
logger.setLevel(logging.ERROR)


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def train_agent(epoch, restore_prior_from='./data/Frag_GL/Prior_RNN_frag_GL_epoch_10.ckpt',
                restore_agent_from='./data/Frag_GL/Prior_RNN_frag_GL_epoch_10.ckpt',
                scoring_function='tanimoto',
                scoring_function_kwargs=None,
                save_dir=None, learning_rate=0.0005,
                batch_size=16, n_steps=20001,
                num_processes=0, sigma=60,
                experience_replay=True):
    voc = Vocabulary(init_from_file="./data/fragments_Voc.csv")

    logger = VizardLog('data/logs_GL_agent_function2')

    start_time = time.time()

    Prior = RNN(voc)
    Agent = RNN(voc)

    # By default restore Agent to same model as Prior, but can restore from already trained Agent too.
    # Saved models are partially on the GPU, but if we dont have cuda enabled we can remap these
    # to the CPU.
    if torch.cuda.is_available():
        Prior.rnn.load_state_dict(torch.load(restore_prior_from,map_location={'cuda:0':'cuda:0'}))
        Agent.rnn.load_state_dict(torch.load(restore_agent_from))
    else:
        Prior.rnn.load_state_dict(torch.load(restore_prior_from, map_location=lambda storage, loc: storage))
        Agent.rnn.load_state_dict(torch.load(restore_agent_from, map_location=lambda storage, loc: storage))

    # We dont need gradients with respect to Prior
    for param in Prior.rnn.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(Agent.rnn.parameters(), lr=0.0001)


    # For policy based RL, we normally train on-policy and correct for the fact that more likely actions
    # occur more often (which means the agent can get biased towards them). Using experience replay is
    # therefor not as theoretically sound as it is for value based RL, but it seems to work well.
    experience = Experience(voc)

    # Log some network weights that can be dynamically plotted with the Vizard bokeh app
    logger.log(Agent.rnn.gru_2.weight_ih.cpu().data.numpy()[::100], "init_weight_GRU_layer_2_w_ih")
    logger.log(Agent.rnn.gru_2.weight_hh.cpu().data.numpy()[::100], "init_weight_GRU_layer_2_w_hh")
    logger.log(Agent.rnn.embedding.weight.cpu().data.numpy()[::30], "init_weight_GRU_embedding")
    logger.log(Agent.rnn.gru_2.bias_ih.cpu().data.numpy(), "init_weight_GRU_layer_2_b_ih")
    logger.log(Agent.rnn.gru_2.bias_hh.cpu().data.numpy(), "init_weight_GRU_layer_2_b_hh")

    print("Model initialized, starting training...")

    # Scoring_function
    # scoring_function = get_scoring_function('st_abs')
    # scoring_function2 = get_scoring_function('gsk3')
    smiles_save = []
    expericence_step_index = []
    score_list = []
    for step in range(n_steps):

        # Sample from Agent
        seqs, agent_likelihood, entropy = Agent.sample(batch_size=batch_size)

        # Remove duplicates, ie only consider unique seqs
        unique_idxs = unique(seqs)
        seqs = seqs[unique_idxs]
        agent_likelihood = agent_likelihood[unique_idxs]
        entropy = entropy[unique_idxs]

        # Get prior likelihood and score
        prior_likelihood,_ = Prior.likelihood(Variable(seqs))
        smiles = seq_to_smiles_frag(seqs, voc)

        score1,score2 = get_scoring_function('st_abs')(smiles)
       
        qed = qed_func()(smiles)
      

        sa = np.array([float(x < 4.0) for x in sa_func()(smiles)],
                      dtype=np.float32)  # to keep all reward components between [0,1]
        
        score = score1 + score2 + qed + sa
        score_list.append(np.mean(score))
        # 判断是否为success分子，并储存
        success_score = multi_scoring_functions_one_hot_dual(smiles, ['st_abs', 'qed', 'sa'])
        itemindex = list(np.where(success_score == 4))
        success_smiles = np.array(smiles)[itemindex]
        success_smiles = [i.tolist()[0] for i in success_smiles if i.size!=0]
        smiles_save.extend(success_smiles)
        expericence_step_index = expericence_step_index + len(success_smiles) * [step]

        # TODO
        if step+1 >= n_steps:
            print('num: ', len(set(smiles_save)))
            save_smiles_df = pd.concat([pd.DataFrame(smiles_save), pd.DataFrame(expericence_step_index)], axis=1)
            save_smiles_df.to_csv('./data/Frag_GL_agent_function2/' + 'epoch_'+str(epoch) +'_smiles.csv', index=False, header=False)
            pd.DataFrame(score_list).to_csv('./data/Frag_GL_agent_function2/' + 'epoch_'+str(epoch) +'_scores.csv', index=False, header=False)
            break


        # Calculate augmented likelihood
        augmented_likelihood = prior_likelihood + sigma * Variable(score)
        loss = torch.pow((augmented_likelihood - agent_likelihood), 2)

        # Experience Replay
        # First sample
        if experience_replay and len(experience) > 4:
            exp_seqs, exp_score, exp_prior_likelihood = experience.sample(4)
            # print( exp_seqs, exp_score, exp_prior_likelihood)
            # index_= []
            # for i in range(4):
            #     if exp_seqs[i,:].sum() > 0:
            #         index_.append(i)
            index_ = [i for i in range(4) if exp_seqs[i,:].sum() > 0]
            exp_seqs, exp_score, exp_prior_likelihood = exp_seqs[index_,:], exp_score[index_], exp_prior_likelihood[index_]
            # print(exp_seqs, exp_score, exp_prior_likelihood)
            exp_agent_likelihood, exp_entropy = Agent.likelihood(exp_seqs.long())
            exp_augmented_likelihood = exp_prior_likelihood + sigma * exp_score
            exp_loss = torch.pow((Variable(exp_augmented_likelihood) - exp_agent_likelihood), 2)
            loss = torch.cat((loss, exp_loss), 0)
            agent_likelihood = torch.cat((agent_likelihood, exp_agent_likelihood), 0)

        # Then add new experience
        prior_likelihood = prior_likelihood.data.cpu().numpy()
        new_experience = zip(smiles, score, prior_likelihood)
        experience.add_experience(new_experience)

        # Calculate loss
        loss = loss.mean()

        # Add regularizer that penalizes high likelihood for the entire sequence
        loss_p = - (1 / agent_likelihood).mean()
        loss += 5 * 1e3 * loss_p

        # Calculate gradients and make an update to the network weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Convert to numpy arrays so that we can print them
        augmented_likelihood = augmented_likelihood.data.cpu().numpy()
        agent_likelihood = agent_likelihood.data.cpu().numpy()

        # Print some information for this step
        time_elapsed = (time.time() - start_time) / 3600
        time_left = (time_elapsed * ((n_steps - step) / (step + 1)))

        print("\n Step {},  Fraction valid SMILES: {:5.2f},  Time elapsed: {:.2f}h, Time left: {:.2f}h".format(step, fraction_valid_smiles(smiles) * 100, time_elapsed, time_left))
        print("  Agent    Prior   Target   Score             SMILES")

        try:
            if step*batch_size <= 20000:  #keep recored 20000 molecules at lest
                
                for i in range(batch_size):
                    print(" {:6.2f}   {:6.2f}  {:6.2f}  {:6.2f}     {}".format(agent_likelihood[i],
                                                                       prior_likelihood[i],
                                                                       augmented_likelihood[i],
                                                                       score[i],
                                                                        smiles[i]))
            else:
                for i in range(10):
                    print(" {:6.2f}   {:6.2f}  {:6.2f}  {:6.2f}     {}".format(agent_likelihood[i],
                                                                        prior_likelihood[i],
                                                                        augmented_likelihood[i],
                                                                        score[i],
                                                                        smiles[i])) 
        except Exception as e:
            print (e)
            
            
        

if __name__ == "__main__":
    for i in range(1,4):
        train_agent(epoch=i)

