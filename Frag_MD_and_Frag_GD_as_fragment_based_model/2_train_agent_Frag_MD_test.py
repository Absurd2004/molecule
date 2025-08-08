#!/usr/bin/env python

import torch
import numpy as np
import pandas as pd
import time
import os
from shutil import copyfile
import wandb

from models.model import RNN
from MCMG_utils.data_structs_fragment import Vocabulary, Experience

from properties import multi_scoring_functions_one_hot_dual
from properties import multi_scoring_functions_one_hot_dual_test
from properties import multi_scoring_functions_one_hot_dual_logP
from properties import get_scoring_function, qed_func, sa_func
from MCMG_utils.utils import Variable, seq_to_smiles, fraction_valid_smiles, unique, seq_to_smiles_frag

#from vizard_logger import VizardLog
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')
import logging
logger = logging.getLogger()
logger.setLevel(logging.ERROR)


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def train_agent(epoch, restore_prior_from='./data/Frag_ML/Prior_RNN_frag_ML.ckpt',
                restore_agent_from='./data/Frag_ML/Prior_RNN_frag_ML.ckpt',
                scoring_function='tanimoto',
                scoring_function_kwargs=None,
                save_dir=None, learning_rate=0.0005,
                batch_size=16, n_steps=20001,
                num_processes=0, sigma=60,
                experience_replay=True,
                use_wandb=False):
    voc = Vocabulary(init_from_file="./data/fragments_Voc2.csv")

    os.makedirs('./data/logs_ML_agent_function1',exist_ok=True)
    os.makedirs('./data/Agent_ML_training_logs', exist_ok=True)
    log_file = f'./data/Agent_ML_training_logs/epoch_{epoch}_detailed_log.txt'
    #logger = VizardLog('./data/logs_ML_agent_function1')

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
    #logger.log(Agent.rnn.gru_2.weight_ih.cpu().data.numpy()[::100], "init_weight_GRU_layer_2_w_ih")
    #logger.log(Agent.rnn.gru_2.weight_hh.cpu().data.numpy()[::100], "init_weight_GRU_layer_2_w_hh")
    #logger.log(Agent.rnn.embedding.weight.cpu().data.numpy()[::30], "init_weight_GRU_embedding")
    #logger.log(Agent.rnn.gru_2.bias_ih.cpu().data.numpy(), "init_weight_GRU_layer_2_b_ih")
    #logger.log(Agent.rnn.gru_2.bias_hh.cpu().data.numpy(), "init_weight_GRU_layer_2_b_hh")

    print("Model initialized, starting training...")

    # Scoring_function
    # scoring_function = get_scoring_function('st_abs')
    # scoring_function2 = get_scoring_function('gsk3')
    smiles_save = []
    expericence_step_index = []
    score_list = []

    if use_wandb:
        wandb.init(project="RL_fragment", name=f"train_agent_epoch_{epoch}", config={
            "batch_size": batch_size,
            "num_steps": n_steps,
            "sigma": sigma,
        })

    for step in tqdm(range(n_steps), total=n_steps, desc=f"Epoch {epoch}", unit="step"):

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

        score1,score2 = get_scoring_function('st_hl_f2')(smiles)
        logP = get_scoring_function('logP')(smiles)
        #symmetry = get_scoring_function('symmetry')(smiles)
       
        qed = qed_func()(smiles)
      

        sa = np.array([float(x < 4.0) for x in sa_func()(smiles)],
                      dtype=np.float32)  # to keep all reward components between [0,1]
        
        #score = score1 + score2 + qed + sa
        #score = logP + symmetry + qed + sa
        score = score1 + score2 + logP + qed + sa
        score_list.append(np.mean(score))
        # 判断是否为success分子，并储存
        #success_score = multi_scoring_functions_one_hot_dual(smiles, ['st_hl_f2','logP', 'qed', 'sa'])
        success_score = multi_scoring_functions_one_hot_dual_test(smiles, ['st_hl_f2','logP', 'qed', 'sa'])
        #success_score = multi_scoring_functions_one_hot_dual_logP(smiles, ['logP', 'qed', 'sa'])
        itemindex = list(np.where(success_score == 5))
        success_smiles = np.array(smiles)[itemindex]
        success_smiles = [i.tolist()[0] for i in success_smiles if i.size!=0]
        smiles_save.extend(success_smiles)
        expericence_step_index = expericence_step_index + len(success_smiles) * [step]

        # TODO
        if step+1 >= n_steps:
            print('num: ', len(set(smiles_save)))
            save_smiles_df = pd.concat([pd.DataFrame(smiles_save), pd.DataFrame(expericence_step_index)], axis=1)
            os.makedirs('./data/Frag_ML_agent_function1_little', exist_ok=True)
            save_smiles_df.to_csv('./data/Frag_ML_agent_function1_little/' + 'epoch_'+str(epoch) +'_smiles.csv', index=False, header=False)
            pd.DataFrame(score_list).to_csv('./data/Frag_ML_agent_function1_little/' + 'epoch_'+str(epoch) +'_scores.csv', index=False, header=False)

            os.makedirs('./data/Agent_ML_models_little', exist_ok=True)
            torch.save(Agent.rnn.state_dict(), f'./data/Agent_ML_models_little/Agent_RNN_epoch_{epoch}.ckpt')
            print(f"Agent model saved to ./data/Agent_ML_models_little/Agent_RNN_epoch_{epoch}.ckpt")
            break


        # Calculate augmented likelihood
        augmented_likelihood = prior_likelihood + sigma * Variable(score)
        loss = torch.pow((augmented_likelihood - agent_likelihood), 2)

        if use_wandb:
            wandb.log({
                "loss": float(loss.mean().item()),
                "avg_score": float(np.mean(score)),
                "avg_score1": float(np.mean(score1)),
                "avg_score2": float(np.mean(score2)),
                "avg_logp": float(np.mean(logP)),
                "avg_qed": float(np.mean(qed)),
                "avg_sa": float(np.mean(sa)),
                "fraction_valid": fraction_valid_smiles(smiles),
                "num_success_smiles": len(smiles_save),
                "step": step,
                "epoch": epoch
            })

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

        if step % 100 == 0:  # 每100步显示一次
            print(f"Step: {step}/{n_steps}, Valid SMILES: {fraction_valid_smiles(smiles)*100:.1f}%, "
                  f"Avg Score: {np.mean(score):.2f}, Success: {len(success_smiles)}, "
                  f"Time: {time_elapsed:.2f}h/{time_left:.2f}h")

        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"\n=== Step: {step} ===\n")
            f.write(f"Fraction valid SMILES: {fraction_valid_smiles(smiles)*100:.1f}%\n")
            f.write(f"Time elapsed: {time_elapsed:.2f}h, Time left: {time_left:.2f}h\n")
            f.write(f"Average score: {np.mean(score):.4f}\n")
            f.write(f"Success molecules: {len(success_smiles)}\n")
            f.write("Agent    Prior   Target   Score             SMILES\n")
            
            try:
                if step*batch_size <= 20000:  # 前20000个分子记录详细信息
                    for i in range(min(len(agent_likelihood), batch_size)):
                        f.write(f"{agent_likelihood[i]:6.2f}   {prior_likelihood[i]:6.2f}  "
                               f"{augmented_likelihood[i]:6.2f}  {score[i]:6.2f}     {smiles[i]}\n")
                else:
                    for i in range(min(len(agent_likelihood), 10)):  # 只记录前10个
                        f.write(f"{agent_likelihood[i]:6.2f}   {prior_likelihood[i]:6.2f}  "
                               f"{augmented_likelihood[i]:6.2f}  {score[i]:6.2f}     {smiles[i]}\n")
            except Exception as e:
                f.write(f"Error writing molecules: {e}\n")
            
            f.write("\n")
    if use_wandb:
        wandb.finish()
            
            
        

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Main script for running the model")
    parser.add_argument('--num-steps', action='store', dest='n_steps', type=int,
                        default=3001)
    parser.add_argument('--batch-size', action='store', dest='batch_size', type=int,
                        default=16)
    parser.add_argument('--sigma', action='store', dest='sigma', type=int,
                        default=60)

    parser.add_argument('--use-wandb', action='store_true', dest='use_wandb',
                        help='Use wandb for logging. Default: False')
    # parser.add_argument('--epoch', action='store', dest='epoch', type=int,
    #                     default=1)
    arg_dict = vars(parser.parse_args())

    for i in range(1,4):
        train_agent(epoch=i,**arg_dict)

