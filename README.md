# Frag_MD/GD and MD/GD 
Effective Generation of Heavy-Atom-Free Triplet Photosensitizers Containing Multiple Intersystem Crossing Mechanisms Based on Machine Learning


### 📦 Download Data

The dataset used in this project is publicly available via Zenodo:

🔗 [Download from Zenodo (DOI: 10.5281/zenodo.15638748)](https://zenodo.org/records/15638748)

Please download and unzip the file into the `./data/` directory:

```bash
unzip data.zip -d ./data/
```


## Environment
- python = 3.8.3
- pytroch = 1.6.0
- RDKit
- numpy
- pandas



## How to run？
Task 1 is de novo design focused on searching new conjugated motifs and task 2 is a conjugated-motif-based method focused on conjugated motif derivation. A reinforce learning (RL) workflow was adopted which mainly consist with a graph convolutional neural network (GCNN) as prediction model and recurrent neural network (RNN) for molecule generation.
The constrained conditions for a successful triplet PSs include four parts:
- ΔE<sub>ST</sub> ≤ 0.30
- E<sub>abs</sub> ≤ 2.48
- QED ≥ 0.38
- SA ≤ 4.0


### Task 1 


```
cd Frag_MD_and_Frag_GD_as_fragment_based_model

# 1.0 preparing

python 00_train_prior_Transformer.py --train-data {your_training_data_path} --valid-data {your_valid_data_path} --save-prior-path {path_to_save_prior_model}

python 01_prior_Transformer_generating_molecules.py --prior {piror_model_path} --save_molecules_path {save_molecules_path}

01_processing_for_fragments_Voc2_for_data_split.ipynb



# 2.0 train and prediction

##### Frag_GD and Frag_GB

python 1_train_prior_Frag_GD.py --train-data {your_training_data_path} --save-middle-path {path_to_save_middle_model}

python 2_train_agent_Frag_GD.py  --num-steps 5000 --batch-size 128 --middle {path_of_middle_model} --agent {path_to_save_agent_model} ---save-file-path{save_smiles}

python 2_train_agent_Frage_GB.py  --num-steps 5000 --batch-size 128 --middle {path_of_middle_model} --agent {path_to_save_agent_model} ---save-file-path{save_smiles}

##### Frag_MD and Frag_MB

python 1_train_prior_Frag_MD.py --train-data {your_training_data_path} --save-middle-path {path_to_save_middle_model}

python 2_train_agent_Frag_MD.py  --num-steps 5000 --batch-size 128 --middle {path_of_middle_model} --agent {path_to_save_agent_model} ---save-file-path{save_smiles}

python 2_train_agent_Frag_MB.py  --num-steps 5000 --batch-size 128 --middle {path_of_middle_model} --agent {path_to_save_agent_model} ---save-file-path{save_smiles}


```


### Task 2

```
cd MD_and_GD_as_char_based_model


# 1.0 preparing

python 00_train_prior_Transformer.py --train-data {your_training_data_path} --valid-data {your_valid_data_path} --save-prior-path {path_to_save_prior_model}

python 01_prior_Transformer_generating_molecules.py --prior {piror_model_path} --save_molecules_path {save_molecules_path}

01_processing_for_fragments_Voc2_for_data_split.ipynb



# 2.0 train and prediction

# GD and GB

python 1_train_prior_GD.py --train-data {your_training_data_path} --save-middle-path {path_to_save_middle_model}

python 2_train_agent_GD.py  --num-steps 5000 --batch-size 128 --middle {path_of_middle_model} --agent {path_to_save_agent_model} ---save-file-path{save_smiles}

python 2_train_agent_GB.py  --num-steps 5000 --batch-size 128 --middle {path_of_middle_model} --agent {path_to_save_agent_model} ---save-file-path{save_smiles}

# MD and MB

python 1_train_prior_MD.py --train-data {your_training_data_path} --save-middle-path {path_to_save_middle_model}

python 2_train_agent_MD.py  --num-steps 5000 --batch-size 128 --middle {path_of_middle_model} --agent {path_to_save_agent_model} ---save-file-path{save_smiles}

python 2_train_agent_MB.py  --num-steps 5000 --batch-size 128 --middle {path_of_middle_model} --agent {path_to_save_agent_model} ---save-file-path{save_smiles}


```




## References

1. Wang J, Zeng Y, Sun H, Wang J, Wang X, Jin R, Wang M, Zhang X, Cao D, Chen X, Hsieh C-Y, Hou T (2023) Molecular Generation with Reduced Labeling through Constraint Architecture. J. Chem. Inf. Model. 63:3319−3327 [https://pubs.acs.org/doi/10.1021/acs.jcim.3c00579](https://pubs.acs.org/doi/10.1021/acs.jcim.3c00579)
2. Wang J, Hsieh C-Y, Wang M, Wang X, Wu Z, Jiang D, Liao B, Zhang X, Yang B, He Q, Cao D, Chen X, Hou T (2021) Multi-constraint molecular generation based on conditional transformer, knowledge distillation and reinforcement learning. Nat. Mach. Intell. 3:914–922 | [https://www.nature.com/articles/s42256-021-00403-1](https://www.nature.com/articles/s42256-021-00403-1)
