import os

# Encode a fragment.
from rdkit import Chem
from mol_utils import join_fragments
from mol_utils import split_molecule
import pandas as pd
from tqdm import tqdm

def encode_molecule(m):
    fs_list = []
    count = 0
    error_index = []
    for index, i in enumerate(tqdm(m, desc="Encoding molecules")):
        try:
            fs = [Chem.MolToSmiles(f) for f in split_molecule(Chem.MolFromSmiles(i))]
            # encoded = "-".join([encodings[f] for f in fs])
            fs_list.extend(fs)
            # print(index)
        except Exception as e:
            error_index.append(index)
  
    return fs_list, error_index

def decode_molecule(enc):
    fs_list = []
    for index, i in enumerate(enc):
        fs = [Chem.MolFromSmiles(x) for x in i]
        fs = join_fragments(fs)
        fs_list.append(Chem.MolToSmiles(fs))
        print(index)
    return fs_list

def decode_molecule_test(enc):
    fs_list = []
    fs = [Chem.MolFromSmiles(x) for x in enc]
    fs = join_fragments(fs)
    fs_list.append(Chem.MolToSmiles(fs))
    return fs_list


train_data = pd.read_csv('/home/yangyongkun/Fragment_and_char_based_PSs_generation/MD_and_GD_as_char_based_model/data/train_data_smiles_qed_sa.csv')
valid_data = pd.read_csv('/home/yangyongkun/Fragment_and_char_based_PSs_generation/MD_and_GD_as_char_based_model/data/valid_data_smiles_qed_sa.csv')
test_data = pd.read_csv('/home/yangyongkun/Fragment_and_char_based_PSs_generation/MD_and_GD_as_char_based_model/data/test_data_smiles_qed_sa.csv')

train_smiles = train_data['smiles'].values.flatten().tolist()
valid_smiles = valid_data['smiles'].values.flatten().tolist()
test_smiles = test_data['smiles'].values.flatten().tolist()



# tain_data
enc,drop_index = encode_molecule(train_smiles)

with open('./fragment_utils/fragments_counts.txt',"a") as f:
    f.write("the fragment number of train data (not drop duplicated):  " + str(len(enc))+'\n')
    f.close()
# print(len(drop_index))
# print(drop_index)
# train_data_filtered = train_data.drop(drop_index)

# print(len(train_data),len(train_data_filtered))
# print(train_data.index[:100])
train_data_filtered = train_data.drop(drop_index)
train_data_filtered.to_csv('./fragment_utils/train_1.csv')

output1 = set(enc)



# valid_data
enc,drop_index = encode_molecule(valid_smiles)

with open('./fragment_utils/fragments_counts.txt',"a") as f:
    f.write("the fragment number of valid data (not drop duplicated):  " + str(len(enc))+'\n')
    f.close()

valid_data_filtered = valid_data.drop(drop_index)
valid_data_filtered.to_csv('./fragment_utils/valid_1.csv')

output2 = set(enc)


# test_data

enc,drop_index = encode_molecule(test_smiles)

with open('./fragment_utils/fragments_counts.txt',"a") as f:
    f.write("the fragment number of test data (not drop duplicated):  " + str(len(enc))+'\n')
    f.close()

test_data_filtered = test_data.drop(drop_index)
test_data_filtered.to_csv('./fragment_utils/test_1.csv')

output3 = set(enc)


output = output1 | output2 | output3
output = list(output)
pd.DataFrame(output).to_csv('./fragment_utils/fragments_Voc_1.csv',header=False,index=False)

# frg_list = ['C[Yb]',  '[Yb]C1CCN([Lu])CC1', '[Yb]N[Lu]', '[Yb]c1cc([Lu])ccc1[Ta]','[Yb]c1cc2c([Lu])ncnc2[nH]1', '[Yb]O[Lu]', '[Yb]c1ccc2oc([Lu])nc2c1', '[Yb]N[Lu]', '[Yb]c1ccc([Lu])cc1', 'Cl[Yb]', 'C[Yb]']

# dec = decode_molecule_test(enc)

# print(dec)