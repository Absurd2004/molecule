import numpy as np
import pandas as pd
import random
import re
import pickle
from rdkit import Chem
import sys
import time
import torch
from torch.utils.data import Dataset
from fragment_utils.mol_utils import MOL_SPLIT_START, split_molecule
from typing import List, Tuple, Optional

from .utils import Variable

from fragment_utils.mol_utils import split_molecule, join_fragments

class Vocabulary(object):
    """A class for handling encoding/decoding from SMILES to an array of indices"""

    def __init__(self, init_from_file=None, max_length=140):
        # self.special_tokens = ['EOS', 'GO']
        self.special_tokens = ['EOS', 'GO', 'high_QED', 'low_QED',
                               'good_SA', 'bad_SA']
        self.additional_chars = set()
        self.chars = self.special_tokens
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        self.reversed_vocab = {v: k for k, v in self.vocab.items()}
        self.max_length = max_length
        if init_from_file: self.init_from_file(init_from_file)
        #输出self.vocab和reversed_vocab的内容
        #print(f"Vocabulary initialized with {len(self.vocab)} tokens.", flush=True)
        #print(f"Vocabulary: {self.vocab}", flush=True)


    def encode(self, char_list):
        """Takes a list of characters (eg '[NH]') and encodes to array of indices"""
        smiles_matrix = np.zeros(len(char_list), dtype=np.float32)
        for i, char in enumerate(char_list):
            #print(f"Encoding character: {char}", flush=True)
            smiles_matrix[i] = self.vocab[char]
        return smiles_matrix

    def decode(self, matrix):
        """Takes an array of indices and returns the corresponding SMILES"""
        chars = []
        for i in matrix:
            if i == self.vocab['EOS']: break
            chars.append(self.reversed_vocab[i])
        smiles = "".join(chars)
        smiles = smiles.replace("L", "Cl").replace("R", "Br")
        return smiles
    def decode_frag(self, matrix):
        """Takes an array of indices and returns the corresponding SMILES"""
        chars = []
        for i in matrix:
            if i == self.vocab['EOS']: break
            chars.append(self.reversed_vocab[i])
        smiles = decode_molecule(chars)
        return smiles
    def tokenize(self, smiles):
        """Takes a SMILES and return a list of characters/tokens"""
        # regex = '(\[[^\[\]]{1,6}\])'
        # smiles = replace_halogen(smiles)
        # char_list = re.split(regex, smiles)
        tokenized =encode_molecule(smiles) 
        # for char in char_list:
        #     if char.startswith('['):
        #         tokenized.append(char)
        #     else:
        #         chars = [unit for unit in char]
        #         [tokenized.append(unit) for unit in chars]
        tokenized.append('EOS')
        return tokenized

    def add_characters(self, chars):
        """Adds characters to the vocabulary"""
        for char in chars:
            self.additional_chars.add(char)
        char_list = list(self.additional_chars)
        char_list.sort()
        self.chars = char_list + self.special_tokens
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        self.reversed_vocab = {v: k for k, v in self.vocab.items()}

    # def token_conditional_token(self, con_list):
    #     for i, char in enumerate(con_list):
    #         smiles_matrix[i] = self.vocab[char]


    def init_from_file(self, file):
        """Takes a file containing \n separated characters to initialize the vocabulary"""
        with open(file, 'r') as f:
            chars = f.read().split()
        self.add_characters(chars)

    def __len__(self):
        return len(self.chars)

    def __str__(self):
        return "Vocabulary containing {} tokens: {}".format(len(self), self.chars)


class MolData(Dataset):
    """Custom PyTorch Dataset that takes a file containing SMILES.

        Args:
                fname : path to a file containing \n separated SMILES.
                voc   : a Vocabulary instance

        Returns:
                A custom PyTorch dataset for training the Prior.
    """

    def __init__(self, fname, voc):
        self.voc = voc
        # self.smiles = []
        df = pd.read_csv(fname)
        self.smiles = df['smiles'].values.tolist()
        # convert conditional to token
        self.con = df[['qed', 'sa']]
        self.con = self.condition_convert(self.con).values.tolist()


    def __getitem__(self, i):
        con_token = self.con[i]
        mol = self.smiles[i]
        tokenized = self.voc.tokenize(mol)
        # add token to smilesxian
        tokenized = con_token + tokenized
        # encoded
        encoded = self.voc.encode(tokenized)
        return Variable(encoded)

    def __len__(self):
        return len(self.smiles)

    def __str__(self):
        return "Dataset containing {} structures.".format(len(self))

    @classmethod
    def collate_fn(cls, arr):
        """Function to take a list of encoded sequences and turn them into a batch"""
        max_length = max([seq.size(0) for seq in arr])
        collated_arr = Variable(torch.zeros(len(arr), max_length))
        for i, seq in enumerate(arr):
            collated_arr[i, :seq.size(0)] = seq
        return collated_arr

    def condition_convert(self, con_df):
        # convert to 0, 1
        # con_df['drd2'][con_df['drd2'] >= 0.5] = 1
        # con_df['drd2'][con_df['drd2'] < 0.5] = 0
        con_df['qed'][con_df['qed'] >= 0.38] = 1
        con_df['qed'][con_df['qed'] < 0.38] = 0
        con_df['sa'][con_df['sa'] <= 4.0] = 1
        con_df['sa'][con_df['sa'] > 4.0] = 0

        # convert to token

        # con_df['drd2'][con_df['drd2'] == 1] = 'is_DRD2'
        # con_df['drd2'][con_df['drd2'] == 0] = 'not_DRD2'
        con_df['qed'][con_df['qed'] == 1] = 'high_QED'
        con_df['qed'][con_df['qed'] == 0] = 'low_QED'
        con_df['sa'][con_df['sa'] == 1] = 'good_SA'
        con_df['sa'][con_df['sa'] == 0] = 'bad_SA'

        return con_df


class Experience(object):
    """Class for prioritized experience replay that remembers the highest scored sequences
       seen and samples from them with probabilities relative to their scores."""

    def __init__(self, voc, max_size=100):
        self.memory = []
        self.max_size = max_size
        self.voc = voc

    def add_experience(self, experience):
        """Experience should be a list of (smiles, score, prior likelihood) tuples"""
        self.memory.extend(experience)
        if len(self.memory) > self.max_size:
            # Remove duplicates
            idxs, smiles = [], []
            for i, exp in enumerate(self.memory):
                if exp[0] not in smiles:
                    idxs.append(i)
                    smiles.append(exp[0])
            self.memory = [self.memory[idx] for idx in idxs]
            # Retain highest scores
            self.memory.sort(key=lambda x: x[1], reverse=True)
            self.memory = self.memory[:self.max_size]
            print("\nBest score in memory: {:.2f}".format(self.memory[0][1]))

    def sample(self, n):
        """Sample a batch size n of experience"""
        if len(self.memory) < n:
            raise IndexError('Size of memory ({}) is less than requested sample ({})'.format(len(self), n))
        else:
            scores = [x[1] for x in self.memory]
            sample = np.random.choice(len(self), size=n, replace=False, p=scores / np.sum(scores))
            sample = [self.memory[i] for i in sample]
            smiles = [x[0] for x in sample]
            scores = [x[1] for x in sample]
            prior_likelihood = [x[2] for x in sample]
        tokenized = [self.voc.tokenize(smile) for smile in smiles]
        encoded = [Variable(self.voc.encode(tokenized_i)) for tokenized_i in tokenized]
        encoded = MolData.collate_fn(encoded)
        return encoded, np.array(scores), np.array(prior_likelihood)

    def initiate_from_file(self, fname, scoring_function, Prior):
        """Adds experience from a file with SMILES
           Needs a scoring function and an RNN to score the sequences.
           Using this feature means that the learning can be very biased
           and is typically advised against."""
        with open(fname, 'r') as f:
            smiles = []
            for line in f:
                smile = line.split()[0]
                if Chem.MolFromSmiles(smile):
                    smiles.append(smile)
        scores = scoring_function(smiles)
        tokenized = [self.voc.tokenize(smile) for smile in smiles]
        encoded = [Variable(self.voc.encode(tokenized_i)) for tokenized_i in tokenized]
        encoded = MolData.collate_fn(encoded)
        prior_likelihood, _ = Prior.likelihood(encoded.long())
        prior_likelihood = prior_likelihood.data.cpu().numpy()
        new_experience = zip(smiles, scores, prior_likelihood)
        self.add_experience(new_experience)

    def print_memory(self, path):
        """Prints the memory."""
        print("\n" + "*" * 80 + "\n")
        print("         Best recorded SMILES: \n")
        print("Score     Prior log P     SMILES\n")
        with open(path, 'w') as f:
            f.write("SMILES Score PriorLogP\n")
            for i, exp in enumerate(self.memory[:100]):
                if i < 50:
                    print("{:4.2f}   {:6.2f}        {}".format(exp[1], exp[2], exp[0]))
                    f.write("{} {:4.2f} {:6.2f}\n".format(*exp))
        print("\n" + "*" * 80 + "\n")

    def __len__(self):
        return len(self.memory)


def replace_halogen(string):
    """Regex to replace Br and Cl with single letters"""
    br = re.compile('Br')
    cl = re.compile('Cl')
    string = br.sub('R', string)
    string = cl.sub('L', string)

    return string


def tokenize(smiles):
    """Takes a SMILES string and returns a list of tokens.
    This will swap 'Cl' and 'Br' to 'L' and 'R' and treat
    '[xx]' as one token."""
    regex = '(\[[^\[\]]{1,6}\])'
    smiles = replace_halogen(smiles)
    char_list = re.split(regex, smiles)
    tokenized = []
    for char in char_list:
        if char.startswith('['):
            tokenized.append(char)
        else:
            chars = [unit for unit in char]
            [tokenized.append(unit) for unit in chars]
    tokenized.append('EOS')
    return tokenized


def canonicalize_smiles_from_file(fname):
    """Reads a SMILES file and returns a list of RDKIT SMILES"""
    with open(fname, 'r') as f:
        smiles_list = []
        for i, line in enumerate(f):
            if i % 100000 == 0:
                print("{} lines processed.".format(i))
            smiles = line.split(" ")[0]
            mol = Chem.MolFromSmiles(smiles)
            if filter_mol(mol):
                smiles_list.append(Chem.MolToSmiles(mol))
        print("{} SMILES retrieved".format(len(smiles_list)))
        return smiles_list


def filter_mol(mol, max_heavy_atoms=50, min_heavy_atoms=10, element_list=[6, 7, 8, 9, 16, 17, 35]):
    """Filters molecules on number of heavy atoms and atom types"""
    if mol is not None:
        num_heavy = min_heavy_atoms < mol.GetNumHeavyAtoms() < max_heavy_atoms
        elements = all([atom.GetAtomicNum() in element_list for atom in mol.GetAtoms()])
        if num_heavy and elements:
            return True
        else:
            return False


def write_smiles_to_file(smiles_list, fname):
    """Write a list of SMILES to a file."""
    with open(fname, 'w') as f:
        for smiles in smiles_list:
            f.write(smiles + "\n")


def filter_on_chars(smiles_list, chars):
    """Filters SMILES on the characters they contain.
       Used to remove SMILES containing very rare/undesirable
       characters."""
    smiles_list_valid = []
    for smiles in smiles_list:
        tokenized = tokenize(smiles)
        if all([char in chars for char in tokenized][:-1]):
            smiles_list_valid.append(smiles)
    return smiles_list_valid


def filter_file_on_chars(smiles_fname, voc_fname):
    """Filters a SMILES file using a vocabulary file.
       Only SMILES containing nothing but the characters
       in the vocabulary will be retained."""
    smiles = []
    with open(smiles_fname, 'r') as f:
        for line in f:
            smiles.append(line.split()[0])
    print(smiles[:10])
    chars = []
    with open(voc_fname, 'r') as f:
        for line in f:
            chars.append(line.split()[0])
    print(chars)
    valid_smiles = filter_on_chars(smiles, chars)
    with open(smiles_fname + "_filtered", 'w') as f:
        for smiles in valid_smiles:
            f.write(smiles + "\n")


def combine_voc_from_files(fnames):
    """Combine two vocabularies"""
    chars = set()
    for fname in fnames:
        with open(fname, 'r') as f:
            for line in f:
                chars.add(line.split()[0])
    with open("_".join(fnames) + '_combined', 'w') as f:
        for char in chars:
            f.write(char + "\n")


def construct_vocabulary(smiles_list):
    """Returns all the characters present in a SMILES file.
       Uses regex to find characters/tokens of the format '[x]'."""
    add_chars = set()
    for i, smiles in enumerate(smiles_list):
        regex = '(\[[^\[\]]{1,6}\])'
        smiles = replace_halogen(smiles)
        char_list = re.split(regex, smiles)
        for char in char_list:
            if char.startswith('['):
                add_chars.add(char)
            else:
                chars = [unit for unit in char]
                [add_chars.add(unit) for unit in chars]

    print("Number of characters: {}".format(len(add_chars)))
    with open('data/Voc_RE', 'w') as f:
        for char in add_chars:
            f.write(char + "\n")
    return add_chars

def encode_molecule(m):
    fs = [Chem.MolToSmiles(f) for f in split_molecule(Chem.MolFromSmiles(m))]
    return fs

def decode_molecule(enc):
    fs = [Chem.MolFromSmiles(x) for x in enc]
    try:
        fs = join_fragments(fs)
    except Exception as e:
        fs = None
    return fs
# if __name__ == "__main__":
#     smiles_file = sys.argv[1]
#     print("Reading smiles...")
#     smiles_list = canonicalize_smiles_from_file(smiles_file)
#     print("Constructing vocabulary...")
#     voc_chars = construct_vocabulary(smiles_list)
#     write_smiles_to_file(smiles_list, "data/mols_filtered.smi")

def _count_cut_sites(mol: Chem.Mol) -> int:
    """统计片段里的切割位点标记（mol_utils 以 AtomicNum >= MOL_SPLIT_START 标注）"""
    return sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() >= MOL_SPLIT_START)
def _ring_count(mol: Chem.Mol) -> int:
    return mol.GetRingInfo().NumRings()
def _choose_scaffold(frag_mols: List[Chem.Mol]) -> Tuple[Optional[int], List[int]]:
    """
    选择 scaffold 的规则（片段级）：
      1) 优先含环 (ring_count>0)
      2) 切割位点数(标记原子数)最大
      3) 若并列：环数最大
      4) 若并列：重原子数最大
      5) 若并列：canonical SMILES 字典序最小
    返回 (scaffold_idx, 其余下标列表)
    """
    if not frag_mols:
        return None, []
    stats = []
    for idx, m in enumerate(frag_mols):
        cs = _count_cut_sites(m)
        rc = _ring_count(m)
        hv = m.GetNumHeavyAtoms()
        smi = Chem.MolToSmiles(m, canonical=True)
        stats.append((idx, cs, rc, hv, smi))
    ring_subset = [x for x in stats if x[2] > 0]
    candidates = ring_subset if ring_subset else stats
    candidates.sort(key=lambda x: (-x[1], -x[2], -x[3], x[4]))
    chosen = candidates[0][0]
    others = [i for i, *_ in stats if i != chosen]
    return chosen, others


def split_scaffold_decorations(smiles: str) -> Tuple[Optional[str], List[str]]:
    """
    将分子 SMILES 切分为 scaffold 和 decorations
    使用现有的 split_molecule 和 选择规则
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, []
    
    try:
        fragments = split_molecule(mol)
        if not fragments:
            return None, []
    except Exception:
        return None, []
    
    # 使用现有的选择逻辑
    scaffold_idx, decoration_indices = _choose_scaffold(fragments)
    if scaffold_idx is None:
        return None, []
    
    scaffold_smi = Chem.MolToSmiles(fragments[scaffold_idx], canonical=True)
    decoration_smis = [Chem.MolToSmiles(fragments[i], canonical=True) 
                      for i in decoration_indices]
    
    return scaffold_smi, decoration_smis
class DecoratorVocabulary:
    """
    封装 scaffold 和 decoration 的两个独立词汇表，参考 LibInvent 设计
    但使用片段级 token 而非字符级
    """
    def __init__(self, scaffold_vocabulary, decoration_vocabulary):
        self.scaffold_vocabulary = scaffold_vocabulary
        self.decoration_vocabulary = decoration_vocabulary

    @classmethod
    def from_files(cls, scaffold_vocab_file: str, decoration_vocab_file: str):
        """从文件创建词汇表对，与 LibInvent 接口一致"""
        scaffold_voc = Vocabulary(init_from_file=scaffold_vocab_file)
        decoration_voc = Vocabulary(init_from_file=decoration_vocab_file)
        return cls(scaffold_voc, decoration_voc)

    def len_scaffold(self):
        """返回 scaffold 词汇表长度，与 LibInvent 接口一致"""
        return len(self.scaffold_vocabulary)

    def len_decoration(self):
        """返回 decoration 词汇表长度，与 LibInvent 接口一致"""
        return len(self.decoration_vocabulary)
    
    def len(self):
        """返回两个词汇表长度的元组，与 LibInvent 接口一致"""
        return (self.len_scaffold(), self.len_decoration())
    
class DecoratorDataset(Dataset):
    """
    片段级的 Scaffold-Decoration 数据集
    输入：CSV 文件（包含 'smiles' 列）
    输出：(scaffold_tensor, decoration_tensor) 对
    """
    def __init__(self, fname, decorator_vocabulary, smiles_col='smiles', min_decorations=0):
        self.vocabulary = decorator_vocabulary
        self._samples = []
        df = pd.read_csv(fname)
        smiles_list = df[smiles_col].values.tolist()
        self.con = df[['qed', 'sa']]
        self.con = self.condition_convert(self.con).values.tolist()

        for i, smi in enumerate(smiles_list):
            # 在线切分 scaffold 和 decorations
            scaffold_smi, decoration_smis = split_scaffold_decorations(smi)
            if scaffold_smi is None or len(decoration_smis) < min_decorations:
                continue
                
            
                # 使用与 MolData 相同的 tokenize 逻辑
            scaffold_tokens = [scaffold_smi]
                
            decoration_tokens = decoration_smis + ['EOS']
                
            scaffold_encoded = self.vocabulary.scaffold_vocabulary.encode(scaffold_tokens)
            decoration_encoded = self.vocabulary.decoration_vocabulary.encode(decoration_tokens)
                
                # 存储样本和对应的条件 token
            self._samples.append((
                Variable(torch.tensor(scaffold_encoded, dtype=torch.float32)),
                Variable(torch.tensor(decoration_encoded, dtype=torch.float32)),
                self.con[i]  # 保存条件 token
            ))
            
    
    def __getitem__(self, i):
        """返回 (scaffold_tensor, decoration_tensor)，与 MolData 格式类似"""
        scaffold_tensor, decoration_tensor, con_token = self._samples[i]
        
        # 在 scaffold 前添加条件 token（与 MolData 逻辑一致）
        scaffold_with_con = torch.cat([
            Variable(torch.tensor(self.vocabulary.scaffold_vocabulary.encode(con_token), dtype=torch.float32)),
            scaffold_tensor
        ])

        return scaffold_with_con, decoration_tensor

    
    def __len__(self):
        return len(self._samples)

    def __str__(self):
        return f"DecoratorData containing {len(self)} scaffold-decoration pairs."

    @staticmethod
    def collate_fn(batch):
        """批处理函数，分别对 scaffold 和 decoration 进行 padding"""
        scaffolds, decorations = zip(*batch)
        
        # 复用 MolData 的 collate 逻辑
        scaffold_batch = MolData.collate_fn(scaffolds)
        decoration_batch = MolData.collate_fn(decorations)
        
        return scaffold_batch, decoration_batch
    
    def condition_convert(self, con_df):
        # convert to 0, 1
        # con_df['drd2'][con_df['drd2'] >= 0.5] = 1
        # con_df['drd2'][con_df['drd2'] < 0.5] = 0
        con_df['qed'][con_df['qed'] >= 0.38] = 1
        con_df['qed'][con_df['qed'] < 0.38] = 0
        con_df['sa'][con_df['sa'] <= 4.0] = 1
        con_df['sa'][con_df['sa'] > 4.0] = 0

        # convert to token

        # con_df['drd2'][con_df['drd2'] == 1] = 'is_DRD2'
        # con_df['drd2'][con_df['drd2'] == 0] = 'not_DRD2'
        con_df['qed'][con_df['qed'] == 1] = 'high_QED'
        con_df['qed'][con_df['qed'] == 0] = 'low_QED'
        con_df['sa'][con_df['sa'] == 1] = 'good_SA'
        con_df['sa'][con_df['sa'] == 0] = 'bad_SA'

        return con_df