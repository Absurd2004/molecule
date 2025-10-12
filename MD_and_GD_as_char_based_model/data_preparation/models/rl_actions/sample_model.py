from typing import List

import numpy as np
import torch.utils.data as tud

import models.dataset as md
from models.rl_actions import BaseAction
from dto import SampledSequencesDTO
from utils.scaffold import remove_attachment_point_numbers,get_indices_of_unique_smiles,randomize_scaffold_smiles

class SampleModel(BaseAction):
    def __init__(self, model, batch_size: int, logger=None, randomize=False, sample_uniquely=True):
        """
        Creates an instance of SampleModel.
        :params model: A model instance (better in scaffold_decorating mode).
        :params batch_size: Batch size to use.
        :return:
        """
        super().__init__(logger)
        self.model = model
        self._batch_size = batch_size
        self._randomize = randomize
        self._sample_uniquely = sample_uniquely

    def run(self, scaffold_list: List[str]) -> List[SampledSequencesDTO]:
        """
        Samples the model for the given number of SMILES.
        :params scaffold_list: A list of scaffold SMILES.
        :return: A list of SampledSequencesDTO.
        """
        scaffold_list = self._randomize_scaffolds(scaffold_list) if self._randomize else scaffold_list
        clean_scaffolds = [remove_attachment_point_numbers(scaffold) for scaffold in scaffold_list]
        #print(f"Sampling {len(clean_scaffolds)} scaffolds")
        #print(f"all scaffolds: {clean_scaffolds}")
        dataset = md.Dataset(clean_scaffolds, self.model.vocabulary.scaffold_vocabulary,
                             self.model.vocabulary.scaffold_tokenizer)
        dataloader = tud.DataLoader(dataset, batch_size=len(dataset), shuffle=False, collate_fn=md.Dataset.collate_fn)

        device = next(self.model.network.parameters()).device
        for batch in dataloader:
            scaffold_batch, length_batch = batch
            scaffold_batch = scaffold_batch.to(device)
            batch = (scaffold_batch, length_batch)
            sampled_sequences = []

            for _ in range(self._batch_size):
                packed = self.model.sample_decorations(*batch)
                #print(f"Sampled {len(packed)} decorations")
                #print(f"all decorations: {packed}")
                #assert False
                for scaffold, decoration, nll in packed:
                    sampled_sequences.append(SampledSequencesDTO(scaffold, decoration, nll))
                
            if self._sample_uniquely:
                sampled_sequences = self._sample_unique_sequences(sampled_sequences)

            return sampled_sequences
    
    def _sample_unique_sequences(self, sampled_sequences: List[SampledSequencesDTO]) -> List[SampledSequencesDTO]:
        strings = ["".join([ss.scaffold, ss.decoration]) for index, ss in enumerate(sampled_sequences)]
        unique_idxs = get_indices_of_unique_smiles(strings)
        sampled_sequences_np = np.array(sampled_sequences)
        unique_sampled_sequences = sampled_sequences_np[unique_idxs]
        return unique_sampled_sequences.tolist()
    
    def _randomize_scaffolds(self, scaffolds: List[str]):
        return [randomize_scaffold_smiles(smi) for smi in scaffolds]