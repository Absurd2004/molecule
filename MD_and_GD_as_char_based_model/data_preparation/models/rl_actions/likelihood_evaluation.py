from typing import List

import torch.utils.data as tud
from models.dataset import DecoratorDataset
from models.rl_actions import BaseAction
from dto import SampledSequencesDTO

class LikelihoodEvaluation(BaseAction):

    def __init__(self, model, batch_size, logger=None):
        """
        Creates an instance of CalculateNLLsFromModel.
        :param model: A generative model instance.
        :param batch_size: Batch size to use.
        :return:
        """
        super().__init__(logger)
        self.model = model
        self.batch_size = batch_size
    
    def run(self, scaffold_decoration_list: List[SampledSequencesDTO]):
        scaffold_decoration_list = [[ss.scaffold, ss.decoration] for ss in scaffold_decoration_list]
        dataset = DecoratorDataset(scaffold_decoration_list, self.model.vocabulary)
        dataloader = tud.DataLoader(dataset, batch_size=len(dataset), collate_fn=DecoratorDataset.collate_fn,
                                    shuffle=False)
        
        device = next(self.model.network.parameters()).device
        for scaffold_batch, decorator_batch in dataloader:
            scaffold_padded, scaffold_lengths = scaffold_batch
            decorator_padded, decorator_lengths = decorator_batch

            scaffold_batch = (scaffold_padded.to(device), scaffold_lengths)
            decorator_batch = (decorator_padded.to(device), decorator_lengths)

            nll = self.model.likelihood(*scaffold_batch, *decorator_batch)
            #print(f"Calculated NLLs for batch of size {scaffold_batch[0].size(0)}")
            #print(f"NLLs: {nll}")
            #assert False, "check nll"
            return scaffold_batch, decorator_batch, nll

