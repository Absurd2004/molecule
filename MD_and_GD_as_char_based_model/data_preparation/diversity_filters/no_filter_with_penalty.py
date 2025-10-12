from copy import deepcopy
from typing import List

import numpy as np
# The import below is a deal breaker
# from reinvent_scoring.scoring.score_summary import FinalSummary

from diversity_filters.base_diversity_filter import BaseDiversityFilter
from diversity_filters.diversity_filter_parameters import DiversityFilterParameters
from dto import SampledSequencesDTO
from scoring_strategy.summary import ScoreSummary
from utils.scaffold import convert_to_rdkit_smiles


class NoFilterWithPenalty(BaseDiversityFilter):
    """Penalize repeatedly generated compounds."""

    def __init__(self, parameters: DiversityFilterParameters):
        super().__init__(parameters)

    def update_score(self, score_summary: ScoreSummary, sampled_sequences: List[SampledSequencesDTO], step=0) -> np.ndarray:
        score_summary = deepcopy(score_summary)
        scores = score_summary.total_score
        smiles = score_summary.scored_smiles

        for i in score_summary.valid_idxs:
            smiles[i] = convert_to_rdkit_smiles(smiles[i])
            scores[i] = 0.5*scores[i] if self._smiles_exists(smiles[i]) else scores[i]

        for i in score_summary.valid_idxs:
            if scores[i] >= self.parameters.minscore:
                decorations = f'{sampled_sequences[i].scaffold}|{sampled_sequences[i].decoration}'
                self._add_to_memory(i, scores[i], smiles[i], decorations, score_summary.component_scores, step)
        return scores
