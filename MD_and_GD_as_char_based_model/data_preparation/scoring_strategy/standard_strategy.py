from __future__ import annotations

from typing import List, Tuple

import numpy as np

import utils.chem as uc

from dto import SampledSequencesDTO
from scoring_strategy.base_scoring_strategy import BaseScoringStrategy
from configurations.configurations import ScoringStrategyConfiguration
from utils.scaffold import join_joined_attachments, to_smiles
from scoring_strategy.score_function import composite_qed_sa_score

class StandardScoringStrategy(BaseScoringStrategy):
    def __init__(self, strategy_configuration: ScoringStrategyConfiguration, logger):
        super().__init__(strategy_configuration, logger)
        weights = getattr(strategy_configuration, "scoring_function", {}) or {}
        self._component_weights: Tuple[float, float] = tuple(weights.get("weights", (1.0, 1.0)))
    
    def evaluate(self, sampled_sequences: List[SampledSequencesDTO], step) -> np.ndarray:
        scores = self._apply_scoring_function(sampled_sequences)
        if hasattr(self.diversity_filter, "update_score"):
            scores = self.diversity_filter.update_score(scores, sampled_sequences, step)
        return scores
    
    def _apply_scoring_function(self, sampled_sequences: List[SampledSequencesDTO]) -> np.ndarray:
        molecules = [
            join_joined_attachments(sample.scaffold, sample.decoration)
            for sample in sampled_sequences
        ]
        smiles = [uc.to_smiles(molecule) if molecule else "INVALID" for molecule in molecules]
        final_score = composite_qed_sa_score(smiles, weights=self._component_weights)
        return np.asarray(final_score)