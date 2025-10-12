from __future__ import annotations

from typing import List, Tuple

import numpy as np

import utils.chem as uc

from dto import SampledSequencesDTO
from scoring_strategy.base_scoring_strategy import BaseScoringStrategy
from configurations.configurations import ScoringStrategyConfiguration
from utils.scaffold import join_joined_attachments
from scoring_strategy.score_function import composite_qed_sa_score
from scoring_strategy.summary import ScoreSummary

class StandardScoringStrategy(BaseScoringStrategy):
    def __init__(self, strategy_configuration: ScoringStrategyConfiguration, logger):
        super().__init__(strategy_configuration, logger)
        weights = getattr(strategy_configuration, "scoring_function", {}) or {}
        self._component_weights: Tuple[float, float] = tuple(weights.get("weights", (1.0, 1.0)))
    
    def evaluate(self, sampled_sequences: List[SampledSequencesDTO], step) -> ScoreSummary:
        summary = self._apply_scoring_function(sampled_sequences)
        #assert False, "check evaluate"
        if hasattr(self.diversity_filter, "update_score"):
            summary.total_score = self.diversity_filter.update_score(summary, sampled_sequences, step)
            #print(f"Scores after diversity filter: {summary.total_score}")
            #assert False,"check after diversity filter"
        return summary
    
    def _apply_scoring_function(self, sampled_sequences: List[SampledSequencesDTO]) -> ScoreSummary:
        molecules = [
            join_joined_attachments(sample.scaffold, sample.decoration)
            for sample in sampled_sequences
        ]
        print(f"Applying scoring function to {len(molecules)} molecules")
        smiles = [uc.to_smiles(molecule) if molecule else "INVALID" for molecule in molecules]
        total_score, component_scores = composite_qed_sa_score(smiles, weights=self._component_weights)
        print(f"Total scores: {total_score}")
        valid_idxs = [idx for idx, smi in enumerate(smiles) if smi != "INVALID"]
        print(f"Valid SMILES count: {len(valid_idxs)}")
        return ScoreSummary(
            total_score=np.asarray(total_score),
            component_scores=component_scores,
            scored_smiles=smiles,
            valid_idxs=valid_idxs,
        )
