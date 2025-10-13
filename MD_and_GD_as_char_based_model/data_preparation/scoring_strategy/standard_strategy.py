from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

import utils.chem as uc

from dto import SampledSequencesDTO
from scoring_strategy.base_scoring_strategy import BaseScoringStrategy
from configurations.configurations import ScoringStrategyConfiguration
from utils.scaffold import join_joined_attachments
from prediction_model.kagnn_gap_predictor import KAGnnGapPredictor
from scoring_strategy.score_function import composite_qed_sa_score, multiple_score
from scoring_strategy.summary import ScoreSummary

class StandardScoringStrategy(BaseScoringStrategy):
    def __init__(self, strategy_configuration: ScoringStrategyConfiguration, logger):
        super().__init__(strategy_configuration, logger)
        scoring_cfg: Dict[str, object] = (
            getattr(strategy_configuration, "scoring_function", {}) or {}
        )

        self._use_kan = bool(scoring_cfg.get("use_kan"))
        self._component_weights = self._parse_weights(scoring_cfg.get("weights"))
        self._kan_config = self._extract_kan_config(scoring_cfg)
        self._kan_predictor = (
            KAGnnGapPredictor(self._kan_config) if self._use_kan else None
        )
        #assert False,"successfully create _kan_predictor"
    
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
        if self._use_kan:
            total_score, component_scores = multiple_score(
                smiles,
                weights=self._component_weights,
                config=self._kan_config,
                predictor=self._kan_predictor,
            )
        else:
            total_score, component_scores = composite_qed_sa_score(
                smiles, weights=self._ensure_composite_weights()
            )
        print(f"Total scores: {total_score}")
        print(f"size of total scores: {len(total_score)}")
        valid_idxs = [idx for idx, smi in enumerate(smiles) if smi != "INVALID"]
        print(f"Valid SMILES count: {len(valid_idxs)}")
        return ScoreSummary(
            total_score=np.asarray(total_score),
            component_scores=component_scores,
            scored_smiles=smiles,
            valid_idxs=valid_idxs,
        )

    def _parse_weights(self, raw_weights) -> Tuple[float, ...]:
        if raw_weights is None:
            return (1.0,) if self._use_kan else (1.0, 1.0)

        if isinstance(raw_weights, (int, float)):
            weights = (float(raw_weights),)
        else:
            weights = tuple(float(w) for w in raw_weights)

        if not weights:
            return (1.0,) if self._use_kan else (1.0, 1.0)

        if self._use_kan:
            return weights

        if len(weights) < 2:
            return tuple(list(weights) + [1.0] * (2 - len(weights)))

        return weights[:2]

    def _ensure_composite_weights(self) -> Tuple[float, float]:
        if len(self._component_weights) >= 2:
            return self._component_weights[:2]
        if len(self._component_weights) == 1:
            return (self._component_weights[0], 1.0)
        return (1.0, 1.0)

    def _extract_kan_config(self, scoring_cfg: Dict[str, object]) -> Dict[str, object]:
        if not self._use_kan:
            return {}
        ignored = {"weights", "use_kan"}
        return {k: v for k, v in scoring_cfg.items() if k not in ignored}
