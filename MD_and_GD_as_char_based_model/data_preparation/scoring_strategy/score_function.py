from __future__ import annotations

import json
from typing import Dict, List, Optional, Tuple

import numpy as np
from rdkit import Chem
from rdkit import DataStructs
from rdkit import rdBase
from rdkit.Chem import AllChem
import rdkit.Chem.QED as QED
from rdkit.Contrib.SA_Score import sascorer

from prediction_model.kagnn_gap_predictor import KAGnnGapPredictor


_KAGNN_CACHE: Dict[str, Optional[KAGnnGapPredictor]] = {"predictor": None, "fingerprint": None}


def _fingerprint_config(config: Optional[Dict[str, object]]) -> str:
    if not config:
        return "__default__"
    try:
        return json.dumps(config, sort_keys=True, default=str)
    except TypeError:
        return str(sorted(config.items()))


def _get_kagnn_predictor(config: Optional[Dict[str, object]]) -> KAGnnGapPredictor:
    fingerprint = _fingerprint_config(config)
    cached_fingerprint = _KAGNN_CACHE.get("fingerprint")
    predictor = _KAGNN_CACHE.get("predictor")
    if predictor is None or cached_fingerprint != fingerprint:
        predictor = KAGnnGapPredictor(config)
        _KAGNN_CACHE["predictor"] = predictor
        _KAGNN_CACHE["fingerprint"] = fingerprint
    return predictor  # type: ignore[return-value]



class qed_func():

    def __call__(self, smiles_list):
        scores = []
        for smiles in smiles_list:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    qed =0
                else:
                    try:
                        qed = QED.qed(mol)
                    except:
                        qed = 0
            except:
                qed = 0
            scores.append(qed)
        return np.float32(scores)

class sa_func():

    def __call__(self, smiles_list):
        scores = []
        for smiles in smiles_list:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    scores.append(100)
                else:
                    scores.append(sascorer.calculateScore(mol))
            except:
                scores.append(100)
        return np.float32(scores)


def composite_qed_sa_score(smiles_list, weights=(1.0, 1.0)):
    """Combine QED and SA scores into a single 0-1 desirability score while exposing components."""

    qed_scores = qed_func()(smiles_list)
    sa_scores = sa_func()(smiles_list)

    qed_transformed = np.where(qed_scores >= 0.4, 1.0, qed_scores).astype(np.float32)

    sa_transformed = np.ones_like(sa_scores, dtype=np.float32)
    mask = sa_scores > 4.0
    if np.any(mask):
        scaled = 1.0 / (1.0 + np.exp(1.0 * (sa_scores[mask] - 4.0)))
        sa_transformed[mask] = np.minimum(1.0, 2.0 * scaled)

    w_qed, w_sa = weights
    weight_sum = w_qed + w_sa if (w_qed + w_sa) != 0 else 1.0
    combined = (w_qed * qed_transformed + w_sa * sa_transformed) / weight_sum
    component_scores = {
        "qed": qed_transformed.astype(np.float32),
        "sa": sa_transformed.astype(np.float32),
    }
    return combined.astype(np.float32), component_scores


def multiple_score(
    smiles_list: List[str],
    weights: Tuple[float, ...] | List[float] = (1.0,),
    config: Optional[Dict[str, object]] = None,
    predictor: Optional[KAGnnGapPredictor] = None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """General scorer pipeline backed by KA-GNN ST Gap results.

    The current implementation exposes a single component (normalized ST Gap),
    but follows the same weighted aggregation pattern as `composite_qed_sa_score`.
    """

    cfg = dict(config or {})
    predictor = predictor or _get_kagnn_predictor(cfg)
    if predictor is None:
        raise RuntimeError("KA-GNN predictor could not be initialized")

    raw_scores = predictor.score(smiles_list).astype(np.float32)
    print(f"raw_scores: {raw_scores}")
    finite_mask = np.isfinite(raw_scores)

    scale = float(cfg.get("scale", 1.0))
    scale = max(scale, 1e-6)

    normalized = np.zeros_like(raw_scores, dtype=np.float32)
    if np.any(finite_mask):
        gaps = np.maximum(raw_scores[finite_mask], 0.0)
        normalized[finite_mask] = 1.0 / (1.0 + gaps / scale)

    fallback = float(cfg.get("invalid_value", 0.0))
    normalized[~finite_mask] = fallback

    component_scores = {
        "st_gap": normalized.astype(np.float32),
        "st_gap_raw": np.where(finite_mask, raw_scores, fallback).astype(np.float32),
    }

    weight_list = [float(w) for w in (weights if isinstance(weights, (list, tuple)) else (weights,))]

    weighted_sum = np.zeros_like(normalized, dtype=np.float32)
    total_weight = 0.0
    primary_components = [name for name in component_scores if not name.endswith("_raw")]

    for idx, name in enumerate(primary_components):
        weight = weight_list[idx] if idx < len(weight_list) else 0.0
        if weight == 0.0:
            continue
        weighted_sum += weight * component_scores[name]
        total_weight += weight

    if total_weight == 0.0:
        return component_scores["st_gap"], component_scores

    combined = weighted_sum / total_weight
    return combined.astype(np.float32), component_scores