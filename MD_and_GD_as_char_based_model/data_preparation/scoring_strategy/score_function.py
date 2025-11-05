from __future__ import annotations

import itertools
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


def _average_decoration_similarity(decoration: str) -> float:
    parts = [part for part in (decoration or "").split("|") if part]
    if len(parts) < 2:
        return 1.0

    fps = []
    for part in parts:
        mol = Chem.MolFromSmiles(part)
        if mol is None:
            continue
        fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048))

    if len(fps) < 2:
        return 0.0 if not fps else 1.0

    similarities = [DataStructs.TanimotoSimilarity(a, b) for a, b in itertools.combinations(fps, 2)]
    return float(np.mean(similarities)) if similarities else 0.0


def _decoration_pair_reward(decoration: str) -> float:
    parts = [part for part in (decoration or "").split("|") if part]
    if not parts:
        return 0.0

    mols: List[Chem.Mol] = []
    for part in parts:
        mol = Chem.MolFromSmiles(part)
        if mol is None:
            return 0.0
        mols.append(mol)

    fps = [AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048) for mol in mols]

    def _is_identical(a, b, tol: float = 1e-9) -> bool:
        return DataStructs.TanimotoSimilarity(a, b) >= 1.0 - tol

    if len(parts) == 3:
        all_same = _is_identical(fps[0], fps[1]) and _is_identical(fps[1], fps[2])
        if all_same or _is_identical(fps[1], fps[2]):
            return 1.0
        return 0.0

    clusters: List[List[int]] = []
    for idx, fp in enumerate(fps):
        assigned = False
        for cluster in clusters:
            if _is_identical(fp, fps[cluster[0]]):
                cluster.append(idx)
                assigned = True
                break
        if not assigned:
            clusters.append([idx])

    cluster_sizes = sorted(len(cluster) for cluster in clusters)
    if cluster_sizes == [len(parts)]:
        return 1.0
    if len(parts) == 4:
        pair_0_2 = _is_identical(fps[0], fps[2])
        pair_1_3 = _is_identical(fps[1], fps[3])
        if pair_0_2 and pair_1_3:
            return 1.0
        if cluster_sizes == [2, 2]:
            return 0.5
        if cluster_sizes == [1, 1, 2]:
            return 0.25
        return 0.0
    if len(parts) == 6:
        pair_indices = [(0, 3), (1, 4), (2, 5)]
        pair_matches = [
            1 if _is_identical(fps[first], fps[second]) else 0
            for first, second in pair_indices
        ]
        match_count = sum(pair_matches)
        if match_count == 3:
            return 1.0
        if match_count == 2:
            return 0.5
        if match_count == 1:
            return 0.25
        return 0.0
    if cluster_sizes == [2, 2]:
        return 1.0
    return 0.0


def _decoration_exceeds_ring_threshold(decoration: str, threshold: int = 3) -> bool:
    parts = [part for part in (decoration or "").split("|") if part]
    for part in parts:
        mol = Chem.MolFromSmiles(part)
        if mol is None:
            continue
        if mol.GetRingInfo().NumRings() > threshold:
            return True
    return False


def multiple_score(
    smiles_list: List[str],
    weights: Tuple[float, ...] | List[float] = (0.5, 0.3, 0.2),
    config: Optional[Dict[str, object]] = None,
    predictor: Optional[KAGnnGapPredictor] = None,
    decorations: Optional[List[str]] = None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """General scorer pipeline backed by KA-GNN ST Gap results.

    The current implementation exposes a single component (normalized ST Gap),
    but follows the same weighted aggregation pattern as `composite_qed_sa_score`.
    """

    cfg = dict(config or {})
    if isinstance(weights, (list, tuple)):
        weight_values = [float(w) for w in weights]
    else:
        weight_values = [float(weights)]
    st_gap_weight = weight_values[0] if weight_values else 0.0

    smiles_count = len(smiles_list)
    fallback = float(cfg.get("invalid_value", 0.0))

    raw_scores = np.full(smiles_count, fallback, dtype=np.float32)
    normalized = np.full(smiles_count, fallback, dtype=np.float32)
    finite_mask = np.zeros(smiles_count, dtype=bool)

    if st_gap_weight != 0.0:
        predictor = predictor or _get_kagnn_predictor(cfg)
        if predictor is None:
            raise RuntimeError("KA-GNN predictor could not be initialized")

        raw_scores = predictor.score(smiles_list).astype(np.float32)
        finite_mask = np.isfinite(raw_scores)

        scale = float(cfg.get("scale", 1.0))
        scale = max(scale, 1e-6)

        normalized = np.zeros_like(raw_scores, dtype=np.float32)
        if np.any(finite_mask):
            gaps = np.maximum(raw_scores[finite_mask], 0.0)
            normalized[finite_mask] = 1.0 / (1.0 + gaps / scale)
        normalized[~finite_mask] = fallback

    print(f"raw_scores: {raw_scores}")
    print(f"normalized: {normalized}")

    if decorations is None:
        decorations = [""] * len(smiles_list)
    elif len(decorations) != len(smiles_list):
        raise ValueError("Length of decorations must match length of smiles_list")

    charge_scores = np.full_like(normalized, fallback)
    symmetry_scores = np.full_like(normalized, fallback)
    ring_penalty_mask = np.zeros_like(normalized, dtype=bool)

    for idx, (smi, decoration) in enumerate(zip(smiles_list, decorations)):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue

        decoration_parts = [part for part in (decoration or "").split("|") if part]
        if len(decoration_parts) == 1:
            pos_charge = any(atom.GetFormalCharge() > 0 for atom in mol.GetAtoms())
            neg_charge = any(atom.GetFormalCharge() < 0 for atom in mol.GetAtoms())
            if pos_charge and neg_charge:
                charge_scores[idx] = 1.0
            elif pos_charge or neg_charge:
                charge_scores[idx] = 0.5
            else:
                charge_scores[idx] = 0.0
        else:
            formal_charge = Chem.GetFormalCharge(mol)
            if formal_charge == 0:
                charge_scores[idx] = 0.0
            elif formal_charge % 2 == 0:
                charge_scores[idx] = 1.0
            else:
                charge_scores[idx] = 0.5
        symmetry_scores[idx] = _decoration_pair_reward(decoration)
        if _decoration_exceeds_ring_threshold(decoration):
            ring_penalty_mask[idx] = True
    print(f"charge_scores: {charge_scores}")
    print(f"symmetry_scores: {symmetry_scores}")

    component_scores = {
        "st_gap": normalized.astype(np.float32),
        "charge_score": charge_scores.astype(np.float32),
        "symmetry_score": symmetry_scores.astype(np.float32),
        "st_gap_raw": np.where(finite_mask, raw_scores, fallback).astype(np.float32),
    }

    weighted_sum = np.zeros_like(normalized, dtype=np.float32)
    total_weight = 0.0
    primary_components = [name for name in component_scores if not name.endswith("_raw")]

    for idx, name in enumerate(primary_components):
        weight = weight_values[idx] if idx < len(weight_values) else 0.0
        if weight == 0.0:
            continue
        weighted_sum += weight * component_scores[name]
        total_weight += weight

    if total_weight == 0.0:
        combined = component_scores["st_gap"].copy()
    else:
        combined = weighted_sum / total_weight

    if np.any(ring_penalty_mask):
        combined = combined.astype(np.float32)
        combined[ring_penalty_mask] = 0.0
    component_scores["decoration_ring_penalty"] = np.where(
        ring_penalty_mask,
        0.0,
        1.0,
    ).astype(np.float32)

    return combined.astype(np.float32), component_scores