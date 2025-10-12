from dataclasses import dataclass
from typing import Dict, List

import numpy as np


@dataclass
class ScoreSummary:
    total_score: np.ndarray
    component_scores: Dict[str, np.ndarray]
    scored_smiles: List[str]
    valid_idxs: List[int]
