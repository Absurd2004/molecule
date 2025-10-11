import numpy as np
from rdkit import Chem
from rdkit import rdBase
from rdkit.Chem import AllChem
from rdkit import DataStructs
import rdkit.Chem.QED as QED
from rdkit.Contrib.SA_Score import sascorer

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
    """Combine QED and SA scores into a single 0-1 desirability score.

    Args:
        smiles_list: Iterable of SMILES strings.
        weights: Tuple (w_qed, w_sa) controlling the contribution of each component.

    Returns:
        numpy.ndarray: Weighted score per input SMILES, float32 in [0, 1].
    """

    qed_scores = qed_func()(smiles_list)
    sa_scores = sa_func()(smiles_list)

    qed_transformed = np.where(qed_scores >= 0.4, 1.0, qed_scores)

    sa_transformed = np.ones_like(sa_scores, dtype=np.float32)
    mask = sa_scores > 4.0
    if np.any(mask):
        scaled = 1.0 / (1.0 + np.exp(1.0 * (sa_scores[mask] - 4.0)))
        sa_transformed[mask] = np.minimum(1.0, 2.0 * scaled)

    w_qed, w_sa = weights
    weight_sum = w_qed + w_sa if (w_qed + w_sa) != 0 else 1.0
    combined = (w_qed * qed_transformed + w_sa * sa_transformed) / weight_sum
    return combined.astype(np.float32)