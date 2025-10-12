from typing import Dict

import numpy as np

try:  # pragma: no cover - optional dependency
    import pandas as pd
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    pd = None


TOTAL_SCORE_COLUMN = "total_score"


class DiversityFilterMemory:

    def __init__(self):
        self._memory_records = []

    def update(
        self,
        indx: int,
        score: float,
        smile: str,
        scaffold: str,
        components: Dict[str, np.ndarray],
        step: int,
    ) -> None:
        component_scores = {name: float(values[indx]) for name, values in components.items()}
        component_scores[TOTAL_SCORE_COLUMN] = float(score)
        if not self.smiles_exists(smile):
            self._add_to_memory(step, smile, scaffold, component_scores)

    def _add_to_memory(self, step: int, smile: str, scaffold: str, component_scores: Dict[str, float]):
        record = {
            **component_scores,
            "Step": step,
            "Scaffold": scaffold,
            "SMILES": smile,
        }
        self._memory_records.append(record)

    def get_memory(self):
        if pd is None:
            return list(self._memory_records)
        return pd.DataFrame(self._memory_records)

    def set_memory(self, memory):
        if pd is not None and isinstance(memory, pd.DataFrame):
            self._memory_records = memory.to_dict(orient="records")
        elif isinstance(memory, list):
            self._memory_records = [dict(row) for row in memory]
        else:
            raise TypeError("memory must be a pandas DataFrame or a list of dicts")

    def smiles_exists(self, smiles: str):
        if len(self._memory_records) == 0:
            return False
        return any(record["SMILES"] == smiles for record in self._memory_records)

    def scaffold_instances_count(self, scaffold: str):
        return sum(1 for record in self._memory_records if record["Scaffold"] == scaffold)

    def number_of_scaffolds(self):
        return len({record["Scaffold"] for record in self._memory_records})

    def number_of_smiles(self):
        return len({record["SMILES"] for record in self._memory_records})
