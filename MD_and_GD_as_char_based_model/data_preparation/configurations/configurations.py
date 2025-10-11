from typing import Any, Dict, Iterable, List, Tuple
from dataclasses import dataclass

@dataclass
class LearningStrategyConfiguration:
    name: str
    parameters: dict = None





@dataclass
class ScoringStrategyConfiguration:
    diversity_filter: DiversityFilterParameters
    scoring_function: ScoringFuncionParameters
    name: str


@dataclass
class ReinforcementLearningConfiguration:
    actor: str
    critic: str
    scaffolds: List[str]
    learning_strategy: LearningStrategyConfiguration
    scoring_strategy: ScoringStrategyConfiguration
    n_steps: int = 1000
    learning_rate: float = 0.0001
    batch_size: int = 128
    randomize_scaffolds: bool = False