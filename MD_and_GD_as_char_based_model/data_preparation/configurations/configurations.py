from typing import Any, Dict, Iterable, List, Tuple
from dataclasses import dataclass, field
from diversity_filters.diversity_filter_memory import DiversityFilterMemory
from diversity_filters.diversity_filter_parameters import DiversityFilterParameters

@dataclass
class LearningStrategyConfiguration:
    name: str
    parameters: dict = None





@dataclass
class ScoringStrategyConfiguration:
    diversity_filter: DiversityFilterParameters
    name: str
    scoring_function: Dict[str, Any]


@dataclass
class ReinforcementLearningConfiguration:
    actor: str
    critic: str
    scaffolds: List[str]
    learning_strategy: LearningStrategyConfiguration
    scoring_strategy: ScoringStrategyConfiguration
    output_dir: str
    n_steps: int = 1000
    learning_rate: float = 0.0001
    batch_size: int = 128
    randomize_scaffolds: bool = False
    wandb: Dict[str, Any] = field(default_factory=dict)