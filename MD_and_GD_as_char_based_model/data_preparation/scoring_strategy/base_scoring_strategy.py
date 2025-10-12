from abc import ABC, abstractmethod
from typing import List


from diversity_filters.diversity_filter import DiversityFilter

from configurations.configurations import ScoringStrategyConfiguration
from dto import SampledSequencesDTO
from scoring_strategy.summary import ScoreSummary


class BaseScoringStrategy(ABC):
    def __init__(self, strategy_configuration: ScoringStrategyConfiguration, logger):
        self._configuration = strategy_configuration

        self.diversity_filter = DiversityFilter(strategy_configuration.diversity_filter)

        self.scoring_function = ""
        self.logger = logger

    @abstractmethod
    def evaluate(self, sampled_sequences: List[SampledSequencesDTO], step: int) -> ScoreSummary:
        raise NotImplementedError("evaluate method is not implemented")

    def save_filter_memory(self):
        # TODO: might be good to consider separating the memory from the actual filter
        self.logger.save_filter_memory(self.diversity_filter)
