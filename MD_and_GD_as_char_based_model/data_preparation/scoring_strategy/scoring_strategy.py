from configurations.configurations import ScoringStrategyConfiguration
from scoring_strategy.base_scoring_strategy import BaseScoringStrategy
from scoring_strategy.standard_strategy import StandardScoringStrategy


class ScoringStrategy:

    def __new__(cls, strategy_configuration: ScoringStrategyConfiguration, logger) -> BaseScoringStrategy:
        scoring_strategy_type = "standard"
        if scoring_strategy_type == strategy_configuration.name:
            return StandardScoringStrategy(strategy_configuration, logger)