from diversity_filters.no_filter_with_penalty import NoFilterWithPenalty
from diversity_filters.base_diversity_filter import BaseDiversityFilter
from diversity_filters.diversity_filter_parameters import DiversityFilterParameters


class DiversityFilter:

    def __new__(cls, parameters: DiversityFilterParameters) -> BaseDiversityFilter:
        if isinstance(parameters, dict):
            parameters = DiversityFilterParameters(**parameters)

        all_filters = {
            "NoFilterWithPenalty": NoFilterWithPenalty,
            "no_filter_with_penalty": NoFilterWithPenalty,
        }

        div_filter = all_filters.get(parameters.name)
        if div_filter is None:
            raise KeyError(f"Unknown diversity filter '{parameters.name}'")
        return div_filter(parameters)
