"""Reinforcement-learning entry point mirroring Lib-INVENT's structure.

本脚本保持与 Lib-INVENT `running_modes/reinforcement_learning` 接口一致，
但具体实现留待后续补充，以便与当前项目逐步融合。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple
from models.rl_actions import LikelihoodEvaluation, SampleModel
from dto import SampledSequencesDTO
from configurations.configurations import LearningStrategyConfiguration as LearningStrategyConfig
from configurations.configurations import ReinforcementLearningConfiguration as ReinforcementLearningConfig
from configurations.configurations import ScoringStrategyConfiguration as ScoringStrategyConfig
from learning_strategy.dap_strategy import DAPStrategy
from scoring_strategy.scoring_strategy import StandardScoringStrategy
from scoring_strategy.summary import ScoreSummary
import time
import models.actions as ma
import models.model as mm






# ---------------------------------------------------------------------------
# Reinforcement-learning runner (结构占位)
# ---------------------------------------------------------------------------


class ReinforcementLearning:
	"""训练主循环，占位实现保持与 Lib-INVENT 相同的方法命名。"""

	def __init__(
		self,
		actor: Any,
		critic: Any,
		configuration: ReinforcementLearningConfig,
		logger: Any,
	) -> None:
		self.actor = actor
		self.critic = critic
		self.configuration = self._double_single_scaffold_hack(configuration)
		self.logger = logger

		self.optimizer = create_optimizer(self.actor, self.configuration)
		self.learning_strategy = DAPStrategy(self.critic, self.optimizer, self.configuration.learning_strategy, self.logger)
		self.scoring_strategy = StandardScoringStrategy(
            strategy_configuration=self.configuration.scoring_strategy,
            logger=self.logger,
        )



	def run(self) -> None:
		assert False,"check befor rl"
		start_time = time.time()
		for step in range( self.configuration.n_steps):
			sampled_sequences = self._sampling()
			score_summary = self._scoring(sampled_sequences, step)
			actor_nlls, critic_nlls, augmented_nlls = self._updating(sampled_sequences, score_summary)
		#finalize_run(self.scoring_strategy)

	def _sampling(self) -> Iterable[Any]:
		sampling_action = SampleModel(self.actor, self.configuration.batch_size, self.logger,
                                      self.configuration.randomize_scaffolds)
		sampled_sequences = sampling_action.run(self.configuration.scaffolds)
		return sampled_sequences

	def _scoring(self, sampled_sequences, step: int) -> ScoreSummary:
		return self.scoring_strategy.evaluate(sampled_sequences, step)

	def _updating(self, sampled_sequences: Iterable[Any], score_summary: ScoreSummary) -> Tuple[Any, Any, Any]:
		scaffold_batch, decorator_batch, actor_nlls = self._calculate_likelihood(sampled_sequences)
		total_scores = score_summary.total_score
		actor_nlls, critic_nlls, augmented_nlls = self.learning_strategy.run(
			scaffold_batch, decorator_batch, total_scores, actor_nlls
		)
		return actor_nlls, critic_nlls, augmented_nlls
		
	def _calculate_likelihood(self, sampled_sequences: List[SampledSequencesDTO]):
		nll_calculation_action = LikelihoodEvaluation(self.actor, self.configuration.batch_size, self.logger)
		encoded_scaffold, encoded_decorators, nlls = nll_calculation_action.run(sampled_sequences)
		return encoded_scaffold, encoded_decorators, nlls

	@staticmethod
	def _double_single_scaffold_hack(configuration: ReinforcementLearningConfig) -> ReinforcementLearningConfig:
		"""保持 Lib-INVENT 中的单 scaffold 对齐 hack。"""

		if len(configuration.scaffolds) == 1:
			configuration.scaffolds *= 2
			configuration.batch_size = max(int(configuration.batch_size / 2), 1)
		return configuration


# ---------------------------------------------------------------------------
# Factory helpers (占位实现)
# ---------------------------------------------------------------------------


def create_optimizer(actor: Any, configuration: ReinforcementLearningConfig) -> Any:
	"""创建优化器的占位函数。"""

	from importlib import import_module

	try:
		torch = import_module("torch")
	except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
		raise ImportError("PyTorch is required to create the optimizer; install it before running RL.") from exc

	params = actor.network.parameters() if hasattr(actor, "network") else actor.parameters()
	return torch.optim.Adam(params, lr=configuration.learning_rate)












def create_logger(configuration: ReinforcementLearningConfig) -> Any:
	raise NotImplementedError("create_logger needs project-specific implementation")


# ---------------------------------------------------------------------------
# Step helpers (占位实现)
# ---------------------------------------------------------------------------


def prepare_run(configuration: ReinforcementLearningConfig, logger: Any) -> int:
	raise NotImplementedError("prepare_run needs project-specific implementation")


def finalize_run(scoring_strategy: Any) -> None:
	raise NotImplementedError("finalize_run needs project-specific implementation")




# ---------------------------------------------------------------------------
# Configuration loading
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Run reinforcement learning fine-tuning")
	parser.add_argument(
		"--config",
		type=Path,
		default = "./configs/rl_configs.json",
		help="Path to a JSON configuration file describing the RL run.",
	)
	return parser.parse_args()


def load_raw_config(config_path: Path) -> Dict[str, Any]:
	with config_path.expanduser().open("r", encoding="utf-8") as handle:
		return json.load(handle)


def build_configuration(raw_config: Dict[str, Any]) -> ReinforcementLearningConfig:
	"""将 JSON 字段映射到 `ReinforcementLearningConfig`。"""

	actor_path = Path(raw_config["actor"]).expanduser()
	critic_path = Path(raw_config["critic"]).expanduser()
	scaffolds = _load_scaffolds(raw_config["scaffolds"])
	output_dir = Path(raw_config.get("output_dir", "./rl_runs")).expanduser()

	learning_strategy = LearningStrategyConfig(
		name=raw_config.get("learning_strategy", {}).get("name", ""),
		parameters=raw_config.get("learning_strategy", {}).get("parameters", {}),
	)
	scoring_strategy = ScoringStrategyConfig(
		name=raw_config.get("scoring_strategy", {}).get("name", ""),
		diversity_filter=raw_config.get("scoring_strategy", {}).get("diversity_filter", {}),
		scoring_function=raw_config.get("scoring_strategy", {}).get("scoring_function", {}),
	)

	return ReinforcementLearningConfig(
		actor=actor_path,
		critic=critic_path,
		scaffolds=scaffolds,
		learning_strategy=learning_strategy,
		scoring_strategy=scoring_strategy,
		output_dir=output_dir,
		n_steps=int(raw_config.get("n_steps", 1000)),
		learning_rate=float(raw_config.get("learning_rate", 1e-4)),
		batch_size=int(raw_config.get("batch_size", 128)),
		randomize_scaffolds=bool(raw_config.get("randomize_scaffolds", False)),
	)


def _load_scaffolds(source: Any) -> List[str]:
	if isinstance(source, list):
		return [str(scaffold) for scaffold in source]
	if isinstance(source, str):
		path = Path(source).expanduser()
		raise NotImplementedError("File-based scaffold loading to be implemented")
	raise TypeError("`scaffolds` must be a list or a path to a file")


# ---------------------------------------------------------------------------
# Model & logger bootstrap (占位实现)
# ---------------------------------------------------------------------------


def load_models(configuration: ReinforcementLearningConfig) -> Tuple[Any, Any]:
	"""加载 actor 与 critic 模型。"""

	actor = load_model_from_checkpoint(configuration.actor, mode="train")
	critic = load_model_from_checkpoint(configuration.critic, mode="eval")
	return actor, critic


def load_model_from_checkpoint(checkpoint_path: Path, mode: str) -> Any:
	"""从 checkpoint 加载模型的占位函数。"""

	model = mm.DecoratorModel.load_from_file(checkpoint_path,mode=mode)
	return model

# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------


def main() -> None:
	args = parse_args()
	raw_config = load_raw_config(args.config)
	configuration = build_configuration(raw_config)
	#logger = create_logger(configuration)
	logger = None
	actor, critic = load_models(configuration)
	rl_runner = ReinforcementLearning(actor=actor, critic=critic, configuration=configuration, logger=logger)
	rl_runner.run()


if __name__ == "__main__":
	main()
