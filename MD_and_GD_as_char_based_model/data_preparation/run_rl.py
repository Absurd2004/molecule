"""Reinforcement-learning entry point mirroring Lib-INVENT's structure.

本脚本保持与 Lib-INVENT `running_modes/reinforcement_learning` 接口一致，
但具体实现留待后续补充，以便与当前项目逐步融合。
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from dataclasses import dataclass, field, asdict
import importlib
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple, Optional
import numpy as np
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
import wandb
import torch
import random






# ---------------------------------------------------------------------------
# Reinforcement-learning runner (结构占位)
# ---------------------------------------------------------------------------

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


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
		#assert False,"check befor rl"
		print(f"Running RL with {len(self.configuration.scaffolds)} scaffolds")
		start_time = time.time()
		first_actor_param = next(self.actor.network.parameters(), None)
		first_critic_param = next(self.critic.network.parameters(), None)
		actor_device = first_actor_param.device if first_actor_param is not None else "unknown"
		critic_device = first_critic_param.device if first_critic_param is not None else "unknown"
		print(f"self.actor on {actor_device}")
		print(f"self.critic on {critic_device}")
		for step in range( self.configuration.n_steps):
			step_start = time.time()
			sampled_sequences = self._sampling()
			memory_before = self._memory_snapshot()
			#assert False,"check after sampling"
			score_summary = self._scoring(sampled_sequences, step)
			memory_after = self._memory_snapshot()
			loss_value, actor_nlls, critic_nlls, augmented_nlls = self._updating(sampled_sequences, score_summary)
			step_end = time.time()
			self._log_step(
				step=step,
				sampled_sequences=sampled_sequences,
				score_summary=score_summary,
				loss_value=loss_value,
				actor_nlls=actor_nlls,
				critic_nlls=critic_nlls,
				augmented_nlls=augmented_nlls,
				memory_before=memory_before,
				memory_after=memory_after,
				step_duration=step_end - step_start,
				total_elapsed=step_end - start_time,
			)
			if (step + 1) % 10 == 0:
				self._export_diversity_memory()
		#finalize_run(self.scoring_strategy)
		self._export_diversity_memory()

	def _sampling(self) -> Iterable[Any]:
		sampling_action = SampleModel(self.actor, self.configuration.batch_size, self.logger,
                                      self.configuration.randomize_scaffolds)
		print(f"start sampling {self.configuration.batch_size} sequences")
		sampled_sequences = sampling_action.run(self.configuration.scaffolds)
		print(f"sampled {len(sampled_sequences)} sequences")
		return sampled_sequences

	def _scoring(self, sampled_sequences, step: int) -> ScoreSummary:
		return self.scoring_strategy.evaluate(sampled_sequences, step)

	def _updating(self, sampled_sequences: Iterable[Any], score_summary: ScoreSummary) -> Tuple[Any, Any, Any]:
		scaffold_batch, decorator_batch, actor_nlls = self._calculate_likelihood(sampled_sequences)
		total_scores = score_summary.total_score
		loss_value, actor_nlls, critic_nlls, augmented_nlls = self.learning_strategy.run(
			scaffold_batch, decorator_batch, total_scores, actor_nlls
		)
		return loss_value, actor_nlls, critic_nlls, augmented_nlls
		
	def _calculate_likelihood(self, sampled_sequences: List[SampledSequencesDTO]):
		nll_calculation_action = LikelihoodEvaluation(self.actor, self.configuration.batch_size, self.logger)
		encoded_scaffold, encoded_decorators, nlls = nll_calculation_action.run(sampled_sequences)
		return encoded_scaffold, encoded_decorators, nlls

	def _memory_snapshot(self) -> Dict[str, int]:
		diversity_filter = getattr(self.scoring_strategy, "diversity_filter", None)
		if diversity_filter is None:
			return {"smiles": 0, "scaffolds": 0}

		number_of_smiles = getattr(diversity_filter, "number_of_smiles_in_memory", None)
		number_of_scaffolds = getattr(diversity_filter, "number_of_scaffold_in_memory", None)
		return {
			"smiles": number_of_smiles() if callable(number_of_smiles) else 0,
			"scaffolds": number_of_scaffolds() if callable(number_of_scaffolds) else 0,
		}

	@staticmethod
	def _to_numpy(values: Any) -> np.ndarray:
		if values is None:
			return np.asarray([])
		if hasattr(values, "detach"):
			return values.detach().cpu().numpy()
		if isinstance(values, np.ndarray):
			return values
		return np.asarray(values)

	def _log_step(
		self,
		step: int,
		sampled_sequences: List[SampledSequencesDTO],
		score_summary: ScoreSummary,
		loss_value: float,
		actor_nlls: Any,
		critic_nlls: Any,
		augmented_nlls: Any,
		memory_before: Dict[str, int],
		memory_after: Dict[str, int],
		step_duration: float,
		total_elapsed: float,
	) -> None:
		total_sequences = len(sampled_sequences)
		valid_count = len(score_summary.valid_idxs)
		invalid_count = total_sequences - valid_count
		valid_ratio = (valid_count / total_sequences) if total_sequences else 0.0

		total_scores = score_summary.total_score
		metrics: Dict[str, Any] = {
			"loss/rl": loss_value,
			"time/step_duration_sec": step_duration,
			"time/elapsed_sec": total_elapsed,
			"batch/size": total_sequences,
			"batch/valid_count": valid_count,
			"batch/valid_ratio": valid_ratio,
			"batch/invalid_count": invalid_count,
			"global/step": step + 1,
			"memory/new_smiles": max(memory_after["smiles"] - memory_before["smiles"], 0),
			"memory/new_scaffolds": max(memory_after["scaffolds"] - memory_before["scaffolds"], 0),
			"memory/total_smiles": memory_after["smiles"],
			"memory/total_scaffolds": memory_after["scaffolds"],
		}

		if total_sequences:
			score_array = np.asarray(total_scores, dtype=np.float32)
			metrics.update(
				{
					"score/total_mean": float(np.mean(score_array)),
					"score/total_std": float(np.std(score_array)),
					"score/total_sum": float(np.sum(score_array)),
					"score/total_min": float(np.min(score_array)),
					"score/total_max": float(np.max(score_array)),
				}
			)
			metrics["score/total_hist"] = wandb.Histogram(score_array)

			valid_scores = score_array[score_summary.valid_idxs]
			if valid_scores.size:
				metrics.update(
					{
						"score/valid_mean": float(np.mean(valid_scores)),
						"score/valid_std": float(np.std(valid_scores)),
						"score/valid_min": float(np.min(valid_scores)),
						"score/valid_max": float(np.max(valid_scores)),
					}
				)

		for component_name, component_values in score_summary.component_scores.items():
			component_array = np.asarray(component_values, dtype=np.float32)
			if component_array.size == 0:
				continue
			metrics[f"score/{component_name}_mean"] = float(np.mean(component_array))
			metrics[f"score/{component_name}_std"] = float(np.std(component_array))
			metrics[f"score/{component_name}_min"] = float(np.min(component_array))
			metrics[f"score/{component_name}_max"] = float(np.max(component_array))
			metrics[f"score/{component_name}_hist"] = wandb.Histogram(component_array)

		smiles_list = [f"{seq.scaffold}|{seq.decoration}" for seq in sampled_sequences]
		unique_smiles = len(set(smiles_list))
		unique_ratio = (unique_smiles / total_sequences) if total_sequences else 0.0
		metrics.update(
			{
				"batch/unique_smiles": unique_smiles,
				"batch/unique_ratio": unique_ratio,
			}
		)

		sampled_nlls = np.asarray([seq.nll for seq in sampled_sequences], dtype=np.float32)
		if sampled_nlls.size:
			metrics.update(
				{
					"sampling/nll_mean": float(np.mean(sampled_nlls)),
					"sampling/nll_std": float(np.std(sampled_nlls)),
					"sampling/nll_min": float(np.min(sampled_nlls)),
					"sampling/nll_max": float(np.max(sampled_nlls)),
				}
			)

		actor_array = self._to_numpy(actor_nlls)
		if actor_array.size:
			metrics.update(
				{
					"loss/actor_nll_mean": float(np.mean(actor_array)),
					"loss/actor_nll_std": float(np.std(actor_array)),
					"loss/actor_nll_min": float(np.min(actor_array)),
					"loss/actor_nll_max": float(np.max(actor_array)),
				}
			)

		critic_array = self._to_numpy(critic_nlls)
		if critic_array.size:
			metrics.update(
				{
					"loss/critic_nll_mean": float(np.mean(critic_array)),
					"loss/critic_nll_std": float(np.std(critic_array)),
					"loss/critic_nll_min": float(np.min(critic_array)),
					"loss/critic_nll_max": float(np.max(critic_array)),
				}
			)

		augmented_array = self._to_numpy(augmented_nlls)
		if augmented_array.size:
			metrics.update(
				{
					"loss/augmented_nll_mean": float(np.mean(augmented_array)),
					"loss/augmented_nll_std": float(np.std(augmented_array)),
					"loss/augmented_nll_min": float(np.min(augmented_array)),
					"loss/augmented_nll_max": float(np.max(augmented_array)),
				}
			)

		wandb.log(metrics, step=step + 1)

	def _export_diversity_memory(self) -> None:
		diversity_filter = getattr(self.scoring_strategy, "diversity_filter", None)
		if diversity_filter is None:
			return

		memory = diversity_filter.get_memory_as_dataframe()
		if memory is None:
			return

		columns: List[str] = []
		if hasattr(memory, "to_dict"):
			records = memory.to_dict(orient="records")
			columns = list(getattr(memory, "columns", []))
		elif isinstance(memory, list):
			records = [dict(row) for row in memory]
		else:
			records = []

		output_dir = Path(self.configuration.output_dir)
		output_dir.mkdir(parents=True, exist_ok=True)
		output_file = output_dir / "diversity_memory.csv"

		if not columns:
			columns = sorted({key for record in records for key in record.keys()})
		if not columns:
			columns = ["SMILES"]

		with output_file.open("w", encoding="utf-8", newline="") as handle:
			writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
			writer.writeheader()
			for record in records:
				writer.writerow(record)

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
	parser.add_argument(
		"--seed",
		type=int,
		default=42,
		help="Random seed for reproducibility.",
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
	wandb_config = raw_config.get("wandb", {}) or {}

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
		wandb=wandb_config,
	)


def _load_scaffolds(source: Any) -> List[str]:
	if isinstance(source, list):
		return [str(scaffold) for scaffold in source]
	if isinstance(source, str):
		path = Path(source).expanduser()
		raise NotImplementedError("File-based scaffold loading to be implemented")
	raise TypeError("`scaffolds` must be a list or a path to a file")


def _serialize_configuration(configuration: ReinforcementLearningConfig) -> Dict[str, Any]:
	config_dict = asdict(configuration)
	config_dict.pop("wandb", None)
	config_dict["actor"] = str(configuration.actor)
	config_dict["critic"] = str(configuration.critic)
	config_dict["output_dir"] = str(configuration.output_dir)
	return config_dict


def init_wandb_run(configuration: ReinforcementLearningConfig) -> Optional[Any]:
	wandb_cfg = dict(configuration.wandb or {})
	init_kwargs: Dict[str, Any] = {}
	allowed_keys = (
		"project",
		"entity",
		"group",
		"job_type",
		"tags",
		"notes",
		"dir",
		"id",
		"resume",
		"mode",
	)
	for key in allowed_keys:
		value = wandb_cfg.get(key)
		if value is not None:
			init_kwargs[key] = value

	run_name = wandb_cfg.get("name") or wandb_cfg.get("run_name")
	if run_name is not None:
		init_kwargs["name"] = run_name

	if "project" not in init_kwargs:
		init_kwargs["project"] = "molecule-rl"

	config_payload = _serialize_configuration(configuration)
	return wandb.init(config=config_payload, **init_kwargs)


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
	set_seed(args.seed)
	raw_config = load_raw_config(args.config)
	configuration = build_configuration(raw_config)
	#logger = create_logger(configuration)
	logger = None
	timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
	base_output_dir = Path(configuration.output_dir).expanduser()
	timestamped_dir = base_output_dir / timestamp
	timestamped_dir.mkdir(parents=True, exist_ok=True)
	configuration.output_dir = str(timestamped_dir)
	print(f"RL artifacts will be saved to {configuration.output_dir}")
	wandb_run = None
	try:
		wandb_run = init_wandb_run(configuration)
		actor, critic = load_models(configuration)
		rl_runner = ReinforcementLearning(actor=actor, critic=critic, configuration=configuration, logger=logger)
		rl_runner.run()
	finally:
		if wandb_run is not None:
			wandb.finish()


if __name__ == "__main__":
	main()
