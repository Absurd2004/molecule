"""Random sampling script for baseline comparison.

本脚本从预训练模型随机采样分子，使用与 RL 相同的评分策略，但不进行训练更新。
所有生成的分子及其评分都会保存到 CSV 文件中，便于与强化学习结果对比。
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import time
import random
import torch

from models.rl_actions import SampleModel
from dto import SampledSequencesDTO
from configurations.configurations import ReinforcementLearningConfiguration as ReinforcementLearningConfig
from configurations.configurations import LearningStrategyConfiguration as LearningStrategyConfig
from configurations.configurations import ScoringStrategyConfiguration as ScoringStrategyConfig
from scoring_strategy.scoring_strategy import StandardScoringStrategy
from scoring_strategy.summary import ScoreSummary
import models.model as mm
from run_rl import ReinforcementLearning, load_models


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """设置随机种子以确保可复现性。"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


# ---------------------------------------------------------------------------
# Random sampling class
# ---------------------------------------------------------------------------

class RandomSampling:
    """随机采样器，使用预训练模型采样并评分，不进行训练更新。"""

    def __init__(
        self,
        model: Any,
        configuration: ReinforcementLearningConfig,
        logger: Any = None,
    ) -> None:
        self.model = model
        self.configuration = self._double_single_scaffold_hack(configuration)
        self.logger = logger

        # 初始化评分策略（复用 RL 的评分逻辑）
        self.scoring_strategy = StandardScoringStrategy(
            strategy_configuration=self.configuration.scoring_strategy,
            logger=self.logger,
        )

        # 用于累积所有采样结果
        self.all_samples: List[Dict[str, Any]] = []

    def run(self) -> None:
        """执行随机采样主循环。"""
        print(f"Running random sampling with {len(self.configuration.scaffolds)} scaffolds")
        start_time = time.time()

        first_model_param = next(self.model.network.parameters(), None)
        model_device = first_model_param.device if first_model_param is not None else "unknown"
        print(f"Model on {model_device}")

        for step in range(self.configuration.n_steps):
            step_start = time.time()
            
            # 采样分子
            sampled_sequences = self._sampling()
            
            # 评分
            score_summary = self._scoring(sampled_sequences, step)
            
            # 保存当前 batch 的结果
            self._record_batch(sampled_sequences, score_summary, step)
            self._save_all_results(silent=True)
            
            step_end = time.time()
            
            # 打印进度信息
            self._print_step_info(
                step=step,
                sampled_sequences=sampled_sequences,
                score_summary=score_summary,
                step_duration=step_end - step_start,
                total_elapsed=step_end - start_time,
            )

        # 保存所有结果到 CSV
        self._save_all_results()
        
        # 导出 diversity memory（如果存在）
        self._export_diversity_memory()

        total_time = time.time() - start_time
        print(f"\nRandom sampling completed in {total_time:.2f} seconds")
        print(f"Total molecules generated: {len(self.all_samples)}")
        print(f"Results saved to: {self.configuration.output_dir}")

    def _sampling(self) -> List[SampledSequencesDTO]:
        """从模型中采样分子序列。"""
        sampling_action = SampleModel(
            self.model,
            self.configuration.batch_size,
            self.logger,
            self.configuration.randomize_scaffolds
        )
        print(f"Sampling {self.configuration.batch_size} sequences...")
        sampled_sequences = sampling_action.run(self.configuration.scaffolds)
        print(f"Sampled {len(sampled_sequences)} sequences")
        return sampled_sequences

    def _scoring(self, sampled_sequences: List[SampledSequencesDTO], step: int) -> ScoreSummary:
        """计算采样序列的评分。"""
        return self.scoring_strategy.evaluate(sampled_sequences, step)

    def _record_batch(
        self,
        sampled_sequences: List[SampledSequencesDTO],
        score_summary: ScoreSummary,
        step: int,
    ) -> None:
        """记录当前 batch 的采样结果和评分。"""
        total_scores = score_summary.total_score
        component_scores = score_summary.component_scores

        for idx, seq in enumerate(sampled_sequences):
            record: Dict[str, Any] = {
                "step": step,
                "scaffold": seq.scaffold,
                "decoration": seq.decoration,
                "smiles": f"{seq.scaffold}|{seq.decoration}",
                "nll": seq.nll,
                "total_score": total_scores[idx] if idx < len(total_scores) else 0.0,
            }

            # 添加各个评分组件
            for component_name, component_values in component_scores.items():
                if idx < len(component_values):
                    record[f"score_{component_name}"] = component_values[idx]

            self.all_samples.append(record)

    def _save_all_results(self, *, silent: bool = False) -> None:
        """将所有采样结果保存到 CSV 文件。"""
        if not self.all_samples:
            print("No samples to save.")
            return

        output_dir = Path(self.configuration.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "random_sampling_results.csv"

        # 确定所有可能的列名
        all_keys = set()
        for record in self.all_samples:
            all_keys.update(record.keys())
        
        # 定义列顺序：step, scaffold, decoration, smiles, nll, total_score, 其他评分
        priority_columns = ["step", "scaffold", "decoration", "smiles", "nll", "total_score"]
        other_columns = sorted(all_keys - set(priority_columns))
        fieldnames = [col for col in priority_columns if col in all_keys] + other_columns

        with output_file.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for record in self.all_samples:
                writer.writerow(record)

        if not silent:
            print(f"\nSaved {len(self.all_samples)} samples to {output_file}")

    def _export_diversity_memory(self) -> None:
        """导出 diversity filter 的内存（如果存在）。"""
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

        print(f"Exported diversity memory to {output_file}")

    def _print_step_info(
        self,
        step: int,
        sampled_sequences: List[SampledSequencesDTO],
        score_summary: ScoreSummary,
        step_duration: float,
        total_elapsed: float,
    ) -> None:
        """打印每一步的统计信息。"""
        total_sequences = len(sampled_sequences)
        valid_count = len(score_summary.valid_idxs)
        invalid_count = total_sequences - valid_count
        valid_ratio = (valid_count / total_sequences) if total_sequences else 0.0

        total_scores = score_summary.total_score
        if total_sequences and len(total_scores):
            score_array = np.asarray(total_scores, dtype=np.float32)
            score_mean = float(np.mean(score_array))
            score_std = float(np.std(score_array))
            score_min = float(np.min(score_array))
            score_max = float(np.max(score_array))
        else:
            score_mean = score_std = score_min = score_max = 0.0

        print(
            f"Step {step + 1}/{self.configuration.n_steps} | "
            f"Time: {step_duration:.2f}s | "
            f"Valid: {valid_count}/{total_sequences} ({valid_ratio:.2%}) | "
            f"Score: {score_mean:.3f}±{score_std:.3f} [{score_min:.3f}, {score_max:.3f}]"
        )

    @staticmethod
    def _double_single_scaffold_hack(configuration: ReinforcementLearningConfig) -> ReinforcementLearningConfig:
        """保持与 RL 相同的单 scaffold hack。"""
        if len(configuration.scaffolds) == 1:
            configuration.scaffolds *= 2
            configuration.batch_size = max(int(configuration.batch_size / 2), 1)
        return configuration


# ---------------------------------------------------------------------------
# Configuration and model loading
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="Run random sampling from pretrained model")
    parser.add_argument(
        "--config",
        type=Path,
        default="./configs/rl_configs.json",
        help="Path to a JSON configuration file (same format as RL config).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--rl-steps-before-sampling",
        type=int,
        default=0,
        help="Number of RL fine-tuning steps to run before switching to pure random sampling.",
    )
    return parser.parse_args()


def load_raw_config(config_path: Path) -> Dict[str, Any]:
    """从 JSON 文件加载配置。"""
    with config_path.expanduser().open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_configuration(raw_config: Dict[str, Any]) -> ReinforcementLearningConfig:
    """构建配置对象（复用 RL 配置结构）。"""
    actor_path = Path(raw_config["actor"]).expanduser()
    critic_path = Path(raw_config.get("critic", raw_config["actor"])).expanduser()
    scaffolds = _load_scaffolds(raw_config["scaffolds"])
    output_dir = Path(raw_config.get("output_dir", "./random_sampling_runs")).expanduser()

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
        wandb={},  # 不使用 wandb
    )


def _load_scaffolds(source: Any) -> List[str]:
    """加载 scaffolds 列表。"""
    if isinstance(source, list):
        return [str(scaffold) for scaffold in source]
    if isinstance(source, str):
        path = Path(source).expanduser()

        raise NotImplementedError("File-based scaffold loading to be implemented")
    raise TypeError("`scaffolds` must be a list or a path to a file")


def load_model_from_checkpoint(checkpoint_path: Path, mode: str = "eval") -> Any:
    """从 checkpoint 加载模型。"""
    model = mm.DecoratorModel.load_from_file(checkpoint_path, mode=mode)
    return model


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def main() -> None:
    """主入口函数。"""
    args = parse_args()
    set_seed(args.seed)

    # 加载配置
    raw_config = load_raw_config(args.config)
    configuration = build_configuration(raw_config)

    # 创建带时间戳的输出目录
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base_output_dir = Path(configuration.output_dir).expanduser()
    timestamped_dir = base_output_dir / timestamp
    timestamped_dir.mkdir(parents=True, exist_ok=True)
    configuration.output_dir = str(timestamped_dir)
    print(f"Random sampling results will be saved to {configuration.output_dir}")

    rl_steps_before_sampling = max(int(args.rl_steps_before_sampling), 0)

    # 根据需要运行少量强化学习步骤以提升采样模型
    if rl_steps_before_sampling > 0:
        print(f"Running {rl_steps_before_sampling} RL warmup steps before random sampling")

        original_steps = configuration.n_steps
        original_output_dir = configuration.output_dir

        warmup_dir = Path(configuration.output_dir) / "rl_warmup"
        warmup_dir.mkdir(parents=True, exist_ok=True)
        configuration.n_steps = rl_steps_before_sampling
        configuration.output_dir = str(warmup_dir)

        actor, critic = load_models(configuration)
        rl_runner = ReinforcementLearning(actor=actor, critic=critic, configuration=configuration, logger=None)
        rl_runner.run()

        configuration.n_steps = original_steps
        configuration.output_dir = original_output_dir

        model = actor.set_mode("eval")
        print("Warmup RL finished; using fine-tuned actor for sampling")
    else:
        model = load_model_from_checkpoint(configuration.actor, mode="eval")
        model.set_mode("eval")
        print(f"Loaded pretrained model from {configuration.actor}")

    # 运行随机采样
    sampler = RandomSampling(model=model, configuration=configuration, logger=None)
    sampler.run()


if __name__ == "__main__":
    main()
