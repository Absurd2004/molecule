"""Plot cumulative hit counts over steps with optional time-to-hit bar chart inset.

This script mirrors the styling of ``total_score_trend.py`` but focuses on
tracking how many molecules meet a specified set of criteria over time.

Requirements
------------
* Input CSV files must include the columns ``Step``, ``SMILES``, ``st_gap_raw``,
  ``charge_score``, and ``symmetry_score``.
* For each run (CSV), the cumulative number of molecules satisfying the provided
  thresholds is computed as we iterate through steps in ascending order.
* An inset bar chart (bottom-right corner) summarises the time-to-hit metrics for
  configurable thresholds (e.g., first hit, 10 hits, 50 hits).

Usage example
-------------

.. code-block:: bash

    python Hit_count.py \
        --config configs/hit_count.json \
        --bias 0 \
        --st-gap-threshold 0.3 \
        --charge-threshold 1.0 \
        --symmetry-threshold 1.0 \
        --hit-thresholds 1 10 50 \
        --output rl_runs/hit_count/hit_count.png

Config file (JSON/YAML) example
-------------------------------

.. code-block:: json

    {
      "curves": [
        {
          "csv": "run_a.csv",
          "label": "Run A",
          "step_range": [0, 5000],
          "color": [220, 20, 60]
        },
        {
          "csv": "run_b.csv",
          "label": "Run B",
          "step_range": [1000, 6000],
          "color": [72, 61, 139]
        }
      ]
    }

"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

try:
	import matplotlib.pyplot as plt  # type: ignore[import]
except ModuleNotFoundError as exc:  # pragma: no cover - import guard
	raise ModuleNotFoundError(
		"matplotlib is required to run Hit_count.py. Please install it via 'pip install matplotlib'."
	) from exc

import numpy as np

from mpl_toolkits.axes_grid1.inset_locator import inset_axes

try:
	import pandas as pd  # type: ignore[import]
except ModuleNotFoundError as exc:  # pragma: no cover - import guard
	raise ModuleNotFoundError(
		"pandas is required to run Hit_count.py. Please install it via 'pip install pandas'."
	) from exc


@dataclass
class CurveSpec:
	"""Configuration describing one line on the chart."""

	csv: Path
	label: str
	step_start: float
	step_end: float
	color: Tuple[float, float, float]

	@classmethod
	def from_dict(cls, data: dict) -> "CurveSpec":
		try:
			csv_path = Path(data["csv"]).expanduser()
			label = data["label"]
			step_range = data["step_range"]
			color_values = data["color"]
		except KeyError as exc:  # pragma: no cover - defensive branch
			missing = exc.args[0]
			raise ValueError(f"Missing required key '{missing}' in curve config") from exc

		if len(step_range) != 2:
			raise ValueError("'step_range' must contain exactly two numbers: [start, end]")

		step_start, step_end = map(float, step_range)
		if step_start > step_end:
			raise ValueError("step_range start must be less than or equal to end")

		if len(color_values) != 3:
			raise ValueError("'color' must be an RGB sequence of length 3")

		color = tuple(float(channel) / 255.0 for channel in color_values)
		if any(channel < 0.0 or channel > 1.0 for channel in color):
			raise ValueError("RGB color channels must be between 0 and 255")

		return cls(
			csv=csv_path,
			label=str(label),
			step_start=step_start,
			step_end=step_end,
			color=color,
		)


def load_config(config_path: Path) -> Sequence[CurveSpec]:
	"""Load curve specifications from a JSON or YAML config file."""

	config_text = config_path.read_text(encoding="utf-8")

	if config_path.suffix.lower() in {".yaml", ".yml"}:
		try:
			import yaml  # type: ignore
		except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
			raise RuntimeError(
				"PyYAML is required for YAML configs. Install it or provide a JSON config."
			) from exc

		raw_config = yaml.safe_load(config_text)
	else:
		raw_config = json.loads(config_text)

	if not isinstance(raw_config, dict) or "curves" not in raw_config:
		raise ValueError("Configuration file must contain a top-level 'curves' list")

	curves = raw_config["curves"]
	if not isinstance(curves, Iterable):
		raise ValueError("'curves' must be a list of curve definitions")

	result: List[CurveSpec] = []
	for idx, curve_data in enumerate(curves):
		if not isinstance(curve_data, dict):
			raise ValueError(f"Curve entry at index {idx} must be an object")
		result.append(CurveSpec.from_dict(curve_data))

	if not result:
		raise ValueError("Configuration must define at least one curve")

	return result


def compute_cumulative_hits(
	df: pd.DataFrame,
	step_start: float,
	step_end: float,
	st_gap_threshold: float,
	charge_threshold: float,
	symmetry_threshold: float,
	bias: float,
	step_col: str,
	st_gap_col: str,
	charge_col: str,
	symmetry_col: str,
) -> pd.DataFrame:
	"""Compute cumulative hit counts for molecules meeting the thresholds."""

	required_columns = {step_col, st_gap_col, charge_col, symmetry_col}
	missing_columns = required_columns.difference(df.columns)
	if missing_columns:
		raise ValueError(f"CSV is missing required columns: {', '.join(sorted(missing_columns))}")

	mask = (df[step_col] >= step_start) & (df[step_col] <= step_end)
	scoped = df.loc[mask].copy()
	scoped.sort_values(step_col, inplace=True)

	steps: List[float] = []
	counts: List[float] = []
	hit_count = 0

	for row in scoped.itertuples(index=False):
		step_value = getattr(row, step_col)
		if pd.isna(step_value):
			continue

		st_gap_value = getattr(row, st_gap_col)
		charge_value = getattr(row, charge_col)
		symmetry_value = getattr(row, symmetry_col)

		if any(pd.isna(v) for v in (st_gap_value, charge_value, symmetry_value)):
			continue

		if (
			float(st_gap_value) < st_gap_threshold
			and float(charge_value) >= charge_threshold
			and float(symmetry_value) >= symmetry_threshold
		):
			hit_count += 1

		steps.append(step_value)
		counts.append(hit_count + bias)

	return pd.DataFrame({step_col: steps, "cumulative_hits": counts})


def compute_time_to_hits(
	df: pd.DataFrame,
	hit_thresholds: Sequence[int],
	step_col: str,
) -> List[Optional[float]]:
	"""Return the step at which the cumulative hits first reach each threshold."""

	results: List[Optional[float]] = []
	for threshold in hit_thresholds:
		match = df[df["cumulative_hits"] >= threshold]
		if match.empty:
			results.append(None)
		else:
			results.append(float(match.iloc[0][step_col]))
	return results


def plot_curves(
	curves: Sequence[CurveSpec],
	datasets: Sequence[pd.DataFrame],
	time_to_hits: Sequence[Optional[float]],
	hit_thresholds: Sequence[int],
	output_path: Optional[Path],
	step_col: str,
	title: Optional[str] = None,
) -> None:
	"""Plot cumulative hit curves with inset bar chart."""

	fig, ax = plt.subplots(figsize=(10, 6))

	for curve, data in zip(curves, datasets):
		ax.plot(
			data[step_col],
			data["cumulative_hits"],
			color=curve.color,
			linewidth=1.5,
			linestyle="-",
			label=curve.label,
		)

	orig_xlim = ax.get_xlim()
	x_min, x_max = orig_xlim
	x_ticks = ax.get_xticks()
	valid_ticks = [tick for tick in x_ticks if tick > x_min and tick <= x_max]
	if valid_ticks:
		x_right = valid_ticks[0]
		ax.axvspan(
			x_min,
			x_right,
			facecolor=(0.9, 0.9, 0.9, 0.7),
			edgecolor="none",
			zorder=0.5,
		)
		ax.set_xlim(orig_xlim)

	ax.set_xlabel("Step", fontsize=12)
	ax.set_ylabel("Cumulative hit count", fontsize=12)
	if title:
		ax.set_title(title, fontsize=14)

	ax.grid(True, linestyle=(0, (1, 2)), linewidth=0.8)
	ax.spines["top"].set_visible(False)
	ax.spines["right"].set_visible(False)

	ax.legend()
	fig.tight_layout()

	# Inset bar chart for time-to-hit metrics.
	if hit_thresholds:
		inset_ax = inset_axes(
			ax,
			width="38%",
			height="34%",
			loc="lower right",
			borderpad=0.8,
		)

		label_texts: List[str] = []
		for threshold in hit_thresholds:
			if threshold == 1:
				label_texts.append("Time-to-first-hit")
			else:
				label_texts.append(f"Time-to-{threshold}-hits")

		available_times = [time for time in time_to_hits if time is not None]
		max_time = max(available_times, default=0.0)
		values = [time if time is not None else 0.0 for time in time_to_hits]
		positions = np.arange(len(hit_thresholds))
		color_map = plt.get_cmap("tab20")
		colors = [color_map(i % color_map.N) for i in range(len(hit_thresholds))]

		bars = inset_ax.barh(positions, values, height=0.55, color=colors)

		face_rgba = ax.get_facecolor()
		soft_bg = (
			face_rgba[0] * 0.85 + 0.15,
			face_rgba[1] * 0.85 + 0.15,
			face_rgba[2] * 0.85 + 0.15,
			0.9,
		)
		inset_ax.patch.set_facecolor(soft_bg)
		inset_ax.patch.set_edgecolor((0.0, 0.0, 0.0, 0.05))

		inset_ax.set_xlim(0, max_time * 1.1 + (max_time * 0.05 if max_time > 0 else 1.0))
		inset_ax.set_ylim(-0.5, len(hit_thresholds) - 0.5)
		inset_ax.set_xticks([])
		inset_ax.set_yticks([])
		for spine in inset_ax.spines.values():
			spine.set_visible(False)

		yaxis_transform = inset_ax.get_yaxis_transform()
		label_offset = -0.03
		for bar, label_text, time_value in zip(bars, label_texts, time_to_hits):
			y_pos = bar.get_y() + bar.get_height() / 2
			inset_ax.text(
				label_offset,
				y_pos,
				label_text,
				ha="right",
				va="center",
				fontsize=8,
				transform=yaxis_transform,
			)

			if time_value is None:
				text_value = "∞"
				x_pos = inset_ax.get_xlim()[1] * 0.95
			else:
				text_value = f"{time_value:.0f}"
				x_pos = bar.get_width()
			inset_ax.text(
				x_pos,
				y_pos,
				text_value,
				ha="left",
				va="center",
				fontsize=8,
			)

	if output_path:
		output_path.parent.mkdir(parents=True, exist_ok=True)
		fig.savefig(output_path, dpi=300)
	else:
		plt.show()


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Plot cumulative hit counts with time-to-hit inset")
	parser.add_argument("--config", type=Path, default=Path("./configs/hit_count.json"), help="Path to JSON or YAML config file")
	parser.add_argument(
		"--bias",
		type=float,
		default=0.0,
		help="Bias added to each cumulative count per step (default: 0.0)",
	)
	parser.add_argument(
		"--st-gap-threshold",
		type=float,
		default=0.3,
		help="Threshold for st_gap_raw to qualify as a hit (default: 0.3)",
	)
	parser.add_argument(
		"--charge-threshold",
		type=float,
		default=1.0,
		help="Minimum charge_score to qualify as a hit (default: 1.0)",
	)
	parser.add_argument(
		"--symmetry-threshold",
		type=float,
		default=1.0,
		help="Minimum symmetry_score to qualify as a hit (default: 1.0)",
	)
	parser.add_argument(
		"--hit-thresholds",
		nargs="*",
		type=int,
		default=[1, 10, 50],
		help="Hit count thresholds for the inset bar chart (default: 1 10 50)",
	)
	parser.add_argument(
		"--output",
		type=Path,
		default=Path("./rl_runs/hit_count/hit_count.png"),
		help="Optional path to save the figure. If omitted, the plot is shown interactively.",
	)
	parser.add_argument("--title", type=str, default=None, help="Optional title for the chart")
	parser.add_argument(
		"--step-column",
		type=str,
		default="Step",
		help="Name of the step column in the CSV files (default: Step)",
	)
	parser.add_argument(
		"--st-gap-column",
		type=str,
		default="st_gap_raw",
		help="Name of the st_gap column in the CSV files (default: st_gap_raw)",
	)
	parser.add_argument(
		"--charge-column",
		type=str,
		default="charge_score",
		help="Name of the charge score column (default: charge_score)",
	)
	parser.add_argument(
		"--symmetry-column",
		type=str,
		default="symmetry_score",
		help="Name of the symmetry score column (default: symmetry_score)",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()

	curves = load_config(args.config)

	datasets: List[pd.DataFrame] = []
	time_to_hits_per_curve: List[List[Optional[float]]] = []

	for curve in curves:
		if not curve.csv.exists():
			raise FileNotFoundError(f"CSV file not found: {curve.csv}")

		df = pd.read_csv(curve.csv)
		result = compute_cumulative_hits(
			df,
			step_start=curve.step_start,
			step_end=curve.step_end,
			st_gap_threshold=args.st_gap_threshold,
			charge_threshold=args.charge_threshold,
			symmetry_threshold=args.symmetry_threshold,
			bias=args.bias,
			step_col=args.step_column,
			st_gap_col=args.st_gap_column,
			charge_col=args.charge_column,
			symmetry_col=args.symmetry_column,
		)
		datasets.append(result)
		time_to_hits_per_curve.append(
			compute_time_to_hits(result, args.hit_thresholds, step_col=args.step_column)
		)

	# For the inset we use the metrics from the first curve by default; if multiple
	# curves are supplied, we take the minimum time-to-hit across curves for each threshold.
	if not time_to_hits_per_curve:
		aggregated_time_to_hits: List[Optional[float]] = []
	else:
		aggregated_time_to_hits = []
		for idx in range(len(args.hit_thresholds)):
			values = [curve_times[idx] for curve_times in time_to_hits_per_curve if curve_times[idx] is not None]
			aggregated_time_to_hits.append(min(values) if values else None)

	plot_curves(
		curves=curves,
		datasets=datasets,
		time_to_hits=aggregated_time_to_hits,
		hit_thresholds=args.hit_thresholds,
		output_path=args.output,
		step_col=args.step_column,
		title=args.title,
	)


if __name__ == "__main__":
	main()
