"""Utility for plotting top-n st_gap_raw trends across multiple CSV files.

The script expects one or more CSV files that each contain the columns:
	- ``Step``: numeric step indicator
	- ``SMILES``: molecule identifier string
	- ``st_gap_raw``: numeric score

For every CSV we produce one line on the chart. Each line represents, for every
step within a user-provided range, the mean value of the smallest ``n``
``st_gap_raw`` scores observed so far (from the beginning of the selected step
range up to the current step). When fewer than ``n`` unique molecules have been
seen, the point is left blank (NaN).

Usage example (JSON configuration)::

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

Command line::

	python est_trend.py --top-n 10 --config config.json --output trend.png

Config files with extensions ``.json`` as well as ``.yaml``/``.yml`` are
accepted. YAML support requires PyYAML to be installed.
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
		"matplotlib is required to run est_trend.py. Please install it via 'pip install matplotlib'."
	) from exc

import numpy as np

try:
	import pandas as pd  # type: ignore[import]
except ModuleNotFoundError as exc:  # pragma: no cover - import guard
	raise ModuleNotFoundError(
		"pandas is required to run est_trend.py. Please install it via 'pip install pandas'."
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


def compute_top_n_cumulative_mean(
	df: pd.DataFrame,
	top_n: int,
	step_start: float,
	step_end: float,
	bias: float = 0.0,
	step_col: str = "Step",
	value_col: str = "st_gap_raw",
	smiles_col: str = "SMILES",
) -> pd.DataFrame:
	"""Return DataFrame with cumulative mean of top-n values for each step.

	The computation iterates through the rows ordered by ``Step`` and keeps the
	best (minimum) ``st_gap_raw`` encountered so far for every unique ``SMILES``.
	For each step we calculate the mean of the ``n`` smallest scores seen up to
	that point. Points with fewer than ``n`` unique molecules are represented as
	``NaN`` so that they don't show up on the line.
	"""

	if top_n <= 0:
		raise ValueError("top_n must be a positive integer")

	required_columns = {step_col, value_col, smiles_col}
	missing_columns = required_columns.difference(df.columns)
	if missing_columns:
		raise ValueError(f"CSV is missing required columns: {', '.join(sorted(missing_columns))}")

	mask = (df[step_col] >= step_start) & (df[step_col] <= step_end)
	scoped = df.loc[mask].copy()
	scoped.sort_values(step_col, inplace=True)

	steps: List[float] = []
	means: List[float] = []
	best_scores: dict[str, float] = {}

	# Iterate efficiently: keep running pointer instead of re-filtering group each time.
	current_step: Optional[float] = None
	for row in scoped.itertuples(index=False):
		step_value = getattr(row, step_col)
		smiles_value = getattr(row, smiles_col)
		score_value = getattr(row, value_col)

		if pd.isna(step_value) or pd.isna(score_value) or isinstance(score_value, str):
			# Skip invalid rows gracefully.
			continue

		if current_step is None:
			current_step = step_value

		if step_value != current_step:
			_update_summary(current_step, best_scores, top_n, steps, means)
			current_step = step_value

		existing_score = best_scores.get(smiles_value)
		numeric_score = float(score_value) + bias
		#if numeric_score < 0.0:
			#numeric_score = 0.0
		if existing_score is None or numeric_score < existing_score:
			best_scores[smiles_value] = numeric_score

	if current_step is not None:
		_update_summary(current_step, best_scores, top_n, steps, means)

	return pd.DataFrame({step_col: steps, "top_n_mean": means})


def _update_summary(
	step_value: float,
	best_scores: dict[str, float],
	top_n: int,
	steps: List[float],
	means: List[float],
) -> None:
	"""Helper to append the current top-n mean for the given step."""

	steps.append(step_value)
	if len(best_scores) < top_n:
		means.append(math.nan)
		return

	smallest_scores = np.partition(np.fromiter(best_scores.values(), dtype=float), top_n - 1)[:top_n]
	means.append(float(np.mean(smallest_scores)))


def plot_curves(
	curves: Sequence[CurveSpec],
	datasets: Sequence[pd.DataFrame],
	top_n: int,
	output_path: Optional[Path],
	step_col: str,
	title: Optional[str] = None,
) -> None:
	"""Plot the computed trend lines."""

	fig, ax = plt.subplots(figsize=(10, 6))

	for curve, data in zip(curves, datasets):
		ax.plot(
			data[step_col],
			data["top_n_mean"],
			color=curve.color,
			linewidth=2.0,
			linestyle="-",
			label=curve.label,
		)

	# Shade the first grid column to help delineate sections.
	# We derive the grid boundaries from the x-ticks (after plotting so Matplotlib has them ready).
	x_ticks = ax.get_xticks()
	if len(x_ticks) >= 2:
		x_min = ax.get_xlim()[0]
		x_next = x_ticks[1]
		x_right = min(x_next, ax.get_xlim()[1])
		y_bottom, y_top = ax.get_ylim()
		ax.axvspan(
			x_min,
			x_right,
			facecolor=(0.9, 0.9, 0.9, 0.5),
			edgecolor="none",
			zorder=0.5,
		)

	ax.set_xlabel("Step", fontsize=12)
	ax.set_ylabel(f"Mean of top {top_n} st_gap_raw", fontsize=12)
	if title:
		ax.set_title(title, fontsize=14)

	ax.grid(True, linestyle=(0, (1, 2)), linewidth=0.8)
	ax.spines["top"].set_visible(False)
	ax.spines["right"].set_visible(False)

	ax.legend()
	fig.tight_layout()

	if output_path:
		fig.savefig(output_path, dpi=300)
	else:
		plt.show()


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Plot cumulative top-n st_gap_raw trends")
	parser.add_argument("--config", type=Path, default= Path("./configs/est_trend.json"),help="Path to JSON or YAML config file")
	parser.add_argument("--top-n",  type=int, default = 50,help="Number of best molecules to average")
	parser.add_argument(
		"--bias",
		type=float,
		default=0.18,
		help="Bias added to each st_gap_raw value before processing (default: 0.0)",
	)
	parser.add_argument(
		"--output",
		type=Path,
		default= "./rl_runs/pic16_est_trend/est_trend.png",
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
		"--value-column",
		type=str,
		default="st_gap_raw",
		help="Name of the value column to minimize (default: st_gap_raw)",
	)
	parser.add_argument(
		"--smiles-column",
		type=str,
		default="SMILES",
		help="Name of the column carrying SMILES strings (default: SMILES)",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()

	curves = load_config(args.config)

	datasets: List[pd.DataFrame] = []
	for curve in curves:
		if not curve.csv.exists():
			raise FileNotFoundError(f"CSV file not found: {curve.csv}")

		df = pd.read_csv(curve.csv)
		result = compute_top_n_cumulative_mean(
			df,
			top_n=args.top_n,
			step_start=curve.step_start,
			step_end=curve.step_end,
			bias=args.bias,
			step_col=args.step_column,
			value_col=args.value_column,
			smiles_col=args.smiles_column,
		)
		datasets.append(result)

	plot_curves(
		curves=curves,
		datasets=datasets,
		top_n=args.top_n,
		output_path=args.output,
		step_col=args.step_column,
		title=args.title,
	)


if __name__ == "__main__":
	main()
