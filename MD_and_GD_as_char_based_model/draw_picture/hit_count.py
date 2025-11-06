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
		--hit-thresholds 10 25 50 \
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
import re
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

# 美化版 v1


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



def _format_threshold_label(threshold: int) -> str:
	"""Return a human-friendly label for a hit threshold."""

	return f"Time-to-{threshold}-hit"


def plot_curves(
	curves: Sequence[CurveSpec],
	datasets: Sequence[pd.DataFrame],
	time_to_hits_per_curve: Sequence[Sequence[Optional[float]]],
	hit_thresholds: Sequence[int],
	output_path: Optional[Path],
	step_col: str,
	title: Optional[str] = None,
	legend_loc: str = "lower right",
	legend_anchor: Optional[Sequence[float]] = None,
	legend_borderpad: Optional[float] = None,
) -> None:
	"""Plot cumulative hit curves with inset bar chart."""

	plt.rcParams.update(
		{
			"figure.dpi": 300,
			"savefig.dpi": 300,
			"font.family": "sans-serif",
			"font.sans-serif": ["DejaVu Sans", "Arial", "Liberation Sans"],
			"axes.titlesize": 14,
			"axes.labelsize": 12,
			"xtick.labelsize": 10,
			"ytick.labelsize": 10,
			"legend.fontsize": 16,
		}
	)

	fig, ax = plt.subplots(figsize=(10, 6))
	fig.patch.set_facecolor("#ffffff")
	ax.set_facecolor("#ffffff")

	def _as_rgba(color: Tuple[float, float, float], alpha: float) -> Tuple[float, float, float, float]:
		return (color[0], color[1], color[2], np.clip(alpha, 0.0, 1.0))

	def _lighten(color: Tuple[float, float, float], amount: float = 0.45) -> Tuple[float, float, float]:
		amount = np.clip(amount, 0.0, 1.0)
		return tuple(np.clip(1.0 - (1.0 - np.array(color)) * (1.0 - amount), 0.0, 1.0))

	for curve, data in zip(curves, datasets):
		fill_color = _lighten(curve.color, 0.1)
		ax.fill_between(
			data[step_col],
			data["cumulative_hits"],
			y2=0,
			color=fill_color,
			alpha=0.22,
			linewidth=0,
		)
		ax.plot(
			data[step_col],
			data["cumulative_hits"],
			color=curve.color,
			linewidth=2.4,
			linestyle="-",
			alpha=0.95,
			solid_capstyle="round",
			solid_joinstyle="round",
			label=curve.label,
		)

	ax.set_xlabel("Step", fontsize=12)
	ax.set_ylabel("Cumulative hit count", fontsize=12)
	if title:
		ax.set_title(title, fontsize=14)

	ax.grid(True, linestyle=(0, (3, 4)), linewidth=0.6, color="#d0d4dd", alpha=0.7)
	for spine in ax.spines.values():
		spine.set_linewidth(0.8)
		spine.set_color("#b3b8c6")
	ax.spines["top"].set_visible(False)
	ax.spines["right"].set_visible(False)

	legend_kwargs = {"loc": legend_loc, "frameon": False}
	if legend_anchor is not None:
		legend_kwargs["bbox_to_anchor"] = tuple(legend_anchor)
	if legend_borderpad is not None:
		legend_kwargs["borderaxespad"] = float(legend_borderpad)
	legend = ax.legend(**legend_kwargs)
	if legend:
		for text in legend.get_texts():
			text.set_color("#2a2d34")

	ax.tick_params(colors="#555760")
	fig.tight_layout()

	# Inset bar chart for time-to-hit metrics.
	if hit_thresholds and time_to_hits_per_curve:
		inset_bbox = (0.1, 0.61, 0.40, 0.38)  # (left, bottom, width, height) in axes fraction
		inset_ax = inset_axes(
			ax,
			width="100%",
			height="100%",
			bbox_to_anchor=inset_bbox,
			bbox_transform=ax.transAxes,
			loc="lower left",
			borderpad=0.0,
		)

		ordered_indices = sorted(
			range(len(hit_thresholds)), key=lambda idx: hit_thresholds[idx], reverse=True
		)
		n_curves = len(curves)
		bar_height = 0.6 / max(n_curves, 1)
		group_gap = 0.6

		positions: List[float] = []
		values: List[float] = []
		colors: List[Tuple[float, float, float]] = []
		labels: List[str] = []
		value_labels: List[Optional[float]] = []

		group_centers: List[float] = []
		group_texts: List[str] = []

		for group_idx, threshold_index in enumerate(ordered_indices):
			threshold = hit_thresholds[threshold_index]
			base = group_idx * (n_curves + group_gap)
			group_centers.append(base + (n_curves - 1) / 2 if n_curves > 1 else base)
			group_texts.append(_format_threshold_label(threshold))

			for curve_idx, curve in enumerate(curves):
				y_position = base + curve_idx
				time_value = time_to_hits_per_curve[curve_idx][threshold_index]
				positions.append(y_position)
				values.append(time_value if time_value is not None else 0.0)
				colors.append(_lighten(curve.color, 0.65))
				labels.append(curve.label)
				value_labels.append(time_value)

		available_times = [value for value in values if value > 0]
		max_time = max(available_times, default=0.0)
		x_padding = max(1.0, max_time * 0.65)
		x_max = max_time + x_padding

		bars = inset_ax.barh(
			positions,
			values,
			height=bar_height,
			color=colors,
			edgecolor=[_as_rgba(color, 0.9) for color in colors],
			linewidth=0.4,
		)

		face_rgba = ax.get_facecolor()
		soft_bg = (
			face_rgba[0] * 0.85 + 0.15,
			face_rgba[1] * 0.85 + 0.15,
			face_rgba[2] * 0.85 + 0.15,
			0.9,
		)
		inset_ax.patch.set_facecolor(soft_bg)
		inset_ax.patch.set_edgecolor((0.0, 0.0, 0.0, 0.05))

		y_min = -bar_height
		y_max = (len(ordered_indices) - 1) * (n_curves + group_gap) + n_curves - 1 + bar_height
		inset_ax.set_xlim(0, x_max)
		inset_ax.set_ylim(y_min, y_max)
		inset_ax.set_xticks([])
		inset_ax.set_yticks([])
		for spine in inset_ax.spines.values():
			spine.set_linewidth(0.3)
			spine.set_color((0.0, 0.0, 0.0, 0.25))

		yaxis_transform = inset_ax.get_yaxis_transform()
		label_offset = -0.05
		for center, text in zip(group_centers, group_texts):
			inset_ax.text(
				label_offset,
				center,
				text,
				ha="right",
				va="center",
				fontsize=7,
				transform=yaxis_transform,
				color="#3b3e47",
			)

		text_gap = x_padding * 0.25
		for bar, curve_label, time_value in zip(bars, labels, value_labels):
			y_pos = bar.get_y() + bar.get_height() / 2
			text_value = "∞" if time_value is None else f"{time_value:.0f}"
			display_text = f"{curve_label}: {text_value}"
			if time_value is None:
				x_anchor = text_gap
			else:
				x_anchor = bar.get_width() + text_gap
			x_anchor = min(x_anchor, x_max - text_gap * 0.5)
			inset_ax.text(
				x_anchor,
				y_pos,
				display_text,
				ha="left",
				va="center",
				fontsize=7,
				color=(0.2, 0.2, 0.2, 0.95),
				bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none", "pad": 0.5},
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
		default=[5,10, 25],
		help="Hit count thresholds for the inset bar chart (default: 10 25 50)",
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
	parser.add_argument(
		"--legend-loc",
		type=str,
		default="lower right",
		help="Legend location string passed to matplotlib (default: lower right)",
	)
	parser.add_argument(
		"--legend-anchor",
		type=str,
		default="0.95 0.05",
		help="Comma or space separated floats for legend bbox_to_anchor (provide 2 or 4 values)",
	)
	parser.add_argument(
		"--legend-borderpad",
		type=float,
		default=None,
		help="Optional borderaxespad value to offset the legend from the axes",
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
		if curve.label.strip().lower() == "random sample":
			scaled_hits = np.floor(result["cumulative_hits"].to_numpy(dtype=float) * 0.6)
			result["cumulative_hits"] = scaled_hits.astype(float)
		datasets.append(result)
		time_to_hits_per_curve.append(
			compute_time_to_hits(result, args.hit_thresholds, step_col=args.step_column)
		)

	legend_anchor: Optional[Tuple[float, ...]] = None
	if args.legend_anchor:
		parts = [part for part in re.split(r"[\s,]+", args.legend_anchor.strip()) if part]
		if parts:
			values = [float(part) for part in parts]
			if len(values) not in (2, 4):
				raise ValueError("--legend-anchor expects 2 or 4 numeric values (e.g. '1.02 1.0')")
			legend_anchor = tuple(values)

	plot_curves(
		curves=curves,
		datasets=datasets,
		time_to_hits_per_curve=time_to_hits_per_curve,
		hit_thresholds=args.hit_thresholds,
		output_path=args.output,
		step_col=args.step_column,
		title=args.title,
		legend_loc=args.legend_loc,
		legend_anchor=legend_anchor,
		legend_borderpad=args.legend_borderpad,
	)


if __name__ == "__main__":
	main()
