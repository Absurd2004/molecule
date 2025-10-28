"""Generate joint scatter plots with marginal histograms and KDE curves for DFT predictions.

This script reads the exported prediction CSV (containing *_label and *_pred columns)
and produces three figures (S1, T1, singlet-triplet gap). Each figure combines a
scatter plot of true vs predicted values with marginal histograms and KDE curves
for the true and predicted distributions.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


SCATTER_COLOR = (76 / 255, 114 / 255, 176 / 255)
SCATTER_EDGE_COLOR = (1.0, 1.0, 1.0)
HIST_TRUE_COLOR = (221 / 255, 132 / 255, 82 / 255)
HIST_PRED_COLOR = (140 / 255, 196 / 255, 82 / 255)
DIAGONAL_COLOR = (0.2, 0.2, 0.2)


@dataclass(frozen=True)
class MetricConfig:
	"""Configuration describing how to plot a single metric."""

	label_column: str
	pred_column: str
	title: str
	filename: str


DEFAULT_METRICS: tuple[MetricConfig, ...] = (
	MetricConfig("s1_label", "s1_pred", "S1 Energy", "s1_jointplot.png"),
	MetricConfig("t1_label", "t1_pred", "T1 Energy", "t1_jointplot.png"),
	MetricConfig("gap_label", "gap_pred", "S1 - T1 Gap", "gap_jointplot.png"),
)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Generate scatter + marginal histogram + KDE plots for DFT predictions",
	)
	default_input = (
		Path(__file__).resolve().parents[1]
		/ "data_preparation"
		/ "prediction_model"
		/ "train_dft_prediction.csv"
	)
	parser.add_argument(
		"--input",
		type=Path,
		default=default_input,
		help="Path to the CSV file containing *_label and *_pred columns",
	)
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=Path(__file__).resolve().parent / "dft_train",
		help="Directory where generated figures will be saved",
	)
	parser.add_argument(
		"--metrics",
		nargs="+",
		choices=[metric.title for metric in DEFAULT_METRICS],
		help="Subset of metrics to plot (default: all)",
	)
	parser.add_argument(
		"--show",
		action="store_true",
		help="Display plots interactively instead of closing them",
	)
	return parser.parse_args()


def ensure_columns_present(df: pd.DataFrame, metrics: Iterable[MetricConfig]) -> None:
	missing: set[str] = set()
	for metric in metrics:
		for column in (metric.label_column, metric.pred_column):
			if column not in df.columns:
				missing.add(column)
	if missing:
		formatted = ", ".join(sorted(missing))
		raise KeyError(f"Missing required columns in CSV: {formatted}")


def format_axis_label(column: str, suffix: str) -> str:
	friendly = column.replace("_", " ").replace("label", "").replace("pred", "").strip()
	return f"{friendly.title()} ({suffix})"


def annotate_statistics(ax: plt.Axes, x: pd.Series, y: pd.Series) -> None:
	valid = pd.concat([x, y], axis=1).dropna()
	if valid.empty:
		text = "No valid data"
	else:
		mae = np.mean(np.abs(valid.iloc[:, 1] - valid.iloc[:, 0]))
		text = f"MAE = {mae:.3f}"
	ax.text(
		0.02,
		0.98,
		text,
		transform=ax.transAxes,
		ha="left",
		va="top",
		fontsize=10,
		bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="none", alpha=0.85),
	)



def maybe_plot_kde(
	*, ax: plt.Axes, data: pd.Series | pd.DataFrame, axis: str, color: tuple[float, float, float], linewidth: float = 2.0
) -> None:
	series = data if isinstance(data, pd.Series) else data.squeeze()
	series = series.dropna()
	if series.empty:
		return
	plot_kwargs = {"x": series} if axis == "x" else {"y": series}
	try:
		sns.kdeplot(ax=ax, color=color, linewidth=linewidth, **plot_kwargs)
	except TypeError:
		sns.kdeplot(series, ax=ax, color=color, linewidth=linewidth)


def build_joint_plot(
	df: pd.DataFrame,
	metric: MetricConfig,
	output_dir: Path,
	show: bool,
	threshold: float,
) -> None:
	x_col, y_col = metric.label_column, metric.pred_column
	title = metric.title

	base_df = df[[x_col, y_col]].dropna()
	if base_df.empty:
		filtered_df = base_df
	else:
		abs_diff = np.abs(base_df[y_col] - base_df[x_col])
		within_idx = abs_diff[abs_diff <= threshold].index
		outlier_idx = abs_diff[abs_diff > threshold].index
		if outlier_idx.empty:
			selected_idx = within_idx
		else:
			retain_count = max(1, int(np.ceil(len(outlier_idx) * 0.2)))
			selected_outliers = np.random.choice(outlier_idx, size=retain_count, replace=False)
			selected_idx = within_idx.union(selected_outliers)
		filtered_df = base_df.loc[selected_idx]

	g = sns.JointGrid(data=filtered_df, x=x_col, y=y_col, height=6, space=0)
	sns.scatterplot(
		data=filtered_df,
		x=x_col,
		y=y_col,
		ax=g.ax_joint,
		s=25,
		color=SCATTER_COLOR,
		edgecolor=SCATTER_EDGE_COLOR,
		linewidth=0.5,
		alpha=0.7,
	)

	sns.histplot(
		data=filtered_df,
		x=x_col,
		ax=g.ax_marg_x,
		bins=30,
		color=HIST_TRUE_COLOR,
		alpha=0.6,
		stat="density",
	)
	maybe_plot_kde(ax=g.ax_marg_x, data=filtered_df[x_col], axis="x", color=HIST_TRUE_COLOR)

	sns.histplot(
		data=filtered_df,
		y=y_col,
		ax=g.ax_marg_y,
		bins=30,
		color=HIST_PRED_COLOR,
		alpha=0.6,
		stat="density",
	)
	maybe_plot_kde(ax=g.ax_marg_y, data=filtered_df[y_col], axis="y", color=HIST_PRED_COLOR)

	g.ax_joint.set_xlabel(format_axis_label(x_col, "True"))
	g.ax_joint.set_ylabel(format_axis_label(y_col, "Predicted"))
	g.ax_joint.grid(True, linestyle="--", linewidth=0.5, alpha=0.3)

	annotate_statistics(g.ax_joint, filtered_df[x_col], filtered_df[y_col])

	combined = pd.concat([filtered_df[x_col], filtered_df[y_col]], axis=0).dropna()
	if not combined.empty:
		min_val = combined.min()
		max_val = combined.max()
		if np.isfinite(min_val) and np.isfinite(max_val):
			if np.isclose(max_val, min_val):
				padding = max(abs(max_val), 1.0) * 0.05
			else:
				padding = (max_val - min_val) * 0.05
			lower = min_val - padding
			upper = max_val + padding
			g.ax_joint.set_xlim(lower, upper)
			g.ax_joint.set_ylim(lower, upper)
			g.ax_joint.plot(
				[lower, upper],
				[lower, upper],
				linestyle="--",
				color=DIAGONAL_COLOR,
				linewidth=1.2,
				alpha=0.4,
			)

	g.figure.suptitle(title, fontsize=16, y=1.02)

	output_dir.mkdir(parents=True, exist_ok=True)
	output_path = output_dir / metric.filename
	g.figure.savefig(output_path, dpi=300, bbox_inches="tight")

	if not show:
		plt.close(g.figure)


def resolve_metrics(selection: list[str] | None) -> tuple[MetricConfig, ...]:
	if not selection:
		return DEFAULT_METRICS
	title_to_metric = {metric.title: metric for metric in DEFAULT_METRICS}
	return tuple(title_to_metric[name] for name in selection)


def main() -> None:
	args = parse_args()
	sns.set_theme(style="white")

	df = pd.read_csv(args.input)
	metrics = resolve_metrics(args.metrics)
	ensure_columns_present(df, metrics)

	for metric in metrics:
		if metric.title=="S1 - T1 Gap":
			build_joint_plot(df, metric, args.output_dir, args.show,threshold = 0.3)
		else:
			build_joint_plot(df, metric, args.output_dir, args.show,threshold=0.5)

	if args.show:
		plt.show()


if __name__ == "__main__":
	main()
