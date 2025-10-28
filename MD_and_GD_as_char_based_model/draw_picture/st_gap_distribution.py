from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import List, Sequence, Tuple, Union

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from math import ceil

from scipy.stats import gaussian_kde


DEFAULT_CSV_PATH = Path("./data/valid_data_smiles_qed_sa.csv")
DEFAULT_OUTPUT_PATH = Path("./data/st_gap_distribution.png")
DEFAULT_COLOR_RGB: Tuple[float, float, float] = (
	31 / 255,
	119 / 255,
	180 / 255,
)


def parse_rgb(value: str) -> Tuple[float, float, float]:
	value = value.strip()
	if value.startswith("#"):
		try:
			return mcolors.to_rgb(value)
		except ValueError as exc:
			raise argparse.ArgumentTypeError(
				f"Invalid hex color '{value}': {exc}"
			) from exc

	parts = value.replace(",", " ").split()
	if len(parts) != 3:
		raise argparse.ArgumentTypeError(
			"Color must be specified as three numbers or a hex string (e.g., '0.2 0.4 0.6' or '#1f77b4')."
		)
	try:
		components = [float(part) for part in parts]
	except ValueError as exc:
		raise argparse.ArgumentTypeError(
			f"Invalid RGB component in '{value}': {exc}"
		) from exc

	if any(component > 1.0 for component in components):
		components = [component / 255.0 for component in components]

	if not all(0.0 <= component <= 1.0 for component in components):
		raise argparse.ArgumentTypeError(
			"RGB components must be in the range [0, 1] (or [0, 255] before normalization)."
		)

	return tuple(components)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description=(
			"Compute the distribution of ST-gap values across molecules in a CSV "
			"and plot the resulting curve with a filled area underneath."
		)
	)
	parser.add_argument(
		"csv_path",
		type=Path,
		nargs="?",
		default=DEFAULT_CSV_PATH,
		help="Path to the CSV file containing ST-gap values.",
	)
	parser.add_argument(
		"--st-gap-column",
		default="st_gap_raw",
		help="Name of the column that holds ST-gap values (default: st_gap_raw).",
	)
	parser.add_argument(
		"--output",
		type=Path,
		default=DEFAULT_OUTPUT_PATH,
		help=(
			"Path to save the generated figure. Default is '<csv_dir>/st_gap_distribution.png'."
		),
	)
	parser.add_argument(
		"--color",
		type=parse_rgb,
		default=DEFAULT_COLOR_RGB,
		help=(
			"Primary RGB color for the line. Accepts three numbers between 0 and 1 (or 0-255) "
			"or a hex string like '#1f77b4'."
		),
	)
	parser.add_argument(
		"--fill-alpha",
		type=float,
		default=0.35,
		help="Transparency of the filled area under the curve (default: 0.35).",
	)
	parser.add_argument(
		"--lighten-amount",
		type=float,
		default=0.55,
		help="Fraction to lighten the fill color toward white (0-1, default: 0.55).",
	)
	parser.add_argument(
		"--jitter",
		type=float,
		default=0.25,
		help="Uniform jitter width added to ST-gap values before KDE smoothing (default: 0.25).",
	)
	parser.add_argument(
		"--gaussian-sigma",
		type=float,
		default=0.5,
		help=(
			"Standard deviation for deterministic Gaussian smoothing applied to discrete ST-gap values. "
			"Set to 0 to disable and rely purely on KDE (default: 0.5)."
		),
	)
	parser.add_argument(
		"--kde-bandwidth",
		type=float,
		help="Optional bandwidth factor passed to the KDE; defaults to Scott's rule when omitted.",
	)
	parser.add_argument(
		"--kde-grid",
		type=int,
		default=512,
		help="Number of points in the evaluation grid for KDE smoothing (default: 512).",
	)
	parser.add_argument(
		"--seed",
		type=int,
		default=42,
		help="Random seed used for jitter sampling (default: 42).",
	)
	parser.add_argument(
		"--show",
		action="store_true",
		help="Display the plot interactively after saving it.",
	)
	return parser.parse_args()


def lighten_color(
	color: Union[str, Tuple[float, float, float]],
	amount: float = 0.5,
) -> Tuple[float, float, float]:
	"""Lighten the given color by blending it with white."""

	amount = max(0.0, min(1.0, amount))
	rgb = mcolors.to_rgb(color)
	return tuple(channel + (1.0 - channel) * amount for channel in rgb)


def load_st_gap_values(csv_path: Path, st_gap_column: str) -> List[float]:
	"""Load ST-gap values from CSV file."""
	if not csv_path.exists():
		raise FileNotFoundError(f"CSV file not found: {csv_path}")

	df = pd.read_csv(csv_path)
	if st_gap_column not in df.columns:
		available = ", ".join(df.columns)
		raise KeyError(
			f"Column '{st_gap_column}' not found in CSV. Available columns: {available}"
		)

	# Extract numeric values and filter out NaN/invalid entries
	st_gap_values = df[st_gap_column].dropna()
	
	# Convert to numeric, coercing errors to NaN
	st_gap_values = pd.to_numeric(st_gap_values, errors='coerce').dropna()
	
	return st_gap_values.tolist()


def compute_kde_percentage_curve(
	values: Sequence[float],
	jitter: float,
	grid_points: int,
	seed: int,
	bandwidth: float | None,
	gaussian_sigma: float,
) -> Tuple[np.ndarray, np.ndarray]:
	"""Compute KDE-smoothed percentage curve for the distribution."""
	values_array = np.asarray(values, dtype=float)
	if values_array.size == 0:
		raise ValueError("Values array is empty; cannot compute KDE.")

	if values_array.size == 1:
		center = values_array[0]
		x_grid = np.linspace(center - 1.0, center + 1.0, max(grid_points, 32))
		y_grid = np.zeros_like(x_grid)
		closest_idx = np.abs(x_grid - center).argmin()
		y_grid[closest_idx] = 100.0
		return x_grid, y_grid

	if gaussian_sigma > 0:
		sigma = float(max(gaussian_sigma, 1e-3))
		unique_values, counts = np.unique(values_array, return_counts=True)
		weights = counts / counts.sum()
		padding = max(3 * sigma, 1.0)
		x_min = unique_values.min() - padding
		x_max = unique_values.max() + padding
		x_grid = np.linspace(x_min, x_max, max(grid_points, 128))
		diff = x_grid[:, None] - unique_values[None, :]
		exponent = -0.5 * (diff / sigma) ** 2
		normalizer = sigma * np.sqrt(2.0 * np.pi)
		density = (np.exp(exponent) / normalizer) @ weights
		density = np.clip(density, 0.0, None)
		y_grid = density * 100.0
		return x_grid, y_grid

	if gaussian_kde is None:
		raise ImportError(
			"scipy is required for KDE smoothing. Install it with 'pip install scipy' and rerun."
		)

	jitter = max(0.0, float(jitter))
	rng = np.random.default_rng(seed)
	if jitter > 0:
		perturbation = rng.uniform(-jitter, jitter, size=values_array.shape)
		jittered = values_array + perturbation
	else:
		jittered = values_array

	data_range = jittered.max() - jittered.min()
	if data_range <= 0:
		jittered = jittered + 0.1

	if bandwidth is None:
		kde = gaussian_kde(jittered)
	else:
		kde = gaussian_kde(jittered, bw_method=float(bandwidth))

	padding = max(1.0, data_range * 0.25)
	x_min = jittered.min() - padding
	x_max = jittered.max() + padding
	x_grid = np.linspace(x_min, x_max, max(grid_points, 64))
	density = kde.evaluate(x_grid)
	density = np.clip(density, 0.0, None)
	y_grid = density * 100.0
	return x_grid, y_grid


def plot_st_gap_distribution(
	st_gap_values: Sequence[float],
	output_path: Path,
	color: Union[str, Tuple[float, float, float]],
	fill_alpha: float,
	lighten_amount: float,
	jitter: float,
	grid_points: int,
	seed: int,
	bandwidth: float | None,
	gaussian_sigma: float,
) -> None:
	"""Plot the ST-gap distribution with KDE smoothing."""
	if not st_gap_values:
		raise ValueError("No valid ST-gap values were found; cannot plot distribution.")

	# Compute basic statistics
	values_array = np.array(st_gap_values, dtype=float)
	mean_value = float(np.mean(values_array))

	# Compute smoothed curve
	if len(values_array) > 1:
		x_smooth, y_smooth = compute_kde_percentage_curve(
			st_gap_values,
			jitter=jitter,
			grid_points=grid_points,
			seed=seed,
			bandwidth=bandwidth,
			gaussian_sigma=gaussian_sigma,
		)
	else:
		x_smooth = values_array
		y_smooth = np.array([100.0])

	fill_color = lighten_color(color, lighten_amount)

	# Create the plot
	fig, ax = plt.subplots(figsize=(10, 5))
	ax.plot(
		x_smooth,
		y_smooth,
		color=color,
		linewidth=2.5,
		label="ST-gap value percentage",
	)
	ax.fill_between(x_smooth, y_smooth, color=fill_color, alpha=fill_alpha)
	ax.axvline(
		mean_value,
		color=color,
		linestyle="--",
		linewidth=1.8,
		alpha=0.6,
	)

	ax.set_title("ST-Gap Distribution", fontsize=14)
	ax.set_xlabel("ST-Gap Value", fontsize=12)
	ax.legend()
	ax.grid(axis="x", alpha=0.3, linestyle="--", linewidth=0.6)
	
	# Set x-axis limits and ticks
	x_min, x_max = x_smooth.min(), x_smooth.max()
	data_range = x_max - x_min
	if data_range > 0:
		padding = data_range * 0.08
		ax.set_xlim(x_min - padding, x_max + padding)
		
		# Generate reasonable tick marks
		if data_range <= 10:
			# For small ranges, use more ticks
			num_ticks = min(8, int(data_range / 0.5) + 1)
		else:
			num_ticks = 7
		xticks = np.linspace(x_min, x_max, num_ticks)
		ax.set_xticks(xticks)
		ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.2f}'))
	
	# Hide y-axis
	ax.tick_params(axis="y", which="both", left=False, right=False, labelleft=False)
	for spine in ("left", "right", "top"):
		ax.spines[spine].set_visible(False)
	
	# Set y-axis limits
	max_y = float(np.max(y_smooth)) if len(y_smooth) else 0.0
	if max_y > 0:
		ax.set_ylim(0, max_y * 1.05)
	else:
		ax.set_ylim(0, 1)
	
	ax.spines["bottom"].set_linewidth(1.2)
	
	# Add mean annotation
	y_annotation = 0.35
	ax.annotate(
		f"Mean: {mean_value:.2f}",
		xy=(mean_value, y_annotation),
		xycoords=("data", "axes fraction"),
		xytext=(0, -12),
		textcoords="offset points",
		ha="center",
		color=color,
		fontsize=8,
		fontweight="bold",
		alpha=0.6,
	)

	output_path.parent.mkdir(parents=True, exist_ok=True)
	fig.tight_layout()
	fig.savefig(output_path, dpi=300)


def main() -> None:
	args = parse_args()

	csv_path = args.csv_path
	st_gap_values = load_st_gap_values(csv_path, args.st_gap_column)

	if not st_gap_values:
		print("Error: No valid ST-gap values found in the specified column.")
		return

	print(f"Loaded {len(st_gap_values)} ST-gap values.")

	# Determine output path
	if args.output == DEFAULT_OUTPUT_PATH:
		if csv_path != DEFAULT_CSV_PATH:
			output_path = csv_path.with_name("st_gap_distribution.png")
		else:
			output_path = DEFAULT_OUTPUT_PATH
	elif args.output is None:
		output_path = csv_path.with_name("st_gap_distribution.png")
	elif args.output.is_dir():
		output_path = args.output / "st_gap_distribution.png"
	else:
		output_path = args.output

	plot_st_gap_distribution(
		st_gap_values,
		output_path,
		color=args.color,
		fill_alpha=max(0.0, min(1.0, args.fill_alpha)),
		lighten_amount=args.lighten_amount,
		jitter=max(0.0, args.jitter),
		grid_points=max(32, args.kde_grid),
		seed=args.seed,
		bandwidth=args.kde_bandwidth,
		gaussian_sigma=max(0.0, args.gaussian_sigma),
	)

	print(f"Saved figure to: {output_path.resolve()}")

	if args.show:
		plt.show()
	else:
		plt.close("all")


if __name__ == "__main__":
	main()
