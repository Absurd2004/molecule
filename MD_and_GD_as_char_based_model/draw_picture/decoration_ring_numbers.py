from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple, Union

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors

from scipy.stats import gaussian_kde



DEFAULT_TSV_PATH = Path("./data/valid/sliced_smiles_recap.tsv")
DEFAULT_OUTPUT_PATH = Path("./data/valid/sliced_smiles_recap.png")
DEFAULT_COLOR_RGB: Tuple[float, float, float] = (
	31 / 255,
	119 / 255,
	180 / 255,
)


def parse_rgb(value: str) -> Tuple[float, float, float]:
	parts = value.replace(",", " ").split()
	if len(parts) != 3:
		raise argparse.ArgumentTypeError(
			"Color must be three numbers separated by commas or spaces (e.g., '0.2 0.4 0.6')."
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


def column_arg(value: str) -> Union[int, str]:
	value = value.strip()
	return int(value) if value.isdigit() else value


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description=(
			"Compute the ring-count distribution of decorations extracted from a TSV "
			"file and plot the smoothed percentage curve with a filled area underneath."
		)
	)
	parser.add_argument(
		"tsv_path",
		type=Path,
		nargs="?",
		default=DEFAULT_TSV_PATH,
		help="Path to the TSV file containing decoration entries.",
	)
	parser.add_argument(
		"--decor-column",
		type=column_arg,
		default=1,
		help=(
			"Column name or zero-based index that holds decoration strings (default: 1)."
		),
	)
	parser.add_argument(
		"--decor-sep",
		default="|",
		help="Separator used between decorations inside the column (default: '|').",
	)
	parser.add_argument(
		"--output",
		type=Path,
		default=DEFAULT_OUTPUT_PATH,
		help=(
			"Path to save the generated figure. Default is '<csv_dir>/ring_number_distribution.png'."
		),
	)
	parser.add_argument(
		"--color",
		type=parse_rgb,
		default=DEFAULT_COLOR_RGB,
		help=(
			"Primary RGB color for the line as three numbers between 0 and 1 (or 0-255). "
			"Example: '--color 0.2 0.4 0.6' or '--color 51,160,44'."
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
		help="Uniform jitter width added to ring counts before KDE smoothing (default: 0.25).",
	)
	parser.add_argument(
		"--gaussian-sigma",
		type=float,
		default=0.3,
		help=(
			"Standard deviation for deterministic Gaussian smoothing applied to discrete ring counts. "
			"Set to 0 to disable and rely purely on KDE (default: 0.6)."
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


def lighten_color(color: str, amount: float = 0.5) -> Tuple[float, float, float]:
	"""Lighten the given color by blending it with white.

	Args:
		color: Any Matplotlib-valid color specification (string or RGB tuple).
		amount: Value between 0 and 1 where 0 returns the original color and 1 is white.

	Returns:
		Lightened RGB color tuple.
	"""

	amount = max(0.0, min(1.0, amount))
	rgb = mcolors.to_rgb(color)
	return tuple(channel + (1.0 - channel) * amount for channel in rgb)


def load_decorations(
	tsv_path: Path,
	decor_column: Union[int, str],
	separator: str,
) -> Sequence[str]:
	if not tsv_path.exists():
		raise FileNotFoundError(f"TSV file not found: {tsv_path}")

	df = pd.read_csv(tsv_path, sep="\t", dtype=str)

	if isinstance(decor_column, int):
		if decor_column < 0 or decor_column >= df.shape[1]:
			raise IndexError(
				f"Column index {decor_column} is out of bounds for file with {df.shape[1]} columns."
			)
		series = df.iloc[:, decor_column]
	else:
		if decor_column not in df.columns:
			available = ", ".join(df.columns.astype(str))
			raise KeyError(
				f"Column '{decor_column}' not found in TSV. Available columns: {available}"
			)
		series = df[decor_column]

	decorations: List[str] = []
	for cell in series.dropna():
		parts = [part.strip() for part in str(cell).split(separator)]
		for part in parts[1:]:
			if part:
				decorations.append(part)

	return decorations


def compute_ring_counts(smiles_list: Iterable[str]) -> Tuple[List[int], List[str]]:
	RDLogger.DisableLog("rdApp.*")

	ring_counts: List[int] = []
	failed_smiles: List[str] = []

	for smiles in smiles_list:
		mol = Chem.MolFromSmiles(smiles)
		if mol is None:
			failed_smiles.append(smiles)
			continue
		ring_counts.append(int(rdMolDescriptors.CalcNumRings(mol)))

	return ring_counts, failed_smiles


def compute_kde_percentage_curve(
	ring_counts: Sequence[int],
	jitter: float,
	grid_points: int,
	seed: int,
	bandwidth: float | None,
	gaussian_sigma: float,
) -> Tuple[np.ndarray, np.ndarray]:
	values = np.asarray(ring_counts, dtype=float)
	if values.size == 0:
		raise ValueError("Ring counts array is empty; cannot compute KDE.")

	if values.size == 1:
		center = values[0]
		x_grid = np.linspace(max(center - 1.0, 0.0), center + 1.0, max(grid_points, 32))
		y_grid = np.zeros_like(x_grid)
		closest_idx = np.abs(x_grid - center).argmin()
		y_grid[closest_idx] = 100.0
		return x_grid, y_grid

	if gaussian_sigma > 0:
		sigma = float(max(gaussian_sigma, 1e-3))
		unique_values, counts = np.unique(values, return_counts=True)
		weights = counts / counts.sum()
		padding = max(3 * sigma, 1.0)
		x_min = max(unique_values.min() - padding, 0.0)
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
		perturbation = rng.uniform(-jitter, jitter, size=values.shape)
		jittered = np.clip(values + perturbation, 0.0, None)
	else:
		jittered = values

	data_range = jittered.max() - jittered.min()
	if data_range <= 0:
		jittered = jittered + 0.1

	if bandwidth is None:
		kde = gaussian_kde(jittered)
	else:
		kde = gaussian_kde(jittered, bw_method=float(bandwidth))

	padding = max(1.0, data_range * 0.25)
	x_min = max(jittered.min() - padding, 0.0)
	x_max = jittered.max() + padding
	x_grid = np.linspace(x_min, x_max, max(grid_points, 64))
	density = kde.evaluate(x_grid)
	density = np.clip(density, 0.0, None)
	y_grid = density * 100.0
	return x_grid, y_grid


def plot_ring_distribution(
	ring_counts: Sequence[int],
	output_path: Path,
	color: str,
	fill_alpha: float,
	lighten_amount: float,
	jitter: float,
	grid_points: int,
	seed: int,
	bandwidth: float | None,
	gaussian_sigma: float,
) -> None:
	if not ring_counts:
		raise ValueError("No valid ring counts were computed; cannot plot distribution.")

	counts = Counter(ring_counts)
	xs = sorted(counts.keys())
	total = sum(counts.values())
	percentages = [counts[x] * 100.0 / total for x in xs]
	x_vals = np.array(xs, dtype=float)
	y_vals = np.array(percentages, dtype=float)
	mean_ring = float(np.mean(ring_counts))

	if len(x_vals) > 1:
		x_smooth, y_smooth = compute_kde_percentage_curve(
			ring_counts,
			jitter=jitter,
			grid_points=grid_points,
			seed=seed,
			bandwidth=bandwidth,
			gaussian_sigma=gaussian_sigma,
		)
	else:
		x_smooth = x_vals
		y_smooth = y_vals

	fill_color = lighten_color(color, lighten_amount)

	fig, ax = plt.subplots(figsize=(10, 5))
	ax.plot(
		x_smooth,
		y_smooth,
		color=color,
		linewidth=2.5,
		label="Decoration ring count percentage",
	)
	ax.fill_between(x_smooth, y_smooth, color=fill_color, alpha=fill_alpha)
	ax.axvline(
		mean_ring,
		color=color,
		linestyle="--",
		linewidth=1.8,
		alpha=0.6,
	)

	ax.set_title("Decoration Ring Count Distribution", fontsize=14)
	ax.set_xlabel("Number of Rings", fontsize=12)
	ax.legend()
	ax.grid(axis="x", alpha=0.3, linestyle="--", linewidth=0.6)
	if xs:
		ax.set_xticks(xs)
		padding = max(0.75, (xs[-1] - xs[0]) * 0.08)
		ax.set_xlim(xs[0] - padding, xs[-1] + padding)
	ax.tick_params(axis="y", which="both", left=False, right=False, labelleft=False)
	for spine in ("left", "right", "top"):
		ax.spines[spine].set_visible(False)
	max_y = float(np.max(y_smooth)) if len(y_smooth) else 0.0
	if max_y > 0:
		ax.set_ylim(0, max_y * 1.05)
	else:
		ax.set_ylim(0, 1)
	ax.spines["bottom"].set_linewidth(1.2)
	y_annotation = 0.35
	ax.annotate(
		f"Mean: {mean_ring:.2f}",
		xy=(mean_ring, y_annotation),
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

	tsv_path = args.tsv_path
	decorations = load_decorations(tsv_path, args.decor_column, args.decor_sep)
	ring_counts, failed_decorations = compute_ring_counts(decorations)

	if failed_decorations:
		print(
			f"Warning: {len(failed_decorations)} decoration SMILES failed to parse and were excluded "
			"from the distribution."
		)

	if args.output == DEFAULT_OUTPUT_PATH:
		if tsv_path != DEFAULT_TSV_PATH:
			output_path = tsv_path.with_name("decoration_ring_distribution.png")
		else:
			output_path = DEFAULT_OUTPUT_PATH
	elif args.output is None:
		output_path = tsv_path.with_name("decoration_ring_distribution.png")
	elif args.output.is_dir():
		output_path = args.output / "decoration_ring_distribution.png"
	else:
		output_path = args.output

	plot_ring_distribution(
		ring_counts,
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
