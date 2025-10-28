"""Generate a t-SNE embedding for molecules from configurable CSV files."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from matplotlib.colors import to_rgb

# ---------------------------------------------------------------------------
# Configure default dataset options here for quick experimentation.
# Provide matching lists for paths, labels, and colors. CLI arguments can
# override any of them at runtime (see --csv/--label/--color).
# ---------------------------------------------------------------------------
DEFAULT_DATASET_PATHS: list[Path | str] = ["./data_preparation/prediction_model/dft/Photosensitizers_DA.csv","./data_preparation/prediction_model/dft/Photosensitizers_DAD.csv"]
DEFAULT_DATASET_LABELS: list[str] = ["DA","DAD"]
DEFAULT_DATASET_COLORS: list[tuple[float, float, float]] = []
DEFAULT_DATASET_MAX_SAMPLES: list[int | None] = []

# Additional colors will be drawn cyclically from this palette if fewer colors
# are provided than datasets.
DEFAULT_COLOR_CYCLE: tuple[tuple[float, float, float], ...] = (
	(76 / 255, 114 / 255, 176 / 255),
	(221 / 255, 132 / 255, 82 / 255),
	(85 / 255, 168 / 255, 104 / 255),
	(196 / 255, 78 / 255, 82 / 255),
	(129 / 255, 114 / 255, 179 / 255),
	(147 / 255, 120 / 255, 96 / 255),
	(218 / 255, 139 / 255, 195 / 255),
	(140 / 255, 140 / 255, 140 / 255),
	(204 / 255, 185 / 255, 116 / 255),
	(100 / 255, 181 / 255, 205 / 255),
)


@dataclass(frozen=True)
class DatasetConfig:
	label: str
	path: Path
	color: tuple[float, float, float]
	max_samples: int | None


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Plot a t-SNE embedding for molecules from CSV files",
	)
	parser.add_argument(
		"--csv",
		dest="csvs",
		type=Path,
		action="append",
		help="CSV file containing a 'SMILES' column (pass multiple times for more datasets)",
		default=None,
	)
	parser.add_argument(
		"--label",
		dest="labels",
		action="append",
		help="Label for the dataset in the same order as --csv",
		default=None,
	)
	parser.add_argument(
		"--color",
		dest="colors",
		action="append",
		help="Matplotlib-compatible color for each dataset (hex/name); order must match --csv",
		default=None,
	)
	parser.add_argument(
		"--output",
		type=Path,
		default=Path(__file__).resolve().parent / "tsne_plot.png",
		help="Path to save the generated PNG figure",
	)
	parser.add_argument("--radius", type=int, default=2, help="Radius for the Morgan fingerprint")
	parser.add_argument("--n-bits", type=int, default=2048, help="Number of bits for the Morgan fingerprint")
	parser.add_argument("--pca-dim", type=int, default=50, help="Target dimensionality after PCA reduction")
	parser.add_argument("--perplexity", type=float, default=50.0, help="Perplexity parameter for t-SNE")
	parser.add_argument("--seed", type=int, default=13, help="Random seed for PCA, t-SNE, and sampling")
	parser.add_argument("--point-size", type=float, default=18.0, help="Marker size for scatter plot points")
	parser.add_argument(
		"--max-samples",
		dest="max_samples",
		type=int,
		action="append",
		help=(
			"Optional maximum number of molecules per dataset. Provide a single value to apply "
			"to every dataset or repeat the flag so counts align with --csv order."
		),
		default=None,
	)
	return parser.parse_args()


def resolve_dataset_configs(args: argparse.Namespace) -> list[DatasetConfig]:
	csv_inputs = args.csvs if args.csvs else DEFAULT_DATASET_PATHS
	label_inputs = args.labels if args.labels else DEFAULT_DATASET_LABELS
	color_inputs = args.colors if args.colors else DEFAULT_DATASET_COLORS
	max_samples_inputs = args.max_samples if args.max_samples is not None else DEFAULT_DATASET_MAX_SAMPLES

	if not csv_inputs:
		raise ValueError(
			"No dataset CSVs supplied. Use --csv or populate DEFAULT_DATASET_PATHS in the script.",
		)

	if label_inputs and len(label_inputs) != len(csv_inputs):
		raise ValueError("Number of labels must match number of CSV files")

	if color_inputs and len(color_inputs) != len(csv_inputs):
		raise ValueError("Number of colors must match number of CSV files")

	if max_samples_inputs and len(max_samples_inputs) not in (1, len(csv_inputs)):
		raise ValueError("Number of --max-samples values must be 1 or match number of CSV files")

	max_samples_list: list[int | None]
	if not max_samples_inputs:
		max_samples_list = [None] * len(csv_inputs)
	elif len(max_samples_inputs) == 1:
		max_samples_list = [max_samples_inputs[0]] * len(csv_inputs)
	else:
		max_samples_list = list(max_samples_inputs)

	for value in max_samples_list:
		if value is not None and value <= 0:
			raise ValueError("--max-samples values must be positive integers")

	labels = (
		list(label_inputs)
		if label_inputs
		else [f"Dataset {idx + 1}" for idx in range(len(csv_inputs))]
	)

	base_colors = list(color_inputs) if color_inputs else list(DEFAULT_COLOR_CYCLE)
	if not base_colors:
		raise ValueError(
			"No colors provided. Supply --color options or fill DEFAULT_DATASET_COLORS/DEFAULT_COLOR_CYCLE.",
		)

	if len(base_colors) < len(csv_inputs):
		repeats = math.ceil(len(csv_inputs) / len(base_colors))
		base_colors = (base_colors * repeats)[: len(csv_inputs)]
	else:
		base_colors = base_colors[: len(csv_inputs)]

	configs: list[DatasetConfig] = []
	for label, csv_path, color, max_count in zip(labels, csv_inputs, base_colors, max_samples_list):
		configs.append(
			DatasetConfig(
				label=label,
				path=Path(csv_path),
				color=to_rgb(color),
				max_samples=max_count,
			),
		)

	return configs


def load_smiles_column(path: Path) -> pd.Series:
	df = pd.read_csv(path)
	if "SMILES" not in df.columns:
		raise KeyError(f"CSV file '{path}' must contain a 'SMILES' column")
	return df["SMILES"].dropna().astype(str).str.strip()


def smiles_to_fingerprints(smiles: Sequence[str], radius: int, n_bits: int) -> tuple[np.ndarray, list[str]]:
	generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
	bit_vector = np.zeros((n_bits,), dtype=int)
	rows: list[np.ndarray] = []
	valid_smiles: list[str] = []
	for smi in smiles:
		mol = Chem.MolFromSmiles(smi)
		if mol is None:
			continue
		fp = generator.GetFingerprint(mol)
		bit_vector.fill(0)
		DataStructs.ConvertToNumpyArray(fp, bit_vector)
		rows.append(bit_vector.astype(np.float32))
		valid_smiles.append(smi)
	if not rows:
		return np.zeros((0, n_bits), dtype=np.float32), []
	return np.stack(rows, axis=0), valid_smiles


def subsample_indices(total: int, max_samples: int | None, seed: int) -> np.ndarray:
	if max_samples is None or total <= max_samples:
		return np.arange(total)
	rng = np.random.default_rng(seed)
	return np.sort(rng.choice(total, size=max_samples, replace=False))


def apply_pca(features: np.ndarray, components: int, seed: int) -> np.ndarray:
	if features.shape[0] <= 1:
		return features.astype(np.float32)
	n_components = min(components, features.shape[1], features.shape[0] - 1)
	if n_components < 1:
		return features.astype(np.float32)
	pca = PCA(n_components=n_components, random_state=seed)
	return pca.fit_transform(features)


def generalized_tanimoto_distance(features: np.ndarray) -> np.ndarray:
	if features.shape[0] == 0:
		return np.zeros((0, 0), dtype=np.float32)
	features = features.astype(np.float32, copy=False)
	dot_products = features @ features.T
	squared_norms = np.sum(features * features, axis=1)
	denominator = squared_norms[:, None] + squared_norms[None, :] - dot_products
	similarity = np.divide(
		dot_products,
		denominator,
		out=np.ones_like(dot_products, dtype=np.float32),
		where=denominator > 1e-12,
	)
	distance = 1.0 - similarity
	np.fill_diagonal(distance, 0.0)
	return np.clip(distance, 0.0, 1.0)


def run_tsne(distance_matrix: np.ndarray, perplexity: float, seed: int) -> np.ndarray:
	if distance_matrix.shape[0] == 0:
		return np.zeros((0, 2), dtype=np.float32)
	effective_perplexity = min(perplexity, max(5.0, (distance_matrix.shape[0] - 1) / 3.0))
	tsne = TSNE(
		n_components=2,
		metric="precomputed",
		perplexity=effective_perplexity,
		random_state=seed,
		init="random",
	)
	return tsne.fit_transform(distance_matrix)


def plot_embedding(
	coords: np.ndarray,
	source_flags: np.ndarray,
	configs: Sequence[DatasetConfig],
	output_path: Path,
	point_size: float,
) -> None:
	if coords.shape[0] == 0:
		raise ValueError("No coordinates to plot – check input data")
	output_path.parent.mkdir(parents=True, exist_ok=True)
	fig, ax = plt.subplots(figsize=(8, 6), dpi=120)

	for idx, cfg in enumerate(configs):
		mask = source_flags == idx
		if not np.any(mask):
			continue
		ax.scatter(
			coords[mask, 0],
			coords[mask, 1],
			s=point_size,
			c=[cfg.color],
			label=cfg.label,
			alpha=0.78,
			edgecolors="white",
			linewidths=0.3,
		)

	ax.set_xlabel("t-SNE 1")
	ax.set_ylabel("t-SNE 2")
	ax.set_title("Molecular t-SNE (Morgan + PCA + Jaccard)")
	ax.legend(frameon=False)
	ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.3)
	fig.tight_layout()
	fig.savefig(output_path, dpi=300)
	plt.close(fig)


def main() -> None:
	args = parse_args()
	configs = resolve_dataset_configs(args)

	feature_blocks: list[np.ndarray] = []
	source_flags: list[int] = []
	active_configs: list[DatasetConfig] = []
	for cfg_index, cfg in enumerate(configs):
		smiles_series = load_smiles_column(cfg.path)
		sample_limit = cfg.max_samples
		if sample_limit is not None and smiles_series.size > sample_limit:
			indices = subsample_indices(smiles_series.size, sample_limit, args.seed + cfg_index)
			smiles_series = smiles_series.iloc[indices].reset_index(drop=True)
			print(
				f"[info] Subsampled {smiles_series.size} molecules from '{cfg.path}' "
				f"(requested max {sample_limit})",
			)
		smiles_list = smiles_series.tolist()
		features, valid_smiles = smiles_to_fingerprints(smiles_list, args.radius, args.n_bits)
		dropped = smiles_series.size - len(valid_smiles)
		if dropped:
			print(f"[warning] Dropped {dropped} invalid SMILES in '{cfg.path}'")
		if features.size == 0:
			continue
		current_index = len(active_configs)
		feature_blocks.append(features)
		active_configs.append(cfg)
		source_flags.extend([current_index] * features.shape[0])

	if not feature_blocks:
		raise ValueError("No valid molecules found in either CSV file")

	all_features = np.concatenate(feature_blocks, axis=0)
	source_flags_array = np.asarray(source_flags, dtype=int)

	reduced = apply_pca(all_features, args.pca_dim, args.seed)
	distance_matrix = generalized_tanimoto_distance(reduced)
	tsne_coords = run_tsne(distance_matrix, args.perplexity, args.seed)

	plot_embedding(
		tsne_coords,
		source_flags_array,
		active_configs,
		args.output,
		args.point_size,
	)

	print(f"Saved t-SNE plot to: {args.output}")


if __name__ == "__main__":
	main()

