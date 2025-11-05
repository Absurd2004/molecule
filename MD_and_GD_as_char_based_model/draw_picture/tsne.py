"""Generate a t-SNE embedding for molecules from configurable CSV files."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # type: ignore[import]
from rdkit import Chem, DataStructs  # type: ignore[import]
from rdkit.Chem import rdFingerprintGenerator  # type: ignore[import]
from sklearn.decomposition import PCA  # type: ignore[import]
from sklearn.manifold import TSNE  # type: ignore[import]

from matplotlib.colors import to_rgb

# ---------------------------------------------------------------------------
# Configure default dataset options here for quick experimentation.
# Provide matching lists for paths, labels, and colors. CLI arguments can
# override any of them at runtime (see --csv/--label/--color).
# ---------------------------------------------------------------------------
DEFAULT_CONFIG_PATH = Path("./configs/tsne.json")

DEFAULT_DATASET_PATHS: list[Path | str] = ["./data_preparation/prediction_model/dft/Photosensitizers_DA.csv","./data_preparation/prediction_model/dft/Photosensitizers_DAD.csv"]
DEFAULT_DATASET_LABELS: list[str] = ["DA","DAD"]
DEFAULT_DATASET_COLORS: list[tuple[float, float, float]] = []
DEFAULT_DATASET_MAX_SAMPLES: list[int | None] = []
DEFAULT_DATASET_ALPHAS: list[float | None] = []

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

DEFAULT_ALPHA: float = 0.78


@dataclass(frozen=True)
class DatasetConfig:
	label: str
	path: Path
	color: tuple[float, float, float]
	max_samples: int | None
	alpha: float


def _normalise_color(value: object) -> tuple[float, float, float]:
	if isinstance(value, str):
		return to_rgb(value)
	if isinstance(value, Sequence) and len(value) == 3:
		rgb = tuple(float(channel) for channel in value)
		if any(channel > 1.0 for channel in rgb):
			rgb = tuple(channel / 255.0 for channel in rgb)
		return rgb  # type: ignore[return-value]
	raise ValueError(
		"Color values must be a Matplotlib-compatible string or a sequence of three numbers",
	)


def _ensure_color_cycle(colors: Sequence[tuple[float, float, float]], count: int) -> list[tuple[float, float, float]]:
	if not colors:
		base = list(DEFAULT_COLOR_CYCLE)
	else:
		base = list(colors)
	if count <= len(base):
		return base[:count]
	repeats = (count + len(base) - 1) // len(base)
	tiled: list[tuple[float, float, float]] = list(base) * repeats
	return tiled[:count]


def _ensure_alpha_list(alphas: Sequence[float | None], count: int) -> list[float]:
	if count <= 0:
		return []
	if not alphas:
		return [DEFAULT_ALPHA] * count
	processed = [DEFAULT_ALPHA if value is None else float(value) for value in alphas]
	for alpha in processed:
		if not (0.0 <= alpha <= 1.0):
			raise ValueError("Alpha values must be between 0 and 1")
	if len(processed) == 1:
		return processed * count
	if len(processed) < count:
		repeats = (count + len(processed) - 1) // len(processed)
		tiled = processed * repeats
		return tiled[:count]
	return processed[:count]


def load_config(path: Optional[Path]) -> Optional[dict]:
	if path is None:
		return None
	if not path.exists():
		return None
	text = path.read_text(encoding="utf-8").strip()
	if not text:
		return None
	return json.loads(text)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Plot a t-SNE embedding for molecules from CSV files",
	)
	parser.add_argument(
		"--config",
		type=Path,
		default=DEFAULT_CONFIG_PATH,
		help="Path to JSON config file describing datasets (default: ./configs/tsne.json)",
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
		"--alpha",
		dest="alphas",
		action="append",
		type=float,
		help="Transparency for each dataset (0-1]; order must match --csv",
		default=None,
	)
	parser.add_argument(
		"--output",
		type=Path,
		default=Path(__file__).resolve().parent / "tsne_plot.png",
		help="Path to save the generated PNG figure",
	)
	parser.add_argument(
		"--figsize",
		type=float,
		nargs=2,
		default=(10.0, 6.0),
		metavar=("WIDTH", "HEIGHT"),
		help="Figure size in inches as WIDTH HEIGHT (default: 10 6)",
	)
	parser.add_argument("--radius", type=int, default=2, help="Radius for the Morgan fingerprint")
	parser.add_argument("--n-bits", type=int, default=2048, help="Number of bits for the Morgan fingerprint")
	parser.add_argument("--pca-dim", type=int, default=30, help="Target dimensionality after PCA reduction")
	parser.add_argument("--perplexity", type=float, default=50.0, help="Perplexity parameter for t-SNE")
	parser.add_argument("--seed", type=int, default=13, help="Random seed for PCA, t-SNE, and sampling")
	parser.add_argument("--point-size", type=float, default=11.0, help="Marker size for scatter plot points")
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


def _configs_from_defaults() -> list[DatasetConfig]:
	default_colors = _ensure_color_cycle(DEFAULT_DATASET_COLORS, len(DEFAULT_DATASET_PATHS))
	default_alphas = _ensure_alpha_list(DEFAULT_DATASET_ALPHAS, len(DEFAULT_DATASET_PATHS))
	default_max_samples = (
		list(DEFAULT_DATASET_MAX_SAMPLES)
		if DEFAULT_DATASET_MAX_SAMPLES
		else [None] * len(DEFAULT_DATASET_PATHS)
	)
	configs: list[DatasetConfig] = []
	for label, csv_path, color, max_samples, alpha in zip(
		DEFAULT_DATASET_LABELS,
		DEFAULT_DATASET_PATHS,
		default_colors,
		default_max_samples,
		default_alphas,
	):
		configs.append(
			DatasetConfig(
				label=label,
				path=Path(csv_path),
				color=tuple(color),
				max_samples=max_samples,
				alpha=float(alpha),
			),
		)
	return configs


def _configs_from_cli(args: argparse.Namespace) -> list[DatasetConfig]:
	csv_inputs = list(args.csvs or [])
	if not csv_inputs:
		return []

	label_inputs = list(args.labels or [])
	color_inputs = list(args.colors or [])
	max_samples_inputs = list(args.max_samples or [])
	alpha_inputs = list(args.alphas or [])

	if label_inputs and len(label_inputs) != len(csv_inputs):
		raise ValueError("Number of --label values must match number of --csv files")
	if color_inputs and len(color_inputs) not in (1, len(csv_inputs)):
		raise ValueError("Number of --color values must be 1 or match number of --csv files")
	if max_samples_inputs and len(max_samples_inputs) not in (1, len(csv_inputs)):
		raise ValueError("Number of --max-samples values must be 1 or match number of --csv files")
	if alpha_inputs and len(alpha_inputs) not in (1, len(csv_inputs)):
		raise ValueError("Number of --alpha values must be 1 or match number of --csv files")

	labels = label_inputs if label_inputs else [f"Dataset {idx + 1}" for idx in range(len(csv_inputs))]

	if color_inputs:
		if len(color_inputs) == 1:
			base_colors = _ensure_color_cycle([_normalise_color(color_inputs[0])], len(csv_inputs))
		else:
			base_colors = [_normalise_color(value) for value in color_inputs]
	else:
		base_colors = _ensure_color_cycle([], len(csv_inputs))

	if max_samples_inputs:
		if len(max_samples_inputs) == 1:
			max_samples_list = [int(max_samples_inputs[0])] * len(csv_inputs)
		else:
			max_samples_list = [int(value) for value in max_samples_inputs]
	else:
		max_samples_list = [None] * len(csv_inputs)

	if alpha_inputs:
		alpha_list = _ensure_alpha_list(alpha_inputs, len(csv_inputs))
	else:
		alpha_list = _ensure_alpha_list([], len(csv_inputs))

	for value in max_samples_list:
		if value is not None and value <= 0:
			raise ValueError("--max-samples must be positive")

	configs: list[DatasetConfig] = []
	for idx, (label, csv_path) in enumerate(zip(labels, csv_inputs)):
		configs.append(
			DatasetConfig(
				label=label,
				path=Path(csv_path),
				color=tuple(base_colors[idx]),
				max_samples=max_samples_list[idx],
				alpha=float(alpha_list[idx]),
			),
		)
	return configs


def _configs_from_config(config_data: dict) -> list[DatasetConfig]:
	if not isinstance(config_data, dict):
		raise ValueError("Config file must contain a JSON object")
	entries = config_data.get("datasets")
	if entries is None:
		raise ValueError("Config must include a 'datasets' list")
	if not isinstance(entries, Iterable):
		raise ValueError("'datasets' must be iterable")

	configs: list[DatasetConfig] = []
	for idx, raw in enumerate(entries):
		if not isinstance(raw, dict):
			raise ValueError(f"Dataset entry at index {idx} must be an object")
		try:
			csv_path = Path(raw["csv"]).expanduser()
			label = str(raw["label"])
		except KeyError as exc:
			raise ValueError(f"Dataset entry {idx} is missing required key '{exc.args[0]}'") from exc

		color_value = raw.get("color")
		if color_value is None:
			color = DEFAULT_COLOR_CYCLE[idx % len(DEFAULT_COLOR_CYCLE)]
		else:
			color = _normalise_color(color_value)

		max_samples_value = raw.get("max_samples")
		max_samples = None if max_samples_value is None else int(max_samples_value)
		if max_samples is not None and max_samples <= 0:
			raise ValueError("max_samples must be positive in config")

		alpha_value = raw.get("alpha")
		alpha = DEFAULT_ALPHA if alpha_value is None else float(alpha_value)
		if not (0.0 <= alpha <= 1.0):
			raise ValueError("alpha must be between 0 and 1 in config")

		configs.append(
			DatasetConfig(
				label=label,
				path=csv_path,
				color=tuple(color),
				max_samples=max_samples,
				alpha=alpha,
			),
		)

	if not configs:
		raise ValueError("Config must define at least one dataset")
	return configs


def resolve_dataset_configs(args: argparse.Namespace, config_data: Optional[dict]) -> list[DatasetConfig]:
	cli_configs = _configs_from_cli(args)
	if cli_configs:
		return cli_configs
	if config_data:
		return _configs_from_config(config_data)
	return _configs_from_defaults()


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
	figsize: tuple[float, float],
) -> None:
	if coords.shape[0] == 0:
		raise ValueError("No coordinates to plot – check input data")
	output_path.parent.mkdir(parents=True, exist_ok=True)
	fig, ax = plt.subplots(figsize=figsize, dpi=120)

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
			alpha=cfg.alpha,
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
	config_data = load_config(args.config)
	configs = resolve_dataset_configs(args, config_data)

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
		(float(args.figsize[0]), float(args.figsize[1])),
	)

	print(f"Saved t-SNE plot to: {args.output}")


if __name__ == "__main__":
	main()

