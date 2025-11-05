"""Visualise novelty distributions as raincloud plots.

This utility mirrors the style and ergonomics of the other plotting scripts in
``draw_picture`` while focusing on *novelty* – defined here simply as

    novelty = 1 − mean(Tanimoto(candidate, training set))

Because both the generated molecules and the reference training library can be
large, the script allows per-dataset subsampling just like ``tsne.py``.

Input data sources can be declared either via command-line flags (``--csv``,
``--label`` …) or through a JSON/YAML config file (default:
``configs/raincloud.json``) using the schema::

	{
	  "reference": {
		"csv": "./data/train.csv",
		"max_samples": 2000
	  },
	  "curves": [
		{
		  "csv": "./rl_runs/rl.csv",
		  "label": "RL",
		  "step_range": [50, 499],
		  "color": [88, 182, 233],
		  "max_samples": 1500
		}
	  ]
	}

Only ``curves`` is required – ``reference`` can be provided on the command
line instead with ``--reference-csv`` and ``--reference-max-samples``.
Consistency with the rest of the project includes:

* Shared default colour cycle.
* Optional step filtering (``step_range`` mirrors diversity plots).
* Detailed console feedback about discarded SMILES and sampling decisions.

Example usage::

	python raincloud.py \
		--reference-csv ./data_preparation/train.csv \
		--csv ./rl_runs/pic16.csv --label RL --max-samples 1500 \
		--csv ./rl_runs/random.csv --label "Random" --color "#8e7cc0" \
		--output ./rl_runs/plots/novelty_raincloud.png

The resulting figure places method labels on the vertical axis and novelty on
the horizontal axis, combining half-violins, boxplots, and jittered points.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:  # pragma: no cover - dependency guards
	import pandas as pd  # type: ignore[import]
except ModuleNotFoundError as exc:  # pragma: no cover - runtime guard
	raise ModuleNotFoundError(
		"pandas is required to run raincloud.py. Install it via 'pip install pandas'.",
	) from exc

try:  # pragma: no cover - dependency guards
	import matplotlib.pyplot as plt  # type: ignore[import]
except ModuleNotFoundError as exc:  # pragma: no cover - runtime guard
	raise ModuleNotFoundError(
		"matplotlib is required to run raincloud.py. Install it via 'pip install matplotlib'.",
	) from exc

try:  # pragma: no cover - dependency guards
	import seaborn as sns  # type: ignore[import]
except ModuleNotFoundError as exc:  # pragma: no cover - runtime guard
	raise ModuleNotFoundError(
		"seaborn is required to run raincloud.py. Install it via 'pip install seaborn'.",
	) from exc

try:  # pragma: no cover - dependency guards
	import ptitprince as pt  # type: ignore[import]
except ModuleNotFoundError as exc:  # pragma: no cover - runtime guard
	raise ModuleNotFoundError(
		"ptitprince is required to render raincloud plots. Install it via 'pip install ptitprince'.",
	) from exc

try:  # pragma: no cover - dependency guards
	from rdkit import Chem  # type: ignore[import]
	from rdkit.Chem import DataStructs, rdFingerprintGenerator  # type: ignore[import]
except ModuleNotFoundError as exc:  # pragma: no cover - runtime guard
	raise ModuleNotFoundError(
		"RDKit is required to compute novelty scores. Install it, e.g., 'conda install -c rdkit rdkit'.",
	) from exc

from matplotlib.colors import to_hex, to_rgb


DEFAULT_CONFIG_PATH = Path("./configs/raincloud.json")
DEFAULT_COLOR_CYCLE: Tuple[Tuple[float, float, float], ...] = (
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
class ReferenceSpec:
	path: Path
	max_samples: Optional[int]


@dataclass(frozen=True)
class RaincloudDatasetSpec:
	label: str
	path: Path
	color: Tuple[float, float, float]
	max_samples: Optional[int]
	step_range: Optional[Tuple[float, float]]


def _normalise_color(value: object) -> Tuple[float, float, float]:
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


def _ensure_color_cycle(colors: Sequence[Tuple[float, float, float]], count: int) -> List[Tuple[float, float, float]]:
	if not colors:
		base = list(DEFAULT_COLOR_CYCLE)
	else:
		base = list(colors)
	if count <= len(base):
		return base[:count]
	repeats = (count + len(base) - 1) // len(base)
	tiled: List[Tuple[float, float, float]] = list(base) * repeats
	return tiled[:count]


def _parse_step_ranges(raw_ranges: Optional[Sequence[Sequence[float]]], expected: int) -> List[Optional[Tuple[float, float]]]:
	if not raw_ranges:
		return [None] * expected
	if len(raw_ranges) not in (1, expected):
		raise ValueError("Number of --step-range entries must match datasets or be a single global range")
	if len(raw_ranges) == 1:
		start, end = raw_ranges[0]
		if start > end:
			raise ValueError("step_range start must be <= end")
		return [(float(start), float(end))] * expected
	result: List[Optional[Tuple[float, float]]] = []
	for start, end in raw_ranges:
		if start > end:
			raise ValueError("step_range start must be <= end")
		result.append((float(start), float(end)))
	return result


def load_config(config_path: Optional[Path]) -> Optional[dict]:
	if config_path is None:
		return None
	if not config_path.exists():
		raise FileNotFoundError(f"Config file not found: {config_path}")
	text = config_path.read_text(encoding="utf-8")
	if config_path.suffix.lower() in {".yaml", ".yml"}:
		try:
			import yaml  # type: ignore[import]
		except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
			raise RuntimeError(
				"PyYAML is required for YAML configs. Install it or provide a JSON config.",
			) from exc
		return yaml.safe_load(text)
	return json.loads(text)


def _specs_from_config(config: dict) -> Tuple[Optional[ReferenceSpec], List[RaincloudDatasetSpec]]:
	if not isinstance(config, dict):
		raise ValueError("Config must be a dictionary")

	reference_section = config.get("reference")
	reference: Optional[ReferenceSpec] = None
	if reference_section is not None:
		if not isinstance(reference_section, dict):
			raise ValueError("'reference' section must be an object")
		try:
			ref_path = Path(reference_section["csv"]).expanduser()
		except KeyError as exc:
			raise ValueError("'reference' section requires a 'csv' key") from exc
		max_samples_value = reference_section.get("max_samples")
		if max_samples_value is not None and int(max_samples_value) <= 0:
			raise ValueError("reference max_samples must be positive")
		reference = ReferenceSpec(path=ref_path, max_samples=int(max_samples_value) if max_samples_value else None)

	curves = config.get("curves")
	if curves is None:
		raise ValueError("Config must contain a 'curves' list")
	if not isinstance(curves, Iterable):
		raise ValueError("'curves' must be an iterable of objects")

	specs: List[RaincloudDatasetSpec] = []
	for idx, entry in enumerate(curves):
		if not isinstance(entry, dict):
			raise ValueError(f"Curve entry at index {idx} must be an object")
		try:
			csv_path = Path(entry["csv"]).expanduser()
			label = str(entry["label"])
		except KeyError as exc:
			missing = exc.args[0]
			raise ValueError(f"Missing required key '{missing}' in curve definition {idx}") from exc

		color_value = entry.get("color")
		if color_value is None:
			color = DEFAULT_COLOR_CYCLE[idx % len(DEFAULT_COLOR_CYCLE)]
		else:
			color = _normalise_color(color_value)

		step_range_value = entry.get("step_range")
		step_range: Optional[Tuple[float, float]] = None
		if step_range_value is not None:
			if not isinstance(step_range_value, Sequence) or len(step_range_value) != 2:
				raise ValueError("'step_range' must contain exactly two numbers")
			start, end = map(float, step_range_value)
			if start > end:
				raise ValueError("step_range start must be <= end")
			step_range = (start, end)

		max_samples_raw = entry.get("max_samples")
		max_samples = None
		if max_samples_raw is not None:
			max_samples = int(max_samples_raw)
			if max_samples <= 0:
				raise ValueError("max_samples must be positive")

		specs.append(
			RaincloudDatasetSpec(
				label=label,
				path=csv_path,
				color=color,
				max_samples=max_samples,
				step_range=step_range,
			),
		)

	if not specs:
		raise ValueError("Config must define at least one curve")
	return reference, specs


def _specs_from_args(args: argparse.Namespace) -> Tuple[Optional[ReferenceSpec], List[RaincloudDatasetSpec]]:
	csv_inputs: List[Path] = list(args.csvs or [])
	if not csv_inputs:
		return None, []

	label_inputs = list(args.labels or [])
	color_inputs = list(args.colors or [])
	step_range_inputs = list(args.step_ranges or [])
	max_samples_inputs = list(args.max_samples or [])

	if label_inputs and len(label_inputs) != len(csv_inputs):
		raise ValueError("Number of --label values must match number of --csv files")
	if color_inputs and len(color_inputs) not in (1, len(csv_inputs)):
		raise ValueError("Number of --color values must be 1 or match number of --csv files")
	if step_range_inputs and len(step_range_inputs) not in (1, len(csv_inputs)):
		raise ValueError("Number of --step-range values must be 1 or match number of --csv files")
	if max_samples_inputs and len(max_samples_inputs) not in (1, len(csv_inputs)):
		raise ValueError("Number of --max-samples values must be 1 or match number of --csv files")

	labels = (
		label_inputs
		if label_inputs
		else [f"Dataset {idx + 1}" for idx in range(len(csv_inputs))]
	)

	if color_inputs:
		if len(color_inputs) == 1:
			colors = _ensure_color_cycle([_normalise_color(color_inputs[0])], len(csv_inputs))
		else:
			colors = [_normalise_color(value) for value in color_inputs]
	else:
		colors = _ensure_color_cycle([], len(csv_inputs))

	step_ranges = _parse_step_ranges(step_range_inputs, len(csv_inputs))

	if max_samples_inputs:
		raw_values: List[int]
		if len(max_samples_inputs) == 1:
			raw_values = [int(max_samples_inputs[0])] * len(csv_inputs)
		else:
			raw_values = [int(value) for value in max_samples_inputs]
		for value in raw_values:
			if value <= 0:
				raise ValueError("--max-samples must be positive")
		max_samples_list = raw_values
	else:
		max_samples_list = [None] * len(csv_inputs)

	specs = [
		RaincloudDatasetSpec(
			label=label,
			path=Path(csv).expanduser(),
			color=colors[idx],
			max_samples=max_samples_list[idx],
			step_range=step_ranges[idx],
		)
		for idx, (label, csv) in enumerate(zip(labels, csv_inputs))
	]

	reference = None
	if args.reference_csv:
		max_samples = args.reference_max_samples
		if max_samples is not None and max_samples <= 0:
			raise ValueError("--reference-max-samples must be positive")
		reference = ReferenceSpec(path=Path(args.reference_csv).expanduser(), max_samples=max_samples)

	return reference, specs


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Raincloud plot for novelty distributions")
	parser.add_argument(
		"--config",
		type=Path,
		default=DEFAULT_CONFIG_PATH,
		help="Path to JSON or YAML config file (default: ./configs/raincloud.json)",
	)
	parser.add_argument(
		"--csv",
		dest="csvs",
		type=Path,
		action="append",
		default=None,
		help="Dataset CSV containing a SMILES column; repeat flag for multiple methods",
	)
	parser.add_argument(
		"--label",
		dest="labels",
		action="append",
		default=None,
		help="Label for each --csv dataset (aligns with order of --csv flags)",
	)
	parser.add_argument(
		"--color",
		dest="colors",
		action="append",
		default=None,
		help="Colour for each dataset (hex/name/RGB); provide one or repeat to match datasets",
	)
	parser.add_argument(
		"--step-range",
		dest="step_ranges",
		action="append",
		nargs=2,
		type=float,
		default=None,
		metavar=("START", "END"),
		help="Optional step range [START END] to filter rows before sampling",
	)
	parser.add_argument(
		"--max-samples",
		dest="max_samples",
		action="append",
		type=int,
		default=None,
		help="Maximum molecules per dataset; provide once for global limit or repeat per dataset",
	)
	parser.add_argument(
		"--reference-csv",
		type=Path,
		default=None,
		help="Reference/training set CSV (required if config lacks 'reference')",
	)
	parser.add_argument(
		"--reference-max-samples",
		type=int,
		default=None,
		help="Optional subsample size for the reference/training set",
	)
	parser.add_argument(
		"--smiles-column",
		type=str,
		default="SMILES",
		help="Name of the column containing SMILES strings (default: SMILES)",
	)
	parser.add_argument(
		"--step-column",
		type=str,
		default="Step",
		help="Name of the step column used with step_range filtering (default: Step)",
	)
	parser.add_argument(
		"--fingerprint-radius",
		type=int,
		default=2,
		help="Radius for Morgan fingerprints (default: 2)",
	)
	parser.add_argument(
		"--fingerprint-bits",
		type=int,
		default=2048,
		help="Bit length for Morgan fingerprints (default: 2048)",
	)
	parser.add_argument(
		"--seed",
		type=int,
		default=13,
		help="Random seed for sampling (default: 13)",
	)
	parser.add_argument(
		"--output",
		type=Path,
		default=Path("./raincloud_plot.png"),
		help="Output PNG file path (default: ./raincloud_plot.png)",
	)
	parser.add_argument(
		"--title",
		type=str,
		default="Novelty raincloud",
		help="Title for the figure",
	)
	parser.add_argument(
		"--xlabel",
		type=str,
		default="Novelty",
		help="Label for the novelty axis",
	)
	parser.add_argument(
		"--point-size",
		type=float,
		default=2.2,
		help="Scatter point size for the rain portion",
	)
	parser.add_argument(
		"--rain-alpha",
		type=float,
		default=0.6,
		help="Alpha (opacity) for jittered points (default: 0.6)",
	)
	parser.add_argument(
		"--bandwidth",
		type=float,
		default=0.2,
		help="Bandwidth for the KDE in the cloud component",
	)
	parser.add_argument(
		"--viol-width",
		type=float,
		default=0.6,
		help="Half violin width for the cloud component",
	)
	parser.add_argument(
		"--move",
		type=float,
		default=0.2,
		help="Offset applied to jittered points to prevent overlap with the cloud",
	)
	parser.add_argument(
		"--dpi",
		type=int,
		default=300,
		help="Resolution of the saved figure (default: 300)",
	)
	parser.add_argument(
		"--figsize",
		type=float,
		nargs=2,
		default=(10.0, 6.0),
		metavar=("WIDTH", "HEIGHT"),
		help="Figure size in inches as WIDTH HEIGHT (default: 10 6)",
	)
	return parser.parse_args()


def _load_smiles_series(
	path: Path,
	smiles_column: str,
	step_column: str,
	step_range: Optional[Tuple[float, float]],
) -> pd.Series:
	if not path.exists():
		raise FileNotFoundError(f"CSV file not found: {path}")
	df = pd.read_csv(path)
	if smiles_column not in df.columns:
		raise KeyError(f"CSV '{path}' must contain column '{smiles_column}'")

	if step_range is not None:
		if step_column not in df.columns:
			raise KeyError(
				f"CSV '{path}' lacks column '{step_column}' required for step_range filtering",
			)
		start, end = step_range
		df = df[(df[step_column] >= start) & (df[step_column] <= end)]

	series = df[smiles_column].dropna().astype(str).str.strip()
	invalid_mask = series.str.upper().eq("INVALID")
	if invalid_mask.any():
		series = series[~invalid_mask]
	return series.reset_index(drop=True)


def _subsample_series(series: pd.Series, max_samples: Optional[int], seed: int) -> pd.Series:
	if max_samples is None or len(series) <= max_samples:
		return series
	rng = np.random.default_rng(seed)
	indices = rng.choice(len(series), size=max_samples, replace=False)
	indices.sort()
	return series.iloc[indices].reset_index(drop=True)


def _get_morgan_generator(radius: int, n_bits: int) -> rdFingerprintGenerator.FingerprintGenerator:
	return rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)


def _smiles_to_fingerprints(
	smiles_list: Sequence[str],
	generator: rdFingerprintGenerator.FingerprintGenerator,
) -> Tuple[List[DataStructs.cDataStructs.ExplicitBitVect], int]:
	fingerprints: List[DataStructs.cDataStructs.ExplicitBitVect] = []
	dropped = 0
	for smiles in smiles_list:
		mol = Chem.MolFromSmiles(smiles)
		if mol is None:
			dropped += 1
			continue
		fingerprints.append(generator.GetFingerprint(mol))
	return fingerprints, dropped


def _compute_novelty_scores(
	candidate_fps: Sequence[DataStructs.cDataStructs.ExplicitBitVect],
	reference_fps: Sequence[DataStructs.cDataStructs.ExplicitBitVect],
) -> List[float]:
	if not reference_fps:
		raise ValueError("Reference set is empty after preprocessing – cannot compute novelty")
	novelties: List[float] = []
	for fp in candidate_fps:
		sims = DataStructs.BulkTanimotoSimilarity(fp, reference_fps)
		if not sims:
			novelties.append(float("nan"))
		else:
			mean_similarity = float(np.mean(sims))
			novelties.append(1.0 - mean_similarity)
	return novelties


def _prepare_datasets(
	specs: Sequence[RaincloudDatasetSpec],
	reference: ReferenceSpec,
	smiles_column: str,
	step_column: str,
	radius: int,
	n_bits: int,
	seed: int,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
	generator = _get_morgan_generator(radius=radius, n_bits=n_bits)

	reference_series = _load_smiles_series(
		reference.path,
		smiles_column=smiles_column,
		step_column=step_column,
		step_range=None,
	)
	reference_series = _subsample_series(reference_series, reference.max_samples, seed)
	reference_fps, ref_dropped = _smiles_to_fingerprints(reference_series.tolist(), generator)
	if ref_dropped:
		print(f"[warning] Dropped {ref_dropped} invalid SMILES from reference '{reference.path}'")
	print(f"[info] Using {len(reference_fps)} reference molecules from '{reference.path}'")

	rows: List[Dict[str, float]] = []
	palette: Dict[str, str] = {}
	method_order: Dict[str, int] = {}

	for method_index, spec in enumerate(specs):
		smiles_series = _load_smiles_series(
			spec.path,
			smiles_column=smiles_column,
			step_column=step_column,
			step_range=spec.step_range,
		)

		sampled = _subsample_series(smiles_series, spec.max_samples, seed + method_index)
		candidate_fps, dropped = _smiles_to_fingerprints(sampled.tolist(), generator)
		if dropped:
			print(f"[warning] Dropped {dropped} invalid SMILES for '{spec.label}' from '{spec.path}'")
		if not candidate_fps:
			print(f"[warning] No valid molecules for '{spec.label}' – skipping")
			continue

		novelties = _compute_novelty_scores(candidate_fps, reference_fps)
		method_order[spec.label] = len(method_order)
		palette[spec.label] = to_hex(spec.color)

		for novelty in novelties:
			rows.append({"method": spec.label, "novelty": novelty})

		print(
			f"[info] '{spec.label}': {len(candidate_fps)} molecules retained (from {len(smiles_series)} loaded)"
			+ (f", step_range={spec.step_range}" if spec.step_range else "")
			+ (f", max_samples={spec.max_samples}" if spec.max_samples else ""),
		)

	if not rows:
		raise ValueError("No novelty scores computed – ensure datasets contain valid SMILES")

	data = pd.DataFrame(rows)
	data["novelty"] = pd.to_numeric(data["novelty"], errors="coerce")
	data = data.dropna(subset=["novelty"])
	if data.empty:
		raise ValueError("All novelty values are NaN – this should not happen")

	return data, palette


def _plot_raincloud(
	data: pd.DataFrame,
	palette: Dict[str, str],
	output_path: Path,
	title: str,
	xlabel: str,
	figsize: Tuple[float, float],
	dpi: int,
	bandwidth: float,
	viol_width: float,
	move: float,
	point_size: float,
	rain_alpha: float,
) -> None:
	sns.set(style="whitegrid", context="talk")
	fig, ax = plt.subplots(figsize=figsize, dpi=120)

	order = list(palette.keys())
	palette_list = [palette[label] for label in order]

	pt.RainCloud(
		x="method",
		y="novelty",
		data=data,
		order=order,
		orient="h",
		palette=palette_list,
		bw=bandwidth,
		width_viol=viol_width,
		move=move,
		rain_alpha=rain_alpha,
		point_size=point_size,
		box_showfliers=False,
		box_whis=1.5,
		box_showmeans=True,
		box_meanprops={"linestyle": "--", "linewidth": 1.2},
	)

	ax.set_xlabel(xlabel)
	ax.set_ylabel("")
	ax.set_title(title)
	sns.despine(trim=True)
	fig.tight_layout()

	output_path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(output_path, dpi=dpi)
	plt.close(fig)
	print(f"Saved raincloud plot to: {output_path}")


def main() -> None:
	args = parse_args()

	config_data: Optional[dict] = None
	specs: List[RaincloudDatasetSpec]
	reference: Optional[ReferenceSpec]

	try:
		config_data = load_config(args.config) if args.config else None
	except FileNotFoundError:
		if args.csvs:
			config_data = None
		else:
			raise

	reference = None
	dataset_specs: List[RaincloudDatasetSpec] = []

	if config_data:
		ref_from_config, specs_from_config = _specs_from_config(config_data)
		reference = ref_from_config
		dataset_specs = specs_from_config

	ref_from_args, specs_from_args = _specs_from_args(args)
	if specs_from_args:
		dataset_specs = specs_from_args
	if ref_from_args:
		reference = ref_from_args

	if not dataset_specs:
		raise ValueError("No datasets specified. Use --csv/--label or provide a config file")

	if reference is None:
		raise ValueError("Reference CSV not provided. Use --reference-csv or define it in the config")

	data, palette = _prepare_datasets(
		specs=dataset_specs,
		reference=reference,
		smiles_column=args.smiles_column,
		step_column=args.step_column,
		radius=args.fingerprint_radius,
		n_bits=args.fingerprint_bits,
		seed=args.seed,
	)

	_plot_raincloud(
		data=data,
		palette=palette,
		output_path=args.output,
		title=args.title,
		xlabel=args.xlabel,
		figsize=(float(args.figsize[0]), float(args.figsize[1])),
		dpi=args.dpi,
		bandwidth=args.bandwidth,
		viol_width=args.viol_width,
		move=args.move,
		point_size=args.point_size,
		rain_alpha=args.rain_alpha,
	)


if __name__ == "__main__":
	main()
