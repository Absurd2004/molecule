"""Draw radar charts for batch metrics (diversity, uniqueness, validity).

The chart combines six metrics for each dataset:

* ``IntDiv1`` and ``IntDiv2`` – diversity of molecular SMILES sampled from the
  provided CSV (random subset of size ``sample_size``).
* ``Uniq.`` – ratio of unique SMILES within the configured step range.
* ``Validity`` – valid molecule count divided by ``batch_size * step_count``.
* ``Dec Uniq.`` – mean uniqueness ratio of decorations grouped by attachment
  site after splitting the ``Scaffold`` column on ``"|"``.
* ``Dec Div`` – mean IntDiv1 of decorations grouped by attachment site.

Configuration is supplied via JSON/YAML. Example::

    {
      "title": "Run RL",
      "output": "./rl_runs/diversity/radar.png",
      "batch_size": 512,
      "sample_size": 5096,
      "radars": [
        {
          "label": "RL",
          "csv": "./rl_runs/pic16_est_trend/diversity_memory.csv",
          "step_range": [50, 499],
          "color": [65, 105, 225]
        }
      ]
    }

Command-line arguments provide overrides for key parameters.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, MutableMapping, Optional, Sequence, Tuple

try:
	import matplotlib.pyplot as plt  # type: ignore[import]
except ModuleNotFoundError as exc:  # pragma: no cover - import guard
	raise ModuleNotFoundError(
		"matplotlib is required to run radar.py. Please install it via 'pip install matplotlib'."
	) from exc

import numpy as np

try:
	import pandas as pd  # type: ignore[import]
except ModuleNotFoundError as exc:  # pragma: no cover - import guard
	raise ModuleNotFoundError(
		"pandas is required to run radar.py. Please install it via 'pip install pandas'."
	) from exc

try:
	from rdkit import Chem  # type: ignore[import]
	from rdkit.Chem import DataStructs, rdFingerprintGenerator  # type: ignore[import]
except ModuleNotFoundError as exc:  # pragma: no cover - import guard
	raise ModuleNotFoundError(
		"RDKit is required to compute diversity metrics. Install it via 'conda install -c rdkit rdkit' or similar."
	) from exc


METRIC_ORDER = ("IntDiv1", "IntDiv2", "Uniq.", "Validity", "Dec Uniq.", "Dec Div")


@dataclass
class RadarSpec:
	"""Configuration describing one dataset on the radar chart."""

	csv: Path
	label: str
	step_start: int
	step_end: int
	color: Tuple[float, float, float]

	@classmethod
	def from_dict(cls, data: MutableMapping[str, object]) -> "RadarSpec":
		try:
			csv_path = Path(str(data["csv"]))
			label = str(data["label"])
			step_range = data["step_range"]
			color_values = data["color"]
		except KeyError as exc:  # pragma: no cover - defensive branch
			missing = exc.args[0]
			raise ValueError(f"Missing required key '{missing}' in radar config entry") from exc

		if not isinstance(step_range, Sequence) or len(step_range) != 2:
			raise ValueError("'step_range' must contain exactly two numbers: [start, end]")

		step_start, step_end = map(int, step_range)
		if step_start > step_end:
			raise ValueError("'step_range' start must be less than or equal to end")

		if not isinstance(color_values, Sequence) or len(color_values) != 3:
			raise ValueError("'color' must be an RGB sequence of length 3")

		color = tuple(float(channel) / 255.0 for channel in color_values)
		if any(channel < 0.0 or channel > 1.0 for channel in color):
			raise ValueError("RGB color channels must be between 0 and 255")

		return cls(
			csv=csv_path.expanduser(),
			label=label,
			step_start=step_start,
			step_end=step_end,
			color=color,
		)


def load_config(config_path: Path) -> Dict[str, object]:
	"""Load and return the configuration dictionary (JSON or YAML)."""

	text = config_path.read_text(encoding="utf-8")
	if config_path.suffix.lower() in {".yaml", ".yml"}:
		try:
			import yaml  # type: ignore
		except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
			raise RuntimeError(
				"PyYAML is required for YAML configs. Install it or provide a JSON config."
			) from exc

		raw_config = yaml.safe_load(text)
	else:
		raw_config = json.loads(text)

	if not isinstance(raw_config, dict):
		raise ValueError("Configuration file must contain a top-level object")

	if "radars" not in raw_config:
		raise ValueError("Configuration must define a 'radars' list")

	radars = raw_config.get("radars")
	if not isinstance(radars, Iterable):
		raise ValueError("'radars' must be an iterable of radar entries")

	specs = [RadarSpec.from_dict(entry) for entry in radars]
	if not specs:
		raise ValueError("Configuration must include at least one radar entry")

	raw_config["radars"] = specs
	return raw_config


@lru_cache(maxsize=None)
def _get_morgan_generator(radius: int, n_bits: int) -> rdFingerprintGenerator.FingerprintGenerator:
	"""Return a cached Morgan fingerprint generator for the given parameters."""

	return rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)


def _morgan_fp(smiles: str, radius: int, n_bits: int) -> Optional[DataStructs.cDataStructs.ExplicitBitVect]:
	"""Return RDKit Morgan fingerprint for ``smiles`` or ``None`` if invalid."""

	mol = Chem.MolFromSmiles(smiles)
	if mol is None:
		return None
	generator = _get_morgan_generator(radius=radius, n_bits=n_bits)
	return generator.GetFingerprint(mol)


def _pairwise_tanimotos(
	fps: Sequence[Optional[DataStructs.cDataStructs.ExplicitBitVect]],
) -> List[float]:
	"""Compute pairwise Tanimoto similarities, ignoring missing fingerprints."""

	values: List[float] = []
	for i in range(len(fps)):
		fp_i = fps[i]
		if fp_i is None:
			continue
		for j in range(i + 1, len(fps)):
			fp_j = fps[j]
			if fp_j is None:
				continue
			values.append(DataStructs.TanimotoSimilarity(fp_i, fp_j))
	return values


def _compute_intdiv(
	items: Sequence[str],
	radius: int,
	n_bits: int,
	cache: Dict[Tuple[str, int, int], Optional[DataStructs.cDataStructs.ExplicitBitVect]],
) -> Tuple[float, float]:
	"""Return IntDiv1/IntDiv2 for the provided SMILES/decorations list."""

	fps: List[Optional[DataStructs.cDataStructs.ExplicitBitVect]] = []
	for item in items:
		key = (item, radius, n_bits)
		if key not in cache:
			cache[key] = _morgan_fp(item, radius=radius, n_bits=n_bits)
		fps.append(cache[key])

	tanimotos = _pairwise_tanimotos(fps)
	if not tanimotos:
		return float("nan"), float("nan")

	arr = np.asarray(tanimotos, dtype=float)
	intdiv1 = 1.0 - arr.mean()
	intdiv2 = 1.0 - np.sqrt(np.square(arr).mean())
	return float(intdiv1), float(intdiv2)


def _extract_decorations(scaffold: str) -> List[str]:
	"""Split the scaffold string into decoration fragments (excluding core)."""

	parts = str(scaffold).split("|")
	if len(parts) <= 1:
		return []
	return [fragment.strip() for fragment in parts[1:] if fragment.strip()]


def _mean_or_nan(values: Sequence[float]) -> float:
	"""Return the mean ignoring NaNs; NaN if the sequence is empty or all NaN."""

	arr = np.asarray(values, dtype=float)
	if arr.size == 0:
		return float("nan")
	filtered = arr[~np.isnan(arr)]
	if filtered.size == 0:
		return float("nan")
	return float(filtered.mean())


def compute_metrics(
	df: pd.DataFrame,
	spec: RadarSpec,
	*,
	step_col: str,
	smiles_col: str,
	scaffold_col: str,
	sample_size: int,
	batch_size: int,
	radius: int,
	n_bits: int,
	rng: np.random.Generator,
) -> Dict[str, float]:
	"""Compute the six radar metrics for a given dataset."""

	required_columns = {step_col, smiles_col, scaffold_col}
	missing_columns = required_columns.difference(df.columns)
	if missing_columns:
		raise ValueError(
			f"CSV '{spec.csv}' is missing required columns: {', '.join(sorted(missing_columns))}"
		)

	mask = (df[step_col] >= spec.step_start) & (df[step_col] <= spec.step_end)
	scoped = df.loc[mask].copy()
	scoped = scoped.dropna(subset=[smiles_col])
	scoped = scoped.reset_index(drop=True)

	valid_count = len(scoped)
	if valid_count == 0:
		return {metric: float("nan") for metric in METRIC_ORDER}

	sample_n = min(sample_size, valid_count)
	indices = rng.choice(valid_count, size=sample_n, replace=False) if sample_n < valid_count else np.arange(valid_count)
	sampled = scoped.iloc[indices].copy()

	cache: Dict[Tuple[str, int, int], Optional[DataStructs.cDataStructs.ExplicitBitVect]] = {}
	intdiv1, intdiv2 = _compute_intdiv(sampled[smiles_col].tolist(), radius, n_bits, cache)

	# Uniqueness ratio
	unique_smiles = scoped[smiles_col].nunique(dropna=True)
	uniq_ratio = unique_smiles / float(valid_count) if valid_count else float("nan")

	# Validity: valid molecules divided by expected total molecules
	step_count = spec.step_end - spec.step_start + 1
	denominator = batch_size * step_count
	validity = valid_count / float(denominator) if denominator > 0 else float("nan")

	# Decoration uniqueness and diversity from the sampled subset
	by_site: Dict[int, List[str]] = defaultdict(list)
	for scaffold in sampled[scaffold_col].dropna():
		for idx, decoration in enumerate(_extract_decorations(scaffold)):
			by_site[idx].append(decoration)

	dec_unique_values: List[float] = []
	dec_div_values: List[float] = []
	for decorations in by_site.values():
		if not decorations:
			continue
		unique_ratio = len(set(decorations)) / float(len(decorations))
		dec_unique_values.append(unique_ratio)
		dec_div, _ = _compute_intdiv(decorations, radius, n_bits, cache)
		dec_div_values.append(dec_div)

	dec_unique = _mean_or_nan(dec_unique_values)
	dec_diversity = _mean_or_nan(dec_div_values)

	return {
		"IntDiv1": intdiv1,
		"IntDiv2": intdiv2,
		"Uniq.": float(uniq_ratio),
		"Validity": float(validity),
		"Dec Uniq.": dec_unique,
		"Dec Div": dec_diversity,
	}


def plot_radar(
	specs: Sequence[RadarSpec],
	metrics_map: Sequence[Dict[str, float]],
	*,
	output_path: Optional[Path],
	title: Optional[str],
) -> None:
	"""Render the radar chart for the provided metrics."""

	n_metrics = len(METRIC_ORDER)
	angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False)
	angles = np.concatenate([angles, angles[:1]])

	fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"})

	max_value = 0.0
	for spec, metrics in zip(specs, metrics_map):
		values = [metrics.get(metric, float("nan")) for metric in METRIC_ORDER]
		finite_vals = [value for value in values if np.isfinite(value)]
		if finite_vals:
			max_value = max(max_value, max(finite_vals))

	max_value = max(max_value, 1.0)
	ax.set_ylim(0, max_value)
	ax.set_xticks(angles[:-1])
	ax.set_xticklabels(METRIC_ORDER)
	levels = np.linspace(0, max_value, num=5, endpoint=True)
	ax.set_yticks(levels)
	ax.set_yticklabels([f"{tick:.2f}" for tick in levels], color="gray")

	for spec, metrics in zip(specs, metrics_map):
		values = [metrics.get(metric, float("nan")) for metric in METRIC_ORDER]
		filled = np.nan_to_num(values, nan=0.0)
		filled = np.append(filled, filled[0])
		ax.plot(angles, filled, color=spec.color, linewidth=2, label=spec.label)
		ax.fill(angles, filled, color=spec.color, alpha=0.2)

	if title:
		ax.set_title(title, fontsize=14, pad=20)

	ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
	fig.tight_layout()

	if output_path:
		output_path.parent.mkdir(parents=True, exist_ok=True)
		fig.savefig(output_path, dpi=300)
	else:
		plt.show()


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Plot a radar chart of molecular batch metrics")
	parser.add_argument(
		"--config",
		type=Path,
		default=Path("./configs/radar.json"),
		help="Path to JSON or YAML configuration file",
	)
	parser.add_argument(
		"--output",
		type=Path,
		help="Optional override for the output image path",
	)
	parser.add_argument("--sample-size", type=int, help="Override sample size k for diversity computation")
	parser.add_argument("--batch-size", type=int, help="Override batch size per step")
	parser.add_argument("--fingerprint-radius", type=int, default=2, help="Morgan fingerprint radius (default: 2)")
	parser.add_argument("--fingerprint-bits", type=int, default=2048, help="Morgan fingerprint bit length (default: 2048)")
	parser.add_argument("--step-column", type=str, default="Step", help="Name of the step column (default: Step)")
	parser.add_argument("--smiles-column", type=str, default="SMILES", help="Name of the SMILES column (default: SMILES)")
	parser.add_argument("--scaffold-column", type=str, default="Scaffold", help="Name of the scaffold column (default: Scaffold)")
	parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling (default: 42)")
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	config = load_config(args.config)

	specs: Sequence[RadarSpec] = config["radars"]  # type: ignore[assignment]

	output_config = config.get("output", "./rl_runs/diversity/radar.png")
	output_path: Optional[Path]
	if output_config in (None, ""):
		output_path = None
	else:
		output_path = Path(str(output_config))
	if args.output is not None:
		output_path = args.output

	sample_size = int(config.get("sample_size", 5096))
	if args.sample_size is not None:
		sample_size = args.sample_size
	if sample_size <= 0:
		raise ValueError("sample_size must be positive")

	batch_size = int(config.get("batch_size", 1))
	if args.batch_size is not None:
		batch_size = args.batch_size
	if batch_size <= 0:
		raise ValueError("batch_size must be positive")

	title = config.get("title") if isinstance(config.get("title"), str) else None

	step_col = args.step_column
	smiles_col = args.smiles_column
	scaffold_col = args.scaffold_column
	radius = args.fingerprint_radius
	n_bits = args.fingerprint_bits

	rng = np.random.default_rng(args.seed)

	metrics: List[Dict[str, float]] = []
	for spec in specs:
		if not spec.csv.exists():
			raise FileNotFoundError(f"CSV file not found: {spec.csv}")
		df = pd.read_csv(spec.csv)
		metrics.append(
			compute_metrics(
				df,
				spec,
				step_col=step_col,
				smiles_col=smiles_col,
				scaffold_col=scaffold_col,
				sample_size=sample_size,
				batch_size=batch_size,
				radius=radius,
				n_bits=n_bits,
				rng=rng,
			)
		)

	for spec, metric_values in zip(specs, metrics):
		print(f"Metrics for {spec.label}:")
		for metric_name in METRIC_ORDER:
			value = metric_values.get(metric_name, float("nan"))
			print(f"  {metric_name}: {value:.4f}" if np.isfinite(value) else f"  {metric_name}: NaN")
		print()

	plot_radar(specs, metrics, output_path=output_path, title=title)


if __name__ == "__main__":
	main()
