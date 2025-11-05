"""Plot batch diversity metrics (IntDiv1/IntDiv2) over steps.

For every input CSV we aggregate the per-step mean of two diversity metrics
(``IntDiv1`` and ``IntDiv2`` by default) and plot both curves on the same chart.
The overall styling mirrors ``total_score_trend.py`` so all plots share a
consistent appearance across the project.

Each dataset contributes two lines to the legend:

* ``<label> IntDiv1`` – solid line using the primary colour from the config.
* ``<label> IntDiv2`` – dashed line using a lightened variant of the primary
  colour so the pair remains visually associated yet distinct.

Config structure (JSON/YAML) matches prior plotting utilities::

    {
      "curves": [
        {
          "csv": "run_a.csv",
          "label": "Run A",
          "step_range": [0, 5000],
          "color": [65, 105, 225]
        }
      ]
    }

Command-line usage example::

    python diversity_per_step.py \
        --config configs/div_per_step.json \
        --output rl_runs/diversity/diversity.png
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
	import matplotlib.pyplot as plt  # type: ignore[import]
except ModuleNotFoundError as exc:  # pragma: no cover - import guard
	raise ModuleNotFoundError(
		"matplotlib is required to run diversity_per_step.py. Please install it via 'pip install matplotlib'."
	) from exc

import numpy as np

try:
	import pandas as pd  # type: ignore[import]
except ModuleNotFoundError as exc:  # pragma: no cover - import guard
	raise ModuleNotFoundError(
		"pandas is required to run diversity_per_step.py. Please install it via 'pip install pandas'."
	) from exc

try:
	from rdkit import Chem  # type: ignore[import]
	from rdkit.Chem import DataStructs, rdFingerprintGenerator  # type: ignore[import]
except ModuleNotFoundError as exc:  # pragma: no cover - import guard
	raise ModuleNotFoundError(
		"RDKit is required to compute diversity metrics. Please install it (e.g., via conda install -c rdkit rdkit)."
	) from exc


@dataclass
class CurveSpec:
	"""Configuration describing one dataset on the chart."""

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


@lru_cache(maxsize=None)
def _get_morgan_generator(radius: int, n_bits: int) -> rdFingerprintGenerator.FingerprintGenerator:
	"""Return a cached Morgan fingerprint generator for the given parameters."""

	return rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)


def _morgan_fp(smiles: str, radius: int, n_bits: int) -> Optional[DataStructs.cDataStructs.ExplicitBitVect]:
	"""Return the RDKit Morgan fingerprint for ``smiles`` or ``None`` if invalid."""

	mol = Chem.MolFromSmiles(smiles)
	if mol is None:
		return None
	generator = _get_morgan_generator(radius=radius, n_bits=n_bits)
	return generator.GetFingerprint(mol)


def _pairwise_tanimotos(
	fps: List[Optional[DataStructs.cDataStructs.ExplicitBitVect]],
) -> List[float]:
	"""Compute all pairwise Tanimoto similarities, skipping invalid fingerprints."""

	values: List[float] = []
	for i, j in combinations(range(len(fps)), 2):
		fp_i = fps[i]
		fp_j = fps[j]
		if fp_i is None or fp_j is None:
			continue
		values.append(DataStructs.TanimotoSimilarity(fp_i, fp_j))
	return values


def _compute_intdiv(
	smiles_list: Sequence[str],
	radius: int,
	n_bits: int,
	cache: Dict[Tuple[str, int, int], Optional[DataStructs.cDataStructs.ExplicitBitVect]],
) -> Tuple[float, float]:
	"""Compute IntDiv1 and IntDiv2 for the given SMILES sequence."""

	fps: List[Optional[DataStructs.cDataStructs.ExplicitBitVect]] = []
	for smiles in smiles_list:
		key = (smiles, radius, n_bits)
		if key not in cache:
			cache[key] = _morgan_fp(smiles, radius=radius, n_bits=n_bits)
		fps.append(cache[key])

	tanimotos = _pairwise_tanimotos(fps)
	if not tanimotos:
		return float("nan"), float("nan")

	arr = np.asarray(tanimotos, dtype=float)
	intdiv1 = 1.0 - arr.mean()
	intdiv2 = 1.0 - np.sqrt(np.square(arr).mean())
	return float(intdiv1), float(intdiv2)


def compute_diversity_per_step(
	df: pd.DataFrame,
	step_start: float,
	step_end: float,
	step_col: str,
	smiles_col: str,
	radius: int,
	n_bits: int,
) -> pd.DataFrame:
	"""Return per-step IntDiv1/IntDiv2 computed from SMILES batches."""

	required_columns = {step_col, smiles_col}
	missing_columns = required_columns.difference(df.columns)
	if missing_columns:
		raise ValueError(f"CSV is missing required columns: {', '.join(sorted(missing_columns))}")

	mask = (df[step_col] >= step_start) & (df[step_col] <= step_end)
	scoped = df.loc[mask, [step_col, smiles_col]].dropna(subset=[smiles_col]).copy()
	if scoped.empty:
		return pd.DataFrame(columns=[step_col, "IntDiv1", "IntDiv2"])

	invalid_mask = (
		scoped[smiles_col]
			.astype(str)
			.str.strip()
			.str.upper()
			.eq("INVALID")
	)
	if invalid_mask.any():
		scoped = scoped.loc[~invalid_mask]
	if scoped.empty:
		return pd.DataFrame(columns=[step_col, "IntDiv1", "IntDiv2"])

	cache: Dict[Tuple[str, int, int], Optional[DataStructs.cDataStructs.ExplicitBitVect]] = {}

	steps: List[float] = []
	intdiv1_values: List[float] = []
	intdiv2_values: List[float] = []

	for step_value, batch in scoped.groupby(step_col, sort=True):
		smiles_list = batch[smiles_col].tolist()
		intdiv1, intdiv2 = _compute_intdiv(
			smiles_list,
			radius=radius,
			n_bits=n_bits,
			cache=cache,
		)
		steps.append(step_value)
		intdiv1_values.append(intdiv1)
		intdiv2_values.append(intdiv2)

	return pd.DataFrame(
		{
			step_col: steps,
			"IntDiv1": intdiv1_values,
			"IntDiv2": intdiv2_values,
		}
	).sort_values(step_col)


def _lighten_color(color: Tuple[float, float, float], factor: float = 0.5) -> Tuple[float, float, float]:
	"""Return a lighter variant of ``color`` by blending with white."""

	return tuple(channel + (1.0 - channel) * factor for channel in color)


def plot_curves(
	curves: Sequence[CurveSpec],
	datasets: Sequence[pd.DataFrame],
	output_path: Optional[Path],
	step_col: str,
	title: Optional[str] = None,
) -> None:
	"""Plot the computed diversity trend lines."""

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
			"legend.fontsize": 7,
		}
	)

	fig, ax = plt.subplots(figsize=(10, 6))
	fig.patch.set_facecolor("#ffffff")
	ax.set_facecolor("#ffffff")

	for curve, data in zip(curves, datasets):
		if data.empty:
			continue

		primary_color = curve.color
		secondary_color = _lighten_color(curve.color, factor=0.6)

		ax.plot(
			data[step_col],
			data["IntDiv1"],
			color=primary_color,
			linewidth=1.5,
			linestyle="-",
			label=f"{curve.label} IntDiv1",
		)

		ax.plot(
			data[step_col],
			data["IntDiv2"],
			color=secondary_color,
			linewidth=1.5,
			linestyle=(0, (5, 2)),
			label=f"{curve.label} IntDiv2",
		)

	ax.set_xlabel("Step", fontsize=12)
	ax.set_ylabel("Mean diversity metric", fontsize=12)
	if title:
		ax.set_title(title, fontsize=14)

	ax.grid(True, linestyle=(0, (1, 2)), linewidth=0.8)
	ax.spines["top"].set_visible(False)
	ax.spines["right"].set_visible(False)

	legend = ax.legend(
		loc="upper left",
		bbox_to_anchor=(0.02, 0.98),
		frameon=False,
		borderaxespad=0.0,
		markerscale=0.9,
	)
	if legend:
		for text in legend.get_texts():
			text.set_color("#2a2d34")
	fig.tight_layout()

	if output_path:
		output_path.parent.mkdir(parents=True, exist_ok=True)
		fig.savefig(output_path, dpi=300)
	else:
		plt.show()


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Plot per-step diversity metrics (IntDiv1/IntDiv2)")
	parser.add_argument(
		"--config",
		type=Path,
		default=Path("./configs/div_per_step.json"),
		help="Path to JSON or YAML config file",
	)
	parser.add_argument(
		"--output",
		type=Path,
		default=Path("./rl_runs/diversity/diversity_per_step.png"),
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
		"--smiles-column",
		type=str,
		default="SMILES",
		help="Name of the column containing SMILES strings (default: SMILES)",
	)
	parser.add_argument(
		"--fingerprint-radius",
		type=int,
		default=2,
		help="Morgan fingerprint radius (default: 2)",
	)
	parser.add_argument(
		"--fingerprint-bits",
		type=int,
		default=2048,
		help="Morgan fingerprint bit length (default: 2048)",
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
		result = compute_diversity_per_step(
			df,
			step_start=curve.step_start,
			step_end=curve.step_end,
			step_col=args.step_column,
			smiles_col=args.smiles_column,
			radius=args.fingerprint_radius,
			n_bits=args.fingerprint_bits,
		)
		datasets.append(result)

	plot_curves(
		curves=curves,
		datasets=datasets,
		output_path=args.output,
		step_col=args.step_column,
		title=args.title,
	)


if __name__ == "__main__":
	main()
