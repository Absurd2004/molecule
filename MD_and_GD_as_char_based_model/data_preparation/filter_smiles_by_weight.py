from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable, List, Sequence

try:
	from rdkit import Chem  # type: ignore[import]
	from rdkit.Chem import Descriptors  # type: ignore[import]
except ModuleNotFoundError as exc:  # pragma: no cover - import guard
	raise ModuleNotFoundError(
		"RDKit is required to compute molecular weights. Install it via 'conda install -c rdkit rdkit'."
	) from exc


def _compute_mol_weight(smiles: str) -> float:
	"""Return the exact molecular weight for a SMILES string or ``float('inf')`` if invalid."""

	mol = Chem.MolFromSmiles(smiles)
	if mol is None:
		return float("inf")
	return Descriptors.ExactMolWt(mol)


def filter_smiles_by_weight(
	input_csv: str | Path,
	*,
	output_name: str = "filter_by_weight.csv",
	rank_start: int = 1,
	rank_end: int = 100,
	smiles_column: str = "SMILES",
) -> Path:
	"""Filter molecules ranked by molecular weight within ``[rank_start, rank_end]``.

	Molecular weights are computed for each SMILES string. Rows are sorted by
	ascending weight and the subset between ``rank_start`` and ``rank_end`` (inclusive,
	one-based ranking) is written to the output CSV. Invalid SMILES are discarded.
	"""

	input_path = Path(input_csv).expanduser().resolve()
	if not input_path.is_file():
		raise FileNotFoundError(f"Input CSV not found: {input_path}")
	if rank_start <= 0 or rank_end <= 0:
		raise ValueError("'rank_start' and 'rank_end' must be positive integers")
	if rank_start > rank_end:
		raise ValueError("'rank_start' must be less than or equal to 'rank_end'")

	with input_path.open(newline="", encoding="utf-8") as handle:
		reader = csv.DictReader(handle)
		fieldnames: Sequence[str] | None = reader.fieldnames
		if not fieldnames:
			raise ValueError("Input CSV is missing a header row.")
		if smiles_column not in fieldnames:
			raise ValueError(f"Column '{smiles_column}' not found in input CSV header.")

		rows: List[dict] = []
		for row in reader:
			smiles = row.get(smiles_column, "")
			weight = _compute_mol_weight(smiles)
			if weight == float("inf"):
				continue
			row["__weight__"] = weight
			rows.append(row)

	if not rows:
		raise ValueError("No valid SMILES found in the input CSV.")

	rows.sort(key=lambda row: row["__weight__"])  # type: ignore[arg-type]

	start_index = rank_start - 1
	end_index = min(rank_end, len(rows))
	if start_index >= len(rows):
		raise ValueError(
			f"rank_start ({rank_start}) exceeds the number of valid SMILES ({len(rows)})"
		)

	selected = rows[start_index:end_index]
	for row in selected:
		row.pop("__weight__", None)

	output_path = input_path.parent / output_name
	with output_path.open("w", newline="", encoding="utf-8") as handle:
		writer = csv.DictWriter(handle, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerows(selected)

	return output_path


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Filter a CSV file for molecules ranked by ascending molecular weight",
	)
	parser.add_argument(
		"csv_path",
		help="Path to the CSV file to filter",
	)
	parser.add_argument(
		"--output-name",
		default="filter_by_weight.csv",
		help="Name of the output CSV file (default: filter_by_weight.csv)",
	)
	parser.add_argument(
		"--rank-start",
		type=int,
		default=1,
		help="One-based starting rank (inclusive) for selected molecules (default: 1)",
	)
	parser.add_argument(
		"--rank-end",
		type=int,
		default=100,
		help="One-based ending rank (inclusive) for selected molecules (default: 100)",
	)
	parser.add_argument(
		"--smiles-column",
		default="SMILES",
		help="Name of the column containing SMILES strings (default: SMILES)",
	)
	return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
	args = _parse_args(argv)
	output_path = filter_smiles_by_weight(
		args.csv_path,
		output_name=args.output_name,
		rank_start=args.rank_start,
		rank_end=args.rank_end,
		smiles_column=args.smiles_column,
	)
	print(f"Filtered rows written to {output_path}")


if __name__ == "__main__":
	main()
