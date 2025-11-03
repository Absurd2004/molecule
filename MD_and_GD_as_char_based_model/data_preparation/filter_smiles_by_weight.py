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
	count: int = 100,
	smiles_column: str = "SMILES",
) -> Path:
	"""Filter molecules with the smallest molecular weights from ``input_csv``.

	The function computes molecular weight for each SMILES entry, selects the ``count``
	rows with the smallest weights (ignoring invalid SMILES), and writes them to a
	CSV file named ``output_name`` in the same directory as ``input_csv``.
	"""

	input_path = Path(input_csv).expanduser().resolve()
	if not input_path.is_file():
		raise FileNotFoundError(f"Input CSV not found: {input_path}")
	if count <= 0:
		raise ValueError("'count' must be a positive integer")

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
	selected = rows[:count]
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
		description="Filter a CSV file for the n molecules with the smallest molecular weights",
	)
	parser.add_argument("--csv_path", default="",help="Path to the CSV file to filter")
	parser.add_argument(
		"--output-name",
		default="filter_by_weight.csv",
		help="Name of the output CSV file (default: filter_by_weight.csv)",
	)
	parser.add_argument(
		"--count",
		type=int,
		default=100,
		help="Number of molecules with smallest weights to keep (default: 100)",
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
		count=args.count,
		smiles_column=args.smiles_column,
	)
	print(f"Filtered rows written to {output_path}")


if __name__ == "__main__":
	main()
