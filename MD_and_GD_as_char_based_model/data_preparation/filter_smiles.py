from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable, Sequence


def _is_close_to_target(value: str, target: float, tolerance: float) -> bool:
	"""Check whether ``value`` (stringified) represents a float close to ``target``."""
	try:
		return abs(float(value) - target) <= tolerance
	except (TypeError, ValueError):
		return False


def filter_symmetry_score(
	input_csv: str | Path,
	*,
	output_name: str = "filter.csv",
	target_value: float = 1.0,
	tolerance: float = 1e-6,
) -> Path:
	"""Filter rows whose ``symmetry_score`` matches ``target_value`` and write to ``filter.csv``.

	The output file is created in the same directory as ``input_csv``.
	"""
	input_path = Path(input_csv).expanduser().resolve()
	if not input_path.is_file():
		raise FileNotFoundError(f"Input CSV not found: {input_path}")

	output_path = input_path.parent / output_name

	with input_path.open(newline="", encoding="utf-8") as handle:
		reader = csv.DictReader(handle)
		fieldnames: Sequence[str] | None = reader.fieldnames
		if not fieldnames:
			raise ValueError("Input CSV is missing a header row.")
		if "symmetry_score" not in fieldnames:
			raise ValueError("Column 'symmetry_score' not found in input CSV header.")

		matched_rows = [row for row in reader if _is_close_to_target(row.get("symmetry_score", ""), target_value, tolerance)]

	with output_path.open("w", newline="", encoding="utf-8") as handle:
		writer = csv.DictWriter(handle, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerows(matched_rows)

	return output_path


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Filter a CSV file for rows where symmetry_score == 1 and write filter.csv",
	)
	parser.add_argument("csv_path", help="Path to the CSV file to filter")
	parser.add_argument(
		"--output-name",
		default="filter.csv",
		help="Name of the output CSV file (default: filter.csv)",
	)
	parser.add_argument(
		"--target-value",
		type=float,
		default=1.0,
		help="Target symmetry_score value to filter for (default: 1.0)",
	)
	parser.add_argument(
		"--tolerance",
		type=float,
		default=1e-6,
		help="Tolerance when comparing floating-point symmetry_score values (default: 1e-6)",
	)
	return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
	args = _parse_args(argv)
	output_path = filter_symmetry_score(
		args.csv_path,
		output_name=args.output_name,
		target_value=args.target_value,
		tolerance=args.tolerance,
	)
	print(f"Filtered rows written to {output_path}")


if __name__ == "__main__":
	main()
