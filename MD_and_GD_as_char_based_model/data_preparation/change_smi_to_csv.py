from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description=(
			"Convert a .smi file (one SMILES per line) into a CSV file with a 'SMILES' column."
		)
	)
	parser.add_argument(
		"smi_path",
		type=Path,
		help="Path to the input .smi file containing SMILES strings (one per line).",
	)
	parser.add_argument(
		"--output",
		"-o",
		type=Path,
		help=(
			"Optional output CSV path. Defaults to the same directory as the input file "
			"with the same stem and a .csv extension."
		),
	)
	parser.add_argument(
		"--encoding",
		default="utf-8",
		help="File encoding used to read the input .smi file (default: utf-8).",
	)
	parser.add_argument(
		"--skip-empty",
		action="store_true",
		help="Skip blank lines when reading the .smi file (default behavior keeps them).",
	)
	return parser.parse_args()


def load_smiles(smi_path: Path, encoding: str, skip_empty: bool) -> list[str]:
	if not smi_path.exists():
		raise FileNotFoundError(f"SMI file not found: {smi_path}")

	with smi_path.open("r", encoding=encoding) as handle:
		lines = [line.strip() for line in handle]

	if skip_empty:
		lines = [line for line in lines if line]

	return lines


def determine_output_path(input_path: Path, explicit_output: Path | None) -> Path:
	if explicit_output is not None:
		if explicit_output.is_dir():
			return explicit_output / f"{input_path.stem}.csv"
		return explicit_output

	return input_path.with_suffix(".csv")


def convert_smi_to_csv(smi_path: Path, output_path: Path, encoding: str, skip_empty: bool) -> Path:
	smiles = load_smiles(smi_path, encoding=encoding, skip_empty=skip_empty)

	if not smiles:
		raise ValueError(
			"No SMILES strings were read from the input file. "
			"Use --skip-empty to remove blank lines or verify the file contents."
		)

	output_path.parent.mkdir(parents=True, exist_ok=True)
	with output_path.open("w", encoding=encoding, newline="") as csv_file:
		writer = csv.writer(csv_file)
		writer.writerow(["SMILES"])
		for smiles_str in smiles:
			writer.writerow([smiles_str])

	return output_path


def main() -> None:
	args = parse_args()
	output_path = determine_output_path(args.smi_path, args.output)

	result_path = convert_smi_to_csv(
		args.smi_path,
		output_path,
		encoding=args.encoding,
		skip_empty=args.skip_empty,
	)

	print(f"Converted '{args.smi_path}' to CSV: {result_path.resolve()}")


if __name__ == "__main__":
	main()
