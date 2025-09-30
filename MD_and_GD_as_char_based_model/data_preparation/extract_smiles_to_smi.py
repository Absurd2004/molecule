#!/usr/bin/env python3
"""Extract SMILES strings from a CSV file into a plain .smi file."""

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read a CSV file, pull the 'smiles' column, and write it to a .smi file.")
    parser.add_argument(
        "--input", "-i", default="./data/test_data_smiles_qed_sa.csv",
        help="Path to the input CSV file containing a 'smiles' column.")
    parser.add_argument(
        "--output", "-o", default="./valid/valid.smi",
        help="Path to the output .smi file that will store SMILES strings.")
    parser.add_argument(
        "--column", "-c", default="smiles",
        help="Column name in the CSV that contains SMILES strings. [default: smiles]")
    return parser.parse_args()


def extract_smiles(input_csv: Path, output_smi: Path, column_name: str) -> int:
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    output_smi.parent.mkdir(parents=True, exist_ok=True)

    extracted = 0
    with input_csv.open("r", encoding="utf-8", newline="") as csv_file, \
            output_smi.open("w", encoding="utf-8", newline="") as smi_file:
        reader = csv.DictReader(csv_file)
        if column_name not in reader.fieldnames:
            raise KeyError(
                f"Column '{column_name}' not found in CSV header: {reader.fieldnames}")

        for row in reader:
            smiles = (row.get(column_name) or "").strip()
            if not smiles:
                continue
            smi_file.write(f"{smiles}\n")
            extracted += 1

    return extracted


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input).resolve()
    output_smi = Path(args.output).resolve()

    count = extract_smiles(input_csv, output_smi, args.column)
    print(f"Wrote {count} SMILES to {output_smi}")


if __name__ == "__main__":
    main()
