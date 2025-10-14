from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Sequence, Tuple

from rdkit import Chem
from rdkit.Chem import Draw


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Render molecules from a CSV column containing SMILES strings."
	)
	parser.add_argument(
		"csv",
		type=Path,
		nargs="?",
		default=Path("./rl_runs/20251014-113340/diversity_memory.csv"),
		help="Path to the input CSV file (default: ./rl_runs/20251014-113340/diversity_memory.csv).",
	)
	parser.add_argument(
		"--column",
		default="SMILES",
		help="Column name containing SMILES strings (default: SMILES).",
	)
	parser.add_argument(
		"--limit",
		type=int,
		default=0,
		help="Maximum number of molecules to render (0 for all, default: 64).",
	)
	parser.add_argument(
		"--size",
		type=int,
		default=300,
		help="Canvas size (pixels) for each molecule image (default: 300).",
	)
	return parser.parse_args()


def read_smiles(csv_path: Path, column: str) -> List[Tuple[int, str]]:
	with csv_path.open("r", encoding="utf-8", newline="") as handle:
		reader = csv.DictReader(handle)
		if reader.fieldnames is None or column not in reader.fieldnames:
			raise SystemExit(f"Column '{column}' not present in {csv_path}")
		smiles: List[Tuple[int, str]] = []
		for line_no, row in enumerate(reader, start=2):
			value = (row.get(column) or "").strip()
			if value:
				smiles.append((line_no, value))
	if not smiles:
		raise SystemExit(f"No SMILES strings found under column '{column}'")
	return smiles


def to_molecules(smiles: Sequence[Tuple[int, str]]) -> List[Tuple[int, Chem.Mol, str]]:
	molecules: List[Tuple[int, Chem.Mol, str]] = []
	for line_no, smi in smiles:
		mol = Chem.MolFromSmiles(smi)
		if mol is None:
			print(f"Skipping invalid SMILES on line {line_no}: {smi}")
			continue
		molecules.append((line_no, mol, smi))
	if not molecules:
		raise SystemExit("No valid SMILES strings could be parsed into molecules.")
	return molecules


def render_individual_images(
	molecules: Sequence[Tuple[int, Chem.Mol, str]],
	csv_path: Path,
	size: int,
) -> Path:
	images_dir = csv_path.parent / "images"
	images_dir.mkdir(parents=True, exist_ok=True)
	for line_no, mol, legend in molecules:
		image = Draw.MolToImage(mol, size=(size, size), legend=legend)
		file_path = images_dir / f"{line_no}.png"
		image.save(file_path)
	return images_dir


def main() -> None:
	args = parse_args()
	smiles = read_smiles(args.csv, args.column)
	if args.limit > 0:
		smiles = smiles[: args.limit]
	molecules = to_molecules(smiles)
	images_dir = render_individual_images(molecules, args.csv, args.size)
	print(f"Saved {len(molecules)} molecule images to {images_dir}")


if __name__ == "__main__":
	main()
