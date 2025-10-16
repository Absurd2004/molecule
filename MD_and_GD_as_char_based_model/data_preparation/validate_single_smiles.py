#!/usr/bin/env python
# coding=utf-8
"""Check whether a SMILES string is valid and save its 2D depiction."""

import argparse
import sys
from pathlib import Path

from rdkit.Chem import Draw  # type: ignore

import utils.chem as uc  # pylint: disable=import-error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a SMILES string and store a 2D depiction image."
    )
    parser.add_argument(
        "--smiles",
        "-s",
        default="[*:0]c1c2nc(c([*:1])c3ccc([nH]3)c([*:2])c3nc(c([*:3])c4ccc1[nH]4)C=C3)C=C2",
        
        help="SMILES string to validate.",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="./data/sample/scaffold.png",
        help="Path to the output image file (PNG or SVG).",
    )
    parser.add_argument(
        "--image-size",
        "-i",
        type=int,
        default=400,
        help="Side length (pixels) for the generated image.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    smiles = (args.smiles or "").strip()
    if not smiles:
        print("No SMILES string provided.", file=sys.stderr)
        return 1

    mol = uc.to_mol(smiles)
    if mol is None:
        print(f"Invalid SMILES string: {smiles}", file=sys.stderr)
        return 1

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix.lower() == ".svg":
        svg = Draw.MolToImage(mol, size=(args.image_size, args.image_size), useSVG=True)
        output_path.write_text(str(svg), encoding="utf-8")
    else:
        image = Draw.MolToImage(mol, size=(args.image_size, args.image_size))
        image.save(str(output_path))

    print(f"Valid SMILES. Image saved to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
