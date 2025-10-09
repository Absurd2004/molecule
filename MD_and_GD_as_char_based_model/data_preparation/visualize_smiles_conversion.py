#!/usr/bin/env python
# coding=utf-8
"""Utility script to visualize original vs. converted SMILES molecules."""

import argparse
from pathlib import Path
from typing import Iterable, Optional, Tuple

from rdkit.Chem import Draw  # type: ignore

import utils.chem as uc  # pylint: disable=import-error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render 2D molecule depictions for original SMILES and their "
            "RDKit-converted counterparts to help manual comparison."
        )
    )
    parser.add_argument(
        "--input-smiles-path",
        "-i",
        default="./data/train/our_smiles.smi",
        help="Path to the input SMILES file (defaults to slice_multi.py configuration).",
    )
    parser.add_argument(
        "--output-folder",
        "-o",
        default="./data/train/our_smiles_visual",
        help="Directory where comparison images will be written.",
    )
    parser.add_argument(
        "--max-molecules",
        "-m",
        type=int,
        default=50,
        help="Maximum number of molecules to process (0 = process all).",
    )
    parser.add_argument(
        "--smiles-variant",
        "-v",
        default="canonical",
        help="Variant parameter forwarded to utils.chem.to_smiles (e.g. canonical, random).",
    )
    parser.add_argument(
        "--image-format",
        choices=("png", "svg"),
        default="png",
        help="Image format for the rendered comparison (PNG or SVG).",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=300,
        help="Side length (in pixels) for each sub-image in the comparison grid.",
    )
    return parser.parse_args()


def iter_smiles(path: str) -> Iterable[str]:
    for smi in uc.read_smi_file(path, ignore_invalid=True):
        if smi:
            yield smi


def to_mol_pair(smiles: str, variant: str) -> Tuple[Optional[str], Tuple[Optional[object], Optional[object]]]:
    mol = uc.to_mol(smiles)
    if mol is None:
        return None, (None, None)

    converted_smiles = uc.to_smiles(mol, variant=variant)
    print(f"Original: {smiles} | Converted ({variant}): {converted_smiles}")
    if not converted_smiles:
        return None, (None, None)

    converted_mol = uc.to_mol(converted_smiles)

    if converted_mol is None:
        return None, (None, None)
    print("success")

    return converted_smiles, (mol, converted_mol)


def save_image(
    mol_pair,
    smiles_pair: Tuple[str, str],
    output_path: Path,
    image_format: str,
    image_size: int,
) -> None:
    legends = [f"Original: {smiles_pair[0]}", f"Converted: {smiles_pair[1]}"]
    sub_img_size = (image_size, image_size)

    if image_format == "svg":
        svg = Draw.MolsToGridImage(
            list(mol_pair),
            molsPerRow=2,
            legends=legends,
            useSVG=True,
            subImgSize=sub_img_size,
        )
        output_path.write_text(str(svg), encoding="utf-8")
    else:
        image = Draw.MolsToGridImage(
            list(mol_pair),
            molsPerRow=2,
            legends=legends,
            useSVG=False,
            subImgSize=sub_img_size,
        )
        image.save(str(output_path))


def main() -> None:
    args = parse_args()

    input_path = Path(args.input_smiles_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input SMILES file not found: {input_path}")

    output_dir = Path(args.output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)

    max_count = max(0, args.max_molecules)
    processed = 0
    skipped = 0

    for idx, original_smiles in enumerate(iter_smiles(str(input_path))):
        if max_count and processed >= max_count:
            break

        converted_smiles, mol_pair = to_mol_pair(original_smiles, args.smiles_variant)
        if converted_smiles is None or mol_pair[0] is None or mol_pair[1] is None:
            skipped += 1
            continue

        file_stem = f"mol_{idx:05d}"
        file_path = output_dir / f"{file_stem}.{args.image_format}"
        save_image(
            (mol_pair[0], mol_pair[1]),
            (original_smiles, converted_smiles),
            file_path,
            args.image_format,
            args.image_size,
        )
        processed += 1

    print(
        f"Done. Generated {processed} comparison images at {output_dir}. Skipped {skipped} molecules due to conversion issues."
    )


if __name__ == "__main__":
    main()
