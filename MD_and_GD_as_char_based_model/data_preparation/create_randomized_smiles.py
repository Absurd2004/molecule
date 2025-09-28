#!/usr/bin/env python

import argparse
import os

import utils.log as ul  # pylint: disable=import-error
import utils.chem as uc  # pylint: disable=import-error
import utils.scaffold as usc  # pylint: disable=import-error

try:
    from tqdm import tqdm  # type: ignore
except ImportError as exc:  # pragma: no cover - dependency guard
    raise ImportError("tqdm is required for progress reporting; install it via 'pip install tqdm'.") from exc


def parse_args():
    """Parses input arguments."""
    parser = argparse.ArgumentParser(
        description="Creates many sets with a given seed.")
    parser.add_argument("--input-smi-path", "-i",
                        help="Path to a SMILES file to convert with scaffolds and decorations.", type=str, default="./data/sliced_smiles.tsv")
    parser.add_argument("--output-smi-folder-path", "-o",
                        help="Path to a folder that will have the converted SMILES files.", type=str, default="./data/randomized_smiles_all")
    parser.add_argument("--num-files", "-n",
                        help="Number of SMILES files to create (numbered from 000 ...) [DEFAULT: 1]",
                        type=int, default=5)
    parser.add_argument("--decorator-type", "-d",
                        help="Type of training set, depending on the decorator TYPES=(single, multi) [DEFAULT: multi].",
                        type=str, default="single")

    return parser.parse_args()



def _to_sliced_mol(row):
    fields = row.strip().split("\t")
    if len(fields) < 2:
        raise ValueError("Row does not contain scaffold and decorations")

    scaffold_smi = fields[0]
    decoration_field = fields[1]

    scaffold_mol = uc.to_mol(scaffold_smi)
    if not scaffold_mol:
        raise ValueError("Invalid scaffold SMILES")

    decoration_smis = decoration_field.split(";") if decoration_field else []
    decorations = {}
    for idx, dec in enumerate(decoration_smis):
        dec_mol = uc.to_mol(dec)
        if not dec_mol:
            raise ValueError("Invalid decoration SMILES")
        decorations[idx] = dec_mol

    return usc.SlicedMol(scaffold_mol, decorations)


def _load_sliced_mols(input_path, logger):
    sliced_mols = []
    with open(input_path, "r") as input_file:
        line_iterator = tqdm(input_file, desc="Loading SMILES", unit="line", leave=False)
        for line_number, line in enumerate(line_iterator, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                sliced_mols.append(_to_sliced_mol(stripped))
            except ValueError as exc:  # pylint: disable=broad-exception-caught
                if logger:
                    logger.warning("Skipping line %d: %s", line_number, exc)
    return sliced_mols


def _format_training_set_row_multi(sliced_mol):
    scaff_smi, dec_smis = sliced_mol.to_smiles(variant="random")

    first_num = usc.get_first_attachment_point(scaff_smi)
    decoration_smi = dec_smis[first_num]

    return (usc.remove_attachment_point_numbers(scaff_smi), usc.remove_attachment_point_numbers(decoration_smi))


def _format_training_set_row_single(sliced_mol):
    scaff_smi, dec_smis = sliced_mol.to_smiles(variant="random")

    attachment_points = usc.get_attachment_points(scaff_smi)
    decorations = []
    for idx in attachment_points:
        decorations.append(usc.remove_attachment_point_numbers(dec_smis[idx]))
    return (usc.remove_attachment_point_numbers(scaff_smi), usc.ATTACHMENT_SEPARATOR_TOKEN.join(decorations))


def main():
    """Main function."""
    args = parse_args()

    sliced_mols = _load_sliced_mols(args.input_smi_path, LOG)

    if not sliced_mols:
        LOG.warning("No valid sliced molecules were loaded from %s", args.input_smi_path)
        return

    os.makedirs(args.output_smi_folder_path, exist_ok=True)

    if args.decorator_type == "single":
        format_func = _format_training_set_row_single
    elif args.decorator_type == "multi":
        format_func = _format_training_set_row_multi
    else:
        raise ValueError("Unsupported decorator type: {}".format(args.decorator_type))

    total_files = args.num_files
    total_smiles = len(sliced_mols)

    file_iterator = tqdm(range(total_files), total=total_files, desc="Files", unit="file")
    for i in file_iterator:
        output_path = os.path.join(args.output_smi_folder_path, "{:03d}.smi".format(i))
        with open(output_path, "w+") as out_file:
            smiles_iterator = tqdm(
                sliced_mols,
                total=total_smiles,
                desc="SMILES {:03d}".format(i),
                unit="mol",
                leave=False
            )
            for sliced_mol in smiles_iterator:
                scaff_smi, dec_smi = format_func(sliced_mol)
                out_file.write("{}\t{}\n".format(scaff_smi, dec_smi))


LOG = ul.get_logger("create_randomized_smiles")
if __name__ == "__main__":
    main()
