#!/usr/bin/env python

import argparse
import os
import random
from multiprocessing import get_context

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
                        help="Path to a SMILES file to convert with scaffolds and decorations.", type=str, default="./data/train/our_smiles_recap.tsv")
    parser.add_argument("--output-smi-folder-path", "-o",
                        help="Path to a folder that will have the converted SMILES files.", type=str, default="./data/train/randomized_smiles_our_smiles")
    parser.add_argument("--num-files", "-n",
                        help="Number of SMILES files to create (numbered from 000 ...) [DEFAULT: 1]",
                        type=int, default=5)
    parser.add_argument("--decorator-type", "-d",
                        help="Type of training set, depending on the decorator TYPES=(single, multi) [DEFAULT: multi].",
                        type=str, default="single")
    default_workers = max(1, os.cpu_count() or 1)
    parser.add_argument("--num-workers", "-w",
                        help="Number of worker processes to use for SMILES processing. [DEFAULT: {}]".format(default_workers),
                        type=int, default=default_workers)

    return parser.parse_args()



def _to_sliced_entry(row):
    fields = row.strip().split("\t")
    if len(fields) < 2:
        raise ValueError("Row does not contain scaffold and decorations")

    scaffold_smi = fields[0]
    decoration_field = fields[1]
    original_smi = fields[2] if len(fields) > 2 else ""

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

    return usc.SlicedMol(scaffold_mol, decorations), original_smi


def _load_smiles_lines(input_path):
    lines = []
    with open(input_path, "r") as input_file:
        line_iterator = tqdm(input_file, desc="Loading SMILES", unit="line", leave=False)
        for line_number, line in enumerate(line_iterator, 1):
            stripped = line.strip()
            if stripped:
                lines.append(stripped)
    return lines


def _format_training_set_row_multi(sliced_entry):
    sliced_mol, original_smi = sliced_entry
    scaff_smi, dec_smis = sliced_mol.to_smiles(variant="random")

    first_num = usc.get_first_attachment_point(scaff_smi)
    decoration_smi = dec_smis[first_num]

    return (
        usc.remove_attachment_point_numbers(scaff_smi),
        usc.remove_attachment_point_numbers(decoration_smi),
        original_smi,
    )


def _format_training_set_row_single(sliced_entry):
    sliced_mol, original_smi = sliced_entry
    scaff_smi, dec_smis = sliced_mol.to_smiles(variant="random")

    attachment_points = usc.get_attachment_points(scaff_smi)
    decorations = []
    for idx in attachment_points:
        decorations.append(usc.remove_attachment_point_numbers(dec_smis[idx]))
    return (
        usc.remove_attachment_point_numbers(scaff_smi),
        usc.ATTACHMENT_SEPARATOR_TOKEN.join(decorations),
        original_smi,
    )


FORMAT_FUNC = None


def _init_worker(decorator_type):
    global FORMAT_FUNC
    FORMAT_FUNC = _format_training_set_row_single if decorator_type == "single" else _format_training_set_row_multi
    random.seed(os.getpid())


def _validate_line(line):
    try:
        _ = _to_sliced_entry(line)
        return line
    except ValueError:
        return None


def _process_randomized_line(line):
    try:
        sliced_entry = _to_sliced_entry(line)
        scaff_smi, dec_smi, original_smi = FORMAT_FUNC(sliced_entry)
        return "{}\t{}\t{}".format(scaff_smi, dec_smi, original_smi)
    except ValueError:
        return None


def main():
    """Main function."""
    args = parse_args()

    raw_lines = _load_smiles_lines(args.input_smi_path)

    if not raw_lines:
        LOG.warning("No valid sliced molecules were loaded from %s", args.input_smi_path)
        return

    os.makedirs(args.output_smi_folder_path, exist_ok=True)

    if args.decorator_type not in {"single", "multi"}:
        raise ValueError("Unsupported decorator type: {}".format(args.decorator_type))

    ctx = get_context("spawn")
    num_workers = max(1, args.num_workers)
    chunk_size = max(1, len(raw_lines) // (num_workers * 4)) if len(raw_lines) >= num_workers else 1

    with ctx.Pool(processes=num_workers) as pool:
        validated = []
        for line in tqdm(pool.imap(_validate_line, raw_lines, chunksize=chunk_size), total=len(raw_lines), desc="Validating", unit="mol", leave=False):
            if line:
                validated.append(line)

    if not validated:
        LOG.warning("All SMILES failed validation for %s", args.input_smi_path)
        return


    invalid_count = len(raw_lines) - len(validated)
    if invalid_count:
        LOG.info("Filtered out %d invalid SMILES", invalid_count)
        print("Filtered out {} invalid SMILES".format(invalid_count))
    else:
        print("No invalid SMILES found")

    raw_lines = validated

    total_files = args.num_files
    total_smiles = len(raw_lines)
    chunk_size = max(1, len(raw_lines) // (num_workers * 4)) if len(raw_lines) >= num_workers else 1

    file_iterator = tqdm(range(total_files), total=total_files, desc="Files", unit="file")
    for i in file_iterator:
        output_path = os.path.join(args.output_smi_folder_path, "{:03d}.smi".format(i))
        with ctx.Pool(processes=num_workers, initializer=_init_worker, initargs=(args.decorator_type,)) as pool:
            results = pool.imap(_process_randomized_line, raw_lines, chunksize=chunk_size)
            with open(output_path, "w+") as out_file:
                for row in tqdm(results, total=total_smiles, desc="SMILES {:03d}".format(i), unit="mol", leave=False):
                    if row:
                        out_file.write(row + "\n")


LOG = ul.get_logger("create_randomized_smiles")
if __name__ == "__main__":
    main()
