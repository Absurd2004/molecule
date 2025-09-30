#!/usr/bin/env python
#  coding=utf-8

import argparse
import json
import os
import random
from collections import OrderedDict
from multiprocessing import get_context
import heapq

try:  # pylint: disable=wrong-import-position
    from tqdm import tqdm  # type: ignore
except ImportError:  # pragma: no cover
    tqdm = None  # type: ignore

import models.actions as ma  # pylint: disable=import-error
import utils.log as ul  # pylint: disable=import-error
import utils.chem as uc  # pylint: disable=import-error
import utils.scaffold as usc  # pylint: disable=import-error

WORKER_ENUMERATOR = None
WORKER_MAX_CUTS = 0


def _init_worker(slice_type, scaffold_conditions, decoration_conditions, max_cuts):
    global WORKER_ENUMERATOR, WORKER_MAX_CUTS
    WORKER_ENUMERATOR = usc.SliceEnumerator(usc.SLICE_SMARTS[slice_type],
                                           scaffold_conditions,
                                           decoration_conditions)
    WORKER_MAX_CUTS = max(1, int(max_cuts))


def _process_line(item):
    if WORKER_ENUMERATOR is None:
        raise RuntimeError("Worker enumerator has not been initialized.")

    index, line = item
    line = line.strip()
    result = OrderedDict()

    if not line:
        return index, result

    fields = line.split("\t")
    if not fields or not fields[0]:
        return index, result

    smiles = fields[0]
    mol = uc.to_mol(smiles)
    if not mol:
        return index, result

    original_smiles = uc.to_smiles(mol)

    for cuts in range(2, WORKER_MAX_CUTS + 1):
        for sliced_mol in WORKER_ENUMERATOR.enumerate(mol, cuts=cuts):
            scaffold_smi, decoration_map = sliced_mol.to_smiles()
            decorations = tuple(
                decoration_map[num]
                for num in sorted(decoration_map)
            )

            key = (scaffold_smi, decorations)
            if key not in result:
                result[key] = {
                    "cuts": cuts,
                    "smiles": original_smiles
                }

    return index, result


def _count_lines(path):
    with open(path, "r") as input_file:
        return sum(1 for _ in input_file)


class SliceDB(ma.Action):

    def __init__(self, input_path, output_path, slice_type,
                 scaffold_conditions, decoration_conditions,
                 max_cuts, num_workers, sample_size=0, sample_seed=42, logger=None):
        ma.Action.__init__(self, logger)

        self.input_path = input_path
        self.output_path = output_path
        self.max_cuts = max_cuts
        self.num_workers = max(1, int(num_workers))
        if slice_type not in usc.SLICE_SMARTS:
            raise ValueError(f"Unknown slice type: {slice_type}")
        self.slice_type = slice_type
        self.scaffold_conditions = scaffold_conditions
        self.decoration_conditions = decoration_conditions
        self.sample_size = max(0, int(sample_size or 0))
        self.sample_seed = int(sample_seed if sample_seed is not None else 42)

    def run(self):
        slices = OrderedDict()
        print(f"start slicing with {self.num_workers} processes")

        selected_lines = None
        total_molecules = None

        if self.sample_size > 0:
            with open(self.input_path, "r") as input_file:
                all_lines = input_file.readlines()

            if not all_lines:
                print("Input file is empty; nothing to sample")
                return []

            if self.sample_size < len(all_lines):
                rng = random.Random(self.sample_seed)
                indices = sorted(rng.sample(range(len(all_lines)), self.sample_size))
                selected_lines = [all_lines[idx] for idx in indices]
                print(f"Sampling {len(selected_lines)} of {len(all_lines)} molecules for slicing")
            else:
                selected_lines = all_lines
                print(f"Requested sample size >= dataset; processing all {len(all_lines)} molecules")

            total_molecules = len(selected_lines)
        elif tqdm is not None:
            try:
                total_molecules = _count_lines(self.input_path)
            except OSError:
                total_molecules = None

        ctx = get_context("spawn")
        init_args = (self.slice_type, self.scaffold_conditions, self.decoration_conditions, self.max_cuts)

        progress = tqdm(desc="Slicing molecules", unit="mol", total=total_molecules) if tqdm is not None else None

        pending_results = []
        next_index = 0
        completed = 0

        chunk_size = max(1, self.num_workers * 4)

        with ctx.Pool(processes=self.num_workers, initializer=_init_worker, initargs=init_args) as pool:
            if selected_lines is not None:
                line_iterator = ((idx, line) for idx, line in enumerate(selected_lines))
                iterator = line_iterator
            else:
                input_file = open(self.input_path, "r")
                iterator = ((idx, line) for idx, line in enumerate(input_file))

            try:
                for index, partial in pool.imap_unordered(_process_line, iterator, chunksize=chunk_size):
                    heapq.heappush(pending_results, (index, partial))

                    while pending_results and pending_results[0][0] == next_index:
                        _, ordered_partial = heapq.heappop(pending_results)
                        if ordered_partial:
                            for key, meta in ordered_partial.items():
                                if key not in slices:
                                    slices[key] = meta
                        next_index += 1

                    if progress is not None:
                        progress.update(1)
                    else:
                        completed += 1
                        if completed % 10000 == 0:
                            print(f"Processed {completed} molecules...", flush=True)
            finally:
                if selected_lines is None:
                    input_file.close()

        while pending_results:
            _, ordered_partial = heapq.heappop(pending_results)
            if ordered_partial:
                for key, meta in ordered_partial.items():
                    if key not in slices:
                        slices[key] = meta
            next_index += 1

        if progress is not None:
            progress.close()
        else:
            print(f"Processed {completed} molecules in total.", flush=True)

        rows = [
            {
                "scaffold": scaffold,
                "decorations": list(decorations),
                "smiles": meta["smiles"],
                "cuts": meta["cuts"]
            }
            for (scaffold, decorations), meta in slices.items()
        ]

        self._log("info", "Obtained %d sliced molecules", len(rows))

        if self.output_path:
            try:
                import pandas as pd  # pylint: disable=import-error  # type: ignore
            except ImportError:  # pragma: no cover
                self._log("warning", "pandas is required to write parquet output; skipping parquet export.")
            else:
                pd.DataFrame(rows).to_parquet(self.output_path, index=False)

        return rows


def parse_args():
    """Parses input arguments."""
    parser = argparse.ArgumentParser(description="Slices the molecules a given way using multi-processing.")
    parser.add_argument("--input-smiles-path", "-i",
                        help="Path to the input file with molecules in SMILES notation.", type=str, default="./data/valid/valid.smi")
    parser.add_argument("--output-parquet-folder", "-o",
                        help="Path to the output Apache Parquet folder.", type=str, default="./data/valid/sliced_parquet_recap")
    parser.add_argument("--output-smiles-path", "-u",
                        help="Path to the output SMILES file.", type=str, default="./data/valid/sliced_smiles_recap.tsv")
    parser.add_argument("--max-cuts", "-c",
                        help="Maximum number of cuts to attempts for each molecule [DEFAULT: 4]", type=int, default=4)
    parser.add_argument("--slice-type", "-s",
                        help="Kind of slicing performed TYPES=(recap, hr) [DEFAULT: hr]", type=str, default="recap")
    parser.add_argument("--conditions-file", "-f",
                        help="JSON file with the filtering conditions for the scaffolds and the decorations.", type=str, default="./condition/condition.json")

    parser.add_argument("--sample-size", "-S",
                        help="Optional number of input molecules to randomly sample before slicing (0 = use all).", type=int, default=1000)
    parser.add_argument("--sample-seed",
                        help="Random seed used when sampling input molecules.", type=int, default=42)

    default_threads = max(1, (os.cpu_count() or 1))
    parser.add_argument("--num-threads", "-t", "--num-processes", "-p",
                        dest="num_threads",
                        help=f"Number of worker processes to use [DEFAULT: {default_threads}]", type=int, default=default_threads)

    return parser.parse_args()


def _to_smiles_rows(row):
    return "{}\t{}\t{}".format(row["scaffold"], ";".join(row["decorations"]), row["smiles"])


def main():
    """Main function."""
    args = parse_args()

    scaffold_conditions = None
    decoration_conditions = None
    if args.conditions_file:
        with open(args.conditions_file, "r") as json_file:
            data = json.load(json_file)
            if "scaffold" in data:
                scaffold_conditions = data["scaffold"]
            if "decoration" in data:
                decoration_conditions = data["decoration"]

    slice_db_action = SliceDB(args.input_smiles_path, args.output_parquet_folder,
                              args.slice_type, scaffold_conditions, decoration_conditions,
                              args.max_cuts, args.num_threads, args.sample_size, args.sample_seed, LOG)
    slice_rows = slice_db_action.run()

    if args.output_smiles_path:
        with open(args.output_smiles_path, "w+") as smiles_file:
            for row in slice_rows:
                smiles_file.write("{}\n".format(_to_smiles_rows(row)))


LOG = ul.get_logger(name="slice_db")
if __name__ == "__main__":
    main()
