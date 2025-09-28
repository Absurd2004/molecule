#!/usr/bin/env python
#  coding=utf-8


import argparse
import json
from collections import OrderedDict

try:  # pylint: disable=wrong-import-position
    from tqdm import tqdm  # type: ignore
except ImportError:  # pragma: no cover
    tqdm = None  # type: ignore

import models.actions as ma  # pylint: disable=import-error
import utils.log as ul  # pylint: disable=import-error
import utils.chem as uc  # pylint: disable=import-error
import utils.scaffold as usc  # pylint: disable=import-error


class SliceDB(ma.Action):

    def __init__(self, input_path, output_path, enumerator, max_cuts, logger=None):
        ma.Action.__init__(self, logger)

        self.input_path = input_path
        self.output_path = output_path
        self.enumerator = enumerator
        self.max_cuts = max_cuts

    def run(self):
        slices = OrderedDict()

        with open(self.input_path, "r") as input_file:
            line_iterator = input_file
            if tqdm is not None:
                line_iterator = tqdm(input_file, desc="Slicing molecules", unit="mol")

            for line in line_iterator:
                fields = line.strip().split("\t")
                if not fields or not fields[0]:
                    continue

                smiles = fields[0]
                mol = uc.to_mol(smiles)
                if not mol:
                    continue

                original_smiles = uc.to_smiles(mol)

                for cuts in range(1, self.max_cuts + 1):
                    for sliced_mol in self.enumerator.enumerate(mol, cuts=cuts):
                        scaffold_smi, decoration_map = sliced_mol.to_smiles()
                        decorations = tuple(
                            decoration_map[num]
                            for num in sorted(decoration_map)
                        )

                        key = (scaffold_smi, decorations)
                        if key not in slices:
                            slices[key] = {
                                "cuts": cuts,
                                "smiles": original_smiles
                            }

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
    parser = argparse.ArgumentParser(description="Slices the molecules a given way.")
    parser.add_argument("--input-smiles-path", "-i",
                        help="Path to the input file with molecules in SMILES notation.", type=str, default="./data/chembl_recap.smi")
    parser.add_argument("--output-parquet-folder", "-o",
                        help="Path to the output Apache Parquet folder.", type=str,default=None)
    parser.add_argument("--output-smiles-path", "-u",
                        help="Path to the output SMILES file.", type=str,default="./data/tmp_smiles_1.tsv")
    parser.add_argument("--max-cuts", "-c",
                        help="Maximum number of cuts to attempts for each molecule [DEFAULT: 4]", type=int, default=4)
    parser.add_argument("--slice-type", "-s",
                        help="Kind of slicing performed TYPES=(recap, hr) [DEFAULT: hr]", type=str, default="recap")

    parser.add_argument("--conditions-file", "-f",
                        help="JSON file with the filtering conditions for the scaffolds and the decorations.", type=str,default="./condition/condition.json")

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

    enumerator = usc.SliceEnumerator(usc.SLICE_SMARTS[args.slice_type], scaffold_conditions, decoration_conditions)
    slice_db_action = SliceDB(args.input_smiles_path, args.output_parquet_folder,
                              enumerator, args.max_cuts, LOG)
    slice_rows = slice_db_action.run()

    if args.output_smiles_path:
        with open(args.output_smiles_path, "w+") as smiles_file:
            for row in slice_rows:
                smiles_file.write("{}\n".format(_to_smiles_rows(row)))


LOG = ul.get_logger(name="slice_db")
if __name__ == "__main__":
    main()
