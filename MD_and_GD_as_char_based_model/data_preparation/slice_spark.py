#!/usr/bin/env python
#  coding=utf-8


import argparse
import json
import os
import sys
import time
from pathlib import Path
from threading import Event, Thread

try:
    from tqdm import tqdm  # type: ignore
except ImportError:  # pragma: no cover
    tqdm = None


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

for path_entry in (SCRIPT_DIR, PROJECT_ROOT):
    str_entry = str(path_entry)
    if str_entry not in sys.path:
        sys.path.insert(0, str_entry)

pythonpath_parts = os.environ.get("PYTHONPATH", "").split(os.pathsep) if os.environ.get("PYTHONPATH") else []
for path_entry in (SCRIPT_DIR, PROJECT_ROOT):
    str_entry = str(path_entry)
    if str_entry not in pythonpath_parts:
        pythonpath_parts.insert(0, str_entry)
if pythonpath_parts:
    os.environ["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)


import pyspark.sql as ps  # pylint: disable=import-error
import pyspark.sql.functions as psf  # pylint: disable=import-error

import models.actions as ma  # pylint: disable=import-error
import utils.log as ul  # pylint: disable=import-error
import utils.chem as uc  # pylint: disable=import-error
import utils.scaffold as usc  # pylint: disable=import-error
import utils.spark as us  # pylint: disable=import-error


class SliceDB(ma.Action):

    def __init__(self, input_path, output_path, enumerator, max_cuts, partitions, logger=None):
        ma.Action.__init__(self, logger)

        self.input_path = input_path
        self.output_path = output_path
        self.enumerator = enumerator
        self.max_cuts = max_cuts
        self.partitions = partitions
        self.slice_count = 0

    def run(self):
        molecule_counter = SC.accumulator(0) if tqdm else None
        input_rdd = SC.textFile(self.input_path).repartition(self.partitions)
        input_cached = False
        total_molecules = None
        monitor_event = None
        monitor_thread = None

        if tqdm:
            input_rdd = input_rdd.cache()
            input_cached = True
            total_molecules = input_rdd.count()
            if total_molecules and molecule_counter is not None:
                monitor_event = Event()
                monitor_thread = Thread(
                    target=_monitor_enumeration_progress,
                    args=(molecule_counter, total_molecules, monitor_event),
                )
                monitor_thread.daemon = True
                monitor_thread.start()

        def _enumerate(row, max_cuts=self.max_cuts, enumerator=self.enumerator, counter=molecule_counter):
            if counter is not None:
                counter.add(1)
            fields = row.split("\t")
            smiles = fields[0]
            mol = uc.to_mol(smiles)
            out_rows = []
            if mol:
                for cuts in range(1, max_cuts + 1):
                    for sliced_mol in enumerator.enumerate(mol, cuts=cuts):
                        scaff_smi, dec_map = sliced_mol.to_smiles()
                        dec_smis = [dec_map[num] for num in sorted(dec_map)]
                        out_rows.append(ps.Row(
                            scaffold=scaff_smi,
                            decorations=dec_smis,
                            smiles=uc.to_smiles(mol),
                            cuts=cuts
                        ))
            return out_rows

        try:
            enumeration_df = SPARK.createDataFrame(
                input_rdd.flatMap(_enumerate)
            ).groupBy("scaffold", "decorations") \
                .agg(psf.first("cuts").alias("cuts"), psf.first("smiles").alias("smiles")) \
                .persist()

            self.slice_count = enumeration_df.count()
            self._log("info", "Obtained %d sliced molecules", self.slice_count)
        finally:
            if monitor_event:
                monitor_event.set()
            if monitor_thread:
                monitor_thread.join()
            if input_cached:
                input_rdd.unpersist(False)

        if self.output_path:
            enumeration_df.write.parquet(self.output_path)

        return enumeration_df


def parse_args():
    """Parses input arguments."""
    parser = argparse.ArgumentParser(description="Slices the molecules a given way.")
    parser.add_argument("--input-smiles-path", "-i",
                        help="Path to the input file with molecules in SMILES notation.", type=str, default="./data/merged_smiles.smi")
    parser.add_argument("--output-parquet-folder", "-o",
                        help="Path to the output Apache Parquet folder.", type=str, default="./data/sliced_parquet")
    parser.add_argument("--output-smiles-path", "-u",
                        help="Path to the output SMILES file.", type=str, default="./data/sliced_smiles.tsv")
    parser.add_argument("--max-cuts", "-c",
                        help="Maximum number of cuts to attempts for each molecule [DEFAULT: 4]", type=int, default=4)
    parser.add_argument("--slice-type", "-s",
                        help="Kind of slicing performed TYPES=(recap, hr) [DEFAULT: hr]", type=str, default="recap")
    parser.add_argument("--num-partitions", "--np",
                        help="Number of Spark partitions to use [DEFAULT: 1000]", type=int, default=1000)
    parser.add_argument("--conditions-file", "-f",
                        help="JSON file with the filtering conditions for the scaffolds and the decorations.", type=str, default="./condition/condition.json")

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
                              enumerator, args.max_cuts, args.num_partitions, LOG)
    slice_df = slice_db_action.run()

    if args.output_smiles_path:
        with open(args.output_smiles_path, "w+") as smiles_file:
            total_slices = getattr(slice_db_action, "slice_count", None)
            iterator = slice_df.rdd.map(_to_smiles_rows).toLocalIterator()
            if tqdm:
                iterator = tqdm(iterator, total=total_slices, desc="Writing slices", unit="slice")
            for row in iterator:
                smiles_file.write("{}\n".format(row))


def _monitor_enumeration_progress(counter, total, done_event):
    if not tqdm:
        return

    progress = tqdm(total=total, desc="Slicing molecules", unit="mol")
    last_value = 0
    try:
        while not done_event.is_set():
            current = counter.value
            if current > last_value:
                progress.update(current - last_value)
                last_value = current
            time.sleep(0.5)
        current = counter.value
        if current > last_value:
            progress.update(current - last_value)
    finally:
        progress.close()


def _configure_spark(builder):
    """Inject driver PYTHONPATH into executor environment."""
    executor_pythonpath = os.environ.get("PYTHONPATH", "")
    if executor_pythonpath:
        builder.config("spark.executorEnv.PYTHONPATH", executor_pythonpath)


LOG = ul.get_logger(name="slice_db")
SPARK, SC = us.SparkSessionSingleton.get("slice_db", params_func=_configure_spark)
if __name__ == "__main__":
    main()
