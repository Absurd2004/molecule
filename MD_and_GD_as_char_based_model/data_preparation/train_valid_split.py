#!/usr/bin/env python
#  coding=utf-8

"""Split sliced molecules into train/valid sets based on scaffold Murcko cores."""

import argparse
import os
import re
from collections import Counter, OrderedDict
from multiprocessing import get_context
from typing import Dict, Iterable, Iterator, Optional, Set, Tuple

try:  # pylint: disable=wrong-import-position
	from tqdm import tqdm  # type: ignore
except ImportError:  # pragma: no cover
	tqdm = None  # type: ignore

try:  # pylint: disable=wrong-import-position
	from rdkit import Chem  # type: ignore
	from rdkit.Chem.Scaffolds import MurckoScaffold  # type: ignore
except ImportError as exc:  # pragma: no cover
	raise ImportError(
		"rdkit is required for train_valid_split. Please install rdkit before running this script."
	) from exc


ATTACHMENT_REGEX = re.compile(r"\[\*(?::\d+)?\]")

WORKER_SEED_MURCKOS: Set[str] = set()


def count_attachment_sites(scaffold_smiles: str) -> int:
	"""Count attachment points in a sliced scaffold SMILES string."""

	if not scaffold_smiles:
		return 0
	return len(ATTACHMENT_REGEX.findall(scaffold_smiles))


def smiles_to_murcko(scaffold_smiles: str) -> Optional[str]:
	"""Convert a scaffold SMILES string to its Murcko scaffold SMILES."""

	if not scaffold_smiles:
		return None

	mol = Chem.MolFromSmiles(scaffold_smiles, sanitize=True)
	if mol is None:
		return None

	murcko_mol = MurckoScaffold.GetScaffoldForMol(mol)
	if murcko_mol is None:
		return None

	murcko_smiles = Chem.MolToSmiles(murcko_mol, isomericSmiles=True)
	return murcko_smiles or None


def _ensure_newline(raw_line: str) -> str:
	return raw_line if raw_line.endswith("\n") else raw_line + "\n"


def select_seed_rows(
	input_path: str,
	attachment_counts: Iterable[int],
) -> Dict[int, Dict[str, str]]:
	"""Select one seed row per attachment count."""

	seeds: Dict[int, Dict[str, str]] = OrderedDict()
	target_counts: Set[int] = set(attachment_counts)

	with open(input_path, "r", encoding="utf-8") as input_file:
		for line_number, raw_line in enumerate(input_file, start=1):
			line = raw_line.rstrip("\n")
			if not line:
				continue

			fields = line.split("\t")
			if len(fields) < 3:
				continue

			scaffold = fields[0]
			attachment_count = count_attachment_sites(scaffold)

			if attachment_count not in target_counts:
				continue

			if attachment_count in seeds:
				continue

			murcko_smiles = smiles_to_murcko(scaffold)
			if murcko_smiles is None:
				continue

			seeds[attachment_count] = {
				"line": line,
				"attachments": str(attachment_count),
				"murcko": murcko_smiles,
				"line_number": str(line_number),
			}

			if len(seeds) == len(target_counts):
				break

	return seeds


def _init_worker(seed_murckos: Iterable[str]) -> None:
	global WORKER_SEED_MURCKOS
	WORKER_SEED_MURCKOS = set(seed_murckos)


def _classify_line(item: Tuple[int, str]) -> Tuple[int, str, Optional[int], str]:
	index, raw_line = item
	line = raw_line.rstrip("\n")
	if not line:
		return index, "skip", None, raw_line

	fields = line.split("\t")
	if len(fields) < 3:
		return index, "skip", None, raw_line

	scaffold = fields[0]
	murcko_smiles = smiles_to_murcko(scaffold)
	if murcko_smiles is None:
		return index, "invalid", None, raw_line

	attachments = count_attachment_sites(scaffold)
	split = "valid" if murcko_smiles in WORKER_SEED_MURCKOS else "train"
	return index, split, attachments, raw_line


def _count_lines(path: str) -> int:
	with open(path, "r", encoding="utf-8") as file_obj:
		return sum(1 for _ in file_obj)


def write_seeds(seed_path: str, seeds: Dict[int, Dict[str, str]]) -> None:
	seed_dir = os.path.dirname(seed_path)
	if seed_dir:
		os.makedirs(seed_dir, exist_ok=True)
	with open(seed_path, "w", encoding="utf-8") as seed_file:
		seed_file.write("attachment_count\tmurcko_scaffold\tline_number\toriginal_line\n")
		for attachments, meta in seeds.items():
			seed_file.write(
				f"{attachments}\t{meta['murcko']}\t{meta['line_number']}\t{meta['line']}\n"
			)


def write_stats(stats_path: str, stats: Dict[str, object], seeds: Dict[int, Dict[str, str]]) -> None:
	stats_dir = os.path.dirname(stats_path)
	if stats_dir:
		os.makedirs(stats_dir, exist_ok=True)
	with open(stats_path, "w", encoding="utf-8") as stats_file:
		for key, value in stats.items():
			if isinstance(value, dict):
				serialized = ",".join(f"{k}:{v}" for k, v in value.items())
				stats_file.write(f"{key}\t{serialized}\n")
			else:
				stats_file.write(f"{key}\t{value}\n")
		stats_file.write(
			"seed_attachment_counts\t{}\n".format(
				",".join(str(count) for count in sorted(seeds.keys()))
			)
		)


def process_dataset(
	input_path: str,
	train_output: str,
	valid_output: str,
	seeds: Dict[int, Dict[str, str]],
	num_workers: int,
) -> Dict[str, int]:
	train_dir = os.path.dirname(train_output)
	valid_dir = os.path.dirname(valid_output)
	if train_dir:
		os.makedirs(train_dir, exist_ok=True)
	if valid_dir:
		os.makedirs(valid_dir, exist_ok=True)

	total_lines = None
	if tqdm is not None:
		try:
			total_lines = _count_lines(input_path)
		except OSError:
			total_lines = None

	ctx = get_context("spawn")
	seed_murckos = [meta["murcko"] for meta in seeds.values()]

	stats_counter: Counter = Counter()
	stats_counter["total_lines"] = 0
	stats_counter["train_lines"] = 0
	stats_counter["valid_lines"] = 0
	stats_counter["invalid_lines"] = 0
	stats_counter["skipped_lines"] = 0

	train_attachment_hist: Counter = Counter()
	valid_attachment_hist: Counter = Counter()

	with open(train_output, "w", encoding="utf-8") as train_file, \
			open(valid_output, "w", encoding="utf-8") as valid_file:
		progress = (
			tqdm(desc="Assigning splits", unit="line", total=total_lines)
			if tqdm is not None
			else None
		)

		with open(input_path, "r", encoding="utf-8") as input_file:
			iterator: Iterator[Tuple[int, str]] = ((idx, line) for idx, line in enumerate(input_file))

			with ctx.Pool(processes=max(1, num_workers), initializer=_init_worker, initargs=(seed_murckos,)) as pool:
				for _, split, attachments, raw_line in pool.imap(_classify_line, iterator, chunksize=256):
					stats_counter["total_lines"] += 1

					if split == "train":
						train_file.write(_ensure_newline(raw_line))
						stats_counter["train_lines"] += 1
						if attachments is not None:
							train_attachment_hist[attachments] += 1
					elif split == "valid":
						valid_file.write(_ensure_newline(raw_line))
						stats_counter["valid_lines"] += 1
						if attachments is not None:
							valid_attachment_hist[attachments] += 1
					elif split == "invalid":
						stats_counter["invalid_lines"] += 1
					else:
						stats_counter["skipped_lines"] += 1

					if progress is not None:
						progress.update(1)

		if progress is not None:
			progress.close()

	stats: Dict[str, object] = {
		"total_lines": stats_counter["total_lines"],
		"train_lines": stats_counter["train_lines"],
		"valid_lines": stats_counter["valid_lines"],
		"invalid_lines": stats_counter["invalid_lines"],
		"skipped_lines": stats_counter["skipped_lines"],
		"train_attachment_hist": dict(sorted(train_attachment_hist.items())),
		"valid_attachment_hist": dict(sorted(valid_attachment_hist.items())),
	}

	return stats


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Split dataset into train/valid by Murcko scaffold seeds.")
	parser.add_argument(
		"--input-tsv",
		"-i",
		default="./data/valid/sliced_smiles.tsv",
		help="Path to the input TSV file with scaffold, decorations, and original molecule.",
	)
	parser.add_argument(
		"--train-output",
		"-tr",
		default="./data/valid/train.tsv",
		help="Path to the output TSV file for the training subset.",
	)
	parser.add_argument(
		"--valid-output",
		"-va",
		default="./data/valid/valid.tsv",
		help="Path to the output TSV file for the validation subset.",
	)
	parser.add_argument(
		"--seed-output",
		"-so",
		default="./data/valid/seeds.tsv",
		help="Path to the TSV file that will contain the selected seed rows.",
	)
	parser.add_argument(
		"--stats-output",
		"-st",
		default="./data/valid/split_stats.txt",
		help="Path to the text file that will contain split statistics.",
	)
	parser.add_argument(
		"--min-sites",
		type=int,
		default=1,
		help="Minimum number of attachment sites to consider for seed selection (inclusive).",
	)
	parser.add_argument(
		"--max-sites",
		type=int,
		default=4,
		help="Maximum number of attachment sites to consider for seed selection (inclusive).",
	)
	parser.add_argument(
		"--num-workers",
		"-p",
		type=int,
		default=max(1, os.cpu_count() or 1),
		help="Number of worker processes to use for Murcko computation.",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()

	if args.min_sites < 0 or args.max_sites < 0:
		raise ValueError("Attachment site bounds must be non-negative integers.")

	if args.max_sites < args.min_sites:
		raise ValueError("max_sites must be greater than or equal to min_sites.")

	attachment_counts = range(args.min_sites, args.max_sites + 1)
	seeds = select_seed_rows(args.input_tsv, attachment_counts)
	if not seeds:
		raise RuntimeError("No valid seeds found within the specified attachment site range.")

	missing_counts = [count for count in attachment_counts if count not in seeds]
	if missing_counts:
		print(
			"Warning: no valid seeds found for attachment counts: {}".format(
				", ".join(str(count) for count in missing_counts)
			)
		)

	write_seeds(args.seed_output, seeds)

	stats = process_dataset(
		args.input_tsv,
		args.train_output,
		args.valid_output,
		seeds,
		args.num_workers,
	)

	write_stats(args.stats_output, stats, seeds)


if __name__ == "__main__":
	main()

