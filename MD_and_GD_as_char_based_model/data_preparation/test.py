"""Utility script to rebuild molecules from scaffold and pipe-separated decorations."""

from pathlib import Path

from rdkit import Chem  # type: ignore

import utils.chem as uc
from utils.scaffold import join_joined_attachments, to_smiles


def process_dataset(input_path: Path, success_path: Path, failure_path: Path) -> None:
	"""Read scaffold/decoration pairs, attempt reconstruction, and store outcomes."""

	successes = []
	failures = []

	mismatch_count = 0

	with input_path.open("r", encoding="utf-8") as handle:
		for line_number, raw_line in enumerate(handle, start=1):
			line = raw_line.strip()
			if not line:
				continue

			parts = line.split("\t")
			if len(parts) < 2:
				failures.append((line_number, line, "missing decoration column", "", ""))
				continue

			scaffold_smi, decorations_smi = parts[0], parts[1]
			original_smi = parts[2] if len(parts) > 2 else ""

			if original_smi:
				original_mol = uc.to_mol(original_smi)
				if original_mol is None:
					failures.append((line_number, line, "invalid original smiles", "", ""))
					continue
			else:
				original_mol = None

			try:
				mol = join_joined_attachments(scaffold_smi, decorations_smi)
			except Exception as exc:  # defensive: capture unexpected RDKit issues
				failures.append((line_number, line, f"exception: {exc}", "", ""))
				continue

			if mol is None:
				failures.append((line_number, line, "join returned None", "", ""))
				continue

			if original_mol is not None:
				try:
					joined_canonical = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
					original_canonical = Chem.MolToSmiles(original_mol, isomericSmiles=True, canonical=True)
				except Exception as exc:  # pragma: no cover - defensive
					failures.append((line_number, line, f"canonicalization error: {exc}", "", ""))
					continue

				if joined_canonical != original_canonical:
					mismatch_count += 1
					failures.append((line_number, line, "reconstruction mismatch", joined_canonical, original_canonical))
					continue
			else:
				joined_canonical = ""
				original_canonical = ""

			joined_smi = to_smiles(mol)
			successes.append((line_number, scaffold_smi, decorations_smi, joined_smi, original_smi))

	with success_path.open("w", encoding="utf-8") as success_file:
		success_file.write("line\tscaffold\tdecorations\tjoined_smiles\toriginal_smiles\n")
		for line_number, scaffold_smi, decorations_smi, joined_smi, original_smi in successes:
			success_file.write(f"{line_number}\t{scaffold_smi}\t{decorations_smi}\t{joined_smi}\t{original_smi}\n")

	with failure_path.open("w", encoding="utf-8") as failure_file:
		failure_file.write("line\traw_entry\treason\tjoined_canonical\toriginal_canonical\n")
		for line_number, raw_entry, reason, joined_canonical, original_canonical in failures:
			failure_file.write(f"{line_number}\t{raw_entry}\t{reason}\t{joined_canonical}\t{original_canonical}\n")

	print(f"Processed {len(successes) + len(failures)} lines.")
	print(f"Successful joins: {len(successes)}")
	print(f"Failed joins: {len(failures)}")

	print(f"Mismatched reconstructions: {mismatch_count}")
	print(f"Success log: {success_path}")
	print(f"Failure log: {failure_path}")


def main() -> None:
	base_dir = Path(__file__).resolve().parent
	input_path = base_dir.parent / "data" / "train"/"randomized_smiles_our_smiles" / "000.smi"
	success_path = base_dir / "joined_success.smi"
	failure_path = base_dir / "joined_failures.tsv"

	process_dataset(input_path, success_path, failure_path)


if __name__ == "__main__":
	main()
