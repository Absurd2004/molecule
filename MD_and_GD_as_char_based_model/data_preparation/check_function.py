from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from typing import Iterable


def print_smiles_charge(smiles: str) -> None:
	"""Print the total formal charge for the molecule represented by ``smiles``.

	Args:
		smiles: A SMILES string describing the molecule.

	If the SMILES string cannot be parsed, a message indicating the error is printed
	instead of a charge value.
	"""
	mol = Chem.MolFromSmiles(smiles)
	if mol is None:
		print(f"Invalid SMILES: {smiles}")
		return

	charge = Chem.GetFormalCharge(mol)
	print(f"SMILES: {smiles} -> Total formal charge: {charge}")


def print_smiles_similarity(smiles_a: str, smiles_b: str) -> None:
	"""Print the Tanimoto similarity between two molecules represented by SMILES.

	Args:
		smiles_a: First SMILES string.
		smiles_b: Second SMILES string.

	If either SMILES cannot be parsed, a warning is printed and the function returns
	without computing similarity.
	"""
	mol_a = Chem.MolFromSmiles(smiles_a)
	mol_b = Chem.MolFromSmiles(smiles_b)
	if mol_a is None:
		print(f"Invalid SMILES: {smiles_a}")
		return
	if mol_b is None:
		print(f"Invalid SMILES: {smiles_b}")
		return

	fp_a = AllChem.GetMorganFingerprintAsBitVect(mol_a, radius=2, nBits=2048)
	fp_b = AllChem.GetMorganFingerprintAsBitVect(mol_b, radius=2, nBits=2048)
	similarity = DataStructs.TanimotoSimilarity(fp_a, fp_b)
	print(
		f"SMILES A: {smiles_a}\nSMILES B: {smiles_b}\nTanimoto similarity: {similarity:.4f}"
	)


def print_two_pair_match(smiles_list: Iterable[str]) -> None:
	"""Classify if four SMILES are two identical pairs or all identical molecules.

	Args:
		smiles_list: Iterable containing exactly four SMILES strings.

	The function clusters RDKit Morgan fingerprints by requiring a Tanimoto
	similarity of 1.0 (within numerical tolerance). It reports whether the four
	structures form:

	- "All identical"  : one cluster containing all four molecules
	- "Two matching pairs": two clusters with two molecules each
	- "Not matching" : any other arrangement (e.g., 3+1, four distinct, etc.)
	"""
	items = list(smiles_list)
	if len(items) != 4:
		print("Expected exactly four SMILES strings.")
		return

	fingerprints = []
	for smi in items:
		mol = Chem.MolFromSmiles(smi)
		if mol is None:
			print(f"Invalid SMILES encountered: {smi}")
			return
		fingerprints.append(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048))

	def _is_identical(a, b, tol: float = 1e-9) -> bool:
		return DataStructs.TanimotoSimilarity(a, b) >= 1.0 - tol

	clusters: list[list[int]] = []
	for idx, fp in enumerate(fingerprints):
		assigned = False
		for cluster in clusters:
			if _is_identical(fp, fingerprints[cluster[0]]):
				cluster.append(idx)
				assigned = True
				break
		if not assigned:
			clusters.append([idx])

	cluster_sizes = sorted(len(cluster) for cluster in clusters)
	if cluster_sizes == [4]:
		print("All identical")
	elif cluster_sizes == [2, 2]:
		print("Two matching pairs")
	else:
		print("Not matching")


def print_smiles_ring_count(smiles: str) -> None:
	"""Print the number of rings present in the molecule described by ``smiles``.

	Args:
		smiles: A SMILES string describing the molecule.

	If the SMILES string cannot be parsed, a message indicating the error is printed
	instead of a ring count value.
	"""
	mol = Chem.MolFromSmiles(smiles)
	if mol is None:
		print(f"Invalid SMILES: {smiles}")
		return

	# RDKit reports the size of the smallest set of smallest rings (SSSR).
	ring_count = mol.GetRingInfo().NumRings()
	print(f"SMILES: {smiles} -> Ring count: {ring_count}")


if __name__ == "__main__":
	# Example usage – replace with any SMILES string you'd like to inspect.
	print_smiles_charge(
		"[*]c1cc[n+](C)cc1"
	)
	print_smiles_similarity("O1c2ccccc2N(c2c(C)c(C)c([*])c(C)c2C)c2ccccc21", "c1cccc2c1N(c1c(C)c(C)c([*])c(C)c1C)c1ccccc1O2")
	print_two_pair_match([
		"[*]c1cc[n+](C)cc1","[*]c1cc[n+](C)cc1","[*]c1cc[n+](C)cc1","[*]c1cc[n+](C)cc1"
	])
	print_smiles_ring_count("[*]c1cc[n+](C)c2ccccc21")
