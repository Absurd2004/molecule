from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs


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


if __name__ == "__main__":
	# Example usage – replace with any SMILES string you'd like to inspect.
	print_smiles_charge(
		"CC1=CC=C(N(C2=CC=C(C)C=C2)C2=CC=C(C3=CC=C(/C=C/C4=CC=[N+](CCCCCCCC[N+]5=CC=C(/C=C/C6=CC=C(C7=CC=C(N(C8=CC=C(C)C=C8)C8=CC=C(C)C=C8)C=C7)S6)C=C5)C=C4)S3)C=C2)C=C1"
	)
	print_smiles_similarity("O1c2ccccc2N(c2c(C)c(C)c([*])c(C)c2C)c2ccccc21", "c1cccc2c1N(c1c(C)c(C)c([*])c(C)c1C)c1ccccc1O2")
