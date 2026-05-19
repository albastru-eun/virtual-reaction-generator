from rdkit import Chem
from rdkit.Chem import rdmolops

def get_atoms_by_distance(mol, center_mapnums, min_dist):
    center_idxs = [
        atom.GetIdx()
        for atom in mol.GetAtoms()
        if atom.GetAtomMapNum() in center_mapnums
    ]
    if not center_idxs:
        return set()

    dists = rdmolops.GetDistanceMatrix(mol)
    selected = set()

    for i, atom in enumerate(mol.GetAtoms()):
        if any(dists[i][c] >= min_dist for c in center_idxs):
            if atom.GetAtomMapNum() > 0:
                selected.add(atom.GetAtomMapNum())
    return selected