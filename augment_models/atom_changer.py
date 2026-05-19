from rdkit import Chem
from rdkit.Chem import rdmolops
from rdkit.Chem import rdchem
import random
import augment_models.functionalizer as functionalizer

ELEMENT_SWAP_RULES = {
    "F": ["Cl", "Br", "I", "C(F)(F)F"], 
    "Cl": ["F", "Br", "I", "C(F)(F)F"],
    "Br": ["F", "Cl", "I", "C(F)(F)F"],
    "I": ["F", "Cl", "Br", "C(F)(F)F"],
    "O": ["Cl", "Br", "I", "NC"], 
    "O_link": ["NC"],
    "N_primary": ["O"],   
    "N_secondary": ["OC"],
    "N_tertiary": ["C"],
    "N_heterocyclic": ["O", "C"],
    "O_heterocyclic": ["NC", "C"],
    "C_aliphatic": ["O", "NC", "[Si](C)(C)"], 
}

def get_atoms_by_distance(mol, center_mapnums, min_dist=2):
    dists = Chem.GetDistanceMatrix(mol)
    mapnum_to_idx = {a.GetAtomMapNum(): a.GetIdx() for a in mol.GetAtoms()}
    center_idxs = [mapnum_to_idx[m] for m in center_mapnums if m in mapnum_to_idx]

    selected_mapnums = set()
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        mapnum = atom.GetAtomMapNum()
        if mapnum == 0:
            continue
        if any(dists[idx][cidx] >= min_dist for cidx in center_idxs):
            selected_mapnums.add(mapnum)
    return selected_mapnums

def find_swappable_atoms(mol, target_symbols, candidate_mapnums):
    swappables = set()
    for atom in mol.GetAtoms():
        if atom.GetAtomMapNum() in candidate_mapnums and atom.GetSymbol() in target_symbols:
            swappables.add(atom.GetAtomMapNum())
    return swappables

def get_atom_class(atom):
    if atom.GetAtomicNum() == 7 and not atom.GetIsAromatic():
        hcount = atom.GetTotalNumHs()
        if atom.GetDegree() == 1 and hcount >= 1:
            return "N_primary"
        elif atom.GetDegree() == 2:
            return "N_secondary"
        elif atom.GetDegree() >= 3:
            return "N_tertiary"
        else:
            return "N"
    elif atom.GetAtomicNum() == 7 and atom.GetIsAromatic():
        hcount = atom.GetTotalNumHs()
        if hcount == 1:
            return "N_heterocyclic"
        else:
            return "N"
    elif atom.GetAtomicNum() == 6 and not atom.GetIsAromatic():
        carbon_neighbors = sum(1 for n in atom.GetNeighbors() if n.GetAtomicNum() == 6)
        if not any(n.GetAtomicNum() in (7, 8) for n in atom.GetNeighbors()) and carbon_neighbors >= 2:
            return "C_aliphatic"
        else:
            return None
    elif atom.GetAtomicNum() == 6 and atom.GetIsAromatic():
        return "O_heterocyclic"
    elif atom.GetAtomicNum() == 8 and not atom.GetIsAromatic():
        hcount = atom.GetTotalNumHs()
        if hcount == 1:
            return "O"
        elif hcount == 0:
            return "O_link"
    elif atom.GetAtomicNum() in [9, 17, 35, 53] and not atom.GetIsAromatic():
        return atom.GetSymbol()
    return None

def smart_swap_atoms(mol_i, mol_o, mapnum_set, element_swap_rules=ELEMENT_SWAP_RULES):
    rw_mol_i = Chem.RWMol(mol_i)
    rw_mol_o = Chem.RWMol(mol_o)

    candidate_atoms = []
    for atom in rw_mol_i.GetAtoms():
        mapnum = atom.GetAtomMapNum()
        if mapnum in mapnum_set:
            atom_class = get_atom_class(atom)
            if atom_class and atom_class in element_swap_rules:
                candidate_atoms.append((mapnum, atom_class))

    if not candidate_atoms:
        return None, None

    mapnum, atom_class = random.choice(candidate_atoms)
    replacements = element_swap_rules[atom_class].copy()
    random.shuffle(replacements)

    atom_idx_i = atom_idx_o = None
    for atom in rw_mol_i.GetAtoms():
        if atom.GetAtomMapNum() == mapnum:
            atom_idx_i = atom.GetIdx()
            break
    for atom in rw_mol_o.GetAtoms():
        if atom.GetAtomMapNum() == mapnum:
            atom_idx_o = atom.GetIdx()
            break
    if atom_idx_i is None or atom_idx_o is None:
        return None, None

    attempt_probs = [1.0, 0.5, 0.5, 0.5, 0.5]
    max_attempts = min(len(attempt_probs), len(replacements))
    used_replacements = set()

    for i in range(max_attempts):
        prob = attempt_probs[i]
        available = [r for r in replacements if r not in used_replacements]
        if not available:
            break
        replacement = random.choice(available)
        used_replacements.add(replacement)

        if random.random() > prob:
            continue

        try:
            frag = Chem.MolFromSmiles(replacement)
            if frag is None or frag.GetNumAtoms() == 0:
                continue

            neighbors_i = [(nbr.GetIdx(), rw_mol_i.GetBondBetweenAtoms(atom_idx_i, nbr.GetIdx()).GetBondType())
                           for nbr in rw_mol_i.GetAtomWithIdx(atom_idx_i).GetNeighbors()]
            neighbors_o = [(nbr.GetIdx(), rw_mol_o.GetBondBetweenAtoms(atom_idx_o, nbr.GetIdx()).GetBondType())
                           for nbr in rw_mol_o.GetAtomWithIdx(atom_idx_o).GetNeighbors()]

            combined_i = Chem.CombineMols(rw_mol_i, frag)
            combined_o = Chem.CombineMols(rw_mol_o, frag)
            combo_rw_i = Chem.RWMol(combined_i)
            combo_rw_o = Chem.RWMol(combined_o)

            frag_offset_i = rw_mol_i.GetNumAtoms()
            frag_offset_o = rw_mol_o.GetNumAtoms()
            new_atom_i = frag_offset_i
            new_atom_o = frag_offset_o

            for n_idx, bond_type in neighbors_i:
                combo_rw_i.AddBond(n_idx, new_atom_i, bond_type)
            for n_idx, bond_type in neighbors_o:
                combo_rw_o.AddBond(n_idx, new_atom_o, bond_type)

            combo_rw_i.GetAtomWithIdx(new_atom_i).SetAtomMapNum(mapnum)
            combo_rw_o.GetAtomWithIdx(new_atom_o).SetAtomMapNum(mapnum)
            combo_rw_i.RemoveAtom(atom_idx_i)
            combo_rw_o.RemoveAtom(atom_idx_o)
            new_mol_i = combo_rw_i.GetMol()
            Chem.SanitizeMol(new_mol_i)
            new_mol_o = combo_rw_o.GetMol()
            Chem.SanitizeMol(new_mol_o)

            return new_mol_i, new_mol_o

        except Exception as e:
            continue

    return None, None
    
def modify_atoms(mol_i, mol_o, reaction_center_mapnums):
    is_conj = functionalizer.is_conjugated(mol_i, reaction_center_mapnums)
    dist = 5 if is_conj else 3
    far_mapnums = get_atoms_by_distance(mol_i, reaction_center_mapnums, dist)

    return smart_swap_atoms(mol_i, mol_o, far_mapnums)