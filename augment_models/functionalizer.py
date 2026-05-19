from rdkit import Chem
from rdkit.Chem import rdFMCS
from rdkit.Chem import rdmolops
from rdkit.Chem import EditableMol
from rdkit.Chem import RWMol
from rdkit.Chem import rdMolDescriptors
import random

def get_mcs_mol(smiles_list):
    mols = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
    res = rdFMCS.FindMCS(mols, timeout=10)
    mcs_mol = Chem.MolFromSmarts(res.smartsString)
    return mcs_mol

def extract_distant_fg(mol, mcs_mol, reaction_center_mapnums, dist):
    fg_mapnums = set()

    mapnum_to_idx = {atom.GetAtomMapNum(): atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomMapNum() > 0}
    center_idxs = [mapnum_to_idx[mn] for mn in reaction_center_mapnums if mn in mapnum_to_idx]

    if not center_idxs:
        return set()

    dmat = rdmolops.GetDistanceMatrix(mol)
    for i, atom in enumerate(mol.GetAtoms()):
        if atom.GetAtomMapNum() > 0:
            if any(1 <= dmat[i][c] <= dist for c in center_idxs):
                fg_mapnums.add(atom.GetAtomMapNum())
    return fg_mapnums

def is_conjugated(mol, reaction_center):
    ring_info = mol.GetRingInfo()
    atom_idx_map = {
        atom.GetAtomMapNum(): atom.GetIdx()
        for atom in mol.GetAtoms()
        if atom.GetAtomMapNum() > 0
    }

    center_idxs = [atom_idx_map[m] for m in reaction_center if m in atom_idx_map]

    for ring in ring_info.AtomRings():
        if any(idx in center_idxs for idx in ring):
            if all(mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring):
                return True
    return False

def get_valid_attachment(mol, exclude_idxs=None):
    if exclude_idxs is None:
        exclude_idxs = set()
    valid = []
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        if idx in exclude_idxs:
            continue
        atomic_num = atom.GetAtomicNum()
        degree = sum(1 for nbr in atom.GetNeighbors() if nbr.GetAtomicNum() != 1)
        is_aromatic = atom.GetIsAromatic()

        if atomic_num in {5, 6, 7, 8, 14, 16}:
            max_valence = atom.GetExplicitValence()
            
            if is_aromatic and atomic_num == 6:
                max_valence = 3
            elif is_aromatic and atomic_num in (7, 8, 16): #do not use N, O, S in aromatic group
                max_valence = 0
                
            if degree < max_valence:
                valid.append(idx)
        else:
            pass
    return valid

def get_atom_valence(atom):
    bond_valence = sum([b.GetBondTypeAsDouble() for b in atom.GetBonds()])
    h_count = atom.GetTotalNumHs()
    return bond_valence + h_count

def add_fg_to_mol(mol, fg_mol, attach_idx):
    mol = Chem.AddHs(mol)
    fg_mol = Chem.AddHs(fg_mol)
    rw_fg = RWMol(fg_mol)
    fg_attach_idx = None
    
    for atom in rw_fg.GetAtoms():
        if atom.GetAtomicNum() == 92:
            fg_attach_idx = atom.GetIdx()
            break
            
    for atom in rw_fg.GetAtoms():
        if atom.GetAtomicNum() in (0, 92):
            atom.SetAtomicNum(6)
            atom.SetFormalCharge(0)
            atom.SetIsAromatic(False)
            atom.SetNumExplicitHs(3)
            fg_attach_idx = atom.GetIdx()

    if fg_attach_idx is not None:
        rw_fg.RemoveAtom(fg_attach_idx)
        fg_mol = rw_fg.GetMol()
    else:
        for atom in rw_fg.GetAtoms():
            if atom.GetAtomicNum() > 1:
                fg_attach_idx = atom.GetIdx()
                break
        if fg_attach_idx is not None:
            rw_fg.RemoveAtom(fg_attach_idx)
            fg_mol = rw_fg.GetMol()
        else:
            return None

    rw_mol = RWMol(mol)
    
    atom = rw_mol.GetAtomWithIdx(attach_idx)

    excluded_smarts = {
        'OH': '[OX2H]',             # Hydroxy
        'NH2': '[NX3H2]',           # Primary amine
        'NHR': '[NX3H][#6]',        # Secondary amine
        'NR2': '[NX3]([#6])[#6]',   # Tertiary amine
        'F': '[F]', 'Cl': '[Cl]', 'Br': '[Br]', 'I': '[I]'
    }

    excluded_smarts_2 = {
        'OH': '[OX2H]',             # Hydroxy
        'NH2': '[NX3H2]',           # Primary amine
        'NHR': '[NX3H][#6]',        # Secondary amine
        'NR2': '[NX3]([#6])[#6]',   # Tertiary amine
    }
    
    excluded_patterns = [(name, Chem.MolFromSmarts(s)) for name, s in excluded_smarts.items()]
    excluded_patterns_2 = [(name, Chem.MolFromSmarts(s)) for name, s in excluded_smarts_2.items()]
    
    for neighbor in atom.GetNeighbors():
        if neighbor.GetAtomicNum() in (7, 8):
            for name, patt in excluded_patterns:
                if fg_mol.HasSubstructMatch(patt):
                    # print(f"Rejected: FG contains excluded group {name}")
                    return None
        if neighbor.GetAtomicNum() in (9, 17, 35, 53):
            for name, patt in excluded_patterns_2:
                if fg_mol.HasSubstructMatch(patt):
                    # print(f"Rejected: FG contains excluded group {name}")
                    return None
                    
    if atom.GetAtomicNum() in (7, 8):  # N or O
        for name, patt in excluded_patterns:
            if fg_mol.HasSubstructMatch(patt):
                # print(f"Rejected: FG contains excluded group {name}")
                return None
            
    for neighbor in atom.GetNeighbors():
        if neighbor.GetAtomicNum() == 1:
            rw_mol.RemoveAtom(neighbor.GetIdx())
            break  # remove only one H

    mol = rw_mol.GetMol()
    combo = Chem.CombineMols(mol, fg_mol)
    rw_combo = RWMol(combo)

    offset = mol.GetNumAtoms()
      
    rw_combo.AddBond(attach_idx, offset, Chem.BondType.SINGLE)

    new_mol = rw_combo.GetMol()
    new_mol = Chem.AddHs(new_mol)
    
    try:
        Chem.SanitizeMol(new_mol)
    except Chem.rdchem.KekulizeException:
        Chem.SanitizeMol(new_mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)

    rw_new = RWMol(new_mol)
    standard_valence = {'N': 3, 'O': 2}

    for atom in rw_new.GetAtoms():
        sym = atom.GetSymbol()
        if sym in standard_valence:
            current_valence = atom.GetExplicitValence() + atom.GetImplicitValence()
            needed_H = standard_valence[sym] - current_valence
            if needed_H > 0:
                atom_idx = atom.GetIdx()
                for _ in range(needed_H):
                    new_H = Chem.Atom('H')
                    new_H_idx = rw_new.AddAtom(new_H)
                    rw_new.AddBond(atom_idx, new_H_idx, Chem.BondType.SINGLE)
    new_mol = rw_new.GetMol()
    Chem.SanitizeMol(new_mol)

    for atom in new_mol.GetAtoms():
        sym = atom.GetSymbol()
        if sym in ('U', '*'):
            return None
    
        if atom.GetIsAromatic():
            if len(atom.GetNeighbors()) > 3:
                #print(f"[Warning] Aromatic atom {sym} has too many neighbors at atom {atom.GetIdx()}")
                return None
        else:
            valence = atom.GetExplicitValence() + atom.GetImplicitValence()
            max_valence = {
                'C': 4, 'N': 3, 'O': 2, 'F': 1, 'Cl': 1, 'Br': 1, 'I': 1,
            }
            if sym in max_valence and valence > max_valence[sym]:
                #print(f"[Warning] Valence exceeded for {sym} at atom {atom.GetIdx()}: {valence}")
                return None

    new_mol = Chem.RemoveHs(new_mol)
    
    return new_mol