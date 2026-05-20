import random
import copy
from collections import defaultdict
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdmolops
from itertools import product

MAPNUM_OFFSET = 250

def strain_computation(mol, energy_cutoff_per_atom=5): #to prevent making strained molecules
    try:
        mol = Chem.AddHs(mol)
        
        if AllChem.EmbedMolecule(mol, randomSeed=42) != 0:
            return True

        AllChem.UFFOptimizeMolecule(mol, maxIters=200)
        ff = AllChem.UFFGetMoleculeForceField(mol)
        energy = ff.CalcEnergy()

        num_atoms = mol.GetNumAtoms()
        energy_per_atom = energy / num_atoms

        return energy_per_atom > energy_cutoff_per_atom

    except Exception:
        return True

def extract_submol(mol, atom_indices, exclude_mapnum_pairs=None):
    if exclude_mapnum_pairs is None:
        exclude_mapnum_pairs = set()

    rw_mol = Chem.RWMol()
    idx_map = {}
    mapnum_map = {}
    try:
        for atom_idx in atom_indices:
            atom = mol.GetAtomWithIdx(atom_idx)
            new_atom = Chem.Atom(atom.GetAtomicNum())
            new_atom.SetIsAromatic(atom.GetIsAromatic())
            new_atom.SetFormalCharge(atom.GetFormalCharge())
            new_atom.SetChiralTag(atom.GetChiralTag())
            new_atom.SetNumExplicitHs(atom.GetNumExplicitHs())
            new_atom.SetAtomMapNum(atom.GetAtomMapNum())
            new_atom.SetNoImplicit(True)
    
            new_idx = rw_mol.AddAtom(new_atom)
            idx_map[atom_idx] = new_idx
            mapnum_map[atom_idx] = atom.GetAtomMapNum()
    
        for bond in mol.GetBonds():
            a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if a1 in idx_map and a2 in idx_map:
                m1, m2 = mapnum_map[a1], mapnum_map[a2]
                if (m1, m2) in exclude_mapnum_pairs or (m2, m1) in exclude_mapnum_pairs:
                    continue
                rw_mol.AddBond(idx_map[a1], idx_map[a2], bond.GetBondType())
    except Exception as e:
        #print(f'[Error] Extraction failed: {e}')
        return None

    return rw_mol.GetMol()

def get_atoms_by_mapnums(mol, mapnums):
    return [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomMapNum() in mapnums]

def remap_fragment_atoms(fragment, offset=MAPNUM_OFFSET):
    for atom in fragment.GetAtoms():
        old_map = atom.GetAtomMapNum()
        if old_map:
            atom.SetAtomMapNum(old_map + offset)
    return fragment

def replace_fragment(target_mol, frag_to_remove_maps, frag_to_add_mol, old_subs, new_subs, num_bond, offset=MAPNUM_OFFSET):
    atoms_to_remove = [
        atom.GetIdx() for atom in target_mol.GetAtoms()
        if atom.GetAtomMapNum() in frag_to_remove_maps and atom.GetAtomMapNum() not in old_subs
    ]
    
    target_edit = Chem.EditableMol(target_mol)
    for idx in sorted(atoms_to_remove, reverse=True):
        target_edit.RemoveAtom(idx)

    mol_without_frag = target_edit.GetMol()
    combined = Chem.CombineMols(mol_without_frag, frag_to_add_mol)
    rw_mol = Chem.RWMol(combined)

    old_maps = [atom.GetIdx() for atom in rw_mol.GetAtoms() if atom.GetAtomMapNum() in old_subs]
    new_maps = [atom.GetIdx() for atom in rw_mol.GetAtoms() if atom.GetAtomMapNum() >= offset]
    bonds_added = 0
    used = set()
    
    for a_idx, b_idx in product(old_maps, new_maps):
        
        if a_idx is None or b_idx is None:
            #print(f"  Skipped: old_map {a_idx} or new_map {b_idx} not found")
            continue
            
        if a_idx in used or b_idx in used:
            continue

        atom_a = rw_mol.GetAtomWithIdx(a_idx)
        atom_b = rw_mol.GetAtomWithIdx(b_idx)
        
        if atom_a.GetAtomicNum() in (6, 7, 8, 15, 16) and atom_b.GetAtomicNum() in (6, 7):
            if rw_mol.GetBondBetweenAtoms(a_idx, b_idx) is None:
                try:
                    test_mol = Chem.RWMol(Chem.Mol(rw_mol))
                    test_mol.AddBond(a_idx, b_idx, Chem.rdchem.BondType.SINGLE)
                    Chem.SanitizeMol(test_mol.GetMol())
                    rw_mol.AddBond(a_idx, b_idx, Chem.rdchem.BondType.SINGLE)
                except:
                    try:
                        test_mol = Chem.RWMol(Chem.Mol(rw_mol))
                        test_mol.AddBond(a_idx, b_idx, Chem.rdchem.BondType.AROMATIC)
                        Chem.SanitizeMol(test_mol.GetMol())
                        rw_mol.AddBond(a_idx, b_idx, Chem.rdchem.BondType.AROMATIC)
                    except:
                        try:
                            test_mol = Chem.RWMol(Chem.Mol(rw_mol))
                            test_mol.AddBond(a_idx, b_idx, Chem.rdchem.BondType.AROMATIC)
                            Chem.SanitizeMol(test_mol.GetMol())
                            rw_mol.AddBond(a_idx, b_idx, Chem.rdchem.BondType.AROMATIC)
                        except Exception as e:
                            #print(f"Failed to add bond between atoms {a_idx} and {b_idx}: {e}")
                            continue
                bonds_added += 1
                used.add(a_idx)
                used.add(b_idx)

    if bonds_added < num_bond:
        #print("[Error] Less then expected")
        return None
        
    mol = rw_mol.GetMol()
    Chem.SanitizeMol(mol)
    return mol

def mol_without_mapnum(mol):
    mol_copy = copy.deepcopy(mol)
    for atom in mol_copy.GetAtoms():
        atom.SetAtomMapNum(0)
    return mol_copy

def graph_mixer_synt(filtered_groups, n_iter_in, aggressive):
    result = defaultdict(list)
    
    for group_key, rows in tqdm(filtered_groups.items()):
        n_iter = len(rows) * n_iter_in
        seen_inputs = set()
        for fg_size in range(8):
            sub_candidates = [row for row in rows if any(len(fg) == fg_size for fg in row['fg_sub'])]

            if len(sub_candidates) < 2:
                continue

            for _ in range(n_iter):
                random.shuffle(sub_candidates)
                s1, s2 = copy.deepcopy(sub_candidates[0]), copy.deepcopy(sub_candidates[1])
                fg1_info = [(m, s) for m, s in zip(s1['fg_mat'], s1['fg_sub']) if len(s) == fg_size]
                fg2_info = [(m, s) for m, s in zip(s2['fg_mat'], s2['fg_sub']) if len(s) == fg_size]

                if not fg1_info or not fg2_info:
                    continue

                m1, s1_subs = random.choice(fg1_info)
                m2, s2_subs = random.choice(fg2_info)

                mol1_i = Chem.MolFromSmiles(s1['inputs'])
                mol1_o = Chem.MolFromSmiles(s1['output'])
                mol2_i = Chem.MolFromSmiles(s2['inputs'])
                mol2_o = Chem.MolFromSmiles(s2['output'])

                if None in (mol1_i, mol1_o, mol2_i, mol2_o):
                    continue

                try:
                    frag = extract_submol(mol1_i, get_atoms_by_mapnums(mol1_i, m1))
                    frag = remap_fragment_atoms(frag)

                    new_i = replace_fragment(mol2_i, m2, frag, old_subs=s2_subs, new_subs=m1, num_bond=fg_size)
                    new_o = replace_fragment(mol2_o, m2, frag, old_subs=s2_subs, new_subs=m1, num_bond=fg_size)
                    
                    input_smi = Chem.MolToSmiles(new_i, isomericSmiles=True, kekuleSmiles=False, canonical=True)
                    output_smi = Chem.MolToSmiles(new_o, isomericSmiles=True, kekuleSmiles=False, canonical=True)
                    cmp_input_smi = Chem.MolToSmiles(mol_without_mapnum(new_i), isomericSmiles=True, kekuleSmiles=False, canonical=True)
                    #cmp_output_smi = Chem.MolToSmiles(mol_without_mapnum(new_o), isomericSmiles=True, kekuleSmiles=False, canonical=True)
                    cmp_prev_input_smi = Chem.MolToSmiles(mol_without_mapnum(mol2_i), isomericSmiles=True, kekuleSmiles=False, canonical=True)
                    if len(cmp_input_smi.split('.')) != len(cmp_prev_input_smi.split('.')):
                        continue #remove invalid data
                   
                    if cmp_input_smi in seen_inputs:
                        continue
                    if cmp_input_smi == cmp_prev_input_smi:
                        continue
                    seen_inputs.add(cmp_input_smi)


                    new_i_m = Chem.MolFromSmiles(input_smi)
                    new_o_m = Chem.MolFromSmiles(output_smi)

                    if new_i_m is None or new_o_m is None:
                        continue
                        
                    if aggressive == 0:
                        if strain_computation(new_i_m) or strain_computation(new_o_m):
                            continue

                    result[group_key].append({
                        'inputs': input_smi,
                        'mixed_num': fg_size,
                        'output': output_smi,
                        'source_ids': (s1.get('id'), s2.get('id'))
                    })
                except Exception as e:
                    #print(f"Fragment replacement failed: {e}")
                    continue

    return result