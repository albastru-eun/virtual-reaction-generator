from rdkit import Chem
from rdkit.Chem import rdmolops
from rdkit.Chem.rdchem import BondType
from copy import deepcopy
import augment_models.functionalizer as functionalizer
import augment_models.distance as distance
import random

fg_list = {
    'primary_amine': '[NX3H2;!$(NC=O);$([NX3H2]-[C])]',
    'secondary_amine': '[NX3H1;!$(NC=O);$([NX3H1]-[C])]',
    'tertiary_amine': '[NX3H0;!$(NC=O);$([NX3H0]-[C])]',
    'aniline': '[N;H2][c]',              
    'hydroxylamine': '[NX3H0;$([NX3H0]-[O])]',
    'carboxyl': 'C(=O)[OH]',            
    'ester_alkyl': '[CX3](=O)O[CX4]',  
    'ester_aryl': '[CX3](=O)O[c]',     
    'ester_alkyl_rev': 'O[CX3](=O)[CX4]', 
    'ester_aryl_rev': 'O[CX3](=O)[c]',    
    'ether_alkyl_alkyl': '[OD2]([CX4])[CX4]',
    'ether_aryl_aryl': '[OD2]([c])[c]',        
    'ether_alkyl_aryl': '[OD2]([CX4])[c]',                    
    'sulfide': '[SX2]([#6])[#6]',      
    'amide_alkyl_C_to_N': 'C(=O)N([CX4])',  
    'amide_alkyl_N_side': 'N(C=O)[CX4]', 
    'amide_aryl_C_to_N': 'C(=O)N([c])', 
    'amide_aryl_N_side': 'N(C=O)[c]',   
    'amide': 'C(=O)[NH2]',        
    'alkene_2substituted_alkyl': '[CX3](~[CX4])=[CX3]',  
    'alkene_2substituted_both_alkyl': '[CX3](~[CX4])=[CX3](~[CX4])',
    'alkene_3substituted_alkyl': '[CX3](~[CX4])(~[CX4])=[CX3](~[CX4])',
    'alkene_3substituted_rev_alkyl': '[CX3](~[CX4])=[CX3](~[CX4])(~[CX4])',
    'alkene_4substituted_alkyl': '[CX3](~[CX4])(~[CX4])=[CX3](~[CX4])(~[CX4])',
    'alkene_2substituted_aryl': '[CX3](~[c])=[CX3]',  
    'alkene_2substituted_both_aryl': '[CX3](~[c])=[CX3](~[c])',
    'alkene_3substituted_aryl': '[CX3](~[c])(~[c])=[CX3](~[c])',
    'alkene_3substituted_rev_aryl': '[CX3](~[c])=[CX3](~[c])(~[c])',
    'alkene_4substituted_aryl': '[CX3](~[c])(~[c])=[CX3](~[c])(~[c])',
    'alkene_2substituted_mixed': '[CX3](~[#6,c])=[CX3](~[#6,c])',
    'alkyne_terminal': '[CX2]#C[H]',  
    'alkyne_alkyl_substituted': '[CX2]#C[CX4]',  
    'alkyne_aryl_substituted': '[CX2]#C[c]',      
    'imine_aliphatic': '[CX2]=[NX2H;!a]',
    'imine_aromatic': '[CX2]=[NX2H;a]',
    'imine_alkyl_N_substituted': '[CX2]=N([CX4])[H0]', 
    'imine_aryl_N_substituted': '[CX2]=N([c])[H0]',  
    'cyanide': '[CX2]#N',       
    'cyclopropyl': 'C1CC1',      
    'cyclobutyl': 'C1CCC1',         
    'cyclopentyl': 'C1CCCC1',        
    'cyclohexyl': 'C1CCCCC1',          
    'tert-butyl': 'C(C)(C)C',    
    'isopropyl': 'CC(C)',  
    'methyl': 'C',
    'ethyl': 'CC',
    'propyl': 'CCC',
    'butyl': 'CCCC',
    'benzyl': 'c1ccccc1C',               
    'methyl_ketone': '[CX3](=O)C',      
    'aryl_ketone': '[CX3](=O)c',         
    'enone': '[CX3](=O)C=C',          
    'aldehyde': '[CX3H1](=O)[#6]',
    'phenyl': 'c1ccccc1',         
    'naphthyl': 'c1cccc2c1cccc2',   
    'pyridine': 'c1ncccc1',           
    'thiophene': 'c1sccc1',     
    'furan': 'c1occc1',             
    'pyrrole': 'c1[nH]ccc1',              
    'pyrrole_C': 'c1[nC]ccc1',             
    'imidazole': 'c1nc[nH]c1',     
    'imidazole_C': 'c1cnc[nX1]1',       
    'thiazole': 'c1scnc1',  
    'oxazole': 'c1ocnc1',      
    'pyrazole': 'c1cn[nH]c1',              
    'pyrazole_C': 'c1cn[nX1]c1',          
    'pyrimidine': 'c1ncncn1',       
    'pyrazine': 'c1ncccn1',             
    'pyridazine': 'c1nnccc1',         
    'benzothiophene': 'c1cc2ccccc2s1',   
    'indole': 'c1cc2ccccc2[nH]1',    
    'indole_C': 'c1cc2ccccc2n1C',  
    'benzofuran': 'c1cc2ccccc2o1',      
    'quinoline':        'c1ccc(cccc2)c2n1',                 
    'isoquinoline':     'c(ncc1)c2c1cccc2',                 
    'quinoxaline':      'c(n1)cnc2c1cccc2',        
    'quinazoline':      'c(n1)ncc2c1cccc2',               
    'benzimidazole':    'c1nc2ccccc2[nH]1',           
    'benzimidazole_C':    'c(cc1)cc2c1n(C)cn2',           
    'benzothiazole_':    'c1nc2ccccc2s1',            
    'benzoxazole':      'c1nc2ccccc2o1',          
    'indazole':         'c1c([nH]nn2)c2ccc1',            
    'indazole_C':         'c1c([nX1]nn2)c2ccc1',    
    'silyl_primary': '[SiH2]([#6])[#6]',            
    'silyl_secondary': '[SiH]([#6])([#6])[#6]',       
    'silyl_tertiary': '[Si]([#6])([#6])([#6])',        
    'boronic_acid': '[B](O)O',            
    'boronate': '[B](O[#6])O[#6]',            
    'borane': '[BH2]',                       
    'sulfone': 'S(=O)(=O)[#6]',                     
    'sulfoxide': 'S(=O)[#6]',             
    'sulfonamide_primary':      'S(=O)(=O)[N;H2]',              
    'sulfonamide_secondary':    'S(=O)(=O)[N;H1][#6]',           
    'sulfonamide_secondary_2':  '[#6]S(=O)(=O)[N;H1]',         
    'sulfonamide_tertiary':     'S(=O)(=O)N([#6])[#6]',        
    'sulfonamide_tertiary_2':   '[#6]S(=O)(=O)N()[#6]',        
    'sulfonic_acid': '[#6]S(=O)(=O)O',            
    'trifluoromethyl_sulfonate': 'C(F)(F)FS(=O)(=O)O[#6]',     
    'tosyl_group': 'Cc1ccc(cc1)S(=O)(=O)', 
}

def get_deletable_atoms(mol, reaction_center, max_depth):
    deletable = []
    dists = rdmolops.GetDistanceMatrix(mol)

    include_mapnums = distance.get_atoms_by_distance(mol, reaction_center, max_depth)
    check_many_fluorines = 0
    check_many_p_carbons_1 = 0
    check_many_p_carbons_2 = 0

    atoms = list(mol.GetAtoms())
    random.shuffle(atoms)

    for atom in atoms:
        symbol = atom.GetSymbol()
        mapnum = atom.GetAtomMapNum()
        neighbors = atom.GetNeighbors()

        if mapnum not in include_mapnums:
            continue
        elif not atom.GetIsAromatic() and any(mol.GetBondBetweenAtoms(atom.GetIdx(), n.GetIdx()).GetBondType() == BondType.DOUBLE for n in atom.GetNeighbors()):
            #print("[Warning] Neighbors have double bond")
            continue
        elif not atom.GetIsAromatic() and any(mol.GetBondBetweenAtoms(atom.GetIdx(), n.GetIdx()).GetBondType() == BondType.TRIPLE for n in atom.GetNeighbors()):
            #print("[Warning] Neighbors have double bond")
            continue
        elif atom.GetIsAromatic():
            #print("[Warning] Atom is aromatic")
            continue
        elif check_many_fluorines != 1 and symbol=='F':
            deletable.append(mapnum)
            check_many_fluorines = 1
        elif symbol in ['Cl', 'Br', 'I']:
            deletable.append(mapnum)
        elif symbol in ['O', 'N']:
            if atom.GetDegree() < 3:
                neighbors = atom.GetNeighbors()
                if all(n.GetSymbol() in ['C', 'H'] for n in neighbors):
                    has_carbonyl = False
                    for n in neighbors:
                        if n.GetSymbol() == 'C':
                            for nbr2 in n.GetNeighbors():
                                if nbr2.GetSymbol() == 'O':
                                    bond = n.GetBondBetweenAtom(nbr2.GetIdx())
                                    if bond.GetBondType() == Chem.rdchem.BondType.DOUBLE:
                                        has_carbonyl = True
                                        break
                        if has_carbonyl:
                            break
                    if not has_carbonyl:
                        deletable.append(mapnum)
        elif symbol == 'C':
            if atom.GetDegree() == 1 and check_many_p_carbons_1 != 1:
                neighbors = atom.GetNeighbors()
                if all(n.GetSymbol() in ['C', 'H'] for n in neighbors):
                    deletable.append(mapnum)
                    check_many_p_carbons_1 = 1
            if atom.GetDegree() == 2 and check_many_p_carbons_2 != 1:
                neighbors = atom.GetNeighbors()
                if all(n.GetSymbol() in ['C', 'H'] for n in neighbors):
                    deletable.append(mapnum)
                    check_many_p_carbons_2 = 1
    return deletable

def remove_and_patch_atoms(mol_i, mol_o, mapnum_list):
    mol_i = deepcopy(mol_i)
    mol_o = deepcopy(mol_o)
    random.shuffle(mapnum_list)

    for mapnum in mapnum_list:
        atom_idx_i = next((atom.GetIdx() for atom in mol_i.GetAtoms()
                           if atom.HasProp("molAtomMapNumber") and atom.GetAtomMapNum() == mapnum), None)
        atom_idx_o = next((atom.GetIdx() for atom in mol_o.GetAtoms()
                           if atom.HasProp("molAtomMapNumber") and atom.GetAtomMapNum() == mapnum), None)

        if atom_idx_i is None or atom_idx_o is None:
            continue

        atom_i = mol_i.GetAtomWithIdx(atom_idx_i)
        atom_o = mol_o.GetAtomWithIdx(atom_idx_o)
        neighbors_i = [n.GetAtomMapNum() for n in atom_i.GetNeighbors()
                       if n.GetAtomicNum() > 1 and n.HasProp("molAtomMapNumber")]
        neighbors_o = [n.GetAtomMapNum() for n in atom_o.GetNeighbors()
                       if n.GetAtomicNum() > 1 and n.HasProp("molAtomMapNumber")]

        if len(neighbors_i) > 2 or len(neighbors_o) > 2:
            continue

        try:
            rw_mol_i = Chem.RWMol(mol_i)
            rw_mol_o = Chem.RWMol(mol_o)

            rw_mol_i.RemoveAtom(atom_idx_i)
            rw_mol_o.RemoveAtom(atom_idx_o)
        except Exception as e:
            continue

        try:
            map_i = get_mapnum_to_idx_map(rw_mol_i)

            if len(neighbors_i) == 2 and all(n in map_i for n in neighbors_i):
                i, j = map_i[neighbors_i[0]], map_i[neighbors_i[1]]
                rw_mol_i.AddBond(i, j, BondType.SINGLE)
            elif len(neighbors_i) == 1 and neighbors_i[0] in map_i:
                i = map_i[neighbors_i[0]]
                h_idx = rw_mol_i.AddAtom(Chem.Atom(1))  # H
                rw_mol_i.AddBond(i, h_idx, BondType.SINGLE)
            else:
                continue
        except Exception as e:
            continue

        try:
            map_o = get_mapnum_to_idx_map(rw_mol_o)

            if len(neighbors_o) == 2 and all(n in map_o for n in neighbors_o):
                i, j = map_o[neighbors_o[0]], map_o[neighbors_o[1]]
                rw_mol_o.AddBond(i, j, BondType.SINGLE)
            elif len(neighbors_o) == 1 and neighbors_o[0] in map_o:
                i = map_o[neighbors_o[0]]
                h_idx = rw_mol_o.AddAtom(Chem.Atom(1))  # H
                rw_mol_o.AddBond(i, h_idx, BondType.SINGLE)
            else:
                continue
        except Exception as e:
            continue

        try:
            mol_t_i = Chem.RemoveHs(rw_mol_i.GetMol())
            mol_t_o = Chem.RemoveHs(rw_mol_o.GetMol())
            Chem.SanitizeMol(mol_t_i)
            Chem.SanitizeMol(mol_t_o)
            return mol_t_i, mol_t_o
        except Exception as e:
            continue

    return None, None

def mol_from_smarts_or_smiles(s):
    mol = Chem.MolFromSmarts(s)
    if mol is None:
        mol = Chem.MolFromSmiles(s)
    return mol

def get_mapnum_to_idx_map(mol):
    return {atom.GetAtomMapNum(): atom.GetIdx() for atom in mol.GetAtoms() if atom.HasProp('molAtomMapNumber')}

def remove_and_patch_fgs(mol_i, mol_o, fg_dict_for_group, deletable_fgs=fg_list):
    fg_mols_for_group = [Chem.MolFromSmiles(s) for s in fg_dict_for_group]
    fg_mols_for_group = [m for m in fg_mols_for_group if m is not None]
    candidates = []
    
    for name, smarts in deletable_fgs:
        smarts_mol = Chem.MolFromSmarts(smarts)
        if smarts_mol is None:
            continue
        matched = any(
            smarts_mol.HasSubstructMatch(fg_mol) and fg_mol.HasSubstructMatch(smarts_mol)
            for fg_mol in fg_mols_for_group if fg_mol is not None
        )
        if not matched:
            candidates.append(smarts)
    
    if len(candidates) == 0:
        return None, None

    random.shuffle(candidates)

    for fgs in candidates:
        patt = Chem.MolFromSmarts(fgs)
        if not patt:
            continue
        matches = mol_i.GetSubstructMatches(patt)
        if not matches:
            continue

        for match in matches:
            mapnums = [
                mol_i.GetAtomWithIdx(idx).GetAtomMapNum()
                for idx in match
                if mol_i.GetAtomWithIdx(idx).HasProp('molAtomMapNumber')
            ]
            if not mapnums:
                continue

            try:
                rw_mol_i = Chem.RWMol(mol_i)
                rw_mol_o = Chem.RWMol(mol_o)
                map_i = get_mapnum_to_idx_map(rw_mol_i)
                map_o = get_mapnum_to_idx_map(rw_mol_o)

                for mapnum in sorted(mapnums, reverse=True):
                    idx_i = map_i.get(mapnum)
                    idx_o = map_o.get(mapnum)
                    if idx_i is None or idx_o is None:
                        continue

                    atom_i = rw_mol_i.GetAtomWithIdx(idx_i)
                    atom_o = rw_mol_o.GetAtomWithIdx(idx_o)
                    neighbors_i_mapnum = [
                        n.GetAtomMapNum() for n in atom_i.GetNeighbors()
                        if n.GetAtomicNum() > 1 and n.HasProp('molAtomMapNumber')
                    ]
                    neighbors_o_mapnum = [
                        n.GetAtomMapNum() for n in atom_o.GetNeighbors()
                        if n.GetAtomicNum() > 1 and n.HasProp('molAtomMapNumber')
                    ]

                    map_i = get_mapnum_to_idx_map(rw_mol_i)
                    map_o = get_mapnum_to_idx_map(rw_mol_o)
                    neighbors_i = [map_i[n] for n in neighbors_i_mapnum if n in map_i]
                    neighbors_o = [map_o[n] for n in neighbors_o_mapnum if n in map_o]

                    if len(neighbors_i) > 2 or len(neighbors_o) > 2:
                        continue

                    rw_mol_i.RemoveAtom(idx_i)
                    rw_mol_o.RemoveAtom(idx_o)

                    if len(neighbors_i) == 1:
                        h_idx_i = rw_mol_i.AddAtom(Chem.Atom(1))
                        rw_mol_i.AddBond(neighbors_i[0], h_idx_i, BondType.SINGLE)
                        h_idx_o = rw_mol_o.AddAtom(Chem.Atom(1))
                        rw_mol_o.AddBond(neighbors_o[0], h_idx_o, BondType.SINGLE)

                    elif len(neighbors_i) == 2:
                        if not rw_mol_i.GetBondBetweenAtoms(neighbors_i[0], neighbors_i[1]):
                            rw_mol_i.AddBond(neighbors_i[0], neighbors_i[1], BondType.SINGLE)
                        if not rw_mol_o.GetBondBetweenAtoms(neighbors_o[0], neighbors_o[1]):
                            rw_mol_o.AddBond(neighbors_o[0], neighbors_o[1], BondType.SINGLE)

                mol_i_new = rw_mol_i.GetMol()
                mol_o_new = rw_mol_o.GetMol()
                Chem.SanitizeMol(mol_i_new)
                Chem.SanitizeMol(mol_o_new)
                return mol_i_new, mol_o_new

            except Exception as e:
                #print(f'[Error] Functional group remover error: {e}')
                continue

    return None, None

def atom_remover_synt(inputs, n_iter):
    updated_result = []
    seen = set()
    removal_probs = [1.0, 0.8, 0.6]

    for row in inputs:
        try:
            mol_r = Chem.MolFromSmiles(row['inputs'])
            mol_p = Chem.MolFromSmiles(row['output'])
            if mol_r is None or mol_p is None:
                continue

            ids = row['id']
            reaction_center_idx = row['class']
            check_fgs = row['functional_groups']

            is_conj = functionalizer.is_conjugated(mol_r, reaction_center_idx)
            dist = 4 if is_conj else 2

            deletable_atoms_r = get_deletable_atoms(mol_r, reaction_center=reaction_center_idx, max_depth=dist)
            deletable_atoms_p = get_deletable_atoms(mol_p, reaction_center=reaction_center_idx, max_depth=dist)
            deletable_atoms = list(set(deletable_atoms_r) & set(deletable_atoms_p))

            for _ in range(n_iter):
                mol_r_mod, mol_p_mod = mol_r, mol_p

                for prob in removal_probs:
                    if random.random() > prob:
                        break
                    mol_r_mod, mol_p_mod = remove_and_patch_atoms(mol_r_mod, mol_p_mod, deletable_atoms)
                    if mol_r_mod is None or mol_p_mod is None:
                        break

                if mol_r_mod is None or mol_p_mod is None:
                    continue

                smi_r_mod = Chem.MolToSmiles(mol_r_mod, kekuleSmiles=False, canonical=True)
                smi_p_mod = Chem.MolToSmiles(mol_p_mod, kekuleSmiles=False, canonical=True)

                if Chem.MolToSmiles(mol_r) == smi_r_mod:
                    continue
                if (smi_r_mod, smi_p_mod) in seen:
                    continue

                seen.add((smi_r_mod, smi_p_mod))
                new_row = deepcopy(row)
                new_row['id'] = ids
                new_row['inputs'] = smi_r_mod
                new_row['output'] = smi_p_mod
                updated_result.append(new_row)

            for _ in range(n_iter):
                mol_r_mod, mol_p_mod = mol_r, mol_p

                for prob in removal_probs:
                    if random.random() > prob:
                        break
                    mol_r_mod, mol_p_mod = remove_and_patch_fgs(mol_r_mod, mol_p_mod, check_fgs)
                    if mol_r_mod is None or mol_p_mod is None:
                        break

                if mol_r_mod is None or mol_p_mod is None:
                    continue

                smi_r_mod = Chem.MolToSmiles(mol_r_mod, kekuleSmiles=False, canonical=True)
                smi_p_mod = Chem.MolToSmiles(mol_p_mod, kekuleSmiles=False, canonical=True)

                if Chem.MolToSmiles(mol_r) == smi_r_mod:
                    continue
                if (smi_r_mod, smi_p_mod) in seen:
                    continue

                seen.add((smi_r_mod, smi_p_mod))
                new_row = deepcopy(row)
                new_row['id'] = ids
                new_row['inputs'] = smi_r_mod
                new_row['output'] = smi_p_mod
                updated_result.append(new_row)

        except Exception as e:
            # print(f"[Error] Failed to process row with id={row.get('id', 'N/A')}: {e}")
            continue

    return updated_result