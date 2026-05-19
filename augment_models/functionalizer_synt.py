import random
from rdkit import Chem
from tqdm import tqdm
import augment_models.functionalizer as functionalizer
import augment_models.distance as distance
from rdkit import DataStructs
from rdkit.Chem import AllChem

def calculate_similarity(smiles1, smiles2, n):
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)
    if mol1 is None or mol2 is None:
        return 0
    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, n, nBits=32)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, n, nBits=32)
    return DataStructs.TanimotoSimilarity(fp1, fp2)

def matches_special_fg(smiles, special_fg_patterns):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    for patt in special_fg_patterns:
        if mol.HasSubstructMatch(patt):
            return True
    return False

def assign_fg_mapnums(input_mol, output_mol, orig_input_num, orig_output_num, fg_num):
    base_mapnum = 250 + 100 * fg_num
    num_fg_atoms = input_mol.GetNumAtoms() - orig_input_num
    for i in range(num_fg_atoms):
        atom_input = input_mol.GetAtomWithIdx(orig_input_num + i)
        atom_input.SetAtomMapNum(base_mapnum + i)
        atom_output = output_mol.GetAtomWithIdx(orig_output_num + i)
        atom_output.SetAtomMapNum(base_mapnum + i)

def functionalizer_synt(filtered_groups, aug_num, fg_dict, daring_value):
    augmented_reactions = []
    seen_reactions = set()
    debug_logs = []
    special_fgs_smarts_nucleophilic = [
        '[OX2H]',             
        '[NX3H2;!$(NC=O);$([NX3H2]-[C])]', 
        '[NX3H1;!$(NC=O);$([NX3H1]-[C])]',   
        '[N;H2][c]',                     
        '[NX3H0;$([NX3H0]-[O])]',               
        '[SX2H]',                                
        'S(=O)(=O)(O[#6])',
        '[#6]S(=O)(=O)O',
        '[CX3H1](=O)[#6]',
        '[CX3](=O)C'
    ]
    special_fgs_smarts_acidic = [
        'C(=O)[OH]',          
        '[SX2H]'                     
        '[OX2H]',             
        '[CX2]=[NX2H]',        
        '[SiH]',
        '[B](O[H])O',
        '[BH]',
        'S(=O)(=O)(N[H])',
        'S(=O)(=O)(O[H])',
    ]
    special_fg_patterns_n = [Chem.MolFromSmarts(s) for s in special_fgs_smarts_nucleophilic]
    special_fg_patterns_a = [Chem.MolFromSmarts(s) for s in special_fgs_smarts_acidic]

    for group_key, rows in tqdm(filtered_groups.items()):
        if group_key not in fg_dict:
            continue

        fgs = fg_dict[group_key]
        if not fgs:
            continue

        for row in rows:
            mol_r = Chem.MolFromSmiles(row['inputs'])
            mol_p = Chem.MolFromSmiles(row['output'])
            id_real = row['id']
            if mol_r is None or mol_p is None:
                debug_logs.append(f"[SKIP] Invalid SMILES in row {row}")
                continue

            center_mapnums = set()
            if 'class' in row and row['class']:
                center_mapnums = set(row['class']) if isinstance(row['class'], (list, set)) else set([row['class']])
            if not center_mapnums:
                debug_logs.append(f"[SKIP] No center_mapnums found in this row {row}")
                continue

            exclude_idxs = {atom.GetIdx() for atom in mol_r.GetAtoms()
                            if atom.GetAtomMapNum() in center_mapnums}

            is_conj = functionalizer.is_conjugated(mol_r, row['class'])
            dist = 4 if is_conj else 2
            allowed_mapnums = distance.get_atoms_by_distance(mol_r, center_mapnums, dist)

            valid_attach_r = functionalizer.get_valid_attachment(mol_r, exclude_idxs)
            valid_attach_p = functionalizer.get_valid_attachment(mol_p, exclude_idxs)

            valid_attach_r_mapnums = {atom.GetAtomMapNum() for atom in mol_r.GetAtoms() if atom.GetIdx() in valid_attach_r}
            valid_attach_p_mapnums = {atom.GetAtomMapNum() for atom in mol_p.GetAtoms() if atom.GetIdx() in valid_attach_p}

            allowed_valid_mapnums = valid_attach_r_mapnums & valid_attach_p_mapnums & allowed_mapnums

            if not allowed_valid_mapnums:
                continue

            for _ in range(aug_num):
                ref_fg = group_key
                
                candidate_fgs = []

                for fg_smiles in fgs:
                    sim = calculate_similarity(ref_fg, fg_smiles, 2)
                    if sim >= daring_value:
                        continue
                    if not matches_special_fg(fg_smiles, special_fg_patterns_n):
                        if not matches_special_fg(fg_smiles, special_fg_patterns_a):
                            candidate_fgs.append(fg_smiles)

                if not candidate_fgs:
                    continue
                else:
                    fg_smiles = random.choice(candidate_fgs)
                    if random.random() > 0.2:
                        candidate_fgs.remove(fg_smiles)
                        for existing_fgs in candidate_fgs:
                            sim_mol = calculate_similarity(existing_fgs, fg_smiles, 6)
                            if sim_mol > 0.8:
                                candidate_fgs.remove(existing_fgs)
                
                fg_mol = Chem.MolFromSmiles(fg_smiles)
                if fg_mol is None:
                    continue

                for atom in fg_mol.GetAtoms():
                    atom.SetAtomMapNum(0)

                attach_mapnum = random.choice(list(allowed_valid_mapnums))

                try:
                    attach_idx_r = next(atom.GetIdx() for atom in mol_r.GetAtoms() if atom.GetAtomMapNum() == attach_mapnum)
                    attach_idx_p = next(atom.GetIdx() for atom in mol_p.GetAtoms() if atom.GetAtomMapNum() == attach_mapnum)
                except StopIteration:
                    continue

                try:
                    new_input_mol = functionalizer.add_fg_to_mol(mol_r, fg_mol, attach_idx_r)
                    new_output_mol = functionalizer.add_fg_to_mol(mol_p, fg_mol, attach_idx_p)
                except Exception:
                    continue

                if new_input_mol is None or new_output_mol is None:
                    continue

                fg_attach_count = 0
                
                orig_input_num = mol_r.GetNumAtoms()
                orig_output_num = mol_p.GetNumAtoms()

                used_position = []
                used_position.append(attach_mapnum)

                assign_fg_mapnums(new_input_mol, new_output_mol, orig_input_num, orig_output_num, fg_attach_count)

                probabilities = [0.7, 0.6, 0.5, 0.4, 0.3]
                for prob in probabilities:
                    if random.random() > prob:
                        break

                    if not candidate_fgs:
                        continue
                    else:
                        fg_smiles_2 = random.choice(candidate_fgs)
                        if random.random() > 0.2:
                            candidate_fgs.remove(fg_smiles_2)
                            for existing_fgs in candidate_fgs:
                                sim_mol = calculate_similarity(existing_fgs, fg_smiles_2, 6)
                                if sim_mol > 0.8:
                                    candidate_fgs.remove(existing_fgs)

                    fg_mol_2 = Chem.MolFromSmiles(fg_smiles_2)
                    if fg_mol_2 is None:
                        break
                    for atom in fg_mol_2.GetAtoms():
                        atom.SetAtomMapNum(0)

                    allowed_mapnums_second = distance.get_atoms_by_distance(new_input_mol, center_mapnums, dist)
                    new_valid_r = functionalizer.get_valid_attachment(new_input_mol, exclude_idxs)
                    new_valid_p = functionalizer.get_valid_attachment(new_output_mol, exclude_idxs)
                    new_valid_r_mapnums = {atom.GetAtomMapNum() for atom in new_input_mol.GetAtoms() if atom.GetIdx() in new_valid_r}
                    new_valid_p_mapnums = {atom.GetAtomMapNum() for atom in new_output_mol.GetAtoms() if atom.GetIdx() in new_valid_p}
                    new_allowed_mapnums = new_valid_r_mapnums & new_valid_p_mapnums & allowed_mapnums_second - set(used_position)
                    
                    ranges_to_remove = [
                        (250, 350),
                        (350, 450),
                        (550, 650),
                        (650, 750),
                    ]
                    for start, end in ranges_to_remove:
                        count_in_range = sum(start <= pos <= end for pos in used_position)
                        if count_in_range >= 2:
                            new_allowed_mapnums -= set(range(start, end))

                    if not new_allowed_mapnums:
                        break

                    attach_mapnum = random.choice(list(new_allowed_mapnums))

                    try:
                        attach_idx_r = next(atom.GetIdx() for atom in new_input_mol.GetAtoms() if atom.GetAtomMapNum() == attach_mapnum)
                        attach_idx_p = next(atom.GetIdx() for atom in new_output_mol.GetAtoms() if atom.GetAtomMapNum() == attach_mapnum)

                        orig_input_num = new_input_mol.GetNumAtoms()
                        orig_output_num = new_output_mol.GetNumAtoms()

                        new_input_mol = functionalizer.add_fg_to_mol(new_input_mol, fg_mol_2, attach_idx_r)
                        new_output_mol = functionalizer.add_fg_to_mol(new_output_mol, fg_mol_2, attach_idx_p)

                        if new_input_mol is None or new_output_mol is None:
                            break

                        fg_attach_count += 1
                        assign_fg_mapnums(new_input_mol, new_output_mol, orig_input_num, orig_output_num, fg_attach_count)

                    except Exception:
                        break

                    used_position.append(attach_mapnum)

                try:
                    new_input = Chem.MolToSmiles(new_input_mol, kekuleSmiles=False, canonical=True)
                    new_output = Chem.MolToSmiles(new_output_mol, kekuleSmiles=False, canonical=True)
                except Exception:
                    continue

                key = (new_input, new_output)
                if key in seen_reactions or new_input == row['inputs'] or new_output == row['output']:
                    continue
                
                seen_reactions.add(key)
                augmented_reactions.append({
                    'inputs': new_input,
                    'output': new_output,
                    'reaction_center': group_key,
                    'source_row': id_real
                })

    for log in debug_logs[-20:]:
        print(log)

    return augmented_reactions