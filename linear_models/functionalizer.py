from rdkit import Chem
from tqdm import tqdm
import copy
from rdkit.Chem import rdmolops
from rdkit.Chem import RWMol
import csv
import os
import augment_models.functionalizer_synt as functionalizer_synt

def functionalizer_linear_mod(reaction_groups, reaction_groups_val, n_iter, daring_value=0.6):
    aug_num = n_iter
    aug_num_val = n_iter
    filtered_groups = {k: v for k, v in reaction_groups.items() if len(v) >= 2 and k}
    filtered_groups_val = {k: v for k, v in reaction_groups_val.items() if len(v) >= 2 and k}

    fg_smarts_list = {
        'hydroxyl': '[*][OX2H;!$(N)]',                
        'primary_amine': '[*][NX3H2;!$(NC=O);$([NX3H2]-[C])]',
        'secondary_amine': '[*][NX3H1;!$(NC=O);$([NX3H1]-[C])]',
        'tertiary_amine': '[*][NX3H0;!$(NC=O);$([NX3H0]-[C])]',
        'aniline': '[*][N;H2][c]',              
        'hydroxylamine': '[*][NX3H0;$([NX3H0]-[O])]',
        'carboxyl': '[*]C(=O)[OH]',            
        'ester_alkyl': '[*][CX3](=O)O[CX4]',  
        'ester_aryl': '[*][CX3](=O)O[c]',     
        'ester_alkyl_rev': '[*]O[CX3](=O)[CX4]', 
        'ester_aryl_rev': '[*]O[CX3](=O)[c]',    
        'ether_alkyl_alkyl': '[*][OD2]([CX4])[CX4]',
        'ether_aryl_aryl': '[*][OD2]([c])[c]',        
        'ether_alkyl_aryl': '[*][OD2]([CX4])[c]',    
        'thiol': '[*][SX2H]',                   
        'sulfide': '[*][SX2]([#6])[#6]',      
        'amide_alkyl_C_to_N': '[*]C(=O)N([CX4])',  
        'amide_alkyl_N_side': '[*]N(C=O)[CX4]', 
        'amide_aryl_C_to_N': '[*]C(=O)N([c])', 
        'amide_aryl_N_side': '[*]N(C=O)[c]',   
        'amide': '[*]C(=O)[NH2]',        
        'alkene_2substituted_alkyl': '[*][CX3](~[CX4])=[CX3]',  
        'alkene_2substituted_both_alkyl': '[*][CX3](~[CX4])=[CX3](~[CX4])',
        'alkene_3substituted_alkyl': '[*][CX3](~[CX4])(~[CX4])=[CX3](~[CX4])',
        'alkene_3substituted_rev_alkyl': '[*][CX3](~[CX4])=[CX3](~[CX4])(~[CX4])',
        'alkene_4substituted_alkyl': '[*][CX3](~[CX4])(~[CX4])=[CX3](~[CX4])(~[CX4])',
        'alkene_2substituted_aryl': '[*][CX3](~[c])=[CX3]',  
        'alkene_2substituted_both_aryl': '[*][CX3](~[c])=[CX3](~[c])',
        'alkene_3substituted_aryl': '[*][CX3](~[c])(~[c])=[CX3](~[c])',
        'alkene_3substituted_rev_aryl': '[*][CX3](~[c])=[CX3](~[c])(~[c])',
        'alkene_4substituted_aryl': '[*][CX3](~[c])(~[c])=[CX3](~[c])(~[c])',
        'alkene_2substituted_mixed': '[*][CX3](~[#6,c])=[CX3](~[#6,c])',
        'alkyne_terminal': '[*][CX2]#C[H]',  
        'alkyne_alkyl_substituted': '[*][CX2]#C[CX4]',  
        'alkyne_aryl_substituted': '[*][CX2]#C[c]',  
        'trifluoromethyl': '[*]C(F)(F)F',     
        'difluoromethyl': '[*]C(F)F',     
        'fluoromethyl': '[*]CF',     
        'halide_1': '[*][F]',   
        'halide_2': '[*][Cl]',  
        'halide_3': '[*][Br]',    
        'halide_4': '[*][I]',  
        'nitro': '[*][NX3](=O)=O',   
        'phenol': '[*]c[OX2H]',       
        'imine_aliphatic': '[*][CX2]=[NX2H;!a]',
        'imine_aromatic': '[*][CX2]=[NX2H;a]',
        'imine_alkyl_N_substituted': '[*][CX2]=N([CX4])[H0]', 
        'imine_aryl_N_substituted': '[*][CX2]=N([c])[H0]',  
        'cyanide': '[*][CX2]#N',       
        'cyclopropyl': '[*]C1CC1',      
        'cyclobutyl': '[*]C1CCC1',         
        'cyclopentyl': '[*]C1CCCC1',        
        'cyclohexyl': '[*]C1CCCCC1',          
        'tert-butyl': '[*]C(C)(C)C',    
        'isopropyl': '[*]CC(C)',  
        'methyl': '[*]C',
        'ethyl': '[*]CC',
        'propyl': '[*]CCC',
        'butyl': '[*]CCCC',
        'benzyl': '[*]c1ccccc1C',               
        'methyl_ketone': '[*][CX3](=O)C',      
        'aryl_ketone': '[*][CX3](=O)c',         
        'enone': '[*][CX3](=O)C=C',          
        'aldehyde': '[*][CX3H1](=O)[#6]',
        'phenyl': '[*]c1ccccc1',         
        'naphthyl_1': '[*]c1cccc2c1cccc2',   
        'naphthyl_2': '[*]c1ccc2ccccc2c1', 
        'pyridine_1': '[*]c1ncccc1',        
        'pyridine_2': '[*]c1cnccc1',      
        'pyridine_3': '[*]c1ccncc1',    
        'thiophene_1': '[*]c1sccc1',   
        'thiophene_2': '[*]c1cscc1',    
        'furan_1': '[*]c1occc1',         
        'furan_2': '[*]c1cocc1',        
        'pyrrole_1': '[*]c1[nH]ccc1',      
        'pyrrole_2': '[*]c1c[nH]cc1',         
        'pyrrole_C_1': '[*]c1[nC]ccc1',        
        'pyrrole_C_2': '[*]c1c[nC]cc1',       
        'imidazole_1': '[*]c1nc[nH]c1',   
        'imidazole_2': '[*]c1ncc[nH]1',     
        'imidazole_C_1': '[*]c1cnc[nX1]1',    
        'imidazole_C_2': '[*]c1ncc[nX1]1',       
        'thiazole_1': '[*]c1scnc1',  
        'thiazole_2': '[*]c1sccn1',  
        'oxazole_1': '[*]c1ocnc1',   
        'oxazole_2': '[*]c1occn1',    
        'pyrazole_1': '[*]c1cn[nH]c1',       
        'pyrazole_2': '[*]c1ccn[nH]1',         
        'pyrazole_C_1': '[*]c1cn[nX1]c1',      
        'pyrazole_C_2': '[*]c1ccn[nX1]1',        
        'pyrimidine': '[*]c1ncncn1',       
        'pyrazine_1': '[*]c1ncccn1',        
        'pyrazine_2': '[*]c1ncncc1',   
        'pyrazine_3': '[*]c1cncnc1',      
        'pyridazine_1': '[*]c1nnccc1',        
        'pyridazine_2': '[*]c1cnncc1',    
        'benzothiophene_1': '[*]c1cc2ccccc2s1',  
        'benzothiophene_2': '[*]c1c2ccccc2sc1',  
        'benzothiophene_3': '[*]c1cccc2c1ccs2',  
        'benzothiophene_4': '[*]c(cc1)cc2c1scc2',  
        'benzothiophene_5': '[*]c1cc(scc2)c2cc1',  
        'benzothiophene_6': '[*]c1c(scc2)c2ccc1',  
        'indole_1': '[*]c1cc2ccccc2[nH]1',   
        'indole_2': '[*]c1c2ccccc2[nH]c1',  
        'indole_3': '[*]c1c([nH]cc2)c2ccc1',  
        'indole_4': '[*]c1cc([nH]cc2)c2cc1',  
        'indole_5': '[*]c(cc1)cc2c1[nH]cc2',   
        'indole_6': '[*]c1cccc2c1cc[nH]2',  
        'indole_C1': '[*]c1cc2ccccc2n1C',  
        'indole_C2': '[*]c1c2ccccc2n(C)c1',  
        'indole_C3': '[*]c1c(n(C)cc2)c2ccc1',   
        'indole_C4': '[*]c1cc(n(C)cc2)c2cc1',   
        'indole_C5': '[*]c(cc1)cc2c1n(C)cc2',   
        'indole_C6': '[*]c1cccc2c1ccn2C', 
        'benzofuran_1': '[*]c1cc2ccccc2o1',    
        'benzofuran_2': '[*]c1c2ccccc2oc1',    
        'benzofuran_3': '[*]c1c(occ2)c2ccc1',   
        'benzofuran_4': '[*]c1cc(occ2)c2cc1',    
        'benzofuran_5': '[*]c(cc1)cc2c1occ2',  
        'benzofuran_6': '[*]c1cccc2c1cco2',   
        'quinoline_1':        '[*]c1ccc(cccc2)c2n1',          
        'quinoline_2':        '[*]c1cc(cccc2)c2nc1',         
        'quinoline_3':        '[*]c1c(cccc2)c2ncc1',           
        'quinoline_4':        '[*]c1cccc2c1cccn2',          
        'quinoline_5':        '[*]c1ccc(nccc2)c2c1',          
        'quinoline_6':        '[*]c1cc(nccc2)c2cc1',       
        'quinoline_7':        '[*]c1c(nccc2)c2ccc1',         
        'isoquinoline_1':     '[*]c(ncc1)c2c1cccc2',         
        'isoquinoline_2':     '[*]c1cc(cccc2)c2cn1',     
        'isoquinoline_3':     '[*]c1c(cccc2)c2cnc1',       
        'isoquinoline_4':     '[*]c1cccc2c1ccnc2',           
        'isoquinoline_5':     '[*]c1ccc(cncc2)c2c1',      
        'isoquinoline_6':     '[*]c1cc(cncc2)c2cc1',      
        'isoquinoline_7':     '[*]c1c(cncc2)c2ccc1',         
        'quinoxaline_1':      '[*]c(n1)cnc2c1cccc2',     
        'quinoxaline_2':      '[*]c1cccc2nccnc12',      
        'quinoxaline_3':      '[*]c1ccc2nccnc2c1',     
        'quinazoline_1':      '[*]c(n1)ncc2c1cccc2',  
        'quinazoline_2':      '[*]c1ncnc2c1cccc2',  
        'quinazoline_3':      '[*]c1c2cncnc2ccc1',      
        'quinazoline_4':      '[*]c1cc2cncnc2cc1',      
        'quinazoline_5':      '[*]c1ccc2cncnc2c1',      
        'quinazoline_6':      '[*]c1cccc2cncnc12',        
        'benzimidazole_1':    '[*]c1nc2ccccc2[nH]1',   
        'benzimidazole_2':    '[*]c1cccc2c1nc[nH]2',    
        'benzimidazole_3':    '[*]c(cc1)cc2c1[nH]cn2',       
        'benzimidazole_C1':    '[*]c(cc1)cc2c1n(C)cn2',       
        'benzimidazole_C2':    '[*]c1cccc2c1ncn2C',    
        'benzimidazole_C3':    '[*]c1nc2ccccc2n1C',     
        'benzimidazole_C4':    '[*]c1c(n(C)cn2)c2ccc1',      
        'benzimidazole_C5':    '[*]c1cc(n(C)cn2)c2cc1',     
        'benzothiazole_1':    '[*]c1nc2ccccc2s1',      
        'benzothiazole_2':    '[*]c1c(scn2)c2ccc1',         
        'benzothiazole_3':    '[*]c1cc(scn2)c2cc1',          
        'benzothiazole_4':    '[*]c(cc1)cc2c1scn2',        
        'benzothiazole_5':    '[*]c1cccc2c1ncs2',        
        'benzoxazole_1':      '[*]c1nc2ccccc2o1',    
        'benzoxazole_2':      '[*]c1c(ocn2)c2ccc1',        
        'benzoxazole_3':      '[*]c1cc(ocn2)c2cc1',       
        'benzoxazole_4':      '[*]c(cc1)cc2c1ocn2',          
        'benzoxazole_5':      '[*]c1cccc2c1nco2',        
        'indazole_1':         '[*]c1c([nH]nn2)c2ccc1',        
        'indazole_2':         '[*]c1cc([nH]nn2)c2cc1',     
        'indazole_C_1':         '[*]c1c([nX1]nn2)c2ccc1',      
        'indazole_C_2':         '[*]c(cc1)cc2c1[nX1]nn2',       
        'indazole_C_3':         '[*]c1cccc2c1nn[nX1]2',    
        'indazole_C_4':         '[*]c1cc([nX1]nn2)c2cc1',  
        'silyl_primary': '[*][SiH2]([#6])[#6]',            
        'silyl_secondary': '[*][SiH]([#6])([#6])[#6]',       
        'silyl_tertiary': '[*][Si]([#6])([#6])([#6])',        
        'boronic_acid': '[*][B](O)O',            
        'boronate': '[*][B](O[#6])O[#6]',            
        'borane': '[*][BH2]',                       
        'sulfone': '[*]S(=O)(=O)[#6]',                     
        'sulfoxide': '[*]S(=O)[#6]',             
        'sulfonamide_primary':      '[*]S(=O)(=O)[N;H2]',              
        'sulfonamide_secondary':    '[*]S(=O)(=O)[N;H1][#6]',           
        'sulfonamide_secondary_2':  '[#6]S(=O)(=O)[N;H1][*]',         
        'sulfonamide_tertiary':     '[*]S(=O)(=O)N([#6])[#6]',        
        'sulfonamide_tertiary_2':   '[#6]S(=O)(=O)N([*])[#6]',        
        'sulfonic_acid': '[#6]S(=O)(=O)O',            
        'trifluoromethyl_sulfonate': 'C(F)(F)FS(=O)(=O)O[#6]',     
        'tosyl_group': 'Cc1ccc(cc1)S(=O)(=O)[*]', 
    }
    
    def replace_dummy_with_uranium(mol):
        mol = Chem.Mol(mol)
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 0:
                atom.SetAtomicNum(92)
                break
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 0:
                atom.SetAtomicNum(6)
        Chem.SanitizeMol(mol)
        return mol
    
    def replace_aromatic_with_phenyl(mol, aromatic_idxs):
        rw_mol = RWMol(mol)
        phenyl = Chem.MolFromSmiles('c1ccccc1')
        phenyl_rw = RWMol(phenyl)
        
        for i, atom in enumerate(phenyl_rw.GetAtoms()):
            atom.SetAtomMapNum(0)
        phenyl = phenyl_rw.GetMol()
    
        offset = rw_mol.GetNumAtoms()
    
        for idx in aromatic_idxs:
            atom = rw_mol.GetAtomWithIdx(idx)
            neighbors = list(atom.GetNeighbors())
    
            check = 0
            for nbr in neighbors:
                if nbr.GetAtomicNum() == 1:
                    rw_mol.RemoveAtom(nbr.GetIdx())
                    check = 1
                    break
            if check == 0:
                return mol
    
            combo = Chem.CombineMols(rw_mol, phenyl)
            rw_combo = RWMol(combo)
    
            rw_combo.AddBond(idx, offset, Chem.BondType.SINGLE)
    
            rw_mol = rw_combo
            offset = rw_mol.GetNumAtoms()
    
        mol_out = rw_mol.GetMol()
        Chem.SanitizeMol(mol_out)
        return mol_out
        
    def extract_functional_groups(mol):
        fg_smiles_list = []
        natoms = mol.GetNumAtoms()
    
        for name, smarts in fg_smarts_list.items():
            patt = Chem.MolFromSmarts(smarts)
            if patt is None:
                print(f"[Warning] Invalid SMARTS pattern for {name}: {smarts}")
                continue
            matches = mol.GetSubstructMatches(patt)
            for match in matches:
                atom_idxs = list(match)
                if any(idx >= natoms or idx < 0 for idx in atom_idxs):
                    print(f"[Warning] Skipping {name} with out-of-range atom indices: {atom_idxs}")
                    continue
                try:
                    patt_copy = Chem.Mol(patt)
                    
                    try:
                        Chem.SanitizeMol(patt_copy)
                    except Chem.rdchem.KekulizeException:
                        Chem.SanitizeMol(patt_copy, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
    
                    aromatic_idxs = []
                    for atom in mol.GetAtoms():
                        if atom.GetAtomicNum() in (7, 8) and atom.GetTotalNumHs() > 0:
                            if any(neigh.GetIsAromatic() and neigh.GetAtomicNum() == 6 for neigh in atom.GetNeighbors()):
                                if not atom.GetIsAromatic():
                                    aromatic_idxs.append(atom.GetIdx())
                    if aromatic_idxs:
                        patt_copy = replace_aromatic_with_phenyl(patt_copy, aromatic_idxs)
    
                    patt_copy = replace_dummy_with_uranium(patt_copy)
                    smiles = Chem.MolToSmiles(patt_copy, canonical=True, kekuleSmiles=False)
                    if smiles != '[U][U]':
                        fg_smiles_list.append(smiles)
    
                except Exception as e:
                    #print(f"[Error] MolFragmentToSmiles failed for {name} with atoms {atom_idxs}: {e}")
                    continue
    
        return fg_smiles_list
    
    
    fg_dict = {}
    fg_dict_val = {}
    
    for group_key, rows in tqdm(filtered_groups.items()):
        group_fgs = []
    
        for r in rows:
            smiles = r['inputs'].split('.')[0]
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                print(f"[Warning] Invalid SMILES: {smiles}")
                continue
    
            fgs = extract_functional_groups(mol)
    
            if fgs:
                group_fgs.extend(fgs)
    
        if group_fgs:
            fg_dict[group_key] = list(set(group_fgs))
    
    
    for group_key, rows in tqdm(filtered_groups_val.items()):
        group_fgs = []
    
        for r in rows:
            smiles = r['inputs'].split('.')[0]
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                print(f"[Warning] Invalid SMILES: {smiles}")
                continue
    
            fgs = extract_functional_groups(mol)
    
            if fgs:
                group_fgs.extend(fgs)
    
        if group_fgs:
            fg_dict_val[group_key] = list(set(group_fgs))
   
    def chunk_dict(d, chunk_size):
        items = list(d.items())
        for i in range(0, len(items), chunk_size):
            yield dict(items[i:i+chunk_size])
    
    def remove_same_data(data):
        seen = set()
        unique_data = []
        for entry in data:
            tup = tuple((k, entry[k]) for k in sorted(entry))
            if tup not in seen:
                seen.add(tup)
                unique_data.append(entry)
        return unique_data
    
    def append_csv_chunk(data, filename, write_header=True):
        mode = 'a'
        with open(filename, mode, newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['id', 'class', 'input>>output'])          
            formatted_data = []
            for row in data:
                new_row = {
                    'id': row['source_row'],
                    'class': row['reaction_center'],
                    'input>>output': f"{row['inputs']}>>{row['output']}"
                }
                formatted_data.append(new_row)
    
            writer.writerows(formatted_data)
        print(f"[Appended] {len(data)} rows → {filename}")
    
    # ---------- Training ----------
    train_csv_path = 'linear_datasets/train.csv'
    print("[Start] Augmenting training data...")
    
    for i, fg_chunk in enumerate(chunk_dict(filtered_groups, 25)):
        print(f"[Train] Chunk {i+1}")
        chunk_result = functionalizer_synt.functionalizer_synt(fg_chunk, aug_num, fg_dict, daring_value)
        if chunk_result:
            chunk_result = remove_same_data(chunk_result)
            append_csv_chunk(chunk_result, train_csv_path)
    
    # ---------- Validation ----------
    val_csv_path = 'linear_datasets/val.csv'
    print("[Start] Augmenting validation data...")
    
    for i, fg_chunk in enumerate(chunk_dict(filtered_groups_val, 25)):
        print(f"[Val] Chunk {i+1}")
        chunk_result = functionalizer_synt.functionalizer_synt(fg_chunk, aug_num_val, fg_dict_val, daring_value)
        if chunk_result:
            chunk_result = remove_same_data(chunk_result)
            append_csv_chunk(chunk_result, val_csv_path)