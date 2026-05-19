from rdkit import Chem
from tqdm import tqdm
import copy
from rdkit.Chem import rdmolops
from rdkit.Chem import RWMol
from rdkit.Chem import AllChem
import augment_models.distance as distance
import augment_models.graph_mixer as graph_mixer
import csv
import os

def strain_computation(mol, energy_cutoff_per_atom=5): #to prevent making strained molecules
    try:
        mol = Chem.AddHs(mol)
        
        if AllChem.EmbedMolecule(mol, randomSeed=42) != 0:
            return True

        if AllChem.UFFOptimizeMolecule(mol, maxIters=200) != 0:
            return True

        ff = AllChem.UFFGetMoleculeForceField(mol)
        energy = ff.CalcEnergy()

        num_atoms = mol.GetNumAtoms()
        energy_per_atom = energy / num_atoms

        return energy_per_atom > energy_cutoff_per_atom

    except Exception:
        return True

def graph_mixer_linear_mod(reaction_groups, reaction_groups_val, n_iter):
    fg_smarts_list = {
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
        'quinoline': 'c1ccc(cccc2)c2n1',
        'isoquinoline': 'c(ncc1)c2c1cccc2',
        'quinoxaline': 'c(n1)cnc2c1cccc2',
        'quinazoline': 'c(n1)ncc2c1cccc2',
        'benzimidazole': 'c1nc2ccccc2[nH]1',
        'benzimidazole_C': 'c(cc1)cc2c1n(C)cn2',
        'benzothiazole_': 'c1nc2ccccc2s1',
        'benzoxazole': 'c1nc2ccccc2o1',
        'indazole': 'c1c([nH]nn2)c2ccc1',
        'indazole_C': 'c1c([nX1]nn2)c2ccc1',
    }

    def ring_substituents(mol, ring_atoms):
        substituents = set()
        mapnums = set()
        for idx in ring_atoms:
            atom = mol.GetAtomWithIdx(idx)
            for neighbor in atom.GetNeighbors():
                n_idx = neighbor.GetIdx()
                n_mapnums = neighbor.GetAtomMapNum()
                if n_idx not in ring_atoms and neighbor.GetAtomicNum() != 1:
                    substituents.add(n_idx)
                    mapnums.add(n_mapnums)
        return len(substituents), mapnums

    def extract_functional_groups(mol):
        ids = []
        fg_list = []
        mapnum = []
        smarts_id = 0

        for name, smarts in fg_smarts_list.items():
            smarts_id += 1
            patt = Chem.MolFromSmarts(smarts)
            if patt is None:
                continue
                
            matches = mol.GetSubstructMatches(patt)
            for match in matches:
                atom_idxs = list(match)
                if len(atom_idxs) < 4:
                    continue
                try:
                    ring_atoms = atom_idxs  
                    n_substituents, n_mapnums = ring_substituents(mol, ring_atoms)
                    if 1 <= n_substituents <= 7:
                        ids.append(smarts_id)
                        fg_list.append(atom_idxs)
                        mapnum.append(n_mapnums)
                except:
                    continue

        return ids, fg_list, mapnum

    for group_key, rows in tqdm(reaction_groups.items()):
        for r in rows:
            smiles = r['inputs'].split('.')[0]
            fg_site = r['class']
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                r['fgs'] = []
                continue
            ids, fgs, mapnum = extract_functional_groups(mol)
            filtered = distance.get_atoms_by_distance(mol, fg_site, min_dist=2)

            new_ids, new_fgs, new_mn = [], [], []
            for i in range(len(fgs)):
                if set(fgs[i]).issubset(filtered):
                    new_ids.append(ids[i])
                    new_fgs.append(fgs[i])
                    new_mn.append(mapnum[i])

            r['fg_id'] = new_ids
            r['fg_mat'] = new_fgs
            r['fg_sub'] = new_mn

    for group_key, rows in tqdm(reaction_groups_val.items()):
        for r in rows:
            smiles = r['inputs'].split('.')[0]
            fg_site = r['class']
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                r['fgs'] = []
                continue
            ids, fgs, mapnum = extract_functional_groups(mol)
            filtered = distance.get_atoms_by_distance(mol, fg_site, min_dist=2)

            new_ids, new_fgs, new_mn = [], [], []
            for i in range(len(fgs)):
                if set(fgs[i]).issubset(filtered):
                    new_ids.append(ids[i])
                    new_fgs.append(fgs[i])
                    new_mn.append(mapnum[i])

            r['fg_id'] = new_ids
            r['fg_mat'] = new_fgs
            r['fg_sub'] = new_mn

    def chunk_dict(d, chunk_size):
        items = list(d.items())
        for i in range(0, len(items), chunk_size):
            yield dict(items[i:i+chunk_size])

    def filter_strain_rows(chunk_result):
        filtered = {}
        for key, rows in chunk_result.items():
            new_rows = []
            for r in rows:
                mol = Chem.MolFromSmiles(r["output"])
                if mol is None:
                    continue
                if not strain_computation(mol):
                    new_rows.append(r)
            if new_rows:
                filtered[key] = new_rows
        return filtered

    def append_csv_chunk(data, filename, write_header=True):
        mode = 'a'
        with open(filename, mode, newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['id', 'class', 'input>>output'])
            formatted_data = []
            checker = 0
            for group_key, rows in data.items():
                for row in rows:
                    formatted_data.append({
                        'id': row['source_ids'],
                        'class': row['mixed_num'],
                        'input>>output': f"{row['inputs']}>>{row['output']}"
                    })
                    checker += 1
            writer.writerows(formatted_data)
        print(f"[Appended] {checker} rows → {filename}")


    # ---------- Training ----------
    train_csv_path = 'linear_datasets/train.csv'
    print("[Start] Augmenting training data...")

    for i, fg_chunk in enumerate(chunk_dict(reaction_groups, 25)):
        print(f"[Train] Chunk {i+1}")
        chunk_result = graph_mixer.graph_mixer_synt(fg_chunk, n_iter)

        if chunk_result:
            chunk_result = filter_strain_rows(chunk_result)
            append_csv_chunk(chunk_result, train_csv_path)


    # ---------- Validation ----------
    val_csv_path = 'linear_datasets/val.csv'
    print("[Start] Augmenting validation data...")

    for i, fg_chunk in enumerate(chunk_dict(reaction_groups_val, 25)):
        print(f"[Val] Chunk {i+1}")
        chunk_result = graph_mixer.graph_mixer_synt(fg_chunk, n_iter)

        if chunk_result:
            chunk_result = filter_strain_rows(chunk_result)
            append_csv_chunk(chunk_result, val_csv_path)