from collections import defaultdict
from rdkit import Chem
from tqdm import tqdm
import pandas as pd

def remap_atom_map(smiles, mapnum_map):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles

    for atom in mol.GetAtoms():
        old_mapnum = atom.GetAtomMapNum()
        if old_mapnum in mapnum_map:
            atom.SetAtomMapNum(mapnum_map[old_mapnum])
    return Chem.MolToSmiles(mol)

def checkMN(clusters):
    new_groups = defaultdict(list)
    highest = 0

    for group_key, rows in tqdm(clusters.items()):
        new_rows = []

        for row in rows:
            all_mapnums = set()
            remap_targets = set()

            for key in ['inputs', 'output', 'fg_site']:
                smi_str = row.get(key, '')
                if not smi_str:
                    continue
                mol = Chem.MolFromSmiles(smi_str)
                if mol:
                    for atom in mol.GetAtoms():
                        mapnum = atom.GetAtomMapNum()
                        if mapnum >= 0:
                            all_mapnums.add(mapnum)
                            if mapnum >= 250:
                                remap_targets.add(mapnum)

            class_info = set(row.get('class', []))
            all_mapnums.update(class_info)
            remap_targets.update({num for num in class_info if num >= 250})
            current_max = 0
            for i in all_mapnums:
                if i < 250:
                    current_max = max(i, current_max)
            start = current_max + 1

            sorted_targets = sorted(remap_targets)
            mapnum_map = {old: new for new, old in zip(range(start, start + len(sorted_targets)), sorted_targets)}

            def remap_joined_smiles(smi_str):
                parts = smi_str.split('.') if smi_str else []
                remapped = [remap_atom_map(smi, mapnum_map) for smi in parts]
                return '.'.join(remapped)

            new_row = {k: v for k, v in row.items() if k not in ['inputs', 'output', 'fg_site', 'class']}

            new_row['inputs'] = remap_joined_smiles(row.get('inputs', ''))
            new_row['output'] = remap_joined_smiles(row.get('output', ''))
            new_row['fg_site'] = remap_joined_smiles(row.get('fg_site', ''))

            new_row['class'] = {mapnum_map.get(val, val) for val in class_info}
            new_rows.append(pd.Series(new_row))

            if mapnum_map:
                highest = max(highest, max(mapnum_map.values()))

        new_groups[group_key] = new_rows

    return new_groups, highest