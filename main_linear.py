# file load and clustering
import re
import os
import pandas as pd
from collections import defaultdict
from rdkit import Chem
from rdkit import rdBase
import augment_models.functionalizer_synt as functionalizer_synt
import augment_models.reaction_site as reaction_site
import linear_models.atom_changer as AC
import linear_models.atom_remover as AR
import linear_models.functionalizer as FZ
import linear_models.graph_mixer as GM
import linear_models.map_num_cheker as map_num_cheker

def get_module():
    while True:
        try:
            num = int(input("Module number (1~4), 0 for end: "))
            if num < 0 or num > 4:
                print("Out of range! Please enter an integer between 0 and 4.")
                continue
            return num
        except ValueError:
            print("Value error! Please enter an integer between 0 and 4.")

def get_size(say_some):
    while True:
        try:
            num = int(input(f"{say_some}: "))
            if num <= 0:
                print("Out of range! Please enter an integer > 0.")
                continue
            return num
        except ValueError:
            print("Value error! Please enter an integer > 0.")

def split_reaction(smiles_reaction):
    inputs, output = smiles_reaction.split('>>')  # split with '>>'        
    return pd.Series([inputs, output])

def remove_atom_map(smiles):
    return re.sub(r":\d+", "", smiles)

def get_mol(smiles):
    try:
        return Chem.MolFromSmiles(smiles)
    except:
        return None
        
def get_cluster_key(smiles):
    mol = get_mol(smiles)
    if mol:
        return Chem.MolToSmiles(mol, canonical=True)
    return None

num_tries = 0
module = 1
nrows = get_size("nrows (dataset size)")
dataset_train = pd.read_csv('original_datasets/raw_train.csv', nrows=nrows)
dataset_val = pd.read_csv('original_datasets/raw_val.csv', nrows=int(nrows * 0.2))
os.makedirs('linear_datasets', exist_ok=True)
dataset_train.to_csv('linear_datasets/train.csv', index=False)
dataset_val.to_csv('linear_datasets/val.csv', index=False)

while module != 0:
    module = get_module()
    dataset_train = pd.read_csv('linear_datasets/train.csv')
    dataset_val = pd.read_csv('linear_datasets/val.csv')
    similarities = [0.85, 0.85, 0.8, 0.8]
    
    if module != 0:
        similarity_value = similarities[module-1]
    else:
        print("end")
        break
  
    dataset_train[['inputs', 'output']] = dataset_train.iloc[:, 2].apply(split_reaction)
    dataset_val[['inputs', 'output']] = dataset_val.iloc[:, 2].apply(split_reaction)
    smiles_columns = ['inputs', 'output']

    dataset_train[['class', 'fg_site']] = pd.DataFrame(
        dataset_train.apply(
            lambda row: reaction_site.get_reaction_center(row['inputs'], row['output'], depth=1)[:2], axis=1).tolist())
    dataset_val[['class', 'fg_site']] = pd.DataFrame(
        dataset_val.apply(
            lambda row: reaction_site.get_reaction_center(row['inputs'], row['output'], depth=1)[:2], axis=1).tolist())
    
    rdBase.DisableLog('rdApp.*')
    
    reaction_groups = defaultdict(list)
    reaction_groups_val = defaultdict(list)

    for idx, row in dataset_train.iterrows():
        fg_raw = row['fg_site']
        if not fg_raw:
            continue
    
        fg_clean = remove_atom_map(fg_raw)
        key = get_cluster_key(fg_clean)
        assigned_key = None
    
        if key:
            is_assigned = False
            for key_old, group_rows in reaction_groups.items():
                existing_fg_clean = remove_atom_map(group_rows[0]['fg_site'])
                sim = functionalizer_synt.calculate_similarity(existing_fg_clean, fg_clean, 2)
                if sim >= similarity_value:
                    reaction_groups[key_old].append(row)
                    is_assigned = True
                    break
    
            if not is_assigned:
                reaction_groups[key].append(row)
    
    print(f"clusters (train): {len(reaction_groups)}")

    for idx, row in dataset_val.iterrows():
        fg_raw = row['fg_site']
        if not fg_raw:
            continue
    
        fg_clean = remove_atom_map(fg_raw)
        key = get_cluster_key(fg_clean)
    
        if key:
            is_assigned = False
            for key_old, group_rows in reaction_groups_val.items():
                existing_fg_clean = remove_atom_map(group_rows[0]['fg_site'])
                sim = functionalizer_synt.calculate_similarity(existing_fg_clean, fg_clean, 2)
                if sim >= similarity_value:
                    reaction_groups_val[key_old].append(row)
                    is_assigned = True
                    break
    
            if not is_assigned:
                reaction_groups_val[key].append(row)
    
    print(f"clusters (val): {len(reaction_groups_val)}")

    reaction_groups, highest = map_num_cheker.checkMN(reaction_groups)
    reaction_groups_val, highest = map_num_cheker.checkMN(reaction_groups_val)

    if highest >= 250:
        print("MAPNUM is higher than 250, too many fgs attached! (low accuracy)")
        break
              
    if module == 1:
        print("Atom Changer Selected")
        n_iter = get_size("n_iter?: ")
        AC.atom_changer_linear_mod(reaction_groups, reaction_groups_val, n_iter)
        num_tries += 1
        continue
    elif module == 2:
        print("Atom Remover Selected")
        n_iter = get_size("n_iter?: ")
        AR.atom_remover_linear_mod(reaction_groups, reaction_groups_val, n_iter)
        num_tries += 1
        continue        
    elif module == 3:
        print("Functionalizer Selected")
        n_iter = get_size("n_iter?: ")
        FZ.functionalizer_linear_mod(reaction_groups, reaction_groups_val, n_iter)
        num_tries += 1
        continue
    elif module == 4:
        print("Graph Mixer Selected")
        n_iter = get_size("n_iter?: ")
        GM.graph_mixer_linear_mod(reaction_groups, reaction_groups_val, n_iter)
        num_tries += 1
        continue
    elif module == 0:
        print("END augmentation")
        break
    else:
        print("Value error!")
        break

#remove same data
print("Eliminating same data")

def canonical_smiles_list(smiles_str):
    mols = smiles_str.split('.')
    can_smis = []
    for mol in mols:
        try:
            m = Chem.MolFromSmiles(mol)
            if m:
                can_smis.append(Chem.MolToSmiles(m, canonical=True, isomericSmiles=True))
        except:
            pass
    return sorted(can_smis)

def reaction_canonical_key(reaction_str):
    try:
        reactants_reagents, products = reaction_str.split('>>')
    except:
        return reaction_str

    react_reag_cans = canonical_smiles_list(reactants_reagents)
    prod_cans = canonical_smiles_list(products)
    key = '.'.join(react_reag_cans) + '>>' + '.'.join(prod_cans)
    
    return key

dataset_train = pd.read_csv('linear_datasets/train.csv')
dataset_val = pd.read_csv('linear_datasets/val.csv')
dataset_train['reaction_key'] = dataset_train['reactants>reagents>production'].apply(reaction_canonical_key)
dataset_val['reaction_key'] = dataset_val['reactants>reagents>production'].apply(reaction_canonical_key)
dataset_train_unique = dataset_train.drop_duplicates(subset=['reaction_key']).drop(columns=['reaction_key'])
dataset_val_unique = dataset_val.drop_duplicates(subset=['reaction_key']).drop(columns=['reaction_key'])
dataset_train_unique.to_csv('linear_datasets/train.csv', index=False)
dataset_val_unique.to_csv('linear_datasets/val.csv', index=False)

print("Elimination complete")