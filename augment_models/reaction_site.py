from rdkit import Chem
#from rdkit.Chem.rdMolDescriptors import GetFunctionalGroups

#find reaction site
def expand_functional_group(atom, mol, depth):
    fg_atoms = set()
    visited = set()
    queue = [(atom, 0)]

    while queue:
        current_atom, d = queue.pop(0)
        idx = current_atom.GetIdx()
        if idx in visited or d > depth:
            continue
        visited.add(idx)

        map_num = current_atom.GetAtomMapNum()
        if map_num > 0:
            fg_atoms.add(map_num)

        for neighbor in current_atom.GetNeighbors():
            queue.append((neighbor, d + 1))

    return fg_atoms

def get_atom_env_smiles(mol, atom_idx, depth):
    try:
        env = Chem.FindAtomEnvironmentOfRadiusN(mol, depth, atom_idx)
        if not env:
            return ""
        amap = {}
        submol = Chem.PathToSubmol(mol, env, atomMap=amap)
        smiles = Chem.MolToSmiles(submol)
        return smiles
    except:
        return ""
        
def get_neighbors_with_bonds_all(atom, mol):
    neighbors = set()
    for neighbor in atom.GetNeighbors():
        bond = mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx())
        if bond is not None:
            map_num = neighbor.GetAtomMapNum()
            if map_num > 0:
                neighbors.add((map_num, bond.GetBondType()))
            else:
                neighbors.add((f"idx{neighbor.GetIdx()}", bond.GetBondType()))
    return neighbors

def get_reaction_center(input_smiles, output_smiles, depth):
    reactant_mol = Chem.MolFromSmiles(input_smiles)
    product_mol = Chem.MolFromSmiles(output_smiles)

    changed_atoms = set()

    reactant_atoms = {
        atom.GetAtomMapNum(): atom
        for atom in reactant_mol.GetAtoms()
        if atom.GetAtomMapNum() > 0
    }
    product_atoms = {
        atom.GetAtomMapNum(): atom
        for atom in product_mol.GetAtoms()
        if atom.GetAtomMapNum() > 0
    }

    common_keys = reactant_atoms.keys() & product_atoms.keys()

    for map_num in common_keys:
        atom_r = reactant_atoms[map_num]
        atom_p = product_atoms[map_num]

        if atom_r.GetSymbol() != atom_p.GetSymbol() or atom_r.GetFormalCharge() != atom_p.GetFormalCharge():
            changed_atoms.add(map_num)
            continue

        r_neighbors = get_neighbors_with_bonds_all(atom_r, reactant_mol)
        p_neighbors = get_neighbors_with_bonds_all(atom_p, product_mol)

        if r_neighbors != p_neighbors:
            changed_atoms.add(map_num)

    expanded_atoms = set()
    for map_num in changed_atoms:
        if map_num in reactant_atoms:
            atom = reactant_atoms[map_num]
            expanded_atoms.update(expand_functional_group(atom, reactant_mol, depth))
        elif map_num in product_atoms:
            atom = product_atoms[map_num]
            expanded_atoms.update(expand_functional_group(atom, product_mol, depth))

    smiles_envs = []

    for map_num in expanded_atoms:
        if map_num in reactant_atoms:
            atom = reactant_atoms[map_num]
            env_smiles = get_atom_env_smiles(reactant_mol, atom.GetIdx(), depth)
        elif map_num in product_atoms:
            atom = product_atoms[map_num]
            env_smiles = get_atom_env_smiles(product_mol, atom.GetIdx(), depth)
        else:
            continue

        if env_smiles:
            smiles_envs.append(env_smiles)

    smiles_env = '.'.join(smiles_envs) if smiles_envs else ''

    return expanded_atoms, smiles_env