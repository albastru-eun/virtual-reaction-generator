import augment_models.atom_changer as atom_changer
from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd

def atom_changer_process(clusters_dict, n_iter=5):
    all_results = []

    for cluster_name, rows in clusters_dict.items():
        for row in rows:
            reaction_center = row.get("class")
            if not reaction_center or not isinstance(reaction_center, (set, list)):
                #print(f"[Skip] Cluster {cluster_name} no valid reaction center in class")
                continue

            original_smiles_i = [row["inputs"]]
            original_smiles_o = [row["output"]]
            seen_smiles_i = set(original_smiles_i)
            seen_smiles_o = set(original_smiles_o)
            generated_smiles_i = []
            generated_smiles_o = []
            id_real = row["id"]

            for _ in range(n_iter):
                mol_i = Chem.MolFromSmiles(row["inputs"])
                mol_o = Chem.MolFromSmiles(row["output"])

                if mol_i is None:
                    #print(f"[Error] Invalid SMILES in (inputs) {cluster_name}: {row['inputs']}")
                    break
                if mol_o is None:
                    #print(f"[Error] Invalid SMILES in (output) {cluster_name}: {row['output']}")
                    break

                mol_i = Chem.AddHs(mol_i)
                mol_o = Chem.AddHs(mol_o)

                modified_i, modified_o = atom_changer.modify_atoms(mol_i, mol_o, reaction_center)
                

                if modified_i is None or modified_o is None:
                    #print("[Warning] One of the modified molecules is None, skipping.")
                    continue

                try:
                    Chem.SanitizeMol(modified_i)
                    Chem.SanitizeMol(modified_o)
                except Chem.rdchem.KekulizeException:
                    Chem.SanitizeMol(modified_i, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                    Chem.SanitizeMol(modified_o, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                except Chem.AtomValenceException as e:
                    #print(f"[Valence Error] {e}")
                    continue

                modified_i = Chem.RemoveHs(modified_i)
                modified_o = Chem.RemoveHs(modified_o)

                smi_i = Chem.MolToSmiles(modified_i, kekuleSmiles=False, canonical=True)
                smi_o = Chem.MolToSmiles(modified_o, kekuleSmiles=False, canonical=True)

                if smi_i not in seen_smiles_i and smi_o not in seen_smiles_o:
                    seen_smiles_i.add(smi_i)
                    seen_smiles_o.add(smi_o)
                    generated_smiles_i.append(smi_i)
                    generated_smiles_o.append(smi_o)
                #else:
                    #print(f"[Duplicate] Skipped {smi_i} / {smi_o}")

            if generated_smiles_i and generated_smiles_o:
                all_results.append({
                    "cluster": cluster_name,
                    "generated_input": generated_smiles_i,
                    "generated_output": generated_smiles_o,
                    "id": id_real
                })

    return all_results