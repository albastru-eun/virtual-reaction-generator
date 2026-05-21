from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd
from tqdm import tqdm

# -------------------------
# functional group mapping
# -------------------------

REPLACEMENTS = {
    "Cl": ["Br", "I"],
    "Br": ["Cl", "I"],
    "I": ["Br"],
    "F": ["Cl"],
    "OC": ["OCC"],   # OMe -> OEt
}

# -------------------------
# simple validity check
# -------------------------

def is_valid(smiles):

    try:
        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            return False

        Chem.SanitizeMol(mol)

        return True

    except:
        return False

# -------------------------
# augmentation
# -------------------------

def augment_rxn(rxn):

    augmented = []

    reactants, products = rxn.split(">>")

    for fg, replacements in REPLACEMENTS.items():

        if fg in reactants:

            for rep in replacements:

                new_reactants = reactants.replace(fg, rep)

                if is_valid(new_reactants):

                    new_rxn = new_reactants + ">>" + products

                    augmented.append(new_rxn)

    return augmented

# -------------------------
# load rare template data
# -------------------------

df = pd.read_csv("rare_template_reactions.csv")

all_rxns = []

for rxn in tqdm(df["rxn_smiles"]):

    all_rxns.append(rxn)

    augmented = augment_rxn(rxn)

    all_rxns.extend(augmented)

# remove duplicates
all_rxns = list(set(all_rxns))

# save
aug_df = pd.DataFrame({
    "rxn_smiles": all_rxns
})

aug_df.to_csv(
    "fg_augmented_train.csv",
    index=False
)

print("Final size:", len(aug_df))