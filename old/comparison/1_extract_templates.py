import pandas as pd
from rxnutils.chem.reaction import ChemicalReaction
from tqdm import tqdm


def main():

    df = pd.read_csv("train_rxn.csv")

    templates = []
    failed = 0

    for rxn in tqdm(df["rxn_smiles"]):

        try:
            rxn = rxn.strip()

            reaction = ChemicalReaction(rxn)

            reaction.generate_reaction_template(radius=0)

            template = reaction.retro_template.smarts

            templates.append(template)

        except Exception as e:
            print(e)
            templates.append(None)
            failed += 1

    df["template"] = templates

    print(f"Failed reactions: {failed}")

    df = df.dropna(subset=["template"])

    df.to_csv("train_templates.csv", index=False)

    print(df.head())


if __name__ == "__main__":
    main()