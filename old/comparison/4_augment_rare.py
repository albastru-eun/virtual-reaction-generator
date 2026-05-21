import pandas as pd

df = pd.read_csv("train_templates_with_freq.csv")

augmented_rows = []

for _, row in df.iterrows():

    freq = row["template_count"]

    if freq == 1:
        repeat = 20

    elif freq <= 5:
        repeat = 10

    elif freq <= 10:
        repeat = 5

    else:
        repeat = 1

    augmented_rows.extend([row] * repeat)

augmented = pd.DataFrame(augmented_rows)

augmented[["rxn_smiles"]].to_csv(
    "output_datasets\train_template_aug.csv",
    index=False
)

print(len(augmented))