import pandas as pd

df = pd.read_csv("train_templates_with_freq.csv")

rare_df = df[df["template_count"] <= 5]

print(len(rare_df))

rare_df.to_csv("rare_template_reactions.csv", index=False)