import pandas as pd

df = pd.read_csv("train_templates.csv")

freq = df["template"].value_counts()

print(freq.head())

freq_df = freq.reset_index()
freq_df.columns = ["template", "count"]

freq_df.to_csv("template_frequency.csv", index=False)

# attach counts
df["template_count"] = df["template"].map(freq)

df.to_csv("train_templates_with_freq.csv", index=False)

print(df.head())