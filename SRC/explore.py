import pandas as pd
df = pd.read_csv("data/df.csv")

### understanding and cleaning the data set
print(df.shape)
print(df.columns)
print(df["label"].unique())
print(df["label"].value_counts())
print(df["text"].head())
print(df["text"].isnull().sum())
print(df[df["text"].isnull()])


df = df.dropna(subset=["text"])
print(df.shape)
print(df["text"].isnull().sum())

print(df.duplicated().sum())
print(df["text"].duplicated().sum())

print(df.groupby("text")["label"].nunique().value_counts())

conflicts = df.groupby("text")["label"].nunique()
conflicts = conflicts[conflicts > 1]

print(conflicts)

print(
    df[df["text"].isin(conflicts.index)]
    .sort_values("text")
)
print(df.shape)

df = df[~df["text"].isin(conflicts.index)]
df["text"].isin(conflicts.index)
print(df.shape)

df = df.drop_duplicates(subset=["text"])
print(df.shape)

print(df["label"].value_counts())
print(df["label"].value_counts(normalize=True))

df["text_length"] = df["text"].str.len()

print(df["text_length"].describe())

print((df["text_length"] > 10000).sum())
print((df["text_length"] > 100000).sum())

print(
    df[df["text_length"] > 10000]["label"].value_counts()
)
df = df.drop(columns=["text_length"])
print(df.columns)

df.to_csv("data/cleaned_emails.csv", index=False)