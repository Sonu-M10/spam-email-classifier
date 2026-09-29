import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix





# Load datasets
human = pd.read_csv("data/human_corpus_sampled.csv")
llm = pd.read_csv("data/llm_corpus_sampled.csv")

# Add source labels
human["source"] = 0
llm["source"] = 1

# Combine datasets
df = pd.concat([human, llm], ignore_index=True)

print("Combined shape:", df.shape)

print("\nSource counts:")
print(df["source"].value_counts())

print("\nColumns:")
print(df.columns)

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())

print("\nDuplicate text:", df["text"].duplicated().sum())

print("\nLLM model counts:")
print(df[df["source"] == 1]["model"].value_counts())

print("\nText overlap between human and LLM:")
human_text = set(human["text"])
llm_text = set(llm["text"])

print(len(human_text.intersection(llm_text)))

print("\nExample email:")
print(df[["text", "subject", "body", "source"]].iloc[0])

print("\nText length:")
print(df["text"].str.len().describe())

print("\nBody length:")
print(df["body"].str.len().describe())


# Keep only the columns needed for modeling
model_df = df[["text", "source"]].copy()

print("\nModeling dataset shape:")
print(model_df.shape)

print("\nModeling dataset:")
print(model_df.head())

print("\nSource distribution:")
print(model_df["source"].value_counts())



X = model_df["text"]
y = model_df["source"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining size:")
print(X_train.shape)

print("\nTesting size:")
print(X_test.shape)

print("\nTraining source distribution:")
print(y_train.value_counts())

print("\nTesting source distribution:")
print(y_test.value_counts())


vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5),
    max_features=100000
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("\nTF-IDF training shape:")
print(X_train_tfidf.shape)

print("\nTF-IDF testing shape:")
print(X_test_tfidf.shape)


model = LogisticRegression(
    max_iter=1000
)

model.fit(X_train_tfidf, y_train)

joblib.dump(model, "results/ai_detection_model.pkl")
joblib.dump(vectorizer, "results/ai_detection_vectorizer.pkl")

print("\nModel saved to results/ai_detection_model.pkl")
print("Vectorizer saved to results/ai_detection_vectorizer.pkl")

y_pred = model.predict(X_test_tfidf)

print("\nPredictions:")
print(y_pred[:20])


accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:")
print(accuracy)

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Human", "LLM"]
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))