import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# Load datasets
human = pd.read_csv("data/human_corpus_sampled.csv")
llm = pd.read_csv("data/llm_corpus_sampled.csv")


# Label the sources
human["source"] = 0
llm["source"] = 1


# Combine datasets
df = pd.concat([human, llm], ignore_index=True)


print("Dataset shape:", df.shape)

print("\nSource counts:")
print(df["source"].value_counts())


# Text and target
X = df["text"]
y = df["source"]


# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Character-level TF-IDF
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


# Train Logistic Regression
model = LogisticRegression(
    max_iter=1000
)

model.fit(
    X_train_tfidf,
    y_train
)


# Predictions
y_pred = model.predict(X_test_tfidf)


# Results
print("\nAccuracy:")
print(accuracy_score(y_test, y_pred))


print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Human", "LLM"]
    )
)


print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        y_pred
    )
)