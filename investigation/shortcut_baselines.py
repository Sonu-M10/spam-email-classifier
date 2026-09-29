import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

human = pd.read_csv("data/human_corpus_sampled.csv")
llm = pd.read_csv("data/llm_corpus_sampled.csv")

human["label"] = 0
llm["label"] = 1

df = pd.concat([human, llm], ignore_index=True)

def normalize_text(text):
    text = str(text)

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " URL ", text)

    # Remove HTML
    text = re.sub(r"<[^>]+>", " ", text)

    # Replace numbers
    text = re.sub(r"\d+", " NUMBER ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    # Lowercase
    text = text.lower()

    return text.strip()

df["normalized_text"] = df["text"].fillna("").apply(normalize_text)

X = df["normalized_text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5),
    max_features=100000
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

pred = model.predict(X_test_tfidf)

print("Normalized character TF-IDF accuracy:",
      accuracy_score(y_test, pred))