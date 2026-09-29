import pandas as pd
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# --------------------------------------------------
# 1. Load datasets
# --------------------------------------------------

human = pd.read_csv("data/human_corpus_sampled.csv")
llm = pd.read_csv("data/llm_corpus_sampled.csv")

human["source"] = 0
llm["source"] = 1

df = pd.concat([human, llm], ignore_index=True)

print("Dataset shape:", df.shape)

print("\nSource counts:")
print(df["source"].value_counts())


# --------------------------------------------------
# 2. Remove common phishing-topic words
# --------------------------------------------------

topic_words = [
    "tax",
    "taxes",
    "refund",
    "refunds",
    "security",
    "secure",
    "verification",
    "verify",
    "account",
    "accounts",
    "password",
    "login",
    "payment",
    "payments",
    "bank",
    "banking",
    "credit",
    "card",
    "paypal",
    "invoice",
    "invoices",
    "delivery",
    "parcel",
    "package",
    "shipping",
    "shipment",
    "portal",
    "access",
    "document",
    "documents",
    "enrollment",
    "benefits",
    "compliance",
    "hr",
    "human",
    "resources",
    "support",
    "urgent",
    "urgently",
    "click",
    "link",
    "action",
    "confirm",
    "confirmation",
    "identity",
    "information",
    "personal",
    "credentials"
]


def remove_topic_words(text):
    text = str(text)

    for word in topic_words:
        text = re.sub(
            r"\b" + re.escape(word) + r"\b",
            " ",
            text,
            flags=re.IGNORECASE
        )

    text = re.sub(r"\s+", " ", text)

    return text.strip()


df["clean_text"] = df["text"].apply(remove_topic_words)


# --------------------------------------------------
# 3. Show an example
# --------------------------------------------------

print("\nOriginal example:")
print(df["text"].iloc[0][:500])

print("\nAfter removing topic words:")
print(df["clean_text"].iloc[0][:500])


# --------------------------------------------------
# 4. Train/test split
# --------------------------------------------------

X = df["clean_text"]
y = df["source"]

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


# --------------------------------------------------
# 5. TF-IDF
# --------------------------------------------------

vectorizer = TfidfVectorizer(
    max_features=100000,
    stop_words="english"
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("\nTF-IDF training shape:")
print(X_train_tfidf.shape)

print("\nTF-IDF testing shape:")
print(X_test_tfidf.shape)


# --------------------------------------------------
# 6. Train model
# --------------------------------------------------

model = LogisticRegression(max_iter=1000)

model.fit(X_train_tfidf, y_train)


# --------------------------------------------------
# 7. Evaluate
# --------------------------------------------------

y_pred = model.predict(X_test_tfidf)

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
print(confusion_matrix(y_test, y_pred))