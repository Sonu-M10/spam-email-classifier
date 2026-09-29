import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


df = pd.read_csv("data/cleaned_emails.csv")

print(df.shape)
print(df[["label", "text"]].head(10))

X = df["text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
print(X_train.shape)
print(X_test.shape)
print(y_train.value_counts(normalize=True))
print(y_test.value_counts(normalize=True))

vectorizer = TfidfVectorizer(
    max_features=100000
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
print(X_train_tfidf.shape)
print(X_test_tfidf.shape)

model = LogisticRegression(max_iter=1000)

model.fit(X_train_tfidf, y_train)
y_pred = model.predict(X_test_tfidf)
print("Accuracy:", accuracy_score(y_test, y_pred))

print(classification_report(
    y_test,
    y_pred,
    target_names=["Legitimate", "Phishing", "Spam"]
))

cm = confusion_matrix(y_test, y_pred)

print(cm)

import joblib

joblib.dump(model, "results/traditional_model.pkl")
joblib.dump(vectorizer, "results/traditional_vectorizer.pkl")

print("\nTraditional model saved.")
print("Traditional vectorizer saved.")