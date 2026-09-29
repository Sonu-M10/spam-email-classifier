import pandas as pd
import json
import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# Load trained AI detection model
model = joblib.load("results/ai_detection_model.pkl")
vectorizer = joblib.load("results/ai_detection_vectorizer.pkl")


# Load external human-written emails
with open("data/cleaned_emails.json", "r") as f:
    data = json.load(f)

df = pd.DataFrame(data)

# Keep human-written phishing emails
df = df[
    (df["Type"] == "Phishing") &
    (df["Created by"] == "Human")
].copy()

print("External human phishing emails:", len(df))


# Use email body/text
texts = df["Body"].fillna("").astype(str)

# Transform using the EXISTING vectorizer
X = vectorizer.transform(texts)

# Model prediction
predictions = model.predict(X)

# Human = 0, AI = 1
actual = [0] * len(df)

print("\nResults")
print("Accuracy:", accuracy_score(actual, predictions))

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        actual,
        predictions,
        labels=[0, 1]
    )
)

print("\nClassification Report:")
print(
    classification_report(
        actual,
        predictions,
        labels=[0, 1],
        target_names=["Human", "AI"],
        zero_division=0
    )
)