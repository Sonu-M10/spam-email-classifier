import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# Load external validation dataset
df = pd.read_csv("data/Phishing_validation_emails.csv")

print("External dataset:", len(df))
print(df["Email Type"].value_counts())


# Load trained traditional model
model = joblib.load("results/traditional_model.pkl")
vectorizer = joblib.load("results/traditional_vectorizer.pkl")


# Prepare emails
texts = df["Email Text"].fillna("").astype(str)

X = vectorizer.transform(texts)


# External dataset:
# Safe Email = 0
# Phishing Email = 1
y_true = df["Email Type"].map({
    "Safe Email": 0,
    "Phishing Email": 1
})


# Original model predictions:
# 0 = Legitimate
# 1 = Phishing
# 2 = Spam
y_pred_original = model.predict(X)


# Convert the 3-class model into a binary phishing detector:
# Phishing (1) = Phishing
# Legitimate (0) and Spam (2) = Not Phishing
y_pred = (y_pred_original == 1).astype(int)


# Results
print("\nResults")
print("Accuracy:", accuracy_score(y_true, y_pred))


print("\nOriginal Model Prediction Counts:")
print(
    pd.Series(y_pred_original)
    .value_counts()
    .sort_index()
)


print("\nBinary Prediction Counts:")
print(
    pd.Series(y_pred)
    .value_counts()
    .sort_index()
)


print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    )
)


print("\nClassification Report:")
print(
    classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["Safe", "Phishing"],
        zero_division=0
    )
)