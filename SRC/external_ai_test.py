import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load independent GPT dataset
df = pd.read_csv(
    "data/independent_llm_dataset/GPT_Phishing_Email_dataset.csv"
)

# Load trained AI detector
model = joblib.load("results/ai_detection_model.pkl")
vectorizer = joblib.load("results/ai_detection_vectorizer.pkl")

texts = df["content"].fillna("").astype(str)

X = vectorizer.transform(texts)

# Dataset labels:
# 0 = Human/legitimate
# 1 = AI-generated phishing
y_true = df["label"]

y_pred = model.predict(X)

print("Independent GPT dataset:", len(df))

print("\nResults")
print("Accuracy:", accuracy_score(y_true, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))

print("\nClassification Report:")
print(
    classification_report(
        y_true,
        y_pred,
        target_names=["Human", "AI"]
    )
)