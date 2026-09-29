import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load cleaned dataset
df = pd.read_csv("data/cleaned_emails.csv")

X = df["text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

vectorizer = TfidfVectorizer(max_features=100000)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=1000)

model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)

print("Accuracy:", accuracy_score(y_test, y_pred))

feature_names = vectorizer.get_feature_names_out()

for class_index, class_name in enumerate(["Legitimate", "Phishing", "Spam"]):

    coefficients = model.coef_[class_index]

    top_features = coefficients.argsort()[-20:][::-1]

    print(f"\nTop features for {class_name}:")

    for i in top_features:
        print(feature_names[i], coefficients[i])