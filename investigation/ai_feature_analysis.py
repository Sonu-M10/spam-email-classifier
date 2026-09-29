import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

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
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Convert text to TF-IDF
vectorizer = TfidfVectorizer(max_features=100000)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# Get feature names
feature_names = vectorizer.get_feature_names_out()

# Get coefficients
coefficients = model.coef_[0]

# Features most associated with Human
human_features = coefficients.argsort()[:30]

print("\n" + "=" * 60)
print("TOP FEATURES ASSOCIATED WITH HUMAN EMAILS")
print("=" * 60)

for i in human_features:
    print(feature_names[i], coefficients[i])

# Features most associated with LLM
llm_features = coefficients.argsort()[-30:][::-1]

print("\n" + "=" * 60)
print("TOP FEATURES ASSOCIATED WITH LLM EMAILS")
print("=" * 60)

for i in llm_features:
    print(feature_names[i], coefficients[i])