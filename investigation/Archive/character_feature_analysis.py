import pandas as pd
from sklearn.model_selection import train_test_split
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


# Train Logistic Regression
model = LogisticRegression(
    max_iter=1000
)

model.fit(
    X_train_tfidf,
    y_train
)


# Get character features
feature_names = vectorizer.get_feature_names_out()

coefficients = model.coef_[0]


# -----------------------------------------
# HUMAN FEATURES
# -----------------------------------------

human_features = coefficients.argsort()[:30]

print("\n" + "=" * 60)
print("TOP CHARACTER FEATURES ASSOCIATED WITH HUMAN EMAILS")
print("=" * 60)

for i in human_features:
    print(
        repr(feature_names[i]),
        coefficients[i]
    )


# -----------------------------------------
# LLM FEATURES
# -----------------------------------------

llm_features = coefficients.argsort()[-30:][::-1]

print("\n" + "=" * 60)
print("TOP CHARACTER FEATURES ASSOCIATED WITH LLM EMAILS")
print("=" * 60)

for i in llm_features:
    print(
        repr(feature_names[i]),
        coefficients[i]
    )