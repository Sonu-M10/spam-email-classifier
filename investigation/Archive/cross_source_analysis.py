import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# Load datasets
human = pd.read_csv("data/human_corpus_sampled.csv")
llm = pd.read_csv("data/llm_corpus_sampled.csv")


# Label the type of email
human["source_type"] = "Human"
llm["source_type"] = "LLM"


# Combine datasets
df = pd.concat([human, llm], ignore_index=True)


print("Dataset shape:", df.shape)

print("\nHuman source counts:")
print(human["source"].value_counts())

print("\nLLM model counts:")
print(llm["model"].value_counts())


# Get the different human sources
human_sources = human["source"].unique()

print("\nHuman sources:")
print(human_sources)


# Test each human source separately
for test_source in human_sources:

    print("\n" + "=" * 70)
    print("TESTING UNSEEN HUMAN SOURCE:", test_source)
    print("=" * 70)


    # --------------------------------------------------
    # TRAINING DATA
    #
    # All LLM emails
    # +
    # Human emails from every source EXCEPT the test source
    # --------------------------------------------------

    train_df = df[
        (df["source_type"] == "LLM") |
        (
            (df["source_type"] == "Human") &
            (df["source"] != test_source)
        )
    ].copy()


    # --------------------------------------------------
    # TESTING DATA
    #
    # LLM emails
    # +
    # Human emails from the completely unseen source
    # --------------------------------------------------

    test_df = df[
        (df["source_type"] == "LLM") |
        (
            (df["source_type"] == "Human") &
            (df["source"] == test_source)
        )
    ].copy()


    # Human = 0
    # LLM = 1

    train_df["target"] = (
        train_df["source_type"] == "LLM"
    ).astype(int)

    test_df["target"] = (
        test_df["source_type"] == "LLM"
    ).astype(int)


    # Training data
    X_train = train_df["text"]
    y_train = train_df["target"]


    # Testing data
    X_test = test_df["text"]
    y_test = test_df["target"]


    print("\nTraining size:", len(X_train))
    print("Testing size:", len(X_test))


    print("\nTraining labels:")
    print(y_train.value_counts())


    print("\nTesting labels:")
    print(y_test.value_counts())


    # --------------------------------------------------
    # TF-IDF
    # --------------------------------------------------

    vectorizer = TfidfVectorizer(
        max_features=100000
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)


    print("\nTF-IDF training shape:")
    print(X_train_tfidf.shape)

    print("TF-IDF testing shape:")
    print(X_test_tfidf.shape)


    # --------------------------------------------------
    # Logistic Regression
    # --------------------------------------------------

    model = LogisticRegression(
        max_iter=1000
    )

    model.fit(
        X_train_tfidf,
        y_train
    )


    # Predictions
    y_pred = model.predict(X_test_tfidf)


    # --------------------------------------------------
    # Results
    # --------------------------------------------------

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
    