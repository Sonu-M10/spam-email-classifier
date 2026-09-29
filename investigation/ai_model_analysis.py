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


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    human = pd.read_csv("data/human_corpus_sampled.csv")
    llm = pd.read_csv("data/llm_corpus_sampled.csv")

    human["source"] = 0
    llm["source"] = 1

    df = pd.concat([human, llm], ignore_index=True)

    return df


# ============================================================
# COMMON MODEL FUNCTION
# ============================================================

def run_tfidf_model(X, y, stop_words=None, analyzer="word",
                    ngram_range=(1, 1), title="Experiment"):

    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    vectorizer = TfidfVectorizer(
        max_features=100000,
        stop_words=stop_words,
        analyzer=analyzer,
        ngram_range=ngram_range
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print("\nTraining shape:")
    print(X_train_tfidf.shape)

    print("\nTesting shape:")
    print(X_test_tfidf.shape)

    model = LogisticRegression(max_iter=1000)

    model.fit(X_train_tfidf, y_train)

    y_pred = model.predict(X_test_tfidf)

    accuracy = accuracy_score(y_test, y_pred)

    print("\nAccuracy:")
    print(accuracy)

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

    return accuracy


# ============================================================
# EXPERIMENT 1
# WORD TF-IDF
# ============================================================

def word_tfidf_experiment(df):

    return run_tfidf_model(
        X=df["text"],
        y=df["source"],
        title="EXPERIMENT 1: WORD TF-IDF"
    )


# ============================================================
# EXPERIMENT 2
# CHARACTER TF-IDF
# ============================================================

def character_tfidf_experiment(df):

    return run_tfidf_model(
        X=df["text"],
        y=df["source"],
        analyzer="char",
        ngram_range=(3, 5),
        title="EXPERIMENT 2: CHARACTER TF-IDF"
    )


# ============================================================
# EXPERIMENT 3
# CLEANED CHARACTER TF-IDF
# ============================================================

def clean_text(text):

    text = str(text)

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", " ", text)

    # Remove email addresses
    text = re.sub(r"\S+@\S+", " ", text)

    # Remove numbers
    text = re.sub(r"\d+", " ", text)

    # Remove repeated formatting characters
    text = re.sub(r"[_=\-]{2,}", " ", text)

    # Clean extra spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def cleaned_character_experiment(df):

    print("\nCleaning text...")

    df = df.copy()

    df["clean_text"] = df["text"].apply(clean_text)

    print("\nExample original:")
    print(df["text"].iloc[0][:300])

    print("\nExample cleaned:")
    print(df["clean_text"].iloc[0][:300])

    return run_tfidf_model(
        X=df["clean_text"],
        y=df["source"],
        analyzer="char",
        ngram_range=(3, 5),
        title="EXPERIMENT 3: CLEANED CHARACTER TF-IDF"
    )


# ============================================================
# EXPERIMENT 4
# WORD TF-IDF + STOPWORD REMOVAL
# ============================================================

def stopword_experiment(df):

    return run_tfidf_model(
        X=df["text"],
        y=df["source"],
        stop_words="english",
        title="EXPERIMENT 4: WORD TF-IDF + STOPWORD REMOVAL"
    )


# ============================================================
# EXPERIMENT 5
# REMOVE COMMON PHISHING TOPIC WORDS
# ============================================================

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


def topic_control_experiment(df):

    print("\nRemoving common phishing-topic words...")

    df = df.copy()

    df["topic_clean_text"] = df["text"].apply(remove_topic_words)

    return run_tfidf_model(
        X=df["topic_clean_text"],
        y=df["source"],
        stop_words="english",
        title="EXPERIMENT 5: TOPIC/CONTENT CONTROL"
    )


# ============================================================
# EXPERIMENT 6
# HUMAN SOURCE CLASSIFICATION
# ============================================================

def human_source_experiment():

    print("\n" + "=" * 70)
    print("EXPERIMENT 6: HUMAN SOURCE CLASSIFICATION")
    print("=" * 70)

    df = pd.read_csv("data/human_corpus_sampled.csv")

    print("\nHuman dataset shape:")
    print(df.shape)

    print("\nHuman source counts:")
    print(df["source"].value_counts())

    X = df["text"]
    y = df["source"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    vectorizer = TfidfVectorizer(
        max_features=100000,
        stop_words="english"
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print("\nTraining shape:")
    print(X_train_tfidf.shape)

    print("\nTesting shape:")
    print(X_test_tfidf.shape)

    model = LogisticRegression(max_iter=1000)

    model.fit(X_train_tfidf, y_train)

    y_pred = model.predict(X_test_tfidf)

    accuracy = accuracy_score(y_test, y_pred)

    print("\nAccuracy:")
    print(accuracy)

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred
        )
    )

    return accuracy


# ============================================================
# EXPERIMENT 7
# STRICT CROSS-SOURCE VALIDATION
# ============================================================

def strict_cross_source_experiment():

    print("\n" + "=" * 70)
    print("EXPERIMENT 7: STRICT CROSS-SOURCE VALIDATION")
    print("=" * 70)

    human = pd.read_csv("data/human_corpus_sampled.csv")
    llm = pd.read_csv("data/llm_corpus_sampled.csv")

    human_sources = human["source"].unique()
    llm_models = llm["model"].dropna().unique()

    print("\nHuman sources:")
    print(human_sources)

    print("\nLLM models:")
    print(llm_models)

    results = []

    for held_out_human in human_sources:

        for held_out_llm in llm_models:

            print("\n" + "-" * 70)
            print(
                f"Held out Human: {held_out_human} | "
                f"Held out LLM: {held_out_llm}"
            )
            print("-" * 70)

            # ----------------------------------------------
            # Training data
            # ----------------------------------------------

            train_human = human[
                human["source"] != held_out_human
            ]

            train_llm = llm[
                llm["model"] != held_out_llm
            ]

            train_human = train_human.copy()
            train_llm = train_llm.copy()

            train_human["label"] = 0
            train_llm["label"] = 1

            train = pd.concat(
                [train_human, train_llm],
                ignore_index=True
            )

            # ----------------------------------------------
            # Test data
            # ----------------------------------------------

            test_human = human[
                human["source"] == held_out_human
            ]

            test_llm = llm[
                llm["model"] == held_out_llm
            ]

            test_human = test_human.copy()
            test_llm = test_llm.copy()

            test_human["label"] = 0
            test_llm["label"] = 1

            test = pd.concat(
                [test_human, test_llm],
                ignore_index=True
            )

            # ----------------------------------------------
            # TF-IDF
            # ----------------------------------------------

            vectorizer = TfidfVectorizer(
                max_features=100000,
                stop_words="english"
            )

            X_train = vectorizer.fit_transform(
                train["text"]
            )

            X_test = vectorizer.transform(
                test["text"]
            )

            y_train = train["label"]
            y_test = test["label"]

            # ----------------------------------------------
            # Train model
            # ----------------------------------------------

            model = LogisticRegression(
                max_iter=1000
            )

            model.fit(X_train, y_train)

            # ----------------------------------------------
            # Evaluate
            # ----------------------------------------------

            y_pred = model.predict(X_test)

            accuracy = accuracy_score(
                y_test,
                y_pred
            )

            print(f"\nAccuracy: {accuracy:.4f}")

            print("\nConfusion Matrix:")
            print(
                confusion_matrix(
                    y_test,
                    y_pred
                )
            )

            results.append({
                "Human Source": held_out_human,
                "LLM Model": held_out_llm,
                "Accuracy": accuracy
            })

    # ========================================================
    # SUMMARY
    # ========================================================

    results_df = pd.DataFrame(results)

    print("\n\n" + "=" * 70)
    print("STRICT CROSS-SOURCE SUMMARY")
    print("=" * 70)

    print(
        results_df.to_string(index=False)
    )

    print("\nAverage accuracy:")
    print(
        results_df["Accuracy"].mean()
    )

    print("\nMinimum accuracy:")
    print(
        results_df["Accuracy"].min()
    )

    return results_df


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("AI-GENERATED VS HUMAN-WRITTEN PHISHING INVESTIGATION")
    print("=" * 70)

    df = load_data()

    print("\nCombined dataset shape:")
    print(df.shape)

    print("\nSource counts:")
    print(df["source"].value_counts())

    results = {}

    # Experiment 1
    results["Word TF-IDF"] = word_tfidf_experiment(df)

    # Experiment 2
    results["Character TF-IDF"] = character_tfidf_experiment(df)

    # Experiment 3
    results["Cleaned Character TF-IDF"] = cleaned_character_experiment(df)

    # Experiment 4
    results["Stopword Removal"] = stopword_experiment(df)

    # Experiment 5
    results["Topic Control"] = topic_control_experiment(df)

    # Experiment 6
    results["Human Source Classification"] = human_source_experiment()
    
    # Experiment 7
    strict_results = strict_cross_source_experiment()


    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n\n" + "=" * 70)
    print("FINAL EXPERIMENT SUMMARY")
    print("=" * 70)

    for experiment, accuracy in results.items():

        print(
            f"{experiment:<40} {accuracy:.4f}"
        )


if __name__ == "__main__":
    main()