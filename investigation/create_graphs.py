import matplotlib.pyplot as plt
import os


# --------------------------------------------------
# Graph output folder
# --------------------------------------------------

GRAPH_FOLDER = "results/graphs"

os.makedirs(GRAPH_FOLDER, exist_ok=True)


# --------------------------------------------------
# Traditional model accuracy
# --------------------------------------------------

models = [
    "Logistic Regression",
    "Linear SVM",
    "Naive Bayes"
]

accuracy = [
    97.02,
    97.31,
    91.79
]


plt.figure(figsize=(9, 6))

bars = plt.bar(models, accuracy)

plt.title("Traditional NLP Model Accuracy", fontsize=16)
plt.xlabel("Model")
plt.ylabel("Accuracy (%)")

plt.ylim(0, 100)


# Add values above bars
for bar, value in zip(bars, accuracy):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 1,
        f"{value:.2f}%",
        ha="center",
        fontsize=11
    )


plt.tight_layout()


# --------------------------------------------------
# Save graph
# --------------------------------------------------

output_path = os.path.join(
    GRAPH_FOLDER,
    "traditional_model_accuracy.png"
)

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

print(f"Graph saved to: {output_path}")

plt.show()

# --------------------------------------------------
# AI detection recall across datasets
# --------------------------------------------------

datasets = [
    "Original Human vs LLM",
    "Independent GPT",
    "Chinese / DeepSeek"
]

ai_recall = [
    100.00,
    78.99,
    0.00
]


plt.figure(figsize=(9, 6))

bars = plt.bar(datasets, ai_recall)

plt.title("AI Detection Recall Across Datasets", fontsize=16)
plt.xlabel("Dataset")
plt.ylabel("AI Recall (%)")

plt.ylim(0, 100)


# Add values above bars
for bar, value in zip(bars, ai_recall):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 2,
        f"{value:.2f}%",
        ha="center",
        fontsize=11
    )


plt.tight_layout()


# --------------------------------------------------
# Save graph
# --------------------------------------------------

output_path = os.path.join(
    GRAPH_FOLDER,
    "ai_detection_recall.png"
)

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

print(f"Graph saved to: {output_path}")

plt.show()

# --------------------------------------------------
# Shortcut / robustness experiments
# --------------------------------------------------

experiments = [
    "Length only",
    "Formatting features",
    "Normalized\ncharacter TF-IDF"
]

experiment_accuracy = [
    68.37,
    100.00,
    100.00
]


plt.figure(figsize=(9, 6))

bars = plt.bar(experiments, experiment_accuracy)

plt.title("AI Detection Shortcut Experiments", fontsize=16)
plt.xlabel("Experiment")
plt.ylabel("Accuracy (%)")

plt.ylim(0, 100)


# Add values above bars
for bar, value in zip(bars, experiment_accuracy):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 2,
        f"{value:.2f}%",
        ha="center",
        fontsize=11
    )


plt.tight_layout()


# --------------------------------------------------
# Save graph
# --------------------------------------------------

output_path = os.path.join(
    GRAPH_FOLDER,
    "ai_detection_shortcut_experiments.png"
)

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

print(f"Graph saved to: {output_path}")

plt.show()

# --------------------------------------------------
# AI vs Human feature analysis
# --------------------------------------------------

human_features = [
    "my",
    "mail",
    "me",
    "money",
    "usaa",
    "http",
    "monkey",
    "org",
    "here",
    "paypal"
]

llm_features = [
    "your",
    "support",
    "delivery",
    "portal",
    "tax",
    "security",
    "refund",
    "verification",
    "access",
    "compliance"
]


plt.figure(figsize=(10, 7))

# Use positions so both groups can appear on one graph
y_positions = range(len(human_features))

plt.barh(
    list(y_positions),
    range(len(human_features)),
    label="Human-associated features"
)

plt.barh(
    list(y_positions),
    [-x for x in range(1, len(llm_features) + 1)],
    label="LLM-associated features"
)

plt.yticks(
    list(y_positions),
    human_features
)

plt.xlabel("Relative feature strength")
plt.title(
    "Features Associated With Human and LLM Emails",
    fontsize=16
)

plt.legend()

plt.tight_layout()


# --------------------------------------------------
# Save graph
# --------------------------------------------------

output_path = os.path.join(
    GRAPH_FOLDER,
    "ai_feature_analysis.png"
)

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

print(f"Graph saved to: {output_path}")

plt.show()

# --------------------------------------------------
# 4. Top Human vs LLM Features
# --------------------------------------------------

human = pd.read_csv("data/human_corpus_sampled.csv")
llm = pd.read_csv("data/llm_corpus_sampled.csv")

human["source"] = 0
llm["source"] = 1

df = pd.concat([human, llm], ignore_index=True)

X = df["text"]
y = df["source"]

vectorizer = TfidfVectorizer(max_features=100000)

X_tfidf = vectorizer.fit_transform(X)

model = LogisticRegression(max_iter=1000)
model.fit(X_tfidf, y)

feature_names = vectorizer.get_feature_names_out()
coefficients = model.coef_[0]

# Get top 10 Human features
human_indices = coefficients.argsort()[:10]

human_features = pd.DataFrame({
    "feature": feature_names[human_indices],
    "coefficient": coefficients[human_indices]
})

# Get top 10 LLM features
llm_indices = coefficients.argsort()[-10:][::-1]

llm_features = pd.DataFrame({
    "feature": feature_names[llm_indices],
    "coefficient": coefficients[llm_indices]
})

# Combine
feature_data = pd.concat([
    human_features,
    llm_features
])

# Plot
plt.figure(figsize=(10, 8))

plt.barh(
    feature_data["feature"],
    feature_data["coefficient"]
)

plt.axvline(0, linewidth=1)

plt.xlabel("Logistic Regression Coefficient")
plt.ylabel("Feature")
plt.title("Top Features Associated with Human vs. LLM-Generated Emails")

plt.tight_layout()

plt.savefig(
    "results/graphs/human_vs_llm_features.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved: results/graphs/human_vs_llm_features.png")