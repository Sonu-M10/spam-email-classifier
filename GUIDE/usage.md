# Usage Guide

## Before Running Commands

Open the project folder in VS Code and open the terminal.

Make sure the terminal is inside the main project folder:

spam-email-classifier/

On macOS, use `python3` rather than `python`.

---

## 1. Explore the Dataset

To run the dataset exploration script:

python3 SRC/explore.py

This script is used to inspect and explore the email dataset before modeling.

---

## 2. Train the Traditional Spam/Phishing Model

Run:

python3 SRC/preprocess.py

This script:

- Loads the cleaned email dataset
- Splits the data into training and testing sets
- Creates TF-IDF features
- Trains the Logistic Regression model
- Evaluates the model
- Saves the trained model and vectorizer

The saved files are:

results/traditional_model.pkl
results/traditional_vectorizer.pkl

---

## 3. Train the SVM Model

Run:

python3 SRC/svm_model.py

This trains and evaluates the Linear SVM spam/phishing classifier.

Results are saved in:

results/svm_results.txt

---

## 4. Train the Naive Bayes Model

Run:

python3 SRC/naive_bayes.py

This trains and evaluates the Multinomial Naive Bayes spam/phishing classifier.

Results are saved in:

results/naive_bayes_results.txt

---

## 5. Train the Human vs. LLM Detector

Run:

python3 SRC/ai_detection.py

This trains the Human vs. AI phishing detector.

The model uses character-level TF-IDF features and Logistic Regression.

The trained files are saved as:

results/ai_detection_model.pkl
results/ai_detection_vectorizer.pkl

---

## 6. Predict an Individual Email

Run:

python3 SRC/predict.py

The program will ask you to enter an email.

The email will be classified as either:

- Human
- AI

The program also provides a confidence/probability value.

The prediction script uses the pre-trained model and vectorizer stored in:

results/ai_detection_model.pkl
results/ai_detection_vectorizer.pkl

---

## 7. Test the AI Detector on the Independent GPT Dataset

Run:

python3 SRC/external_ai_test.py

This evaluates the saved Human vs. LLM detector on the independent GPT dataset.

This is an external validation experiment and does not retrain the model on the test dataset.

---

## 8. Test the AI Detector on the Independent Human Phishing Dataset

Run:

python3 SRC/external_human_test.py

This evaluates the saved AI detector on the independent human phishing dataset.

The dataset contains human-written phishing emails.

This is a one-class test because the dataset does not contain AI-generated emails.

---

## 9. Test the Traditional Model on the Independent Phishing Dataset

Run:

python3 SRC/external_phishing_test.py

This evaluates the saved traditional spam/phishing model on the independent phishing validation dataset.

The dataset contains:

- Safe emails
- Phishing emails

This test is used to examine how well the traditional model generalizes to an external dataset.

---

# Investigation Scripts

The `investigation/` folder contains experiments used to understand model behavior and investigate the reliability of the results.

These scripts are primarily for research and analysis rather than normal model usage.

---

## 10. AI Model Analysis

Run:

python3 investigation/ai_model_analysis.py

This runs multiple Human vs. LLM experiments, including:

- Word TF-IDF
- Character TF-IDF
- Cleaned character TF-IDF
- Stopword removal
- Topic/content controls
- Human source classification
- Strict cross-source testing

---

## 11. Shortcut Baselines

Run:

python3 investigation/shortcut_baselines.py

This investigates whether simple features can explain the strong Human vs. LLM classification results.

The experiments include:

- Length-only features
- Formatting features
- Normalized character TF-IDF

---

## 12. Traditional Model Analysis

Run:

python3 investigation/traditional_model_analysis.py

This analyzes the traditional spam/phishing model and examines the features associated with:

- Legitimate emails
- Phishing emails
- Spam emails

---

## 13. Create Graphs

Run:

python3 investigation/create_graphs.py

This creates the project's analysis graphs.

Graphs are saved in:

results/graphs/

Current graphs include:

- ai_detection_recall.png
- ai_detection_shortcut_experiments.png
- traditional_model_accuracy.png
- human_vs_llm_features.png

---

# Results and Saved Models

Trained models and vectorizers are stored in:

results/

Important model files:

results/traditional_model.pkl
results/traditional_vectorizer.pkl
results/ai_detection_model.pkl
results/ai_detection_vectorizer.pkl

Experiment results are also stored in the `results/` folder.

Graphs are stored in:

results/graphs/

---

# Recommended Workflow

For someone running the project from the beginning, the general workflow is:

1. Explore the dataset:

python3 SRC/explore.py

2. Train the traditional Logistic Regression model:

python3 SRC/preprocess.py

3. Train the SVM model:

python3 SRC/svm_model.py

4. Train the Naive Bayes model:

python3 SRC/naive_bayes.py

5. Train the Human vs. LLM detector:

python3 SRC/ai_detection.py

6. Test individual emails:

python3 SRC/predict.py

7. Run the external validation experiments:

python3 SRC/external_ai_test.py

python3 SRC/external_human_test.py

python3 SRC/external_phishing_test.py

8. Run the research investigation scripts:

python3 investigation/ai_model_analysis.py

python3 investigation/shortcut_baselines.py

python3 investigation/traditional_model_analysis.py

9. Recreate the project graphs:

python3 investigation/create_graphs.py

---

# Important Notes

- Run all commands from the main `spam-email-classifier` directory.
- Use `python3` on macOS.
- Do not type the terminal prompt when copying commands.
- The external testing scripts use saved models and are intended to evaluate generalization.
- The investigation scripts are separate from the main training and prediction workflow.
- Running a training script again can overwrite the corresponding saved model files.
- Graphs are saved in `results/graphs/`.
