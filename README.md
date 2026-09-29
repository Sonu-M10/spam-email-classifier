# AI-Aware Spam & Phishing Email Classification Using NLP

This project investigates how traditional Natural Language Processing (NLP) and machine learning methods can be used to classify spam and phishing emails, with a particular focus on whether these methods can distinguish AI-generated phishing emails from human-written phishing emails.

The project also examines how well models trained on one collection of emails generalize to completely different datasets, sources, and languages.

## Research Question

Can traditional NLP machine-learning methods distinguish AI-generated phishing emails from human-written phishing emails, and are the observed differences robust across different representations, sources, and LLMs?

## Project Goals

This project has two main parts:

1. Build traditional machine-learning models that classify emails as legitimate, phishing, or spam.
2. Investigate whether traditional NLP models can distinguish AI-generated phishing emails from human-written phishing emails.

The second part is especially important because high performance on a single dataset does not necessarily mean that a model can reliably detect AI-generated writing in real-world emails.

## Dataset

### Traditional Spam & Phishing Dataset

The main classification dataset contains approximately 365,000 emails labeled as:

- 0 — Ham / Legitimate
- 1 — Phishing
- 2 — Spam

After removing missing text, conflicting duplicate labels, and duplicate emails, the cleaned dataset contains approximately 281,000 unique emails.

### Human vs. LLM Phishing Dataset

A separate research corpus contains:

- 5,000 human-written phishing emails
- 4,986 LLM-generated phishing emails

The LLM-generated emails were produced using GPT-4.1, DeepSeek 3.2, and Llama 3.3 70B.

The human emails came from multiple phishing sources, including TREC-07, CEAS_08, Nazario, Nigerian_Fraud, and enron_data_fraud_labeled.

The model uses only the email text for the human-vs-LLM classification experiments.

## Models

### Traditional Email Classification

- Logistic Regression
- Linear Support Vector Machine (SVM)
- Multinomial Naive Bayes

TF-IDF was used to convert email text into numerical features.

### AI-Generated vs. Human-Written Classification

The main AI-era detector uses character-level TF-IDF, 3–5 character n-grams, and Logistic Regression.

Additional experiments were performed using word-level TF-IDF, character-level TF-IDF, normalized character TF-IDF, stopword removal, topic/content controls, simple formatting-based features, cross-source evaluation, and cross-model evaluation.

## Traditional Classification Results

| Model | Accuracy | Phishing Recall | Phishing F1 |
|---|---:|---:|---:|
| Logistic Regression | 97.02% | 94% | 0.95 |
| Linear SVM | 97.31% | 93% | 0.95 |
| Naive Bayes | 91.79% | 72% | 0.81 |

Linear SVM and Logistic Regression both performed substantially better than Naive Bayes on the original dataset.

On the external validation dataset, the traditional classifier achieved 50% accuracy and 0% phishing recall. The model classified all 2,000 external emails as legitimate.

This demonstrates the importance of testing NLP models outside the dataset they were trained on.

## Human vs. LLM Investigation

The initial human-vs-LLM experiments produced extremely high performance on the original research corpus.

A character-level TF-IDF Logistic Regression model achieved:

- Accuracy: 100%
- Human precision: 1.00
- Human recall: 1.00
- LLM precision: 1.00
- LLM recall: 1.00

Because perfect or near-perfect performance can indicate that a model is learning dataset-specific characteristics rather than genuine authorship signals, additional experiments were performed.

### Shortcut Experiments

Simple features including email length, word count, number of lines, punctuation, digits, uppercase characters, spaces, and non-ASCII characters were able to perfectly separate the original human and LLM datasets.

A length-only model achieved approximately 68% accuracy, while the combined formatting features achieved 100%.

After normalizing URLs, HTML, numbers, whitespace, and case, character-level TF-IDF still achieved 100% accuracy on the original corpus.

These results suggest that the original dataset contains strong differences between the human and LLM email collections.

## Cross-Source Evaluation

Across 15 combinations of held-out human source and LLM model:

- Average accuracy: 99.74%
- Lowest accuracy: 98.76%

The results remained extremely high within this corpus, but additional independent datasets showed that this performance did not consistently transfer outside the original research setup.

## Independent Dataset Evaluation

### Independent English GPT Dataset

An independent dataset containing 3,746 GPT-generated and legitimate emails was evaluated using the frozen AI detector.

Results:

- Accuracy: 81.31%
- Human recall: 86%
- AI recall: 79%

Confusion matrix:

[[1089, 180],
 [520, 1957]]

This was substantially lower than the near-perfect results obtained on the original research corpus.

### Independent Chinese / DeepSeek Dataset

A separate dataset contained 2,020 Chinese phishing emails generated using DeepSeek.

The English character-level detector classified 2,020 as Human and 0 as AI. AI recall was therefore 0%.

This result suggests that the detector's learned patterns were highly dependent on the language and characteristics of the data it was trained on.

## Key Findings

1. Traditional NLP models can perform very well on spam and phishing classification when training and testing data come from the same distribution.
2. High internal accuracy does not necessarily mean that a model has learned a general concept of AI-generated writing.
3. Human and LLM phishing datasets can contain strong source, formatting, topic, language, and generation-pipeline differences.
4. The human-vs-LLM detector performed strongly on the original research corpus but showed lower performance on an independent English GPT dataset.
5. The detector failed on the independent Chinese/DeepSeek phishing dataset, showing that the learned features did not generalize across languages and generation sources.
6. High in-dataset accuracy should not be interpreted as reliable AI-authorship detection.

## Project Structure

spam-email-classifier/
├── Data/
│   ├── cleaned_emails.csv
│   ├── cleaned_emails.json
│   ├── corpus_features.csv
│   ├── human_corpus_sampled.csv
│   ├── llm_corpus_sampled.csv
│   ├── Phishing_validation_emails.csv
│   └── independent_llm_dataset/
├── GUIDE/
│   ├── setup.md
│   └── usage.md
├── SRC/
│   ├── preprocess.py
│   ├── svm_model.py
│   ├── naive_bayes.py
│   ├── ai_detection.py
│   ├── predict.py
│   └── external_*_test.py
├── investigation/
│   ├── ai_feature_analysis.py
│   ├── ai_model_analysis.py
│   ├── shortcut_baselines.py
│   ├── traditional_model_analysis.py
│   ├── create_graphs.py
│   └── Archive/
├── results/
│   ├── baseline_results.txt
│   ├── svm_results.txt
│   ├── naive_bayes_results.txt
│   └── graphs/
└── README.md

## Running the Project

Clone:
git clone https://github.com/Sonu-M10/spam-email-classifier.git
cd spam-email-classifier

Create virtual environment:
python3 -m venv .venv
source .venv/bin/activate

Install dependencies:
pip install pandas scikit-learn matplotlib joblib

Run traditional models:
python3 SRC/preprocess.py
python3 SRC/svm_model.py
python3 SRC/naive_bayes.py

Run AI-generated email detector:
python3 SRC/ai_detection.py

Test an individual email:
python3 SRC/predict.py

Additional investigation scripts are located in the investigation/ directory.

## Limitations

The human and LLM datasets were collected through different pipelines, which means differences in source, time period, topics, formatting, language, and generation methods may contribute to model performance.

The independent datasets also differ from the original research corpus, making direct comparisons imperfect.

The project therefore does not claim that traditional NLP can reliably determine whether an email was written by an AI system in general. Instead, it investigates how well these methods distinguish the specific human and LLM-generated phishing corpora used in the experiments.

## Research Takeaway

Traditional NLP methods were able to distinguish human-written and LLM-generated phishing emails extremely well within the original research corpus. However, independent testing showed that this performance does not consistently generalize across datasets, languages, and generation sources.

The main conclusion is that high accuracy on a controlled human-vs-LLM dataset should not be treated as proof that a model can reliably detect AI authorship in real-world phishing emails.

This project highlights the importance of external validation, cross-source testing, and investigating dataset-specific shortcuts when developing AI-generated text detectors.
