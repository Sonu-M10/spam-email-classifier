---
license: apache-2.0
task_categories:
- text-classification
language:
- en
- zh
tags:
- email-classification
pretty_name: LLM-Generated Phishing Email Dataset
size_categories:
- 10K<n<100K
---

# LLM-Generated Phishing Email Dataset

## Dataset Description

This dataset comprises a collection of phishing and legitimate emails generated using Large Language Models (LLMs), specifically DeepSeek for Chinese emails and OpenAI models for English emails. The primary purpose of this dataset is to facilitate research and development in phishing email detection and classification.

The dataset is structured with two key columns:
- `content`: The full text content of the email.
- `label`: A binary classification label indicating the nature of the email:
    - `0`: Legitimate email
    - `1`: Phishing email

## Dataset Structure

The dataset is provided in a tabular format, with each row representing an email and its corresponding label.



## Data Generation

The emails in this dataset are synthetically generated using advanced LLMs. This approach allows for the creation of diverse and realistic email samples, which can be particularly useful for training and evaluating machine learning models for phishing detection, especially in scenarios where real-world labeled data is scarce or sensitive.

- **Chinese Emails**: Generated using the DeepSeek LLM.
- **English Emails**: Generated using OpenAI LLMs.

The generation process aims to produce emails that mimic the characteristics of both legitimate communications and various phishing attempts, including those employing social engineering tactics.

## Usage

This dataset can be used for:
- Training and evaluating machine learning models for phishing email detection.
- Research into LLM-generated text and its applications in cybersecurity.
- Developing and testing natural language processing (NLP) techniques for email analysis.

## Citation

If you use this dataset in your research, please consider citing the original repository or project.