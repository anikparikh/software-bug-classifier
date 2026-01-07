# Software Bug Classifier

This project is a small NLP pipeline that classifies software bug reports by
severity (**Critical**, **Major**, **Minor**) using TF-IDF features and a
Logistic Regression classifier built with scikit-learn.

Given a bug's summary and description, the model predicts a severity label and
a confidence score. The project is organized as a simple, reproducible ML
pipeline with a training script and a CLI prediction tool.

---

## Tech Stack

- Python 3
- pandas
- scikit-learn
- joblib

---

## Project Structure

```text
software-bug-classifier/
  data/
    bugs.csv                 # labeled bug reports (summary, description, label)
  models/
    bug_classifier.joblib    # trained model pipeline (created by train.py)
  src/
    train.py                 # training + evaluation script
    predict.py               # CLI tool for predicting severity
  requirements.txt           # Python dependencies
  README.md
