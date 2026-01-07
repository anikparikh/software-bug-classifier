from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "bugs.csv"
MODEL_FILE = ROOT / "models" / "bug_classifier.joblib"


def load_data():
    df = pd.read_csv(DATA_FILE)

    # make sure required columns exist
    for col in ["summary", "description", "label"]:
        if col not in df.columns:
            raise ValueError(f"Missing column in CSV: {col}")

    # combine summary + description into a single text field
    df["summary"] = df["summary"].fillna("")
    df["description"] = df["description"].fillna("")
    df["text"] = df["summary"] + " " + df["description"]

    df = df.dropna(subset=["label"])
    return df["text"], df["label"]


def build_model():
    """
    Text -> TF-IDF -> Logistic Regression classifier
    Uses class_weight='balanced' to handle label imbalance.
    """
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1, 2))),
            ("clf", LogisticRegression(
                max_iter=1000,
                class_weight="balanced",  # <-- NEW
            )),
        ]
    )



def main():
    print(f"Loading data from {DATA_FILE}...")
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,       # 80/20 split
        random_state=42,
        stratify=y,          # keep class balance consistent
    )

    model = build_model()

    print("Training model...")
    model.fit(X_train, y_train)

    print("Evaluating...")
    y_pred = model.predict(X_test)

    # print detailed report
    print(classification_report(y_test, y_pred))

    # also print a couple of key numbers for README / resume
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    print(f"Accuracy: {acc:.3f}")
    print(f"Macro F1: {macro_f1:.3f}")

    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_FILE)
    print(f"Saved model to {MODEL_FILE}")


if __name__ == "__main__":
    main()
