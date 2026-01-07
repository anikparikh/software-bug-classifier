import sys
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]
MODEL_FILE = ROOT / "models" / "bug_classifier.joblib"


def load_model():
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_FILE}. "
            "Run: python -m src.train first."
        )
    return joblib.load(MODEL_FILE)


def main():
    if len(sys.argv) < 2:
        print('Usage: python -m src.predict "bug summary and description here"')
        sys.exit(1)

    text = sys.argv[1]
    model = load_model()

    pred = model.predict([text])[0]
    if hasattr(model, "predict_proba"):
        confidence = model.predict_proba([text])[0].max()
        print(f"Predicted label: {pred} (confidence {confidence:.2f})")
    else:
        print(f"Predicted label: {pred}")


if __name__ == "__main__":
    main()
