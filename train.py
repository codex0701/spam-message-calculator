"""Train a spam classifier: TF-IDF + (Naive Bayes vs Logistic Regression)."""
import urllib.request
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, f1_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).parent
DATA_URL = ("https://raw.githubusercontent.com/justmarkham/"
            "pycon-2016-tutorial/master/data/sms.tsv")


def load_data() -> pd.DataFrame:
    """Use the full SMS Spam dataset if it can be downloaded, else the tiny sample."""
    tsv = ROOT / "data" / "spam.tsv"
    if not tsv.exists():
        try:
            urllib.request.urlretrieve(DATA_URL, tsv)
            print("Downloaded full SMS Spam dataset.")
        except Exception as e:
            print(f"Download failed ({e}). Using small sample dataset instead.")
    if tsv.exists():
        return pd.read_csv(tsv, sep="\t", names=["label", "message"])
    return pd.read_csv(ROOT / "data" / "sample_spam.csv")


def main():
    df = load_data().dropna()
    df["y"] = (df["label"] == "spam").astype(int)
    print(f"Rows: {len(df)} | spam: {df['y'].sum()} | ham: {(1 - df['y']).sum()}")

    X_train, X_test, y_train, y_test = train_test_split(
        df["message"], df["y"], test_size=0.25, random_state=42, stratify=df["y"])

    models = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    }
    best_name, best_pipe, best_f1 = None, None, -1
    for name, clf in models.items():
        pipe = make_pipeline(
            TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2)), clf)
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        f1 = f1_score(y_test, preds)
        print(f"\n== {name} ==  accuracy={accuracy_score(y_test, preds):.3f}  F1(spam)={f1:.3f}")
        print(classification_report(y_test, preds, target_names=["ham", "spam"], zero_division=0))
        if f1 > best_f1:
            best_name, best_pipe, best_f1 = name, pipe, f1

    print(f"Best model: {best_name}")
    joblib.dump(best_pipe, ROOT / "models" / "spam_model.joblib")

    ConfusionMatrixDisplay.from_estimator(
        best_pipe, X_test, y_test, display_labels=["ham", "spam"], cmap="Blues")
    plt.title(f"Confusion matrix - {best_name}")
    plt.savefig(ROOT / "images" / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    print("Saved model and confusion matrix.")


if __name__ == "__main__":
    main()
