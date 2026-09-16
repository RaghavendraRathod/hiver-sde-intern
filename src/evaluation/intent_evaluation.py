import os
import sys

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# Allow imports when running from the project root.
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.classifier import IntentClassifier


DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "golden_set.csv"
)


def clean_text(text):
    return str(text).lower().strip()


def print_metrics(name, y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    weighted_f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    print(f"Accuracy:     {accuracy:.4f}")
    print(f"Macro-F1:     {macro_f1:.4f}")
    print(f"Weighted-F1:  {weighted_f1:.4f}")

    print("\nPer-intent results:")
    print(
        classification_report(
            y_true,
            y_pred,
            zero_division=0
        )
    )

    return {
        "model": name,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }


def majority_baseline(X_train, y_train, X_test):
    """
    Predict the most frequent training-set intent
    for every test example.
    """

    majority_class = y_train.value_counts().idxmax()

    predictions = [
        majority_class
        for _ in range(len(X_test))
    ]

    return predictions


def tfidf_logistic_baseline(X_train, y_train, X_test):
    """
    Simple ML baseline:
    TF-IDF word features + Logistic Regression.
    """

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_features=15000,
        sublinear_tf=True,
    )

    train_features = vectorizer.fit_transform(
        X_train
    )

    test_features = vectorizer.transform(
        X_test
    )

    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42,
    )

    model.fit(
        train_features,
        y_train
    )

    return model.predict(test_features)


def hybrid_classifier(X_train, y_train, X_test):
    """
    Train the improved hybrid classifier only on the
    training portion, then evaluate on unseen test data.
    """

    classifier = IntentClassifier()

    classifier.train(
        X_train,
        y_train
    )

    predictions = []

    for text in X_test:
        result = classifier.predict(text)
        predictions.append(result["intent"])

    return predictions


def main():

    print("=" * 70)
    print("HIVER SDE INTERN - INTENT CLASSIFIER EVALUATION")
    print("=" * 70)

    # ---------------------------------------------------------
    # Load golden set
    # ---------------------------------------------------------

    df = pd.read_csv(DATA_PATH)

    df = df.dropna(
        subset=[
            "customer_text",
            "intent"
        ]
    ).copy()

    df["customer_text"] = (
        df["customer_text"]
        .map(clean_text)
    )

    print(f"\nTotal labelled examples: {len(df)}")

    print("\nOverall intent distribution:")
    print(
        df["intent"]
        .value_counts()
    )

    # ---------------------------------------------------------
    # Same split for ALL models
    # ---------------------------------------------------------

    X = df["customer_text"]
    y = df["intent"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    print("\nEvaluation split:")
    print(f"Training examples: {len(X_train)}")
    print(f"Test examples:     {len(X_test)}")

    print("\nTest-set intent distribution:")
    print(
        y_test.value_counts()
    )

    results = []

    # ---------------------------------------------------------
    # 1. Majority baseline
    # ---------------------------------------------------------

    majority_predictions = majority_baseline(
        X_train,
        y_train,
        X_test
    )

    results.append(
        print_metrics(
            "1. Majority Class Baseline",
            y_test,
            majority_predictions
        )
    )

    # ---------------------------------------------------------
    # 2. TF-IDF + Logistic Regression
    # ---------------------------------------------------------

    tfidf_predictions = tfidf_logistic_baseline(
        X_train,
        y_train,
        X_test
    )

    results.append(
        print_metrics(
            "2. TF-IDF + Logistic Regression Baseline",
            y_test,
            tfidf_predictions
        )
    )

    # ---------------------------------------------------------
    # 3. Improved Hybrid Classifier
    # ---------------------------------------------------------

    hybrid_predictions = hybrid_classifier(
        X_train,
        y_train,
        X_test
    )

    results.append(
        print_metrics(
            "3. Improved Hybrid Rule + ML Classifier",
            y_test,
            hybrid_predictions
        )
    )

    # ---------------------------------------------------------
    # Comparison
    # ---------------------------------------------------------

    results_df = pd.DataFrame(results)

    print("\n" + "=" * 70)
    print("FINAL MODEL COMPARISON")
    print("=" * 70)

    print(
        results_df.to_string(
            index=False,
            formatters={
                "accuracy": "{:.4f}".format,
                "macro_f1": "{:.4f}".format,
                "weighted_f1": "{:.4f}".format,
            }
        )
    )

    # ---------------------------------------------------------
    # Confusion matrix for final classifier
    # ---------------------------------------------------------

    labels = sorted(
        y.unique()
    )

    cm = confusion_matrix(
        y_test,
        hybrid_predictions,
        labels=labels
    )

    cm_df = pd.DataFrame(
        cm,
        index=labels,
        columns=labels
    )

    print("\n" + "=" * 70)
    print("HYBRID CLASSIFIER CONFUSION MATRIX")
    print("=" * 70)

    print(cm_df)

    # ---------------------------------------------------------
    # Save results
    # ---------------------------------------------------------

    results_dir = os.path.join(
        PROJECT_ROOT,
        "results"
    )

    os.makedirs(
        results_dir,
        exist_ok=True
    )

    results_path = os.path.join(
        results_dir,
        "intent_metrics.csv"
    )

    results_df.to_csv(
        results_path,
        index=False
    )

    confusion_path = os.path.join(
        results_dir,
        "intent_confusion_matrix.csv"
    )

    cm_df.to_csv(
        confusion_path
    )

    print("\nResults saved to:")
    print(results_path)

    print("\nConfusion matrix saved to:")
    print(confusion_path)


if __name__ == "__main__":
    main()