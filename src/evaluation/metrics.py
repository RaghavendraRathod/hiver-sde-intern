import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


RESULTS_PATH = Path("results/end_to_end_results.csv")
OUTPUT_PATH = Path("results/metrics.json")


def main():
    df = pd.read_csv(RESULTS_PATH)

    true_intent = df["true_intent"]
    predicted_intent = df["predicted_intent"]

    true_escalate = df["true_escalate"].astype(bool)
    predicted_escalate = df["predicted_escalate"].astype(bool)

    intent_metrics = {
        "accuracy": round(
            accuracy_score(true_intent, predicted_intent), 4
        ),
        "macro_f1": round(
            f1_score(
                true_intent,
                predicted_intent,
                average="macro",
                zero_division=0,
            ),
            4,
        ),
        "weighted_f1": round(
            f1_score(
                true_intent,
                predicted_intent,
                average="weighted",
                zero_division=0,
            ),
            4,
        ),
        "classification_report": classification_report(
            true_intent,
            predicted_intent,
            output_dict=True,
            zero_division=0,
        ),
    }

    escalation_metrics = {
        "accuracy": round(
            accuracy_score(true_escalate, predicted_escalate), 4
        ),
        "precision": round(
            precision_score(
                true_escalate,
                predicted_escalate,
                zero_division=0,
            ),
            4,
        ),
        "recall": round(
            recall_score(
                true_escalate,
                predicted_escalate,
                zero_division=0,
            ),
            4,
        ),
        "f1": round(
            f1_score(
                true_escalate,
                predicted_escalate,
                zero_division=0,
            ),
            4,
        ),
        "confusion_matrix": confusion_matrix(
            true_escalate,
            predicted_escalate,
        ).tolist(),
    }

    metrics = {
        "evaluation": {
            "dataset": "AmazonHelp golden evaluation set",
            "examples": len(df),
            "evaluation_type": "held_out",
            "random_state": 42,
            "test_size": 0.25,
        },
        "intent_classification": intent_metrics,
        "escalation_decision": escalation_metrics,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("=" * 80)
    print("METRICS EXPORTED")
    print("=" * 80)
    print(f"Input:  {RESULTS_PATH}")
    print(f"Output: {OUTPUT_PATH}")
    print()
    print("Intent classification")
    print(f"  Accuracy:    {intent_metrics['accuracy']}")
    print(f"  Macro-F1:    {intent_metrics['macro_f1']}")
    print(f"  Weighted-F1: {intent_metrics['weighted_f1']}")
    print()
    print("Escalation decision")
    print(f"  Accuracy:  {escalation_metrics['accuracy']}")
    print(f"  Precision: {escalation_metrics['precision']}")
    print(f"  Recall:    {escalation_metrics['recall']}")
    print(f"  F1:        {escalation_metrics['f1']}")
    print()
    print("Saved successfully.")


if __name__ == "__main__":
    main()