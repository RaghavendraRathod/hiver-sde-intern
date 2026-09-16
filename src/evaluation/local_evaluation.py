import os
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)


INPUT_PATH = "results/pipeline_outputs.csv"

INTENT_RESULTS_PATH = "results/local_intent_metrics.csv"
ESCALATION_RESULTS_PATH = "results/local_escalation_metrics.csv"
FAILURES_PATH = "results/local_failures.csv"


def normalize_bool(value):
    """
    Convert CSV boolean-like values into actual Python booleans.
    """
    if isinstance(value, bool):
        return value

    value = str(value).strip().lower()

    if value in {"true", "1", "yes"}:
        return True

    if value in {"false", "0", "no"}:
        return False

    return False


def evaluate_intent(df):
    print("\n" + "=" * 80)
    print("INTENT CLASSIFICATION EVALUATION")
    print("=" * 80)

    y_true = df["true_intent"]
    y_pred = df["predicted_intent"]

    accuracy = accuracy_score(y_true, y_pred)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=sorted(y_true.unique()),
        zero_division=0,
    )

    labels = sorted(y_true.unique())

    macro_f1 = f1.mean()

    weighted_f1 = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average="weighted",
        zero_division=0,
    )[2]

    print(f"\nExamples evaluated: {len(df)}")
    print(f"Accuracy:             {accuracy:.4f}")
    print(f"Macro-F1:             {macro_f1:.4f}")
    print(f"Weighted-F1:          {weighted_f1:.4f}")

    print("\nClassification report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=labels,
            zero_division=0,
        )
    )

    print("\nConfusion matrix:")
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )

    confusion_df = pd.DataFrame(
        cm,
        index=labels,
        columns=labels,
    )

    print(confusion_df.to_string())

    results = pd.DataFrame(
        {
            "intent": labels,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    )

    results["accuracy_overall"] = accuracy
    results["macro_f1"] = macro_f1
    results["weighted_f1"] = weighted_f1

    results.to_csv(
        INTENT_RESULTS_PATH,
        index=False,
    )

    return accuracy, macro_f1, weighted_f1


def evaluate_escalation(df):
    print("\n" + "=" * 80)
    print("ESCALATION EVALUATION")
    print("=" * 80)

    y_true = df["true_escalate"].apply(normalize_bool)
    y_pred = df["escalate"].apply(normalize_bool)

    accuracy = accuracy_score(y_true, y_pred)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        zero_division=0,
    )

    print(f"\nExamples evaluated: {len(df)}")
    print(f"Accuracy:           {accuracy:.4f}")
    print(f"Precision:          {precision:.4f}")
    print(f"Recall:             {recall:.4f}")
    print(f"F1:                 {f1:.4f}")

    print("\nEscalation confusion matrix:")

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[False, True],
    )

    confusion_df = pd.DataFrame(
        cm,
        index=["True: No", "True: Yes"],
        columns=["Pred: No", "Pred: Yes"],
    )

    print(confusion_df.to_string())

    results = pd.DataFrame(
        [
            {
                "metric": "accuracy",
                "value": accuracy,
            },
            {
                "metric": "precision",
                "value": precision,
            },
            {
                "metric": "recall",
                "value": recall,
            },
            {
                "metric": "f1",
                "value": f1,
            },
        ]
    )

    results.to_csv(
        ESCALATION_RESULTS_PATH,
        index=False,
    )

    return accuracy, precision, recall, f1


def analyze_failures(df):
    print("\n" + "=" * 80)
    print("FAILURE ANALYSIS")
    print("=" * 80)

    intent_failures = df[
        df["true_intent"] != df["predicted_intent"]
    ].copy()

    escalation_failures = df[
        df["true_escalate"].apply(normalize_bool)
        != df["escalate"].apply(normalize_bool)
    ].copy()

    print(
        f"\nIntent failures: {len(intent_failures)} / {len(df)}"
    )

    print(
        f"Escalation failures: "
        f"{len(escalation_failures)} / {len(df)}"
    )

    failures = []

    for _, row in intent_failures.iterrows():
        failures.append(
            {
                "customer_tweet_id": row["customer_tweet_id"],
                "failure_type": "intent_classification",
                "customer_message": row["customer_message"],
                "true_intent": row["true_intent"],
                "predicted_intent": row["predicted_intent"],
                "classifier_confidence": row[
                    "classifier_confidence"
                ],
                "true_escalate": row["true_escalate"],
                "predicted_escalate": row["escalate"],
                "true_escalation_reason": row[
                    "true_escalation_reason"
                ],
                "predicted_escalation_reason": row[
                    "escalation_reason"
                ],
                "reply": row["reply"],
            }
        )

    for _, row in escalation_failures.iterrows():
        failures.append(
            {
                "customer_tweet_id": row["customer_tweet_id"],
                "failure_type": "escalation",
                "customer_message": row["customer_message"],
                "true_intent": row["true_intent"],
                "predicted_intent": row["predicted_intent"],
                "classifier_confidence": row[
                    "classifier_confidence"
                ],
                "true_escalate": row["true_escalate"],
                "predicted_escalate": row["escalate"],
                "true_escalation_reason": row[
                    "true_escalation_reason"
                ],
                "predicted_escalation_reason": row[
                    "escalation_reason"
                ],
                "reply": row["reply"],
            }
        )

    failures_df = pd.DataFrame(failures)

    if len(failures_df) > 0:
        failures_df.to_csv(
            FAILURES_PATH,
            index=False,
        )

        print(f"\nFailure examples saved to:")
        print(FAILURES_PATH)

        print("\nFailure examples:")

        display_columns = [
            "customer_tweet_id",
            "failure_type",
            "true_intent",
            "predicted_intent",
            "true_escalate",
            "predicted_escalate",
        ]

        print(
            failures_df[display_columns].to_string(
                index=False
            )
        )

    else:
        print("\nNo failures found.")

    return failures_df


def main():
    print("=" * 80)
    print("LOCAL AUTOMATED EVALUATION")
    print("=" * 80)

    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(
            f"Input file not found: {INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH)

    print(f"\nLoaded: {INPUT_PATH}")
    print(f"Records: {len(df)}")

    required_columns = [
        "customer_tweet_id",
        "customer_message",
        "true_intent",
        "predicted_intent",
        "true_escalate",
        "escalate",
        "true_escalation_reason",
        "escalation_reason",
        "classifier_confidence",
        "reply",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    os.makedirs("results", exist_ok=True)

    intent_metrics = evaluate_intent(df)

    escalation_metrics = evaluate_escalation(df)

    failures = analyze_failures(df)

    print("\n" + "=" * 80)
    print("LOCAL EVALUATION COMPLETE")
    print("=" * 80)

    print("\nIntent metrics:")
    print(f"Accuracy:    {intent_metrics[0]:.4f}")
    print(f"Macro-F1:    {intent_metrics[1]:.4f}")
    print(f"Weighted-F1: {intent_metrics[2]:.4f}")

    print("\nEscalation metrics:")
    print(f"Accuracy:  {escalation_metrics[0]:.4f}")
    print(f"Precision: {escalation_metrics[1]:.4f}")
    print(f"Recall:    {escalation_metrics[2]:.4f}")
    print(f"F1:        {escalation_metrics[3]:.4f}")

    print("\nFiles created:")
    print(f"- {INTENT_RESULTS_PATH}")
    print(f"- {ESCALATION_RESULTS_PATH}")

    if len(failures) > 0:
        print(f"- {FAILURES_PATH}")


if __name__ == "__main__":
    main()