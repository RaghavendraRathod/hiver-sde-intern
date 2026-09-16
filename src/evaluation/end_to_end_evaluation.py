import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from src.classifier import IntentClassifier
from src.escalation import decide_escalation


GOLDEN_PATH = "data/golden_set.csv"
RESULTS_PATH = "results/end_to_end_results.csv"
FAILURES_PATH = "results/end_to_end_failures.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.25


def main():
    print("=" * 80)
    print("END-TO-END HELD-OUT EVALUATION")
    print("=" * 80)

    # ------------------------------------------------------------------
    # 1. Load golden set
    # ------------------------------------------------------------------
    golden = pd.read_csv(GOLDEN_PATH)

    print(f"\nLoaded golden set: {len(golden)} examples")

    # ------------------------------------------------------------------
    # 2. Same fixed train/test split used for intent benchmark
    # ------------------------------------------------------------------
    # Stratify on the joint intent + escalation label.
    # This ensures the held-out set contains both escalation outcomes
    # while preserving the intent distribution as much as possible.
    # Stratify by escalation outcome.
    #
    # We originally considered stratifying on intent + escalation together,
    # but some joint groups contain only one example. That makes
    # train_test_split impossible with that combined label.
    #
    # Stratifying on escalation guarantees that the held-out set contains
    # both escalation outcomes. Intent performance is evaluated separately
    # on the same held-out set.
    train_df, test_df = train_test_split(
        golden,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=golden["escalate"]
    )

    print(f"Training examples: {len(train_df)}")
    print(f"Test examples:     {len(test_df)}")
    print(f"Random state:      {RANDOM_STATE}")

    # ------------------------------------------------------------------
    # 3. Train classifier ONLY on training data
    # ------------------------------------------------------------------
    print("\nTraining hybrid classifier...")

    classifier = IntentClassifier()

    classifier.train(
        train_df["customer_text"].tolist(),
        train_df["intent"].tolist(),
    )

    print("Training complete.")

    # ------------------------------------------------------------------
    # 4. Predict held-out test examples
    # ------------------------------------------------------------------
    results = []

    print("\nRunning predictions...")

    for _, row in test_df.iterrows():

        text = row["customer_text"]

        prediction = classifier.predict(text)

        predicted_intent = prediction["intent"]
        confidence = prediction["confidence"]
        method = prediction["method"]

        predicted_escalate, predicted_reason = decide_escalation(
            text,
            predicted_intent,
            confidence,
        )

        results.append(
            {
                "customer_tweet_id": row["customer_tweet_id"],
                "customer_text": text,

                "true_intent": row["intent"],
                "predicted_intent": predicted_intent,
                "classifier_confidence": confidence,
                "classifier_method": method,

                "true_escalate": str(row["escalate"]).strip().lower() == "yes",
                "predicted_escalate": bool(predicted_escalate),

                "true_escalation_reason": row["escalation_reason"],
                "predicted_escalation_reason": predicted_reason,
            }
        )

    results_df = pd.DataFrame(results)

    # ------------------------------------------------------------------
    # 5. Intent evaluation
    # ------------------------------------------------------------------
    intent_accuracy = accuracy_score(
        results_df["true_intent"],
        results_df["predicted_intent"],
    )

    intent_macro_f1 = f1_score(
        results_df["true_intent"],
        results_df["predicted_intent"],
        average="macro",
        zero_division=0,
    )

    intent_weighted_f1 = f1_score(
        results_df["true_intent"],
        results_df["predicted_intent"],
        average="weighted",
        zero_division=0,
    )

    # ------------------------------------------------------------------
    # 6. Escalation evaluation
    # ------------------------------------------------------------------
    escalation_accuracy = accuracy_score(
        results_df["true_escalate"],
        results_df["predicted_escalate"],
    )

    escalation_precision = precision_score(
        results_df["true_escalate"],
        results_df["predicted_escalate"],
        zero_division=0,
    )

    escalation_recall = recall_score(
        results_df["true_escalate"],
        results_df["predicted_escalate"],
        zero_division=0,
    )

    escalation_f1 = f1_score(
        results_df["true_escalate"],
        results_df["predicted_escalate"],
        zero_division=0,
    )

    # ------------------------------------------------------------------
    # 7. Print results
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("INTENT CLASSIFICATION")
    print("=" * 80)

    print(f"Accuracy:    {intent_accuracy:.4f}")
    print(f"Macro-F1:    {intent_macro_f1:.4f}")
    print(f"Weighted-F1: {intent_weighted_f1:.4f}")

    print("\n" + "=" * 80)
    print("ESCALATION DECISION")
    print("=" * 80)

    print(f"Accuracy:  {escalation_accuracy:.4f}")
    print(f"Precision: {escalation_precision:.4f}")
    print(f"Recall:    {escalation_recall:.4f}")
    print(f"F1:        {escalation_f1:.4f}")

    # ------------------------------------------------------------------
    # 8. Confusion matrix
    # ------------------------------------------------------------------
    cm = confusion_matrix(
        results_df["true_escalate"],
        results_df["predicted_escalate"],
        labels=[False, True],
    )

    print("\nEscalation confusion matrix:")
    print("              Pred No  Pred Yes")
    print(f"True No       {cm[0][0]:7d}  {cm[0][1]:8d}")
    print(f"True Yes      {cm[1][0]:7d}  {cm[1][1]:8d}")

    # ------------------------------------------------------------------
    # 9. Failure analysis
    # ------------------------------------------------------------------
    intent_failures = results_df[
        results_df["true_intent"] != results_df["predicted_intent"]
    ]

    escalation_failures = results_df[
        results_df["true_escalate"] != results_df["predicted_escalate"]
    ]

    print("\n" + "=" * 80)
    print("FAILURE ANALYSIS")
    print("=" * 80)

    print(f"Intent failures:      {len(intent_failures)}/{len(results_df)}")
    print(f"Escalation failures:  {len(escalation_failures)}/{len(results_df)}")

    if len(intent_failures) > 0:
        print("\nIntent failures:")

        for _, row in intent_failures.iterrows():
            print("-" * 80)
            print("Message:", row["customer_text"])
            print("True intent:", row["true_intent"])
            print("Predicted intent:", row["predicted_intent"])
            print("Confidence:", row["classifier_confidence"])

    if len(escalation_failures) > 0:
        print("\nEscalation failures:")

        for _, row in escalation_failures.iterrows():
            print("-" * 80)
            print("Message:", row["customer_text"])
            print("True escalation:", row["true_escalate"])
            print("Predicted escalation:", row["predicted_escalate"])
            print("True reason:", row["true_escalation_reason"])
            print(
                "Predicted reason:",
                row["predicted_escalation_reason"],
            )

    # ------------------------------------------------------------------
    # 10. Save results
    # ------------------------------------------------------------------
    os.makedirs("results", exist_ok=True)

    results_df.to_csv(
        RESULTS_PATH,
        index=False,
    )

    escalation_failures.to_csv(
        FAILURES_PATH,
        index=False,
    )

    print("\n" + "=" * 80)
    print("FILES SAVED")
    print("=" * 80)

    print(f"Full results: {RESULTS_PATH}")
    print(f"Failures:     {FAILURES_PATH}")

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()