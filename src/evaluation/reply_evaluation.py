import os
import sys
import json
import pandas as pd
from sklearn.model_selection import train_test_split

# Allow imports from project root
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.classifier import IntentClassifier
from src.retrieval import HistoricalRetriever
from src.escalation import decide_escalation
from src.agent.reply import generate_reply_with_metadata


GOLDEN_PATH = "data/golden_set.csv"
RESULTS_PATH = "results/reply_evaluation.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.25


def load_data():
    print("Loading golden evaluation set...")

    df = pd.read_csv(GOLDEN_PATH)

    df = df.dropna(
        subset=["customer_text", "intent"]
    ).copy()

    df["customer_text"] = df["customer_text"].astype(str)
    df["intent"] = df["intent"].astype(str)

    print(f"Loaded {len(df)} golden examples.")

    return df


def create_split(df):
    """
    Create the same stratified train/test split
    used by the end-to-end evaluation.
    """

    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["escalate"]
    )

    print(
        f"Training examples: {len(train_df)}"
    )

    print(
        f"Evaluation examples: {len(test_df)}"
    )

    return train_df, test_df


def build_classifier(train_df):
    print("\nTraining intent classifier...")

    classifier = IntentClassifier()

    classifier.train(
        train_df["customer_text"],
        train_df["intent"]
    )

    print("Classifier trained.")

    return classifier


def build_retriever():
    print("\nBuilding historical retrieval index...")

    retriever = HistoricalRetriever()

    return retriever


def get_generic_baseline():
    """
    Generic customer-support baseline.
    """

    return (
        "I'm sorry you're experiencing this issue. "
        "Please provide the relevant order details so we can "
        "better understand the problem and determine the "
        "appropriate next steps."
    )


def get_retrieval_baseline(
    historical_matches
):
    """
    Retrieval baseline:
    use the highest-similarity historical Amazon reply.
    """

    if not historical_matches:
        return ""

    return historical_matches[0].get(
        "amazon_reply",
        ""
    )


def serialize_evidence(
    historical_matches
):
    """
    Convert retrieved historical examples into
    JSON suitable for CSV storage and later judging.
    """

    evidence = []

    for match in historical_matches:

        evidence.append({
            "customer_tweet_id": match.get(
                "customer_tweet_id",
                ""
            ),
            "customer_text": match.get(
                "customer_text",
                ""
            ),
            "amazon_reply": match.get(
                "amazon_reply",
                ""
            ),
            "similarity": float(
                match.get("similarity", 0.0)
            ),
        })

    return json.dumps(
        evidence,
        ensure_ascii=False
    )


def evaluate_agent(
    classifier,
    retriever,
    row
):
    """
    Run the actual support agent on one
    held-out evaluation example.
    """

    customer_message = row["customer_text"]

    # -----------------------------
    # Intent classification
    # -----------------------------

    classification = classifier.predict(
        customer_message
    )

    intent = classification["intent"]

    confidence = float(
        classification["confidence"]
    )

    # -----------------------------
    # Historical retrieval
    # -----------------------------

    historical_matches = retriever.retrieve(
        customer_message,
        top_k=3
    )

    # -----------------------------
    # Escalation
    # -----------------------------

    escalate, escalation_reason = decide_escalation(
        customer_message,
        intent,
        confidence
    )

    # -----------------------------
    # Agent reply
    # -----------------------------

    reply_result = generate_reply_with_metadata(
        customer_message=customer_message,
        intent=intent,
        confidence=confidence,
        historical_matches=historical_matches,
        escalate=escalate,
        escalation_reason=escalation_reason
    )

    return {
        "predicted_intent": intent,

        "confidence": confidence,

        "predicted_escalate": escalate,

        "predicted_escalation_reason":
            escalation_reason,

        "agent_reply":
            reply_result["reply"],

        "agent_reply_source":
            reply_result["source"],

        "gemini_attempts":
            reply_result.get(
                "gemini_attempts",
                0
            ),

        "historical_matches":
            historical_matches,
    }


def main():

    os.makedirs(
        "results",
        exist_ok=True
    )

    # -----------------------------
    # Load data
    # -----------------------------

    df = load_data()

    # -----------------------------
    # Train/test split
    # -----------------------------

    train_df, test_df = create_split(df)

    # -----------------------------
    # Build components
    # -----------------------------

    classifier = build_classifier(
        train_df
    )

    retriever = build_retriever()

    results = []

    print(
        "\nStarting reply evaluation..."
    )

    print("=" * 80)

    for i, (_, row) in enumerate(
        test_df.iterrows(),
        start=1
    ):

        customer_message = row[
            "customer_text"
        ]

        print(
            f"\n[{i}/{len(test_df)}]"
        )

        print(
            "Customer:",
            customer_message
        )

        # -----------------------------
        # Historical retrieval
        # -----------------------------

        historical_matches = retriever.retrieve(
            customer_message,
            top_k=3
        )

        # -----------------------------
        # Baseline 1
        # -----------------------------

        generic_reply = get_generic_baseline()

        # -----------------------------
        # Baseline 2
        # -----------------------------

        retrieval_reply = get_retrieval_baseline(
            historical_matches
        )

        # -----------------------------
        # Actual AI agent
        # -----------------------------

        agent_result = evaluate_agent(
            classifier,
            retriever,
            row
        )

        print(
            "Predicted intent:",
            agent_result["predicted_intent"]
        )

        print(
            "Confidence:",
            round(
                agent_result["confidence"],
                4
            )
        )

        print(
            "Escalate:",
            agent_result["predicted_escalate"]
        )

        print(
            "Reply source:",
            agent_result["agent_reply_source"]
        )

        print(
            "Agent reply:",
            agent_result["agent_reply"]
        )

        # -----------------------------
        # Save evidence
        # -----------------------------

        evidence_json = serialize_evidence(
            historical_matches
        )

        # -----------------------------
        # Save result
        # -----------------------------

        results.append({

            "customer_tweet_id":
                row["customer_tweet_id"],

            "customer_text":
                customer_message,

            "gold_intent":
                row["intent"],

            "gold_escalate":
                row["escalate"],

            "gold_escalation_reason":
                row["escalation_reason"],

            "predicted_intent":
                agent_result[
                    "predicted_intent"
                ],

            "confidence":
                agent_result[
                    "confidence"
                ],

            "predicted_escalate":
                agent_result[
                    "predicted_escalate"
                ],

            "predicted_escalation_reason":
                agent_result[
                    "predicted_escalation_reason"
                ],

            "generic_reply":
                generic_reply,

            "retrieval_reply":
                retrieval_reply,

            "agent_reply":
                agent_result[
                    "agent_reply"
                ],

            "agent_reply_source":
                agent_result[
                    "agent_reply_source"
                ],

            "gemini_attempts":
                agent_result[
                    "gemini_attempts"
                ],

            "historical_evidence":
                evidence_json,
        })

    # -----------------------------
    # Save results
    # -----------------------------

    results_df = pd.DataFrame(
        results
    )

    results_df.to_csv(
        RESULTS_PATH,
        index=False
    )

    print(
        "\n" + "=" * 80
    )

    print(
        "Saved evaluation results to:"
    )

    print(
        RESULTS_PATH
    )

    print(
        f"Total evaluated examples: "
        f"{len(results_df)}"
    )

    print(
        "\nReply source distribution:"
    )

    print(
        results_df[
            "agent_reply_source"
        ].value_counts()
    )


if __name__ == "__main__":
    main()