import os
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


RESULTS_PATH = "results/reply_evaluation.csv"
METRICS_PATH = "results/reply_metrics.csv"


def normalize(text):
    if not isinstance(text, str):
        return ""

    text = text.lower()

    # Remove URLs
    text = re.sub(r"https?://\S+", " ", text)

    # Keep words only
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    return " ".join(text.split())


def token_set(text):
    return set(normalize(text).split())


def lexical_overlap(reply, reference):
    """
    Token-level Jaccard similarity.

    This is used only as a lightweight automated
    reference-overlap proxy, not as a complete
    measure of response quality.
    """

    reply_tokens = token_set(reply)
    reference_tokens = token_set(reference)

    if not reply_tokens or not reference_tokens:
        return 0.0

    intersection = reply_tokens & reference_tokens
    union = reply_tokens | reference_tokens

    return len(intersection) / len(union)


def tfidf_similarity(text_a, text_b):
    """
    TF-IDF cosine similarity between two texts.
    """

    if not text_a or not text_b:
        return 0.0

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2)
    )

    matrix = vectorizer.fit_transform(
        [text_a, text_b]
    )

    return float(
        cosine_similarity(
            matrix[0:1],
            matrix[1:2]
        )[0][0]
    )


def actionability_score(reply):
    """
    Lightweight heuristic for whether a response
    contains an actionable support instruction.
    """

    text = normalize(reply)

    action_phrases = [
        "check",
        "provide",
        "contact",
        "review",
        "tracking",
        "order details",
        "next steps",
        "refund",
        "return",
        "replacement",
        "support agent",
        "account",
        "payment"
    ]

    matches = sum(
        1 for phrase in action_phrases
        if phrase in text
    )

    return min(matches / 3.0, 1.0)


def evaluate():

    print("Loading reply evaluation results...")

    df = pd.read_csv(
        RESULTS_PATH
    )

    print(
        f"Loaded {len(df)} evaluated examples."
    )

    rows = []

    for _, row in df.iterrows():

        customer = row["customer_text"]

        generic = row["generic_reply"]

        retrieval = row["retrieval_reply"]

        agent = row["agent_reply"]

        # Retrieval baseline is also used as a
        # historical reference for the lightweight
        # automated comparison.
        retrieval_overlap = lexical_overlap(
            retrieval,
            customer
        )

        agent_vs_retrieval = lexical_overlap(
            agent,
            retrieval
        )

        generic_vs_retrieval = lexical_overlap(
            generic,
            retrieval
        )

        agent_customer_similarity = tfidf_similarity(
            customer,
            agent
        )

        retrieval_customer_similarity = tfidf_similarity(
            customer,
            retrieval
        )

        generic_customer_similarity = tfidf_similarity(
            customer,
            generic
        )

        rows.append({

            "customer_tweet_id":
                row["customer_tweet_id"],

            "gold_intent":
                row["gold_intent"],

            "predicted_intent":
                row["predicted_intent"],

            "gold_escalate":
                row["gold_escalate"],

            "predicted_escalate":
                row["predicted_escalate"],

            "generic_reply":
                generic,

            "retrieval_reply":
                retrieval,

            "agent_reply":
                agent,

            "generic_actionability":
                actionability_score(generic),

            "retrieval_actionability":
                actionability_score(retrieval),

            "agent_actionability":
                actionability_score(agent),

            "generic_customer_similarity":
                generic_customer_similarity,

            "retrieval_customer_similarity":
                retrieval_customer_similarity,

            "agent_customer_similarity":
                agent_customer_similarity,

            "generic_reference_overlap":
                generic_vs_retrieval,

            "agent_reference_overlap":
                agent_vs_retrieval,

            "retrieval_reference_overlap":
                1.0,

        })

    metrics_df = pd.DataFrame(rows)

    metrics_df.to_csv(
        METRICS_PATH,
        index=False
    )

    print()
    print("=" * 80)
    print("AUTOMATED REPLY METRICS")
    print("=" * 80)

    print("\nMean actionability:")

    print(
        "Generic:",
        round(
            metrics_df[
                "generic_actionability"
            ].mean(),
            4
        )
    )

    print(
        "Retrieval:",
        round(
            metrics_df[
                "retrieval_actionability"
            ].mean(),
            4
        )
    )

    print(
        "Agent:",
        round(
            metrics_df[
                "agent_actionability"
            ].mean(),
            4
        )
    )

    print("\nMean customer-response similarity:")

    print(
        "Generic:",
        round(
            metrics_df[
                "generic_customer_similarity"
            ].mean(),
            4
        )
    )

    print(
        "Retrieval:",
        round(
            metrics_df[
                "retrieval_customer_similarity"
            ].mean(),
            4
        )
    )

    print(
        "Agent:",
        round(
            metrics_df[
                "agent_customer_similarity"
            ].mean(),
            4
        )
    )

    print("\nMean overlap with historical retrieval reply:")

    print(
        "Generic:",
        round(
            metrics_df[
                "generic_reference_overlap"
            ].mean(),
            4
        )
    )

    print(
        "Retrieval:",
        round(
            metrics_df[
                "retrieval_reference_overlap"
            ].mean(),
            4
        )
    )

    print(
        "Agent:",
        round(
            metrics_df[
                "agent_reference_overlap"
            ].mean(),
            4
        )
    )

    print()
    print("=" * 80)

    print(
        f"Saved metrics to: {METRICS_PATH}"
    )


if __name__ == "__main__":
    evaluate()