import pandas as pd


HISTORICAL_PATH = "data/amazon_customer_sample.csv"
GOLDEN_PATH = "data/golden_set.csv"
OUTPUT_PATH = "data/retrieval_corpus.csv"


def main():
    print("Loading historical dataset...")
    historical = pd.read_csv(HISTORICAL_PATH)

    print("Loading golden set...")
    golden = pd.read_csv(GOLDEN_PATH)

    print(f"Historical examples before filtering: {len(historical)}")
    print(f"Golden examples: {len(golden)}")

    # Convert IDs to strings so comparisons are reliable.
    historical_ids = historical["customer_tweet_id"].astype(str)
    golden_ids = set(golden["customer_tweet_id"].astype(str))

    # Remove every golden-set example from the retrieval corpus.
    retrieval_corpus = historical[
        ~historical_ids.isin(golden_ids)
    ].copy()

    # Remove rows without usable customer messages.
    retrieval_corpus = retrieval_corpus.dropna(
        subset=["customer_text"]
    )

    retrieval_corpus["customer_text"] = (
        retrieval_corpus["customer_text"].astype(str)
    )

    retrieval_corpus.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\n" + "=" * 60)
    print("RETRIEVAL CORPUS CREATED")
    print("=" * 60)

    print(f"Original historical examples: {len(historical)}")
    print(f"Golden examples excluded: {len(golden)}")
    print(f"Final retrieval corpus: {len(retrieval_corpus)}")

    print(f"\nSaved to:")
    print(OUTPUT_PATH)

    # Verify there is absolutely no ID overlap.
    retrieval_ids = set(
        retrieval_corpus["customer_tweet_id"].astype(str)
    )

    overlap = retrieval_ids.intersection(golden_ids)

    print(f"\nGolden/retrieval ID overlap: {len(overlap)}")

    if len(overlap) == 0:
        print("PASS: No golden examples are present in retrieval corpus.")
    else:
        print("ERROR: Golden examples are still present!")


if __name__ == "__main__":
    main()