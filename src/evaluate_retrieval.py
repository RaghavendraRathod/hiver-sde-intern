import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


HISTORICAL_PATH = "data/amazon_customer_sample.csv"
GOLDEN_PATH = "data/golden_set.csv"


class HistoricalRetriever:
    def __init__(self, data_path):
        self.df = pd.read_csv(data_path)

        self.df = self.df.dropna(subset=["customer_text"]).copy()
        self.df["customer_text"] = self.df["customer_text"].astype(str)

        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            max_features=10000,
            stop_words="english"
        )

        self.matrix = self.vectorizer.fit_transform(
            self.df["customer_text"]
        )

    def retrieve(self, query, top_k=3, exclude_id=None):
        query_vector = self.vectorizer.transform([query])

        similarities = cosine_similarity(
            query_vector,
            self.matrix
        ).flatten()

        # Prevent the query from retrieving itself.
        if exclude_id is not None:
            matching_rows = self.df[
                self.df["customer_tweet_id"].astype(str)
                == str(exclude_id)
            ]

            for index in matching_rows.index:
                matrix_position = self.df.index.get_loc(index)
                similarities[matrix_position] = -1

        top_indices = similarities.argsort()[-top_k:][::-1]

        results = []

        for index in top_indices:
            row = self.df.iloc[index]

            results.append({
                "customer_tweet_id": int(row["customer_tweet_id"]),
                "customer_text": row["customer_text"],
                "amazon_reply": row["amazon_reply"],
                "similarity": float(similarities[index])
            })

        return results


def main():
    print("Loading data...")

    golden = pd.read_csv(GOLDEN_PATH)
    retriever = HistoricalRetriever(HISTORICAL_PATH)

    print(f"Golden examples: {len(golden)}")
    print(f"Historical examples: {len(retriever.df)}")

    similarity_scores = []

    for _, row in golden.iterrows():

        query = str(row["customer_text"])
        query_id = row["customer_tweet_id"]

        results = retriever.retrieve(
            query,
            top_k=3,
            exclude_id=query_id
        )

        if results:
            similarity_scores.append(results[0]["similarity"])

    print("\n" + "=" * 60)
    print("LEAKAGE-FREE RETRIEVAL EVALUATION")
    print("=" * 60)

    print(f"Evaluated examples: {len(similarity_scores)}")

    if similarity_scores:
        average_similarity = sum(similarity_scores) / len(similarity_scores)

        print(
            f"Average top-1 similarity: "
            f"{average_similarity:.4f}"
        )

        print(
            f"Minimum top-1 similarity: "
            f"{min(similarity_scores):.4f}"
        )

        print(
            f"Maximum top-1 similarity: "
            f"{max(similarity_scores):.4f}"
        )

    # Show several examples so we can manually inspect retrieval quality.
    print("\n" + "=" * 60)
    print("SAMPLE RETRIEVAL RESULTS")
    print("=" * 60)

    for _, row in golden.head(10).iterrows():

        query = str(row["customer_text"])
        query_id = row["customer_tweet_id"]

        print("\n" + "-" * 60)
        print("CUSTOMER:")
        print(query)

        print(f"\nEXPECTED INTENT: {row['intent']}")

        results = retriever.retrieve(
            query,
            top_k=3,
            exclude_id=query_id
        )

        for rank, result in enumerate(results, start=1):
            print(f"\nMatch {rank}")
            print(f"Similarity: {result['similarity']:.4f}")
            print(f"Tweet ID: {result['customer_tweet_id']}")
            print(f"Customer: {result['customer_text']}")
            print(f"Amazon: {result['amazon_reply']}")


if __name__ == "__main__":
    main()