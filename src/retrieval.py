import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DATA_PATH = "data/retrieval_corpus.csv"


class HistoricalRetriever:
    def __init__(self, data_path=DATA_PATH):
        print("Loading clean historical retrieval corpus...")

        self.df = pd.read_csv(data_path)

        self.df = self.df.dropna(
            subset=["customer_text"]
        ).copy()

        self.df["customer_text"] = (
            self.df["customer_text"].astype(str)
        )

        print(f"Loaded {len(self.df)} historical conversations.")

        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            max_features=10000,
            stop_words="english"
        )

        self.matrix = self.vectorizer.fit_transform(
            self.df["customer_text"]
        )

        print("TF-IDF retrieval index created.")

    def retrieve(self, query, top_k=3):
        query_vector = self.vectorizer.transform([query])

        similarities = cosine_similarity(
            query_vector,
            self.matrix
        ).flatten()

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
    retriever = HistoricalRetriever()

    test_queries = [
        "My Amazon package is late and has not arrived yet.",
        "I was charged for something I did not buy.",
        "My Fire TV keeps restarting and freezing."
    ]

    for query in test_queries:
        print("=" * 80)
        print("CUSTOMER QUERY:")
        print(query)

        print("\nTOP HISTORICAL MATCHES:")

        results = retriever.retrieve(
            query,
            top_k=3
        )

        for rank, result in enumerate(results, 1):
            print(f"\n--- Match {rank} ---")
            print(
                f"Similarity: {result['similarity']:.4f}"
            )
            print(
                f"Tweet ID: {result['customer_tweet_id']}"
            )
            print(
                f"Customer: {result['customer_text']}"
            )
            print(
                f"Amazon: {result['amazon_reply']}"
            )


if __name__ == "__main__":
    main()