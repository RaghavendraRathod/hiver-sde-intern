import json
import os

import pandas as pd
from dotenv import load_dotenv

GOLDEN_PATH = "data/golden_set.csv"
OUTPUT_PATH = "results/pipeline_outputs.csv"

load_dotenv()


def main():
    print("Loading golden set...")

    golden = pd.read_csv(GOLDEN_PATH)

    print(f"Golden examples available: {len(golden)}")

    # Use the same fixed sample as the current judge.
    sample = golden.head(20).copy()

    print(f"\nGenerating pipeline outputs for {len(sample)} examples...\n")

    # Import here so the script can be run from the project root.
    from src.pipeline import SupportPipeline

    pipeline = SupportPipeline()

    results = []

    for index, row in sample.iterrows():

        print("=" * 80)
        print(f"Example {index + 1}/{len(sample)}")
        print("=" * 80)

        customer_text = row["customer_text"]

        print("\nCustomer:")
        print(customer_text)

        result = pipeline.analyze(
            customer_text,
            top_k=3,
        )

        print("\nPredicted intent:")
        print(result["intent"])

        print("\nConfidence:")
        print(result["confidence"])

        print("\nEscalation:")
        print(result["escalate"])

        print("\nEscalation reason:")
        print(result["escalation_reason"])

        print("\nGenerated reply:")
        print(result["reply"])

        # Convert historical matches to JSON so the complete
        # pipeline output can be stored in one CSV cell.
        historical_matches = result.get("historical_matches", [])

        results.append(
            {
                "customer_tweet_id": row["customer_tweet_id"],
                "customer_message": customer_text,
                "true_intent": row["intent"],
                "true_escalate": row["escalate"],
                "true_escalation_reason": row["escalation_reason"],
                "predicted_intent": result["intent"],
                "classifier_confidence": result["confidence"],
                "escalate": result["escalate"],
                "escalation_reason": result["escalation_reason"],
                "reply": result["reply"],
                "historical_matches": json.dumps(
                    historical_matches,
                    ensure_ascii=False,
                ),
            }
        )

    output = pd.DataFrame(results)

    os.makedirs(
        os.path.dirname(OUTPUT_PATH),
        exist_ok=True,
    )

    output.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("\n" + "=" * 80)
    print("PIPELINE RECORD GENERATION COMPLETE")
    print("=" * 80)

    print(f"\nSaved: {OUTPUT_PATH}")
    print(f"Records saved: {len(output)}")


if __name__ == "__main__":
    main()