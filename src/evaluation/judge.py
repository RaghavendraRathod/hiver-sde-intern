import os
import time
import json
import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types


INPUT_PATH = "results/pipeline_outputs.csv"
OUTPUT_PATH = "results/judge_results.csv"

load_dotenv()


def create_client():
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set.")

    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            retry_options=types.HttpRetryOptions(
                attempts=1
            )
        )
    )


def judge_reply(client, row):
    prompt = f"""
You are evaluating an AI customer-support agent for a technical hiring assignment.

The agent receives a customer message, predicts an intent, decides whether to
escalate to a human, and drafts a reply grounded in historical support examples.

Evaluate ONLY the generated reply and escalation decision using the evidence
provided below.

CUSTOMER MESSAGE:
{row["customer_message"]}

TRUE INTENT:
{row["true_intent"]}

PREDICTED INTENT:
{row["predicted_intent"]}

CLASSIFIER CONFIDENCE:
{row["classifier_confidence"]}

TRUE ESCALATION:
{row["true_escalate"]}

TRUE ESCALATION REASON:
{row["true_escalation_reason"]}

PREDICTED ESCALATION:
{row["escalate"]}

PREDICTED ESCALATION REASON:
{row["escalation_reason"]}

HISTORICAL SUPPORT EVIDENCE:
{row["historical_matches"]}

GENERATED REPLY:
{row["reply"]}

Score the system using these criteria:

1. Groundedness: 0-2
   0 = reply is unsupported or invents information
   1 = partially grounded
   2 = clearly grounded in the available evidence

2. Relevance: 0-2
   0 = does not address the customer's issue
   1 = partially addresses it
   2 = directly addresses it

3. Helpfulness: 0-2
   0 = not useful
   1 = somewhat useful
   2 = gives a useful next step or resolution

4. Tone: 0-2
   0 = inappropriate, confusing, or unprofessional
   1 = acceptable
   2 = professional and customer-friendly

5. Safety/Escalation: 0-2
   Consider whether escalation was appropriate given the customer message.
   0 = clearly unsafe or seriously inappropriate
   1 = debatable/partially appropriate
   2 = appropriate and safe

6. Overall: 0-10
   Give an overall assessment of the complete response.

Return ONLY valid JSON in exactly this format:

{{
  "groundedness": 0,
  "relevance": 0,
  "helpfulness": 0,
  "tone": 0,
  "safety_escalation": 0,
  "overall": 0,
  "reason": "brief explanation"
}}
"""

    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents=prompt
    )

    text = response.text.strip()

    # Remove markdown code fences if Gemini adds them.
    if text.startswith("```"):
        text = text.replace("```json", "").replace("```", "").strip()

    return json.loads(text)


def main():
    print("=" * 80)
    print("LLM-AS-JUDGE EVALUATION")
    print("=" * 80)

    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(
            f"Input file not found: {INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH)

    print(f"\nLoaded cached pipeline outputs: {len(df)}")
    print("Pipeline will NOT be executed again.")

    client = create_client()

    results = []

    for index, row in df.iterrows():

        print("\n" + "=" * 80)
        print(f"Judging example {index + 1}/{len(df)}")
        print("=" * 80)

        print("\nCustomer:")
        print(row["customer_message"])

        print("\nGenerated reply:")
        print(row["reply"])

        try:
            scores = judge_reply(client, row)

            result = {
                "customer_tweet_id": row["customer_tweet_id"],
                "customer_message": row["customer_message"],
                "true_intent": row["true_intent"],
                "predicted_intent": row["predicted_intent"],
                "classifier_confidence": row["classifier_confidence"],
                "true_escalate": row["true_escalate"],
                "escalate": row["escalate"],
                "true_escalation_reason": row["true_escalation_reason"],
                "escalation_reason": row["escalation_reason"],
                "reply": row["reply"],
                "groundedness": scores["groundedness"],
                "relevance": scores["relevance"],
                "helpfulness": scores["helpfulness"],
                "tone": scores["tone"],
                "safety_escalation": scores["safety_escalation"],
                "overall": scores["overall"],
                "reason": scores["reason"],
                "judge_status": "success",
            }

            results.append(result)

            print("\nJudge scores:")
            print(json.dumps(scores, indent=2))

        except Exception as e:

            print(f"\nJudge failed: {e}")

            results.append({
                "customer_tweet_id": row["customer_tweet_id"],
                "customer_message": row["customer_message"],
                "true_intent": row["true_intent"],
                "predicted_intent": row["predicted_intent"],
                "classifier_confidence": row["classifier_confidence"],
                "true_escalate": row["true_escalate"],
                "escalate": row["escalate"],
                "true_escalation_reason": row["true_escalation_reason"],
                "escalation_reason": row["escalation_reason"],
                "reply": row["reply"],
                "groundedness": None,
                "relevance": None,
                "helpfulness": None,
                "tone": None,
                "safety_escalation": None,
                "overall": None,
                "reason": str(e),
                "judge_status": "failed",
            })

        # Avoid hitting free-tier rate limits too aggressively.
        if index < len(df) - 1:
            time.sleep(15)

    output = pd.DataFrame(results)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    output.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\n" + "=" * 80)
    print("LLM JUDGE COMPLETE")
    print("=" * 80)

    print(f"\nSaved: {OUTPUT_PATH}")
    print(f"Total records: {len(output)}")

    successful = (output["judge_status"] == "success").sum()
    failed = (output["judge_status"] == "failed").sum()

    print(f"Successful judgments: {successful}")
    print(f"Failed judgments: {failed}")

    if successful > 0:
        score_columns = [
            "groundedness",
            "relevance",
            "helpfulness",
            "tone",
            "safety_escalation",
            "overall",
        ]

        print("\nAverage scores:")
        print(output.loc[
            output["judge_status"] == "success",
            score_columns
        ].mean())


if __name__ == "__main__":
    main()