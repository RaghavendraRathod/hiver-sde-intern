import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd
from google import genai
from google.genai import types


# ============================================================
# Configuration
# ============================================================

MODEL = "gemini-3.1-flash-lite"

INPUT_FILE = Path("results/llm_judge_results.csv")
OUTPUT_FILE = Path("results/llm_judge_scored_v2.csv")

MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 2


# ============================================================
# Judge rubric
# ============================================================

SYSTEM_PROMPT = """
You are an expert evaluator of customer-support AI responses.

Evaluate ONE candidate response to a customer's message.

Score the response using the following rubric.

All 1-5 scores mean:

1 = Very poor
2 = Poor
3 = Acceptable but weak
4 = Good
5 = Excellent

Criteria:

helpfulness:
Does the response meaningfully help the customer move toward resolution?

intent_fit:
Does the response correctly address the customer's actual support need?

grounding:
Is the response appropriately grounded in the supplied historical support evidence?
Do NOT reward invented policies, guarantees, or unsupported claims.

factual_safety:
Does the response avoid unsupported claims, fabricated actions, fabricated order information,
false promises, or pretending that something was done when it was not?

escalation_appropriateness:
Given the customer message and the operational escalation decision, is the response appropriate?
A response can be appropriate even when escalation is required, as long as it does not falsely
claim that escalation has already happened.

professionalism:
Is the response clear, concise, polite, and professional?

overall_pass:
Return true only if the response is broadly safe and useful for a real customer-support setting.

failure_reason:
If overall_pass is false, briefly identify the main problem.
If overall_pass is true, return an empty string.

Important:
- Judge only the information supplied in the evaluation prompt.
- Do not assume facts that are not present.
- Do not reward a response merely because it sounds polite.
- Do not penalize a response simply because it is concise.
- Historical replies are evidence of previous support behavior, not authoritative policy.
"""


# ============================================================
# Gemini client
# ============================================================

def get_client():
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. "
            "Set it in PowerShell before running this script."
        )

    return genai.Client(api_key=api_key)


# ============================================================
# Prompt construction
# ============================================================

def build_prompt(row):
    customer_message = str(row.get("customer_text", ""))
    predicted_intent = str(row.get("predicted_intent", ""))
    confidence = str(row.get("confidence", ""))
    predicted_escalation = str(row.get("predicted_escalate", ""))
    escalation_reason = str(
        row.get("predicted_escalation_reason", "")
    )

    historical_evidence = str(
        row.get("historical_evidence", "")
    )

    candidate_type = str(row.get("candidate", ""))
    candidate_response = str(row.get("reply", ""))

    return f"""
Evaluate this customer-support response.

CUSTOMER MESSAGE:
{customer_message}

OPERATIONAL CONTEXT:
Predicted intent: {predicted_intent}
Intent confidence: {confidence}
Predicted escalation: {predicted_escalation}
Escalation reason: {escalation_reason}

HISTORICAL SUPPORT EVIDENCE:
{historical_evidence}

CANDIDATE TYPE:
{candidate_type}

CANDIDATE RESPONSE:
{candidate_response}

Return ONLY a JSON object matching the required schema.
""".strip()


# ============================================================
# JSON schema
# ============================================================

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "helpfulness": {
            "type": "integer",
            "minimum": 1,
            "maximum": 5,
        },
        "intent_fit": {
            "type": "integer",
            "minimum": 1,
            "maximum": 5,
        },
        "grounding": {
            "type": "integer",
            "minimum": 1,
            "maximum": 5,
        },
        "factual_safety": {
            "type": "integer",
            "minimum": 1,
            "maximum": 5,
        },
        "escalation_appropriateness": {
            "type": "integer",
            "minimum": 1,
            "maximum": 5,
        },
        "professionalism": {
            "type": "integer",
            "minimum": 1,
            "maximum": 5,
        },
        "overall_pass": {
            "type": "boolean",
        },
        "failure_reason": {
            "type": "string",
        },
    },
    "required": [
        "helpfulness",
        "intent_fit",
        "grounding",
        "factual_safety",
        "escalation_appropriateness",
        "professionalism",
        "overall_pass",
        "failure_reason",
    ],
}


# ============================================================
# Result validation
# ============================================================

def validate_result(result):
    required_fields = [
        "helpfulness",
        "intent_fit",
        "grounding",
        "factual_safety",
        "escalation_appropriateness",
        "professionalism",
        "overall_pass",
        "failure_reason",
    ]

    for field in required_fields:
        if field not in result:
            raise ValueError(
                f"Missing required field: {field}"
            )

    score_fields = [
        "helpfulness",
        "intent_fit",
        "grounding",
        "factual_safety",
        "escalation_appropriateness",
        "professionalism",
    ]

    for field in score_fields:
        value = result[field]

        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(
                f"{field} must be an integer."
            )

        if value < 1 or value > 5:
            raise ValueError(
                f"{field} must be between 1 and 5."
            )

    if not isinstance(result["overall_pass"], bool):
        raise ValueError(
            "overall_pass must be boolean."
        )

    if not isinstance(result["failure_reason"], str):
        raise ValueError(
            "failure_reason must be a string."
        )

    return True


# ============================================================
# Single judge call
# ============================================================

def judge_one(client, row):
    prompt = build_prompt(row)

    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    response_mime_type="application/json",
                    response_schema=JUDGE_SCHEMA,
                    temperature=0,
                ),
            )

            raw_text = response.text

            if not raw_text:
                raise ValueError(
                    "Model returned empty output."
                )

            result = json.loads(raw_text)

            validate_result(result)

            return result, attempt, None

        except Exception as exc:
            last_error = str(exc)

            if attempt < MAX_RETRIES:
                time.sleep(
                    RETRY_DELAY_SECONDS * attempt
                )

    return None, MAX_RETRIES, last_error


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Evaluate only the first 3 rows.",
    )

    args = parser.parse_args()

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    required_columns = [
        "customer_tweet_id",
        "customer_text",
        "predicted_intent",
        "confidence",
        "predicted_escalate",
        "predicted_escalation_reason",
        "candidate",
        "reply",
        "historical_evidence",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns in {INPUT_FILE}: {missing}"
        )

    print(f"Loaded {len(df)} evaluation rows.")

    if args.smoke_test:
        df = df.head(3).copy()
        print("SMOKE TEST: evaluating only 3 rows.")

    client = get_client()

    results = []

    for position, (_, row) in enumerate(
        df.iterrows(),
        start=1,
    ):
        print(
            f"[{position}/{len(df)}] "
            f"Evaluating {row.get('candidate', 'candidate')}..."
        )

        result, attempts, error = judge_one(
            client,
            row,
        )

        output_row = row.to_dict()

        output_row["judge_model"] = MODEL
        output_row["judge_attempts"] = attempts

        if result is not None:
            output_row.update(result)
            output_row["judge_status"] = "success"

            print(
                f"    success | "
                f"helpfulness={result['helpfulness']} | "
                f"intent_fit={result['intent_fit']} | "
                f"grounding={result['grounding']} | "
                f"overall_pass={result['overall_pass']}"
            )

        else:
            output_row["judge_status"] = "failed"
            output_row["judge_error"] = error

            print(
                f"    FAILED | {error}"
            )

        results.append(output_row)

    output_df = pd.DataFrame(results)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    successful = (
        output_df["judge_status"] == "success"
    ).sum()

    failed = (
        output_df["judge_status"] == "failed"
    ).sum()

    print()
    print("=" * 60)
    print("LLM JUDGE COMPLETE")
    print("=" * 60)
    print(f"Model:       {MODEL}")
    print(f"Rows:        {len(output_df)}")
    print(f"Successful:  {successful}")
    print(f"Failed:      {failed}")
    print(f"Output:      {OUTPUT_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()