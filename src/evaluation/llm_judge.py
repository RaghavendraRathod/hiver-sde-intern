import json
from pathlib import Path

import pandas as pd


INPUT_PATH = "results/reply_evaluation.csv"
OUTPUT_PATH = "results/llm_judge_results.csv"


RUBRIC = """
Score each response from 1 to 5 on:

1. helpfulness
   1 = does not help the customer
   3 = somewhat useful
   5 = directly useful and addresses the issue

2. intent_fit
   1 = clearly misunderstands the customer's issue
   3 = partially addresses it
   5 = directly addresses the customer's issue

3. grounding
   1 = unsupported or contradicts the historical evidence
   3 = partly consistent with evidence
   5 = clearly grounded in the historical evidence

4. factual_safety
   1 = invents facts, policies, actions, or guarantees
   3 = mostly safe but somewhat assumptive
   5 = makes no unsupported claims

5. escalation_appropriateness
   1 = escalation decision is clearly inappropriate
   3 = debatable
   5 = clearly appropriate

6. professionalism
   1 = poor/unprofessional
   3 = acceptable
   5 = clear, concise, professional support language

Also provide:

overall_pass:
true only if the response is reasonably helpful, safe, relevant,
and appropriate for the situation.

failure_reason:
A short explanation if overall_pass is false.
Use an empty string if the response passes.
"""


def format_evidence(raw_evidence):
    """Convert the stored JSON evidence into readable judge context."""
    try:
        evidence = json.loads(raw_evidence)
    except (TypeError, json.JSONDecodeError):
        return "No historical evidence available."

    if not evidence:
        return "No historical evidence available."

    sections = []

    for i, item in enumerate(evidence, start=1):
        sections.append(
            f"""
Historical example {i}
Similarity: {item.get("similarity", "")}

Customer:
{item.get("customer_text", "")}

Amazon reply:
{item.get("amazon_reply", "")}
""".strip()
        )

    return "\n\n".join(sections)


def build_prompt(row, candidate_name, candidate_reply):
    evidence = format_evidence(row["historical_evidence"])

    return f"""
You are evaluating an AI customer-support response.

Do NOT assume that a response is good merely because it resembles a
historical response.

Evaluate whether the candidate response actually addresses the customer's
message and whether its claims are supported by the supplied historical
evidence.

CUSTOMER MESSAGE:
{row["customer_text"]}

HISTORICAL EVIDENCE:
{evidence}

CANDIDATE:
{candidate_name}

CANDIDATE RESPONSE:
{candidate_reply}

ESCALATION DECISION MADE BY THE SYSTEM:
{row["predicted_escalate"]}

ESCALATION REASON:
{row["predicted_escalation_reason"]}

{RUBRIC}

Return ONLY valid JSON in exactly this structure:

{{
  "helpfulness": 1,
  "intent_fit": 1,
  "grounding": 1,
  "factual_safety": 1,
  "escalation_appropriateness": 1,
  "professionalism": 1,
  "overall_pass": false,
  "failure_reason": "short explanation"
}}
""".strip()


def main():
    df = pd.read_csv(INPUT_PATH)

    required_columns = [
        "customer_tweet_id",
        "customer_text",
        "gold_intent",
        "predicted_intent",
        "confidence",
        "gold_escalate",
        "predicted_escalate",
        "predicted_escalation_reason",
        "generic_reply",
        "retrieval_reply",
        "agent_reply",
        "agent_reply_source",
        "historical_evidence",
    ]

    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        raise ValueError(
            f"Missing required columns in {INPUT_PATH}: {missing}"
        )

    rows = []

    candidates = [
        ("Generic", "generic_reply"),
        ("Retrieval", "retrieval_reply"),
        ("Agent", "agent_reply"),
    ]

    for _, row in df.iterrows():

        for candidate_name, reply_column in candidates:
            candidate_reply = str(row[reply_column])

            prompt = build_prompt(
                row=row,
                candidate_name=candidate_name,
                candidate_reply=candidate_reply,
            )

            rows.append(
                {
                    "customer_tweet_id": row["customer_tweet_id"],
                    "candidate": candidate_name,
                    "customer_text": row["customer_text"],
                    "gold_intent": row["gold_intent"],
                    "predicted_intent": row["predicted_intent"],
                    "confidence": row["confidence"],
                    "gold_escalate": row["gold_escalate"],
                    "predicted_escalate": row["predicted_escalate"],
                    "predicted_escalation_reason": row[
                        "predicted_escalation_reason"
                    ],
                    "reply": candidate_reply,
                    "agent_reply_source": row["agent_reply_source"],
                    "historical_evidence": row["historical_evidence"],
                    "judge_prompt": prompt,
                    "helpfulness": None,
                    "intent_fit": None,
                    "grounding": None,
                    "factual_safety": None,
                    "escalation_appropriateness": None,
                    "professionalism": None,
                    "overall_pass": None,
                    "failure_reason": "",
                }
            )

    result = pd.DataFrame(rows)

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_PATH, index=False)

    print(f"Prepared {len(result)} judge prompts.")
    print(f"Rows per candidate: {len(df)}")
    print(f"Candidates: {len(candidates)}")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()