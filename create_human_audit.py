import pandas as pd

# Load LLM judge results
df = pd.read_csv("results/llm_judge_scored_v2.csv")

rows = []
random_seed = 42

# Select 5 LLM-pass + 5 LLM-fail examples from each candidate
for candidate in ["Generic", "Retrieval", "Agent"]:
    candidate_df = df[df["candidate"] == candidate].copy()

    passed = candidate_df[
        candidate_df["overall_pass"] == True
    ].sample(n=5, random_state=random_seed)

    failed = candidate_df[
        candidate_df["overall_pass"] == False
    ].sample(n=5, random_state=random_seed)

    rows.append(pd.concat([passed, failed]))

# Combine and shuffle
audit = (
    pd.concat(rows)
    .sample(frac=1, random_state=random_seed)
    .reset_index(drop=True)
)

# Add blank human-review columns
audit["human_helpfulness"] = ""
audit["human_intent_fit"] = ""
audit["human_grounding"] = ""
audit["human_factual_safety"] = ""
audit["human_escalation_appropriateness"] = ""
audit["human_professionalism"] = ""
audit["human_overall_pass"] = ""
audit["human_failure_reason"] = ""

# Keep only useful audit columns
columns = [
    "customer_tweet_id",
    "candidate",
    "customer_text",
    "reply",
    "gold_intent",
    "predicted_intent",
    "confidence",
    "gold_escalate",
    "predicted_escalate",
    "predicted_escalation_reason",
    "historical_evidence",

    # LLM judge results
    "helpfulness",
    "intent_fit",
    "grounding",
    "factual_safety",
    "escalation_appropriateness",
    "professionalism",
    "overall_pass",
    "failure_reason",

    # Human review fields
    "human_helpfulness",
    "human_intent_fit",
    "human_grounding",
    "human_factual_safety",
    "human_escalation_appropriateness",
    "human_professionalism",
    "human_overall_pass",
    "human_failure_reason",
]

audit[columns].to_csv(
    "results/human_audit.csv",
    index=False
)

print()
print("Human audit created successfully.")
print()
print("Rows per candidate:")
print(audit["candidate"].value_counts().to_string())
print()
print("Total rows:", len(audit))
print("Output: results/human_audit.csv")