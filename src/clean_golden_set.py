import pandas as pd

PATH = "data/golden_set.csv"

df = pd.read_csv(PATH)

# Normalize inconsistent escalation reasons
df["escalation_reason"] = df["escalation_reason"].replace({
    "need_human_support": "needs_human_support",
    "payment_issue": "needs_human_support",
})

df.to_csv(PATH, index=False)

print("Golden set cleaned successfully.")
print()
print(df["escalation_reason"].value_counts())