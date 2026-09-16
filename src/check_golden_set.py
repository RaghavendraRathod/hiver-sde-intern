import pandas as pd

PATH = "data/golden_set.csv"

VALID_INTENTS = [
    "delivery_delay",
    "order_status_tracking",
    "missing_package",
    "wrong_item",
    "damaged_item",
    "return_replacement",
    "refund_issue",
    "payment_billing",
    "account_access",
    "technical_product_issue",
    "other_general",
]

df = pd.read_csv(PATH)

print("=" * 60)
print("GOLDEN SET QUALITY CHECK")
print("=" * 60)

print(f"\nTotal rows: {len(df)}")

# Required columns
required_columns = [
    "customer_tweet_id",
    "customer_text",
    "intent",
    "escalate",
    "escalation_reason",
]

print("\nRequired columns:")
for col in required_columns:
    status = "OK" if col in df.columns else "MISSING"
    print(f"  {col}: {status}")

# Missing values
print("\nMissing values:")
for col in required_columns:
    if col in df.columns:
        print(f"  {col}: {df[col].isna().sum()}")

# Intent distribution
print("\nIntent distribution:")
intent_counts = df["intent"].value_counts()

for intent in VALID_INTENTS:
    count = intent_counts.get(intent, 0)
    percentage = count / len(df) * 100
    print(f"  {intent:25s}: {count:3d} ({percentage:5.1f}%)")

# Invalid intents
invalid = df[~df["intent"].isin(VALID_INTENTS)]

print(f"\nInvalid intent labels: {len(invalid)}")

if len(invalid) > 0:
    print(invalid[["customer_tweet_id", "intent"]].to_string(index=False))

# Escalation distribution
print("\nEscalation distribution:")
print(df["escalate"].value_counts(dropna=False))

# Escalation reasons
print("\nEscalation reasons:")
print(df["escalation_reason"].value_counts(dropna=False))

# Duplicate IDs
duplicates = df[df["customer_tweet_id"].duplicated(keep=False)]

print(f"\nDuplicate tweet IDs: {len(duplicates)}")

if len(duplicates) > 0:
    print(duplicates[["customer_tweet_id", "customer_text"]].to_string(index=False))

print("\n" + "=" * 60)
print("QUALITY CHECK COMPLETE")
print("=" * 60)