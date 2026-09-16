import pandas as pd

INPUT_FILE = "data/amazon_customer_sample.csv"
OUTPUT_FILE = "data/golden_set.csv"

df = pd.read_csv(INPUT_FILE)

print("Available columns:")
print(list(df.columns))

# Select 200 examples reproducibly
golden = df.sample(n=200, random_state=42).copy()

# Add our manual evaluation fields
golden["intent"] = ""
golden["escalate"] = ""
golden["escalation_reason"] = ""

# Keep the columns that actually exist in our dataset
columns_to_keep = [
    "customer_tweet_id",
    "customer_text",
]

# Add the support response only if it exists
if "support_text" in golden.columns:
    columns_to_keep.append("support_text")
elif "response_text" in golden.columns:
    columns_to_keep.append("response_text")
elif "support_response" in golden.columns:
    columns_to_keep.append("support_response")

columns_to_keep.extend([
    "intent",
    "escalate",
    "escalation_reason",
])

golden = golden[columns_to_keep]

golden.to_csv(OUTPUT_FILE, index=False)

print()
print(f"Golden set created: {OUTPUT_FILE}")
print(f"Examples: {len(golden)}")

print("\nFinal columns:")
for column in golden.columns:
    print(f"- {column}")