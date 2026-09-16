import pandas as pd

INPUT_FILE = "data/twcs/twcs.csv"
OUTPUT_FILE = "data/amazon_customer_sample.csv"

print("Loading dataset...")

# Load only the columns we need
df = pd.read_csv(
    INPUT_FILE,
    usecols=[
        "tweet_id",
        "author_id",
        "inbound",
        "created_at",
        "text",
        "response_tweet_id",
        "in_response_to_tweet_id"
    ]
)

print(f"Total rows loaded: {len(df):,}")

# Create lookup table for tweets
tweet_lookup = df.set_index("tweet_id")

# Amazon support tweets
amazon = df[
    (df["author_id"] == "AmazonHelp") &
    (df["inbound"] == False)
].copy()

print(f"Amazon support tweets: {len(amazon):,}")

samples = []

for _, support in amazon.iterrows():

    parent_id = support["in_response_to_tweet_id"]

    if pd.isna(parent_id):
        continue

    try:
        parent_id = int(parent_id)
    except (ValueError, TypeError):
        continue

    if parent_id not in tweet_lookup.index:
        continue

    customer = tweet_lookup.loc[parent_id]

    # We want the parent message to be from a customer
    if customer["inbound"] != True:
        continue

    text = str(customer["text"]).strip()

    if not text:
        continue

    samples.append({
        "customer_tweet_id": parent_id,
        "customer_text": text,
        "amazon_reply": str(support["text"]).strip(),
        "created_at": customer["created_at"]
    })

# Remove duplicate customer messages
result = pd.DataFrame(samples).drop_duplicates(
    subset=["customer_tweet_id"]
)

# Take a manageable sample
result = result.sample(
    n=min(500, len(result)),
    random_state=42
)

result = result.reset_index(drop=True)

result.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

print()
print("Done!")
print(f"Customer conversations extracted: {len(result)}")
print(f"Saved to: {OUTPUT_FILE}")
print()
print("First 10 examples:")
print(result.head(10).to_string(index=False))