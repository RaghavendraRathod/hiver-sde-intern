import pandas as pd

file_path = "data/twcs/twcs.csv"

print("Loading dataset...")

df = pd.read_csv(
    file_path,
    usecols=[
        "tweet_id",
        "author_id",
        "inbound",
        "text",
        "response_tweet_id",
        "in_response_to_tweet_id"
    ]
)

print(f"Total tweets loaded: {len(df):,}")

# Normalize tweet IDs
df["tweet_id"] = df["tweet_id"].astype(str)

# Find Amazon support tweets
amazon = df[df["author_id"] == "AmazonHelp"].copy()

print(f"Amazon support tweets: {len(amazon):,}")

# Create lookup by tweet ID
tweet_lookup = df.set_index("tweet_id")

print("\nSample customer -> Amazon conversations")
print("=" * 100)

shown = 0

for _, support in amazon.iterrows():

    parent_id = support["in_response_to_tweet_id"]

    # Skip if no parent tweet
    if pd.isna(parent_id):
        continue

    parent_id = str(int(parent_id))

    # Parent tweet must exist
    if parent_id not in tweet_lookup.index:
        continue

    customer = tweet_lookup.loc[parent_id]

    # Parent must be an inbound/customer tweet
    if customer["inbound"] != True:
        continue

    print("\nCUSTOMER:")
    print(customer["text"])

    print("\nAMAZON:")
    print(support["text"])

    print("-" * 100)

    shown += 1

    if shown >= 20:
        break

print(f"\nDisplayed {shown} conversations.")