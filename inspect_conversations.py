import pandas as pd
from collections import defaultdict

file_path = "data/twcs/twcs.csv"

brands = [
    "AmazonHelp",
    "AppleSupport",
    "Uber_Support",
    "SpotifyCares",
    "Delta",
    "Tesco",
    "AmericanAir",
    "TMobileHelp",
    "comcastcares",
    "British_Airways"
]

# Store tweet information needed to connect conversations
tweets = {}

print("Loading relevant tweets...")

for chunk in pd.read_csv(
    file_path,
    usecols=[
        "tweet_id",
        "author_id",
        "inbound",
        "text",
        "response_tweet_id",
        "in_response_to_tweet_id"
    ],
    chunksize=100_000
):

    chunk["author_id"] = chunk["author_id"].astype(str)

    # Keep tweets belonging to the selected brands
    brand_rows = chunk[chunk["author_id"].isin(brands)]

    for _, row in brand_rows.iterrows():
        tweet_id = str(row["tweet_id"])

        tweets[tweet_id] = {
            "author_id": row["author_id"],
            "inbound": row["inbound"],
            "text": row["text"],
            "response_tweet_id": row["response_tweet_id"],
            "in_response_to_tweet_id": row["in_response_to_tweet_id"]
        }

print(f"Brand tweets loaded: {len(tweets):,}")

print("\nNow checking conversation relationships...")

# Count conversations initiated by customers
conversation_stats = defaultdict(lambda: {
    "support_tweets": 0,
    "customer_tweets": 0,
    "conversations": set()
})

for tweet_id, tweet in tweets.items():

    brand = tweet["author_id"]

    conversation_stats[brand]["support_tweets"] += 1

    parent = tweet["in_response_to_tweet_id"]

    if pd.notna(parent):
        parent = str(parent)

        conversation_stats[brand]["conversations"].add(parent)

print("\nConversation statistics:")
print("-" * 80)

for brand in brands:

    stats = conversation_stats[brand]

    print(
        f"{brand:20} "
        f"Support tweets: {stats['support_tweets']:7,} | "
        f"Customer conversations: {len(stats['conversations']):7,}"
    )