import pandas as pd

file_path = "data/twcs/twcs.csv"

df = pd.read_csv(
    file_path,
    usecols=[
        "tweet_id",
        "author_id",
        "inbound",
        "text",
        "response_tweet_id",
        "in_response_to_tweet_id"
    ],
    nrows=10000
)

print("\nINBOUND VALUES:")
print(df["inbound"].value_counts(dropna=False))

print("\nINBOUND DATA TYPE:")
print(df["inbound"].dtype)

print("\nSAMPLE AMAZON ROWS:")
amazon = df[df["author_id"] == "AmazonHelp"]

print(
    amazon[
        [
            "tweet_id",
            "author_id",
            "inbound",
            "response_tweet_id",
            "in_response_to_tweet_id",
            "text"
        ]
    ].head(10).to_string(index=False)
)

print("\nSAMPLE ALL ROWS:")
print(
    df[
        [
            "tweet_id",
            "author_id",
            "inbound",
            "response_tweet_id",
            "in_response_to_tweet_id",
            "text"
        ]
    ].head(10).to_string(index=False)
)