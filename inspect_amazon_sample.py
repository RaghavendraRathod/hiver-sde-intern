import pandas as pd

FILE = "data/amazon_customer_sample.csv"

df = pd.read_csv(FILE)

print(f"Total examples: {len(df)}")
print()

for i, row in df.head(50).iterrows():
    print("=" * 100)
    print(f"EXAMPLE {i + 1}")
    print(f"Customer: {row['customer_text']}")
    print(f"Amazon:   {row['amazon_reply']}")
    print()