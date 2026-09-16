import pandas as pd
from collections import Counter

file_path = "data/twcs/twcs.csv"

counts = Counter()
total_rows = 0

print("Scanning dataset...")

for chunk in pd.read_csv(
    file_path,
    usecols=["author_id"],
    chunksize=100_000
):
    counts.update(chunk["author_id"].dropna().astype(str))
    total_rows += len(chunk)

print(f"\nTotal rows: {total_rows:,}")

print("\nTop 30 accounts:")
for name, count in counts.most_common(30):
    print(f"{name}: {count:,}")