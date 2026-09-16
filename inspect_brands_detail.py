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

stats = defaultdict(lambda: {"total": 0, "inbound": 0, "outbound": 0})

print("Analyzing top brands...")

for chunk in pd.read_csv(
    file_path,
    usecols=["author_id", "inbound"],
    chunksize=100_000
):
    chunk["author_id"] = chunk["author_id"].astype(str)

    for brand in brands:
        rows = chunk[chunk["author_id"] == brand]

        if len(rows) > 0:
            stats[brand]["total"] += len(rows)
            stats[brand]["inbound"] += rows["inbound"].sum()
            stats[brand]["outbound"] += (~rows["inbound"]).sum()

print("\nBrand statistics:")
print("-" * 65)

for brand in brands:
    s = stats[brand]

    print(
        f"{brand:20} "
        f"Total: {s['total']:7,} | "
        f"Customer: {s['inbound']:7,} | "
        f"Support: {s['outbound']:7,}"
    )