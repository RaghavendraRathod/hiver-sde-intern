import pandas as pd
import re
from collections import Counter

FILE = "data/amazon_customer_sample.csv"

df = pd.read_csv(FILE)

# Combine customer messages
texts = df["customer_text"].fillna("").astype(str)

# Common customer-support keywords/topics
keywords = {
    "delivery": [
        "deliver", "delivery", "arrive", "arrived", "late",
        "delay", "delayed", "shipping", "shipped"
    ],
    "order": [
        "order", "ordered", "ordering"
    ],
    "package": [
        "package", "parcel", "shipment"
    ],
    "refund": [
        "refund", "money back", "reimburse"
    ],
    "payment": [
        "payment", "paid", "pay", "card", "credit", "bank"
    ],
    "account": [
        "account", "login", "password", "suspended", "locked"
    ],
    "prime": [
        "prime", "membership"
    ],
    "return": [
        "return", "replacement", "replace"
    ],
    "damaged": [
        "damaged", "damage", "broken", "crushed"
    ],
    "missing": [
        "missing", "didn't receive", "not received", "not arrived"
    ],
    "wrong_item": [
        "wrong", "incorrect", "fake"
    ],
    "technical": [
        "error", "problem", "issue", "not working", "doesn't work"
    ],
    "promotion": [
        "promotion", "promo", "discount", "offer"
    ],
    "cancel": [
        "cancel", "cancelled", "canceled"
    ],
    "preorder": [
        "pre-order", "preorder", "preorder"
    ]
}

print(f"Total examples: {len(df)}")
print()
print("Keyword/topic frequency")
print("=" * 60)

results = []

for topic, words in keywords.items():
    count = 0

    for text in texts:
        text_lower = text.lower()

        if any(word in text_lower for word in words):
            count += 1

    results.append((topic, count))

results.sort(key=lambda x: x[1], reverse=True)

for topic, count in results:
    percentage = count / len(df) * 100
    print(f"{topic:15} {count:4} ({percentage:5.1f}%)")

print()
print("Note:")
print("These are keyword-based discovery counts, NOT final intent labels.")