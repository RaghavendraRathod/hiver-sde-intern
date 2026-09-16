import pandas as pd

FILE = "data/amazon_customer_sample.csv"

df = pd.read_csv(FILE)

texts = df["customer_text"].fillna("").astype(str)

topics = {
    "delivery": [
        "deliver", "delivery", "arrive", "arrived",
        "late", "delay", "delayed", "shipping", "shipped"
    ],
    "payment": [
        "payment", "paid", "pay", "card",
        "credit", "bank", "charge", "charged"
    ],
    "return_refund": [
        "return", "refund", "money back",
        "reimburse", "replacement"
    ],
    "account": [
        "account", "login", "password",
        "suspended", "locked"
    ],
    "prime": [
        "prime", "membership"
    ],
    "technical": [
        "error", "problem", "issue",
        "not working", "doesn't work"
    ],
    "wrong_item": [
        "wrong", "incorrect", "fake"
    ],
    "promotion": [
        "promotion", "promo", "discount", "offer"
    ],
}

for topic, keywords in topics.items():

    matches = []

    for i, text in enumerate(texts):

        text_lower = text.lower()

        if any(keyword in text_lower for keyword in keywords):
            matches.append((i, text))

    print("\n" + "=" * 80)
    print(f"TOPIC: {topic.upper()}")
    print(f"MATCHES: {len(matches)}")
    print("=" * 80)

    for i, text in matches[:10]:
        print(f"\nExample {i}:")
        print(text[:500])