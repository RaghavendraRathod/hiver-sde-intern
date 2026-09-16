import pandas as pd
import os

FILE = "data/golden_set.csv"

INTENTS = [
    "delivery_delay",
    "order_status_tracking",
    "missing_package",
    "wrong_item",
    "damaged_item",
    "return_replacement",
    "refund_issue",
    "payment_billing",
    "account_access",
    "technical_product_issue",
    "other_general"
]

ESCALATION_REASONS = {
    "yes": "Human review needed",
    "no": "none"
}


def show_intents():
    print("\nAvailable intents:")
    for i, intent in enumerate(INTENTS, start=1):
        print(f"  {i}. {intent}")


def get_intent():
    while True:
        show_intents()
        choice = input("\nChoose intent number: ").strip()

        try:
            number = int(choice)

            if 1 <= number <= len(INTENTS):
                return INTENTS[number - 1]

        except ValueError:
            pass

        print("Invalid choice. Please enter a number from 1 to 11.")


def get_escalation():
    while True:
        choice = input("\nEscalate to human? (y/n): ").strip().lower()

        if choice == "y":
            return "yes"

        if choice == "n":
            return "no"

        print("Please enter y or n.")


def save(df):
    df.to_csv(FILE, index=False)


def main():

    if not os.path.exists(FILE):
        print(f"ERROR: {FILE} not found.")
        return

    df = pd.read_csv(FILE)

    # Make sure the columns exist
    for column in ["intent", "escalate", "escalation_reason"]:
        if column not in df.columns:
            df[column] = ""

    # Treat empty values as unlabeled
    df["intent"] = df["intent"].fillna("")
    df["escalate"] = df["escalate"].fillna("")
    df["escalation_reason"] = df["escalation_reason"].fillna("")

    # Start from first unlabeled example
    unlabeled = df[
        (df["intent"].str.strip() == "") |
        (df["escalate"].str.strip() == "")
    ]

    print("\n========================================")
    print(" Hiver Golden Set Labeling Tool")
    print("========================================")

    print(f"\nTotal examples: {len(df)}")
    print(f"Already labeled: {len(df) - len(unlabeled)}")
    print(f"Remaining: {len(unlabeled)}")

    if len(unlabeled) == 0:
        print("\nAll examples are already labeled!")
        return

    for index in unlabeled.index:

        row = df.loc[index]

        print("\n")
        print("=" * 70)
        print(f"Example {index + 1} / {len(df)}")
        print("=" * 70)

        print("\nCustomer message:")
        print("-" * 70)
        print(row["customer_text"])
        print("-" * 70)

        print(f"\nTweet ID: {row['customer_tweet_id']}")

        intent = get_intent()

        escalate = get_escalation()

        if escalate == "yes":
            reason = input(
                "\nWhy should this be escalated to a human?\n> "
            ).strip()

            if not reason:
                reason = "Human review needed"

        else:
            reason = "none"

        df.at[index, "intent"] = intent
        df.at[index, "escalate"] = escalate
        df.at[index, "escalation_reason"] = reason

        # Save immediately after every example
        save(df)

        print("\n✓ Saved!")

        print(f"Intent: {intent}")
        print(f"Escalate: {escalate}")
        print(f"Reason: {reason}")

        # Allow user to stop safely
        print("\nPress ENTER to continue.")
        print("Type 'q' to quit after saving this example.")

        command = input("> ").strip().lower()

        if command == "q":
            print("\nProgress saved successfully.")
            print("You can run this script again later to continue.")
            return

    print("\n========================================")
    print(" ALL 200 EXAMPLES ARE LABELED!")
    print("========================================")

    save(df)


if __name__ == "__main__":
    main()