from src.escalation import decide_escalation


tests = [
    (
        "still no any confirmation about refund or exchange my watch",
        "refund_issue",
        0.90,
    ),
    (
        "can you DM about an aftersales issue. Thanks",
        "other_general",
        0.40,
    ),
    (
        "$108 was taken out my account for amazon prime and i didn't sign up for none of that shit",
        "payment_billing",
        0.57,
    ),
    (
        "Amazon Prime membership appeared without my permission and I was charged",
        "payment_billing",
        0.60,
    ),
    (
        "how do I complain about this? Didn't even attempt delivery or bother telling me",
        "delivery_delay",
        0.39,
    ),
]


for i, (text, intent, confidence) in enumerate(tests, start=1):
    result = decide_escalation(text, intent, confidence)

    print("=" * 80)
    print(f"Test {i}")
    print("Message:", text)
    print("Intent:", intent)
    print("Confidence:", confidence)
    print("Result:", result)