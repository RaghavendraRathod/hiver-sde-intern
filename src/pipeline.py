import pandas as pd

from .classifier import IntentClassifier
from .retrieval import HistoricalRetriever
from .escalation import decide_escalation
from .agent.reply import generate_reply


GOLDEN_PATH = "data/golden_set.csv"


class SupportPipeline:
    def __init__(self):
        # Load golden examples for classifier training
        self.golden = pd.read_csv(GOLDEN_PATH)
        self.golden = self.golden.dropna(
            subset=["customer_text", "intent"]
        ).copy()

        # Train intent classifier
        self.classifier = IntentClassifier()
        self.classifier.train(
            self.golden["customer_text"],
            self.golden["intent"]
        )

        # Load historical support conversations
        self.retriever = HistoricalRetriever()

    def analyze(self, customer_message, top_k=3):
        # 1. Classify intent
        classification = self.classifier.predict(customer_message)

        intent = classification["intent"]
        confidence = classification["confidence"]

        # 2. Retrieve similar historical conversations
        historical_matches = self.retriever.retrieve(
            customer_message,
            top_k=top_k
        )

        # 3. Decide whether human escalation is required
        escalate, escalation_reason = decide_escalation(
            customer_message,
            intent,
            confidence
        )

        # 4. Generate grounded reply
        reply = generate_reply(
            customer_message=customer_message,
            intent=intent,
            confidence=confidence,
            historical_matches=historical_matches,
            escalate=escalate,
            escalation_reason=escalation_reason
        )

        # 5. Return complete agent result
        return {
            "customer_message": customer_message,
            "intent": intent,
            "confidence": confidence,
            "historical_matches": historical_matches,
            "escalate": escalate,
            "escalation_reason": escalation_reason,
            "reply": reply,
        }