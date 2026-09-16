import re


# Intents that normally require human intervention.
ESCALATION_INTENTS = {
    "missing_package": "missing_package",
    "wrong_item": "wrong_item",
    "damaged_item": "damaged_item",
    "refund_issue": "delayed_refund",
    "account_access": "needs_human_support",
}


# Explicit requests to speak to/contact human support.
HUMAN_SUPPORT_PATTERNS = [
    r"\b(?:speak|talk|chat|contact|reach)\s+(?:to|with)\s+(?:a\s+)?(?:human|person|agent|support)\b",
    r"\b(?:human|live)\s+agent\b",
    r"\b(?:customer\s+)?service\s+agent\b",
    r"\bcan\s+you\s+dm\b",
    r"\bdm\s+(?:me|about|please)\b",
    r"\bcontact\s+me\b",
    r"\bneed\s+(?:a\s+)?(?:human|person|agent)\b",
    r"\bwant\s+(?:a\s+)?(?:human|person|agent)\b",
    r"\baftersales\s+issue\b",
]


# High-risk unauthorized payment / membership signals.
UNAUTHORIZED_CHARGE_PATTERNS = [
    r"\bcharged\s+without\s+(?:my\s+)?authori[sz]ation\b",
    r"\bcharged\s+without\s+(?:my\s+)?permission\b",
    r"\bcharged\s+without\s+(?:my\s+)?consent\b",
    r"\bwithout\s+(?:my\s+)?permission\b",
    r"\bwithout\s+(?:my\s+)?consent\b",
    r"\bnot\s+authori[sz]ed\b",
    r"\b(?:didn't|did\s+not)\s+sign\s+up\b",
    r"\b(?:didn't|did\s+not)\s+subscribe\b",
    r"\b(?:didn't|did\s+not)\s+join\b",
    r"\b(?:didn't|did\s+not)\s+authorize\b",
    r"\b(?:didn't|did\s+not)\s+make\s+(?:this|the)\s+(?:charge|payment|purchase)\b",
    r"\bunauthori[sz]ed\s+(?:charge|payment|purchase|transaction)\b",
    r"\b(?:unknown|unrecognized|unrecognised)\s+(?:charge|payment|transaction)\b",
    r"\bprime\s+(?:membership|member)\b.*\b(?:didn't|did\s+not|never)\b",
]


# Strong delivery-failure signals.
SEVERE_DELIVERY_PATTERNS = [
    r"\bdidn't\s+(?:even\s+)?attempt\s+delivery\b",
    r"\bdid\s+not\s+(?:even\s+)?attempt\s+delivery\b",
    r"\bno\s+delivery\s+attempt\b",
    r"\bdelivery\s+was\s+not\s+attempted\b",
    r"\bhad\s+to\s+chase\b",
    r"\b(?:courier|driver)\s+never\s+(?:came|arrived|attempted)\b",
]


# Signals that a refund is genuinely delayed/unresolved.
DELAYED_REFUND_PATTERNS = [
    r"\b(?:still|yet)\b.*\b(?:refund|refunded)\b",
    r"\brefund\b.*\b(?:still|not|never)\b",
    r"\brefund\b.*\b(?:waiting|waited|pending|delayed)\b",
]


def _matches_any(text, patterns):
    """Return True if any regex pattern matches the text."""
    text = text.lower()

    for pattern in patterns:
        if re.search(pattern, text):
            return True

    return False


def decide_escalation(text, intent, confidence):
    confidence = float(confidence)
    """
    Decide whether a customer message should be escalated.

    Returns:
        (escalate: bool, reason: str)
    """

    text_lower = text.lower()

    # ---------------------------------------------------------
    # 1. High-risk unauthorized charges take highest priority.
    # ---------------------------------------------------------
    if _matches_any(text_lower, UNAUTHORIZED_CHARGE_PATTERNS):
        return True, "unauthorized_charge"

    # ---------------------------------------------------------
    # 2. Explicit human-support requests.
    # ---------------------------------------------------------
    if _matches_any(text_lower, HUMAN_SUPPORT_PATTERNS):
        return True, "needs_human_support"

    # ---------------------------------------------------------
    # 3. Severe delivery failures.
    # ---------------------------------------------------------
    if _matches_any(text_lower, SEVERE_DELIVERY_PATTERNS):
        return True, "delivery_issue"

    # ---------------------------------------------------------
    # 4. Existing intent-based escalation policy.
    # ---------------------------------------------------------
    if intent in ESCALATION_INTENTS:

        # Refunds should only automatically escalate when
        # there is evidence that the refund is unresolved/delayed.
        if intent == "refund_issue":
            if _matches_any(text_lower, DELAYED_REFUND_PATTERNS):
                return True, "delayed_refund"

            # Avoid escalating every generic refund question.
            return False, "none"

        return True, ESCALATION_INTENTS[intent]

    # ---------------------------------------------------------
    # 5. Very low confidence should be reviewed by a human.
    # ---------------------------------------------------------
    if confidence < 0.20:
        return True, "low_intent_confidence"

    # ---------------------------------------------------------
    # 6. Otherwise, no escalation.
    # ---------------------------------------------------------
    return False, "none"