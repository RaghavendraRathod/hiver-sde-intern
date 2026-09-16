import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types


load_dotenv()


MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

# Set USE_GEMINI=false for deterministic/reproducible evaluation.
# Set USE_GEMINI=true when you specifically want to test real Gemini generation.
USE_GEMINI = os.getenv("USE_GEMINI", "true").lower() == "true"


# Keep the SDK itself from doing repeated retries.
# Our code controls retry behavior explicitly.
client = genai.Client(
    http_options=types.HttpOptions(
        retry_options=types.HttpRetryOptions(
            attempts=1,
            initial_delay=0,
            max_delay=0,
        ),
        timeout=15000,
    )
)


def build_evidence(historical_matches):
    """
    Convert retrieved historical conversations into concise evidence
    that can be provided to Gemini.
    """

    if not historical_matches:
        return "No historical examples were retrieved."

    evidence_blocks = []

    for i, match in enumerate(historical_matches, start=1):
        customer = match.get("customer_text", "").strip()
        response = match.get("amazon_reply", "").strip()
        similarity = match.get("similarity", 0.0)

        evidence_blocks.append(
            f"""Example {i}
Similarity: {similarity:.4f}
Customer: {customer}
Historical Amazon response: {response}"""
        )

    return "\n\n".join(evidence_blocks)


def fallback_reply(
    customer_message,
    intent,
    historical_matches,
    escalate,
    escalation_reason,
):
    """
    Deterministic fallback used when Gemini is unavailable.

    The fallback intentionally avoids inventing specific policies,
    refunds, dates, order details, tracking numbers, or actions
    that are not supported by the evidence.
    """

    if escalate:
        return (
            "I'm sorry you're experiencing this issue. "
            "I've identified that this case should be reviewed "
            "by a human support agent. "
            "A support agent can review the order details and "
            "help with the next steps."
        )

    if intent == "delivery_delay":
        return (
            "I'm sorry your package is delayed. "
            "Please check the current tracking status of your order "
            "for the latest delivery information."
        )

    if intent == "order_status_tracking":
        return (
            "I can help with your order status. "
            "Please check the current tracking information for the "
            "latest update on your order."
        )

    if intent == "missing_package":
        return (
            "I'm sorry you haven't received your package. "
            "Please check the latest delivery and tracking information "
            "for your order."
        )

    if intent == "wrong_item":
        return (
            "I'm sorry you received the wrong item. "
            "Please provide the relevant order details so the issue "
            "can be reviewed and the appropriate next steps determined."
        )

    if intent == "damaged_item":
        return (
            "I'm sorry your item arrived damaged. "
            "Please provide the relevant order details so the issue "
            "can be reviewed and the appropriate next steps determined."
        )

    if intent == "return_replacement":
        return (
            "I can help with your return or replacement request. "
            "Please provide the relevant order details so the next "
            "steps can be determined."
        )

    if intent == "refund_issue":
        return (
            "I'm sorry you're experiencing an issue with your refund. "
            "Please check the order and refund details so the issue "
            "can be investigated."
        )

    if intent == "payment_billing":
        return (
            "I'm sorry you're experiencing a billing issue. "
            "Please check the order and payment details associated "
            "with the transaction so the issue can be investigated."
        )

    if intent == "account_access":
        return (
            "I'm sorry you're having trouble accessing your account. "
            "Please provide the relevant account details so the issue "
            "can be reviewed."
        )

    if intent == "technical_product_issue":
        return (
            "I'm sorry you're having trouble with your product. "
            "Please share the product and the issue you're experiencing "
            "so the appropriate troubleshooting steps can be identified."
        )

    return (
        "I'm sorry you're experiencing this issue. "
        "Please provide the relevant order details so we can better "
        "understand the problem and determine the appropriate next steps."
    )


def is_valid_gemini_reply(text):
    """
    Reject empty, obviously truncated, or internally-revealing Gemini output.
    """

    if not text:
        return False

    text = text.strip()

    if len(text) < 20:
        return False

    # The model should return a customer-facing response only.
    forbidden_phrases = [
        "predicted intent",
        "classifier confidence",
        "historical examples",
        "retrieval system",
        "internal reasoning",
        "as an ai",
    ]

    lowered = text.lower()

    for phrase in forbidden_phrases:
        if phrase in lowered:
            return False

    # Reject common signs of an incomplete generation.
    if text.endswith(("I'm", "you're", "your", "the", "a", "an")):
        return False

    # If the response is extremely short and has no sentence-ending punctuation,
    # treat it as potentially truncated.
    if len(text) < 60 and text[-1] not in ".!?":
        return False

    return True


def _generate_with_gemini(prompt):
    """
    Make a single Gemini request.

    The SDK retry count is already set to 1, and the HTTP timeout prevents
    an unavailable request from hanging indefinitely.
    """

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=250,
        ),
    )

    text = getattr(response, "text", None)

    if not is_valid_gemini_reply(text):
        return None

    return text.strip()


def generate_reply_with_metadata(
    customer_message,
    intent,
    confidence,
    historical_matches,
    escalate,
    escalation_reason,
):
    """
    Generate a support reply and return metadata describing its source.

    Returns:
        {
            "reply": str,
            "source": "gemini" | "fallback",
            "gemini_attempts": int
        }
    """

    evidence = build_evidence(historical_matches)

    escalation_text = (
        f"YES - escalate to a human. Reason: {escalation_reason}"
        if escalate
        else "NO - the case can be auto-handled."
    )

    prompt = f"""
You are an AI customer-support assistant for Amazon.

Write a short, professional reply to the customer.

Customer message:
{customer_message}

Predicted intent:
{intent}

Classifier confidence:
{confidence:.4f}

Escalation decision:
{escalation_text}

Historical Amazon support examples:
{evidence}

Rules:
- Use the historical examples as grounding.
- Do not invent refunds, delivery dates, policies, order details,
  tracking numbers, or actions that are not supported by the evidence.
- If the case is being escalated, do not pretend that the issue
  has already been resolved.
- If historical evidence suggests asking for information such as
  tracking status or carrier information, it is acceptable to ask
  the customer for that information.
- Keep the response concise and customer-friendly.
- Do not mention that you are an AI.
- Do not mention the classifier, confidence score, retrieval system,
  historical examples, or internal reasoning.
- Return only the customer-facing reply.
"""

    # Deterministic evaluation mode.
    if not USE_GEMINI:
        return {
            "reply": fallback_reply(
                customer_message=customer_message,
                intent=intent,
                historical_matches=historical_matches,
                escalate=escalate,
                escalation_reason=escalation_reason,
            ),
            "source": "fallback",
            "gemini_attempts": 0,
        }

    # Real Gemini mode.
    # We deliberately use only a small number of attempts.
    max_attempts = 2

    for attempt in range(1, max_attempts + 1):
        try:
            print(
                f"Gemini request "
                f"(attempt {attempt}/{max_attempts})..."
            )

            text = _generate_with_gemini(prompt)

            if text:
                return {
                    "reply": text,
                    "source": "gemini",
                    "gemini_attempts": attempt,
                }

            print("Gemini returned unusable output.")

        except Exception as error:
            error_text = str(error)

            print(
                f"Gemini request failed "
                f"(attempt {attempt}/{max_attempts}): {error_text}"
            )

            # Quota/rate-limit errors should not be retried.
            if (
                "429" in error_text
                or "RESOURCE_EXHAUSTED" in error_text
                or "quota" in error_text.lower()
                or "rate limit" in error_text.lower()
            ):
                print("Quota/rate limit detected. Stopping Gemini attempts.")
                break

            # Temporary service errors get one controlled retry.
            if (
                "503" in error_text
                or "UNAVAILABLE" in error_text
                or "temporarily unavailable" in error_text.lower()
            ):
                if attempt < max_attempts:
                    print("Retrying Gemini once...")
                    time.sleep(1)
                    continue

            # Other errors are not worth repeatedly retrying.
            break

    print("Gemini unavailable. Using deterministic fallback reply.")

    return {
        "reply": fallback_reply(
            customer_message=customer_message,
            intent=intent,
            historical_matches=historical_matches,
            escalate=escalate,
            escalation_reason=escalation_reason,
        ),
        "source": "fallback",
        "gemini_attempts": max_attempts,
    }


def generate_reply(
    customer_message,
    intent,
    confidence,
    historical_matches,
    escalate,
    escalation_reason,
):
    """
    Backward-compatible wrapper.

    Existing pipeline code can continue receiving a plain string.
    """

    result = generate_reply_with_metadata(
        customer_message=customer_message,
        intent=intent,
        confidence=confidence,
        historical_matches=historical_matches,
        escalate=escalate,
        escalation_reason=escalation_reason,
    )

    return result["reply"]