import re

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


class IntentClassifier:
    """
    Hybrid intent classifier for Amazon customer-support messages.

    Strategy:
    1. High-precision rules handle obvious intent phrases.
    2. TF-IDF + Logistic Regression handles less explicit messages.
    3. Character n-grams help with noisy/social-media-style text.
    """

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
        "other_general",
    ]

    def __init__(self):
        self.word_vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            max_features=15000,
            sublinear_tf=True,
        )

        self.char_vectorizer = TfidfVectorizer(
            lowercase=True,
            analyzer="char_wb",
            ngram_range=(3, 5),
            max_features=15000,
            sublinear_tf=True,
        )

        self.model = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            C=2.0,
            random_state=42,
        )

        self.is_trained = False

    def _clean(self, text):
        """Basic text normalization."""
        text = str(text).lower()
        text = re.sub(r"http\S+|www\S+", " ", text)
        text = re.sub(r"@\w+", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _rule_based_intent(self, text):
        """
        High-precision rules for very distinctive customer-support intents.

        Returns:
            (intent, confidence) or (None, None)
        """

        text = self._clean(text)

        # ---------------------------------------------------------
        # 1. MISSING PACKAGE
        # ---------------------------------------------------------
        missing_patterns = [
            r"\bmarked as delivered\b",
            r"\bsays delivered\b",
            r"\bshows delivered\b",
            r"\bdelivered but\b",
            r"\bdelivered and i\b",
            r"\bdelivered but i\b",
            r"\bnever received\b",
            r"\bnot received\b",
            r"\bdid not receive\b",
            r"\bdidn't receive\b",
            r"\bhaven't received\b",
            r"\bnot arrived\b",
            r"\bnever arrived\b",
        ]

        if any(re.search(pattern, text) for pattern in missing_patterns):
            return "missing_package", 0.95

        # ---------------------------------------------------------
        # 2. WRONG ITEM
        # ---------------------------------------------------------
        wrong_item_patterns = [
            r"\bwrong item\b",
            r"\bwrong product\b",
            r"\bwrong order\b",
            r"\bdifferent item\b",
            r"\bsent me the wrong\b",
            r"\breceived the wrong\b",
            r"\bgot the wrong\b",
            r"\breceived .* instead of\b",
        ]

        if any(re.search(pattern, text) for pattern in wrong_item_patterns):
            return "wrong_item", 0.95

        # ---------------------------------------------------------
        # 3. DAMAGED ITEM
        # ---------------------------------------------------------
        damaged_patterns = [
            r"\bdamaged\b",
            r"\bbroken\b",
            r"\bcracked\b",
            r"\bdefective\b",
            r"\barrived damaged\b",
            r"\barrived broken\b",
        ]

        if any(re.search(pattern, text) for pattern in damaged_patterns):
            return "damaged_item", 0.94

        # ---------------------------------------------------------
        # 4. UNAUTHORIZED / BILLING
        # ---------------------------------------------------------
        unauthorized_patterns = [
            r"\bunauthorized charge\b",
            r"\bunauthorised charge\b",
            r"\bcharged without authorization\b",
            r"\bcharged without authorisation\b",
            r"\bcharged without my authorization\b",
            r"\bcharged without my authorisation\b",
            r"\bcharged .* without\b",
            r"\bcharged me .* without\b",
            r"\bdidn't authorize\b",
            r"\bdid not authorize\b",
            r"\bunknown charge\b",
            r"\bcharge i don't recognize\b",
            r"\bcharge i do not recognize\b",
        ]

        if any(re.search(pattern, text) for pattern in unauthorized_patterns):
            return "payment_billing", 0.97

        # ---------------------------------------------------------
        # 5. REFUND
        # ---------------------------------------------------------
        refund_patterns = [
            r"\brefund\b",
            r"\brefunded\b",
            r"\brefund pending\b",
            r"\brefund hasn't\b",
            r"\brefund has not\b",
            r"\bmoney back\b",
        ]

        if any(re.search(pattern, text) for pattern in refund_patterns):
            return "refund_issue", 0.90

        # ---------------------------------------------------------
        # 6. RETURN / REPLACEMENT
        # ---------------------------------------------------------
        return_patterns = [
            r"\breturn\b",
            r"\breturning\b",
            r"\breplacement\b",
            r"\breplace\b",
            r"\bexchange\b",
        ]

        if any(re.search(pattern, text) for pattern in return_patterns):
            return "return_replacement", 0.88

        # ---------------------------------------------------------
        # 7. ACCOUNT ACCESS
        # ---------------------------------------------------------
        account_patterns = [
            r"\bcan't log in\b",
            r"\bcant log in\b",
            r"\bcannot log in\b",
            r"\bcan't login\b",
            r"\bcant login\b",
            r"\bcannot login\b",
            r"\blogin problem\b",
            r"\blogin issue\b",
            r"\bsign in\b",
            r"\bsign-in\b",
            r"\bpassword\b",
            r"\baccount access\b",
            r"\baccount locked\b",
        ]

        if any(re.search(pattern, text) for pattern in account_patterns):
            return "account_access", 0.91

        # ---------------------------------------------------------
        # 8. ORDER STATUS / TRACKING
        # ---------------------------------------------------------
        tracking_patterns = [
            r"\btracking\b",
            r"\btrack my order\b",
            r"\btrack my package\b",
            r"\bwhere is my order\b",
            r"\bwhere's my order\b",
            r"\bwhere is my package\b",
            r"\bwhere's my package\b",
            r"\border status\b",
            r"\bshipment status\b",
            r"\bdelivery status\b",
            r"\bcheck my order\b",
            r"\bcheck the status\b",
        ]

        if any(re.search(pattern, text) for pattern in tracking_patterns):
            return "order_status_tracking", 0.92

        # ---------------------------------------------------------
        # 9. TECHNICAL PRODUCT ISSUE
        # ---------------------------------------------------------
        technical_patterns = [
            r"\bnot working\b",
            r"\bdoesn't work\b",
            r"\bdoesnt work\b",
            r"\bwon't work\b",
            r"\bwont work\b",
            r"\bfreezing\b",
            r"\bfreezes\b",
            r"\brestarting\b",
            r"\brestarts\b",
            r"\berror\b",
            r"\bcrashing\b",
            r"\bcrashes\b",
            r"\bfire tv\b",
            r"\bdevice problem\b",
            r"\bdevice issue\b",
            r"\bapp problem\b",
            r"\bapp issue\b",
        ]

        if any(re.search(pattern, text) for pattern in technical_patterns):
            return "technical_product_issue", 0.88

        # ---------------------------------------------------------
        # 10. DELIVERY DELAY
        # ---------------------------------------------------------
        delivery_patterns = [
            r"\blate\b",
            r"\bdelayed\b",
            r"\bdelay\b",
            r"\bhasn't arrived\b",
            r"\bhas not arrived\b",
            r"\bwon't arrive\b",
            r"\bwont arrive\b",
            r"\bnot arriving\b",
            r"\boverdue\b",
            r"\bmissed delivery\b",
            r"\bsupposed to arrive\b",
            r"\bexpected today\b",
            r"\bexpected yesterday\b",
        ]

        if any(re.search(pattern, text) for pattern in delivery_patterns):
            return "delivery_delay", 0.86

        return None, None

    def train(self, texts, labels):
        """Train the machine-learning fallback classifier."""

        texts = pd.Series(texts).fillna("").map(self._clean)
        labels = pd.Series(labels).fillna("other_general")

        word_features = self.word_vectorizer.fit_transform(texts)
        char_features = self.char_vectorizer.fit_transform(texts)

        # Combine word and character features.
        from scipy.sparse import hstack

        features = hstack([word_features, char_features])

        self.model.fit(features, labels)

        self.is_trained = True

    def predict(self, text):
        """
        Predict an intent.

        Rules are checked first because they are more reliable for
        highly distinctive support phrases.
        """

        # Try high-precision rules first.
        rule_intent, rule_confidence = self._rule_based_intent(text)

        if rule_intent is not None:
            return {
                "intent": rule_intent,
                "confidence": rule_confidence,
                "method": "rule",
            }

        # Fall back to machine learning.
        if not self.is_trained:
            raise RuntimeError(
                "Classifier has not been trained. Call train() first."
            )

        clean_text = self._clean(text)

        word_features = self.word_vectorizer.transform([clean_text])
        char_features = self.char_vectorizer.transform([clean_text])

        from scipy.sparse import hstack

        features = hstack([word_features, char_features])

        probabilities = self.model.predict_proba(features)[0]

        best_index = probabilities.argmax()

        intent = self.model.classes_[best_index]
        confidence = float(probabilities[best_index])

        return {
            "intent": intent,
            "confidence": confidence,
            "method": "ml",
        }