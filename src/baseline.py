import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
)
from sklearn.model_selection import train_test_split


DATA_PATH = "data/golden_set.csv"


def evaluate_model(name, y_true, y_pred):
    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    print(f"Accuracy:    {accuracy:.4f}")
    print(f"Macro F1:    {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")

    print("\nPer-intent results:")
    print(
        classification_report(
            y_true,
            y_pred,
            zero_division=0
        )
    )


# ---------------------------------------------------------
# Load golden set
# ---------------------------------------------------------

df = pd.read_csv(DATA_PATH)

X = df["customer_text"].fillna("")
y = df["intent"]


print(f"Loaded {len(df)} golden examples.")
print(f"Number of intents: {y.nunique()}")


# ---------------------------------------------------------
# Baseline 1: Majority Class
# ---------------------------------------------------------

majority_class = y.value_counts().idxmax()

print("\nMost common intent:")
print(majority_class)

majority_predictions = [majority_class] * len(y)

evaluate_model(
    "BASELINE 1 — MAJORITY CLASS",
    y,
    majority_predictions
)


# ---------------------------------------------------------
# Baseline 2: TF-IDF + Logistic Regression
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

print("\nTF-IDF train/test split:")
print(f"Training examples: {len(X_train)}")
print(f"Test examples:     {len(X_test)}")


vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=1,
    max_features=10000
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced"
)

model.fit(X_train_tfidf, y_train)

predictions = model.predict(X_test_tfidf)


evaluate_model(
    "BASELINE 2 — TF-IDF + LOGISTIC REGRESSION",
    y_test,
    predictions
)