import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from src.classifier import IntentClassifier


DATA_PATH = "data/golden_set.csv"


def print_results(name, y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )
    weighted_f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)
    print(f"Accuracy:    {accuracy:.4f}")
    print(f"Macro F1:    {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")

    return {
        "model": name,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }


# ---------------------------------------------------------
# Load golden set
# ---------------------------------------------------------

df = pd.read_csv(DATA_PATH)

X = df["customer_text"].fillna("")
y = df["intent"]


print("=" * 70)
print("FAIR MODEL COMPARISON")
print("=" * 70)
print(f"Total examples: {len(df)}")
print(f"Number of intents: {y.nunique()}")


# ---------------------------------------------------------
# ONE SHARED TRAIN/TEST SPLIT
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

print("\nShared train/test split:")
print(f"Training examples: {len(X_train)}")
print(f"Test examples:     {len(X_test)}")
print("Random state:      42")
print("Stratification:    intent")


results = []


# ---------------------------------------------------------
# BASELINE 1 — MAJORITY CLASS
# ---------------------------------------------------------

majority_class = y_train.value_counts().idxmax()

majority_predictions = [majority_class] * len(y_test)

results.append(
    print_results(
        "BASELINE 1 — MAJORITY CLASS",
        y_test,
        majority_predictions
    )
)


# ---------------------------------------------------------
# BASELINE 2 — TF-IDF + LOGISTIC REGRESSION
# ---------------------------------------------------------

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

tfidf_predictions = model.predict(X_test_tfidf)

results.append(
    print_results(
        "BASELINE 2 — TF-IDF + LOGISTIC REGRESSION",
        y_test,
        tfidf_predictions
    )
)


# ---------------------------------------------------------
# MODEL 3 — HYBRID RULE + ML CLASSIFIER
# ---------------------------------------------------------

hybrid = IntentClassifier()

hybrid.train(
    X_train.tolist(),
    y_train.tolist()
)

hybrid_predictions = []

for text in X_test:
    prediction = hybrid.predict(text)
    hybrid_predictions.append(prediction["intent"])


results.append(
    print_results(
        "MODEL 3 — HYBRID RULE + ML CLASSIFIER",
        y_test,
        hybrid_predictions
    )
)


# ---------------------------------------------------------
# COMPARISON TABLE
# ---------------------------------------------------------

comparison = pd.DataFrame(results)

comparison = comparison.sort_values(
    by="macro_f1",
    ascending=False
)

print("\n\n" + "=" * 70)
print("FINAL COMPARISON")
print("=" * 70)

print(
    comparison.to_string(
        index=False,
        formatters={
            "accuracy": "{:.4f}".format,
            "macro_f1": "{:.4f}".format,
            "weighted_f1": "{:.4f}".format,
        }
    )
)


# ---------------------------------------------------------
# SAVE RESULTS
# ---------------------------------------------------------

output_path = "results/model_comparison.csv"

comparison.to_csv(
    output_path,
    index=False
)

print(f"\nSaved comparison to: {output_path}")