\# Decision Log



\## 1. Focused the first implementation on AmazonHelp

The full TWCS dataset contains multiple brands and millions of tweets. I selected AmazonHelp first so that the resolution language and support patterns are more internally consistent and the experiment remains tractable.



\## 2. Used an 11-intent taxonomy

I chose 11 intents to balance coverage and learnability. The taxonomy is detailed enough to distinguish operationally different support problems without creating too many extremely sparse classes.



\## 3. Treated Prime as context rather than an intent

Prime appears frequently in customer messages, but it can occur across delivery, billing, account, and other problems. Therefore, Prime is treated as contextual information rather than a standalone intent.



\## 4. Explicitly separated delivery delay, tracking, and missing package

These three cases can look similar lexically but require different responses. A delayed shipment is different from a request for current tracking information, and both differ from an order marked delivered but not received.



\## 5. Created a 200-example golden set

A manually labeled golden set provides a fixed evaluation target and makes model comparison possible without requiring manual labeling of the entire dataset.



\## 6. Used a fixed stratified split with random seed 42

The same split is used for model comparisons so that differences in accuracy and F1 are attributable to the models rather than different evaluation samples.



\## 7. Used a hybrid rule-plus-ML classifier

Several support intents have strong lexical indicators, such as unauthorized charges, damaged items, wrong items, and missing packages. High-confidence rules capture these cases while TF-IDF + logistic regression handles less explicit messages.



\## 8. Used class-balanced logistic regression

The golden set is imbalanced, with some intents occurring much more frequently than others. `class\_weight="balanced"` was therefore used to reduce the tendency to predict only common classes.



\## 9. Used TF-IDF retrieval before considering embeddings

TF-IDF retrieval is inexpensive, transparent, deterministic, and easy to inspect. It also provides a clear baseline for determining whether semantic retrieval is actually needed.



\## 10. Limited the initial retrieval corpus to 300 historical examples

The goal was to establish an end-to-end historical-resolution retrieval baseline quickly. This also makes retrieval errors easy to inspect before scaling to a much larger corpus.



\## 11. Put explicit human/support requests first in escalation logic

If a customer explicitly asks to speak with a person or support team, the system should not allow a weaker downstream intent rule to suppress that signal.



\## 12. Added a low-confidence escalation rule

Low-confidence predictions can indicate that the classifier does not understand the request. Escalating these cases is safer than presenting an uncertain automated response as definitive.



\## 13. Added a deterministic fallback for unavailable LLM generation

Gemini generation was affected by API quota/rate-limit availability during evaluation. A deterministic fallback keeps the pipeline runnable and makes the limitation explicit rather than hiding failed generations.



\## 14. Evaluated three response strategies

The evaluation compares a generic response baseline, historical retrieval-only responses, and the complete agent pipeline. This separates the value of retrieval from the value of the full decision pipeline.



## 15. Used an LLM judge plus a human audit
The LLM judge evaluates helpfulness, intent fit, grounding, factual safety, escalation appropriateness, professionalism, and overall pass, with gold labels hidden to reduce evaluation leakage. A 30-row human audit was also created from LLM-passed and LLM-failed examples to provide an independent quality check because LLM-based evaluation can disagree with human judgment.

