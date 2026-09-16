\# Hiver SDE Intern — AI Customer Support Agent

\## Evaluation Report



\### 1. Problem Framing



The objective is to build an AI customer-support agent that can classify incoming customer messages, use historical support resolutions as evidence, draft a response, and decide whether the interaction should be automatically handled or escalated to a human.



The Customer Support on Twitter (TWCS) dataset was used, with experiments focused on AmazonHelp conversations.



The system separates the problem into four components:



1\. Intent classification

2\. Historical retrieval

3\. Response generation

4\. Escalation decision



A 200-example manually labelled golden set was created for evaluation.



\---



\## 2. Intent Taxonomy



The final taxonomy contains 11 intents:



\- delivery\_delay

\- order\_status\_tracking

\- missing\_package

\- wrong\_item

\- damaged\_item

\- return\_replacement

\- refund\_issue

\- payment\_billing

\- account\_access

\- technical\_product\_issue

\- other\_general



The taxonomy was derived from inspection of AmazonHelp customer messages.



Particular care was taken to distinguish:



\- delivery delays from tracking/status questions

\- missing packages from ordinary delays

\- Prime membership as context rather than a standalone intent



\---



\## 3. Model Comparison



Three approaches were evaluated on the same stratified evaluation split.



| Model | Accuracy | Macro-F1 | Weighted-F1 |

|---|---:|---:|---:|

| Majority baseline | 0.300 | 0.042 | 0.139 |

| TF-IDF + Logistic Regression | 0.360 | 0.168 | 0.325 |

| Hybrid Rule + ML | 0.500 | 0.377 | 0.463 |



The hybrid classifier combines high-signal rules with TF-IDF-based machine learning.



The results show improvement over both the majority baseline and the pure TF-IDF classifier on this evaluation split.



\---



\## 4. End-to-End Results



A separate held-out evaluation used 50 examples.



\### Intent



\- Accuracy: 0.540

\- Macro-F1: 0.367

\- Weighted-F1: 0.493



\### Escalation



\- Accuracy: 0.460

\- Precision: 0.586

\- Recall: 0.531

\- F1: 0.557



The escalation classifier still produces substantial false positives and false negatives.



False negatives are particularly important when a customer describes a serious delivery or support problem without using an explicit escalation phrase.



\---



\## 5. Response Evaluation



Three response strategies were compared:



1\. Generic baseline

2\. Retrieval-only response

3\. Agent response



The evaluation used 50 held-out messages per candidate.



The LLM judge scored helpfulness, intent fit, grounding, factual safety, escalation appropriateness, professionalism and overall pass/fail.



| Candidate | Pass Rate | Helpfulness | Intent Fit | Grounding | Factual Safety | Escalation Appropriateness | Professionalism |

|---|---:|---:|---:|---:|---:|---:|---:|

| Agent | 72% | 3.08 | 3.62 | 3.84 | 4.48 | 4.50 | 4.68 |

| Generic | 76% | 3.18 | 3.74 | 3.92 | 4.66 | 4.14 | 4.72 |

| Retrieval | 26% | 1.94 | 2.04 | 2.78 | 3.92 | 2.84 | 3.48 |



Scores are on a 1–5 scale except pass rate.



The agent substantially outperformed retrieval-only responses on the judge's overall pass rate.



The generic baseline nevertheless scored slightly higher on overall pass rate.



This means the experiment does not establish that the agent is universally better than the generic baseline.



The agent's escalation appropriateness score was 4.50/5, compared with 4.14/5 for the generic baseline.



\---



\## 6. What Is Misleading About My Headline Number?



A headline such as:



> "The agent achieved a 72% LLM-judge pass rate."



is useful but incomplete.



The number is based on only 50 held-out customer messages per candidate and is produced by an LLM judge rather than real customer satisfaction measurements.



More importantly, Gemini generation was unavailable during the evaluated run because of API quota limitations. The agent therefore used its deterministic fallback response generator.



The fallback generator receives the retrieval context through the pipeline, but its deterministic responses do not directly incorporate the retrieved historical reply into the final text.



Therefore, the 72% result should be interpreted as the quality of the current evaluated pipeline/fallback configuration, not as evidence that a fully LLM-generated, evidence-grounded production agent achieves 72% quality.



The human audit was intentionally small, so the LLM judge should not be treated as ground truth. I created a 30-row human audit across the three response strategies. Because a few examples were revisited during the interactive review, the latest explicit human rating was used for each unique `(tweet_id, candidate)` pair. This produced 22 uniquely mapped human-rated rows; the remaining 8 rows were left unrated rather than inferred.

On these 22 rows, the LLM judge and human reviewer agreed on 12/22 overall-pass decisions (54.5%), with Cohen's kappa of 0.052. Humans marked 16/22 responses as passes, while the LLM judge marked 12/22 as passes. The low agreement reinforces that the LLM judge is a diagnostic signal rather than ground truth.

The golden set itself is manually constructed and has an uneven intent distribution.



\---



\## 7. Main Failure Modes



\### Failure Mode 1 — Irrelevant Retrieval



TF-IDF can retrieve a historically similar message that belongs to a different support situation.



For example, delivery-related complaints can retrieve payment or unrelated support responses.



\*\*Hypothesis:\*\* lexical similarity is insufficient for distinguishing closely related support situations.



\*\*Improvement:\*\* use dense semantic retrieval followed by intent-aware reranking.



\---



\### Failure Mode 2 — Historical Context Leakage



Historical replies can contain customer-specific information that is not applicable to the current customer.



Observed examples include incorrect customer names and unrelated historical context.



\*\*Hypothesis:\*\* raw historical replies are being treated as reusable evidence instead of separating reusable resolution information from case-specific details.



\*\*Improvement:\*\* extract reusable resolution steps and exclude historical identities and case-specific details.



\---



\### Failure Mode 3 — Over-Generic Fallback Responses



The deterministic fallback sometimes gives a generic order-information request even when the customer describes a specific problem.



This is particularly visible for accessibility, delivery and product-specific complaints.



\*\*Hypothesis:\*\* intent-level templates do not contain enough customer-specific context.



\*\*Improvement:\*\* condition response generation on intent plus extracted entities, severity and customer request.



\---



\### Failure Mode 4 — Intent Confusion



Delivery delay, tracking, missing package, damaged delivery and general complaints can overlap.



Short social-media messages make the distinction harder.



\*\*Hypothesis:\*\* several intents have relatively few labelled examples and some messages are noisy or multilingual.



\*\*Improvement:\*\* add contrastive examples for confusing intent pairs and increase minority-intent coverage.



\---



\### Failure Mode 5 — Language Mismatch



Some evaluation messages are not English, while the current response generator can produce English responses.



\*\*Hypothesis:\*\* the current pipeline has no explicit language-detection and language-preservation layer.



\*\*Improvement:\*\* detect customer language before retrieval and generation, then require the response to use the customer's language.



\---



\## 8. Additional Observations



Other observed failure patterns include:



\- unsupported claims about what the support team has already done

\- missing actionable escalation paths

\- failure to recognize already-resolved conversations

\- privacy-sensitive requests being handled with generic information requests

\- ambiguous link-only messages being interpreted too confidently

\- inappropriate reuse of historical customer names or handles



These are important because a response can be linguistically fluent while still being operationally unsafe or irrelevant.



\---



\## 9. One-Week Improvement Plan



\### Days 1–2 — Retrieval



Replace TF-IDF retrieval with dense embeddings.



Add intent-aware retrieval filtering and reranking.



\### Day 3 — Evidence Extraction



Transform historical replies into structured evidence:



\- issue

\- action

\- eligibility

\- caveat

\- escalation condition



\### Day 4 — Grounded Generation



Require every generated recommendation to be supported by retrieved evidence.



Reject unsupported claims.



\### Day 5 — Multilingual Support



Add language detection and language-preserving response generation.



\### Day 6 — Escalation



Develop a dedicated escalation classifier using intent, severity and explicit human-support requests.



\### Day 7 — Evaluation



Expand the golden set, add more minority-intent examples, repeat the human audit and measure human/LLM-judge agreement.



\---



\## 10. Conclusion



The experiments show that combining high-signal rules with a statistical intent classifier improves intent classification over simple baselines.



The response experiment also shows that naive retrieval-only responses can perform poorly when lexical similarity retrieves an incorrect historical resolution.



The current agent improves substantially over retrieval-only responses, while the generic baseline remains competitive.



The main engineering opportunity is therefore not simply adding a larger language model. It is improving evidence selection, preventing historical-context leakage, making responses context-aware, preserving customer language and strengthening escalation safety.



The current results are best viewed as a reproducible prototype evaluation with clearly identified failure modes and a concrete path toward a production-quality support agent.

