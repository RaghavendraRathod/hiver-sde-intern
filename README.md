@"

\# Hiver SDE Intern Take-Home — AI Customer Support Agent



\## Overview



This project implements an AI-assisted customer-support agent using the Customer Support on Twitter (TWCS) dataset.



The pipeline:



1\. Classifies a customer message into a support intent.

2\. Retrieves historically similar customer-support interactions.

3\. Drafts a customer-facing response.

4\. Decides whether the interaction should be auto-handled or escalated.

5\. Evaluates the system against a manually labelled golden set and multiple baselines.



The experiments focus on AmazonHelp support conversations.



\---



\## Project Structure



```text

hiver-sde-intern/

├── data/

│   ├── amazon\_customer\_sample.csv

│   ├── golden\_set.csv

│   ├── retrieval\_corpus.csv

│   ├── sample.csv

│   └── twcs/

│       └── twcs.csv              # local only; ignored by Git

├── results/

│   ├── model\_comparison.csv

│   ├── intent\_metrics.csv

│   ├── intent\_confusion\_matrix.csv

│   ├── end\_to\_end\_results.csv

│   ├── end\_to\_end\_failures.csv

│   ├── reply\_evaluation.csv

│   ├── reply\_metrics.csv

│   ├── llm\_judge\_results.csv

│   ├── llm\_judge\_scored\_v2.csv

│   ├── human\_audit.csv

│   └── metrics.json

├── src/

│   ├── classifier.py

│   ├── retrieval.py

│   ├── escalation.py

│   ├── pipeline.py

│   ├── agent/

│   └── evaluation/

├── test\_escalation.py

└── requirements.txt

