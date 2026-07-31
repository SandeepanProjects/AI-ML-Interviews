# Ragas Explained Properly (Production Guide with Code)

**Ragas (Retrieval-Augmented Generation Assessment)** is one of the most popular frameworks for evaluating **RAG (Retrieval-Augmented Generation)** applications.

Instead of manually reading hundreds of answers, Ragas automatically measures:

* Retrieval quality
* Answer quality
* Hallucinations
* Context usage

It helps answer questions like:

* Did the retriever fetch the right documents?
* Did the LLM use the retrieved documents correctly?
* Did the LLM hallucinate?
* Is the answer correct?

---

# Why Do We Need Ragas?

Imagine you build an enterprise HR chatbot.

```
User:
How many maternity leave days do employees receive?
```

Retriever returns:

```
Document:
Employees receive 180 days of maternity leave.
```

LLM answers:

```
Employees receive 90 days.
```

Without evaluation, you only know the answer is wrong.

Ragas tells you **why** it is wrong.

Was it because:

* Retrieval failed?
* The LLM ignored the context?
* The LLM hallucinated?
* The answer was incomplete?

---

# Where Ragas Fits

```text
                    User Question
                          │
                          ▼
                   Retriever
                          │
                          ▼
                 Retrieved Context
                          │
                          ▼
                        LLM
                          │
                          ▼
                  Generated Answer
                          │
                          ▼
                     Ragas
                          │
      ┌───────────────────┼────────────────────┐
      ▼                   ▼                    ▼
 Context Metrics     Answer Metrics     Hallucination Metrics
```

Ragas is **not part of inference**.

It is an **evaluation framework** that runs after inference.

---

# What Does Ragas Evaluate?

The most commonly used metrics are:

| Metric             | Measures                                | Stage      |
| ------------------ | --------------------------------------- | ---------- |
| Context Precision  | Is retrieved context useful?            | Retrieval  |
| Context Recall     | Was all required information retrieved? | Retrieval  |
| Faithfulness       | Is the answer supported by context?     | Generation |
| Answer Correctness | Is the answer correct?                  | Generation |
| Answer Relevancy   | Does the answer address the question?   | Generation |

Think of them like this:

```
Retriever
     │
     ├── Context Precision
     └── Context Recall

LLM
     │
     ├── Faithfulness
     ├── Answer Correctness
     └── Answer Relevancy
```

---

# Example Dataset

Suppose your HR chatbot receives:

Question

```
How many maternity leave days?
```

Retrieved Context

```
Employees receive 180 days.

Maternity leave applies after one year.
```

Generated Answer

```
Employees receive 180 days.
```

Ground Truth

```
Employees receive 180 days.
```

This single example is enough for Ragas to evaluate.

---

# Installing Ragas

```bash
pip install ragas
```

Also install:

```bash
pip install datasets
```

---

# Creating the Dataset

Ragas expects a dataset containing:

* question
* contexts
* answer
* ground_truth

```python
from datasets import Dataset

dataset = Dataset.from_dict({

    "question":[
        "How many maternity leave days?"
    ],

    "contexts":[[
        "Employees receive 180 days.",
        "Leave applies after one year."
    ]],

    "answer":[
        "Employees receive 180 days."
    ],

    "ground_truth":[
        "Employees receive 180 days."
    ]

})
```

Notice:

contexts is **a list of retrieved chunks**, not one large string.

---

# Running Evaluation

```python
from ragas import evaluate

from ragas.metrics import (

    faithfulness,

    context_precision,

    context_recall,

    answer_correctness

)

result = evaluate(

    dataset,

    metrics=[

        faithfulness,

        context_precision,

        context_recall,

        answer_correctness

    ]

)

print(result)
```

Example output:

```
Faithfulness         0.98

Context Precision    0.90

Context Recall       0.93

Answer Correctness   0.95
```

---

# Understanding Every Metric

## 1. Context Precision

Question:

```
How useful were the retrieved chunks?
```

Retriever:

```
Leave Policy
VPN Guide
Office Rules
```

Only Leave Policy helps.

```
Precision

1 / 3

=

0.33
```

---

## 2. Context Recall

Question:

```
Did retrieval include everything needed?
```

Required:

```
Leave Policy

HR FAQ
```

Retrieved:

```
Leave Policy
```

```
Recall

1 / 2

=

0.5
```

---

## 3. Faithfulness

Question:

```
Did the LLM invent facts?
```

Context

```
Employees receive 180 days.
```

Answer

```
Employees receive 180 days
plus 30 bonus days.
```

"30 bonus days" is unsupported.

Faithfulness decreases.

---

## 4. Answer Correctness

Ground Truth

```
180 days
```

Answer

```
90 days
```

Incorrect.

Score becomes low.

---

# Real Production Workflow

```text
User Question
      │
      ▼
Retriever
      │
      ▼
Retrieved Context
      │
      ▼
LLM
      │
      ▼
Answer
      │
      ▼
Store Evaluation Record
      │
      ▼
Run Ragas
      │
      ▼
Metrics Dashboard
```

Notice:

Ragas usually runs:

* Offline evaluation
* Nightly benchmark jobs
* CI/CD validation
* A/B testing

It typically does **not** run for every production request because it adds cost and latency.

---

# Evaluating Many Questions

```python
questions = [

    "Leave policy",

    "VPN reset",

    "Insurance",

    "Travel reimbursement"

]
```

Generate answers.

```python
records = []

for q in questions:

    docs = retriever.invoke(q)

    answer = rag_chain.invoke(q)

    records.append({

        "question": q,

        "contexts":[

            d.page_content

            for d in docs

        ],

        "answer": answer,

        "ground_truth": gold_answers[q]

    })
```

Create dataset.

```python
dataset = Dataset.from_list(records)
```

Evaluate.

```python
results = evaluate(
    dataset,
    metrics=[
        faithfulness,
        answer_correctness
    ]
)
```

---

# Production Architecture

```text
              User Questions

                    │

                    ▼

            RAG Pipeline

                    ▼

       Generated Answers

                    ▼

           Gold Answers

                    ▼

             Ragas Evaluation

      ┌────────────┼────────────┐

      ▼            ▼            ▼

 Retrieval     Generation     Summary

 Metrics         Metrics      Dashboard
```

---

# CI/CD Integration

Every deployment can automatically evaluate the RAG system.

```text
Git Push

↓

GitHub Actions

↓

Build

↓

Run Ragas

↓

Faithfulness

↓

Context Recall

↓

Answer Correctness

↓

Pass/Fail Deployment
```

Example policy:

```python
if results["faithfulness"] < 0.9:
    raise Exception("Deployment Failed")
```

---

# Monitoring Over Time

Suppose:

Week 1

```
Faithfulness

0.97
```

Week 4

```
Faithfulness

0.74
```

Possible reasons:

* New documents added
* Poor chunking
* Broken embeddings
* Retrieval regression
* Prompt changes

Ragas helps detect these regressions before users notice.

---

# Common Mistakes

### 1. Running Ragas on one example

Wrong.

Evaluate hundreds or thousands of questions.

---

### 2. Ignoring retrieval metrics

Many engineers only look at answer correctness.

Always evaluate:

* Retrieval
* Generation

Separately.

---

### 3. Using synthetic questions only

Use:

* Real production queries
* User feedback
* Human-labeled datasets

---

### 4. No ground truth

Answer Correctness requires a reference answer.

Without one, use metrics such as:

* Faithfulness
* Context Precision
* Context Recall

---

# Best Practices

* Build a gold evaluation dataset from real user queries.
* Run Ragas automatically in CI/CD before deploying changes.
* Track trends over time rather than relying on one score.
* Combine Ragas metrics with latency, token usage, and cost.
* Investigate retrieval metrics first when answers degrade.

---

# Enterprise Evaluation Architecture

```text
                      Production Logs
                             │
                             ▼
                  Evaluation Dataset
                             │
                             ▼
                     RAG Pipeline
                             │
                             ▼
                 Retrieved Documents
                             │
                             ▼
                    Generated Answers
                             │
                             ▼
                         Ragas
     ┌─────────────────────────────────────────┐
     │ Context Precision                       │
     │ Context Recall                          │
     │ Faithfulness                            │
     │ Answer Correctness                      │
     │ Answer Relevancy                        │
     └─────────────────────────────────────────┘
                             │
                             ▼
                Prometheus / Grafana Dashboard
```

---

# Ragas vs Traditional ML Evaluation

| Traditional ML    | RAG Evaluation with Ragas                                     |
| ----------------- | ------------------------------------------------------------- |
| Accuracy          | Answer Correctness                                            |
| Precision         | Context Precision                                             |
| Recall            | Context Recall                                                |
| F1 Score          | Combination of retrieval and generation metrics               |
| Confusion Matrix  | Less applicable; evaluate retrieval and generation separately |
| Manual evaluation | Automated LLM-assisted evaluation                             |

---

# Senior AI Engineer Interview Answer

> **Ragas is an evaluation framework for Retrieval-Augmented Generation (RAG) systems. Unlike traditional evaluation, it measures both retrieval quality and generation quality. For retrieval, I monitor Context Precision and Context Recall to determine whether the retriever returns relevant and complete evidence. For generation, I evaluate Faithfulness to detect hallucinations, Answer Correctness to compare against reference answers, and Answer Relevancy to ensure the response addresses the user's question. In production, I build a gold dataset of real user queries with reference answers, run Ragas in offline evaluation jobs or CI/CD pipelines, and track these metrics over time. If Context Recall drops, I investigate chunking, embeddings, retrieval depth, and hybrid search. If Faithfulness drops while retrieval remains strong, I inspect prompts and model behavior. This systematic approach allows me to identify whether regressions originate from retrieval or generation and maintain high-quality enterprise RAG systems.
