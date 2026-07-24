These three metrics are among the **most important RAG evaluation metrics** because they measure the quality of the **entire RAG pipeline**, not just retrieval.

They answer different questions:

| Metric                | Question Answered                                                  |
| --------------------- | ------------------------------------------------------------------ |
| **Context Precision** | Did we retrieve mostly useful context?                             |
| **Context Recall**    | Did we retrieve all the information needed to answer the question? |
| **Faithfulness**      | Is the generated answer supported by the retrieved context?        |

These metrics are widely used in frameworks like **Ragas**, **DeepEval**, **TruLens**, and enterprise RAG evaluation pipelines.

---

# The Complete RAG Pipeline

```text
                  User Question
                        │
                        ▼
                 Hybrid Retriever
                        │
                        ▼
                 Retrieved Chunks
                        │
                        ▼
                    Reranker
                        │
                        ▼
                Final Context
                        │
                        ▼
                     LLM Answer
                        │
                        ▼
               Evaluation Metrics
        ┌────────┬──────────┬──────────┐
        ▼        ▼          ▼
 Context Precision Context Recall Faithfulness
```

Notice:

* Context Precision evaluates retrieval quality.
* Context Recall evaluates retrieval completeness.
* Faithfulness evaluates answer correctness.

---

# Example

Suppose our documents are

```text
Doc1
JWT Authentication uses access tokens.

Doc2
Refresh tokens generate new access tokens.

Doc3
Docker Compose manages containers.

Doc4
OAuth2 defines authorization.
```

User asks

```text
How does JWT authentication work?
```

Retriever returns

```text
Doc1 ✓

Doc2 ✓

Doc3 ✗
```

LLM answers

```text
JWT uses access tokens.
Refresh tokens can issue new access tokens.
```

Now let's evaluate.

---

# Context Precision

Question

> How much of the retrieved context was actually useful?

Suppose

Retrieved

```text
Doc1

Doc2

Doc3
```

Useful

```text
Doc1

Doc2
```

Irrelevant

```text
Doc3
```

Context Precision

```text
Useful Context
--------------

Retrieved Context

=

2 / 3

=

0.67
```

High precision means

```text
Almost everything retrieved
was useful.
```

Low precision means

```text
Retriever brought
lots of garbage.
```

---

## Code

```python
def context_precision(
    retrieved,
    useful
):

    retrieved = set(retrieved)
    useful = set(useful)

    return len(
        retrieved & useful
    ) / len(retrieved)
```

Example

```python
retrieved = [
    "Doc1",
    "Doc2",
    "Doc3"
]

useful = [
    "Doc1",
    "Doc2"
]

print(
    context_precision(
        retrieved,
        useful
    )
)
```

Output

```text
0.67
```

---

# Context Recall

Question

> Did we retrieve all information needed?

Suppose

Ground truth

```text
Doc1

Doc2

Doc4
```

Retriever

```text
Doc1

Doc2
```

Missing

```text
Doc4
```

Context Recall

```text
Retrieved Useful Context
------------------------

All Useful Context

=

2 / 3

=

0.67
```

---

## Code

```python
def context_recall(
    retrieved,
    relevant
):

    retrieved = set(retrieved)
    relevant = set(relevant)

    return len(
        retrieved &
        relevant
    ) / len(relevant)
```

Example

```python
retrieved = [
    "Doc1",
    "Doc2"
]

relevant = [
    "Doc1",
    "Doc2",
    "Doc4"
]

print(
    context_recall(
        retrieved,
        relevant
    )
)
```

Output

```text
0.67
```

---

# Difference

Imagine

Ground truth

```text
A

B

C

D
```

Retriever

```text
A

B

X

Y
```

Context Precision

```text
Useful

2

Retrieved

4

↓

0.5
```

Context Recall

```text
Retrieved

2

Needed

4

↓

0.5
```

Different questions.

Precision asks

```text
How clean?
```

Recall asks

```text
How complete?
```

---

# Faithfulness

Faithfulness is different.

It asks

> Is the answer supported by the retrieved context?

Suppose

Context

```text
JWT uses access tokens.

Refresh tokens renew access tokens.
```

LLM Answer

```text
JWT uses RSA encryption.

Refresh tokens expire every hour.
```

Problem

Neither sentence appears in context.

These are hallucinations.

Faithfulness

```text
0
```

---

Another example

Context

```text
JWT uses access tokens.

Refresh tokens renew access tokens.
```

Answer

```text
JWT uses access tokens.

Refresh tokens generate
new access tokens.
```

Everything is supported.

Faithfulness

```text
1
```

---

# How Faithfulness Works

Modern evaluation libraries usually break the generated answer into **atomic statements**.

Example

Answer

```text
JWT uses access tokens.

Refresh tokens renew access tokens.

OAuth2 is mandatory.
```

Statements

```text
Statement1

JWT uses access tokens

------------

Statement2

Refresh tokens renew access tokens

------------

Statement3

OAuth2 is mandatory
```

Each statement is checked against the retrieved context.

---

Suppose

```text
Statement1 ✓

Statement2 ✓

Statement3 ✗
```

Faithfulness

```text
Supported

2

Total

3

=

0.67
```

---

## Simple Code

```python
def faithfulness(
    supported,
    total
):

    return supported / total
```

Example

```python
supported = 2
total = 3

print(
    faithfulness(
        supported,
        total
    )
)
```

Output

```text
0.67
```

---

# How Libraries Compute Faithfulness

Libraries like **Ragas** and **DeepEval** use an LLM (or a Natural Language Inference model) to judge whether each statement is entailed by the retrieved context.

Simplified algorithm:

```python
for statement in answer:

    if statement_supported_by_context():
        score += 1

faithfulness = score / total_statements
```

In reality, the support check is done by an LLM prompt or an NLI model rather than string matching.

---

# Example Using Ragas

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    context_precision,
    context_recall
)

results = evaluate(
    dataset,
    metrics=[
        faithfulness,
        context_precision,
        context_recall
    ]
)

print(results)
```

Example output

```text
Faithfulness       0.94

Context Precision  0.88

Context Recall     0.91
```

---

# Real Production Example

Imagine a medical chatbot.

Question

```text
How is diabetes treated?
```

Retriever

```text
Doc1
Insulin therapy

Doc2
Diet changes

Doc3
Exercise

Doc4
Hospital parking rules
```

LLM Answer

```text
Diabetes is treated using
insulin,
diet,
exercise.
```

Evaluation

Context Precision

```text
Useful

3

Retrieved

4

↓

0.75
```

Context Recall

Suppose another relevant document about medication was missed.

```text
Retrieved useful

3

Needed

4

↓

0.75
```

Faithfulness

Every answer sentence is supported.

```text
1.0
```

Even though recall isn't perfect.

---

# Low Faithfulness Example

Context

```text
Insulin treats diabetes.
```

Answer

```text
Stem cell therapy is
the standard treatment.
```

Retriever was correct.

LLM hallucinated.

Scores

```text
Context Precision

1.0

Context Recall

1.0

Faithfulness

0.0
```

This demonstrates an important point:

> **A perfect retriever does not guarantee a faithful answer.**

The LLM can still invent information that is not present in the retrieved context.

---

# How These Metrics Work Together

```text
                 Question
                     │
                     ▼
               Retrieved Context
                     │
      ┌──────────────┴──────────────┐
      ▼                             ▼
Context Precision             Context Recall
      │                             │
      └──────────────┬──────────────┘
                     ▼
              LLM Generates Answer
                     │
                     ▼
                Faithfulness
```

---

# Real Enterprise Evaluation Pipeline

```text
Ground Truth Questions
          │
          ▼
Hybrid Retrieval
(BM25 + Dense)
          │
          ▼
Cross-Encoder Reranker
          │
          ▼
Top-K Context
          │
          ├──────────────┐
          │              │
          ▼              ▼
Context Precision   Context Recall
          │
          ▼
Prompt + LLM
          │
          ▼
Generated Answer
          │
          ▼
Statement Extraction
          │
          ▼
LLM/NLI Verification
          │
          ▼
Faithfulness Score
```

---

# Summary

| Metric                | Measures                                                | High Score Means                                | Low Score Indicates                              |
| --------------------- | ------------------------------------------------------- | ----------------------------------------------- | ------------------------------------------------ |
| **Context Precision** | How much of the retrieved context is relevant           | Retriever returns mostly useful chunks          | Too many irrelevant chunks retrieved             |
| **Context Recall**    | Whether all necessary context was retrieved             | Retriever found nearly all required information | Important supporting chunks were missed          |
| **Faithfulness**      | Whether the answer is grounded in the retrieved context | The answer is supported by evidence             | The LLM hallucinated or added unsupported claims |

In a production RAG system, teams typically monitor these metrics together:

* **Low Context Precision** → Improve chunking, hybrid search, or reranking to reduce irrelevant retrieval.
* **Low Context Recall** → Improve embeddings, chunk size, `top_k`, or retrieval strategy to avoid missing relevant documents.
* **Low Faithfulness** → Improve prompt grounding, context formatting, generation settings, or use constrained generation and citation-based prompting to reduce hallucinations.

A high-quality enterprise RAG system aims to maintain **high context recall**, **high context precision**, and **high faithfulness** simultaneously, because all three are necessary for reliable, trustworthy answers.
