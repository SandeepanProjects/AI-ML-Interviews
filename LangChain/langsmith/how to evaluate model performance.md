Evaluating model performance is one of the **most important responsibilities of a Senior AI Engineer**. A model that achieves **95% accuracy offline** may still fail in production due to hallucinations, latency, cost, or poor user experience.

In interviews, you may be asked:

* How do you evaluate an LLM?
* How do you know your RAG system is good?
* What metrics do you monitor in production?
* How do you detect model degradation?
* How do you compare GPT-4.1 vs GPT-5 vs Claude?

A complete answer should cover **offline evaluation**, **online evaluation**, **human evaluation**, and **production monitoring**.

---

# Why Evaluate Models?

Suppose you build a customer support chatbot.

```text
User

↓

LLM

↓

Answer
```

The answer may be:

* Correct ✅
* Hallucinated ❌
* Too slow ❌
* Too expensive ❌
* Unsafe ❌

Accuracy alone is not enough.

---

# Production Evaluation Pipeline

```text
                Dataset

                   │

                   ▼

          Offline Evaluation

                   │

         ┌─────────┼──────────┐

         ▼         ▼          ▼

     Accuracy   Latency    Cost

                   │

                   ▼

          Deploy to Production

                   │

                   ▼

          Online Monitoring

                   │

        Human Feedback + A/B Tests

                   │

                   ▼

          Continuous Improvement
```

---

# Types of Evaluation

| Type     | Purpose                          |
| -------- | -------------------------------- |
| Offline  | Compare models before deployment |
| Online   | Monitor production performance   |
| Human    | Judge quality and usefulness     |
| Business | Measure real-world impact        |

---

# Offline Evaluation

Offline evaluation uses a labeled dataset.

Example:

```text
Question

↓

Expected Answer

↓

Model Prediction

↓

Compare
```

Dataset:

```python
dataset = [

    {
        "question":"2+2",
        "expected":"4"
    },

    {
        "question":"Capital of France",
        "expected":"Paris"
    }
]
```

---

# Simple Accuracy

```python
correct = 0

for sample in dataset:

    prediction = llm.invoke(
        sample["question"]
    ).content

    if prediction == sample["expected"]:

        correct += 1

accuracy = correct / len(dataset)

print(accuracy)
```

Output:

```text
Accuracy = 95%
```

Works well for classification but not for open-ended generation.

---

# Precision / Recall / F1

Useful for:

* Classification
* Entity extraction
* Intent detection

```python
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
)

precision_score(y_true, y_pred)

recall_score(y_true, y_pred)

f1_score(y_true, y_pred)
```

---

# BLEU Score

Used in:

* Translation
* Text generation

```python
from nltk.translate.bleu_score import sentence_bleu

score = sentence_bleu(

    [["hello","world"]],

    ["hello","world"]

)

print(score)
```

---

# ROUGE

Useful for summarization.

```python
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(
    ["rougeL"]
)

score = scorer.score(

    reference,

    prediction

)
```

---

# BERTScore

Instead of exact word matching:

```text
Car

vs

Automobile
```

BERTScore measures semantic similarity.

```python
from bert_score import score

P, R, F1 = score(

    predictions,

    references,

    lang="en"

)
```

---

# LLM-as-a-Judge

A modern approach is to use another LLM to evaluate outputs.

Prompt:

```text
Evaluate the answer on:

- Correctness
- Completeness
- Helpfulness

Return score from 1-10.
```

Example:

```python
evaluation_prompt = f"""
Question:

{question}

Answer:

{answer}

Score correctness
from 1-10.
"""

judge.invoke(evaluation_prompt)
```

---

# RAG Evaluation

A RAG system has two stages.

```text
Question

↓

Retriever

↓

Documents

↓

LLM

↓

Answer
```

Both need evaluation.

---

## Retrieval Metrics

### Context Precision

How many retrieved documents are relevant?

```text
Retrieved

10 Docs

↓

Relevant

8

↓

Precision = 0.8
```

---

### Context Recall

Did retrieval find everything needed?

```text
Relevant Docs

5

Retrieved

4

↓

Recall = 80%
```

---

### Faithfulness

Did the answer stay grounded in the retrieved documents?

Example:

Documents:

```text
CEO = Sarah
```

Answer:

```text
CEO = John
```

Faithfulness:

```text
0%
```

---

### Answer Relevance

Question:

```text
How do transformers work?
```

Answer:

```text
Python is popular.
```

Low relevance.

---

# Latency Evaluation

Measure response time.

```python
import time

start = time.time()

response = llm.invoke(prompt)

latency = time.time()-start

print(latency)
```

Monitor:

* Average latency
* P95 latency
* P99 latency

---

# Token Usage

```python
usage = response.response_metadata[
    "token_usage"
]

print(
    usage["total_tokens"]
)
```

Monitor:

* Prompt tokens
* Completion tokens
* Cost

---

# Cost Evaluation

```python
PRICE_PER_1K = 0.002

cost = (

usage["total_tokens"]

/1000

)*PRICE_PER_1K
```

Aggregate:

* Cost per user
* Cost per request
* Cost per workflow

---

# Hallucination Detection

Prompt:

```text
Answer only using
retrieved documents.

If unsure,
say you don't know.
```

Evaluation:

```text
Supported by context?

Yes

↓

Faithful

No

↓

Hallucination
```

---

# Human Evaluation

Humans rate answers.

Scale:

```text
Correctness

1-5

Helpfulness

1-5

Safety

1-5

Completeness

1-5
```

Store:

```python
feedback = {

    "rating":5,

    "comment":"Excellent"

}
```

---

# Online Evaluation

Production metrics include:

```text
Requests/sec

Latency

Errors

Retries

Token Usage

Cost

Tool Success

Retriever Success

Hallucination Rate

User Rating
```

---

# A/B Testing

Compare two models.

```text
Users

        │

   50% GPT-4.1

   50% GPT-5

        │

 Compare

Accuracy

Latency

Cost

User Satisfaction
```

Example:

```python
if random() < 0.5:

    model = gpt41

else:

    model = gpt5
```

Measure:

* Response quality
* Cost
* Retention

---

# Drift Detection

Production:

```text
Training Data

↓

Model

↓

Production Data

↓

Distribution Changes

↓

Performance Drops
```

Monitor:

* Data drift
* Concept drift
* Embedding drift

---

# LangSmith Evaluation

LangSmith can evaluate:

```text
Prompt Versions

↓

Datasets

↓

Run Experiments

↓

Compare Models

↓

Score Outputs
```

Metrics include:

* Correctness
* Faithfulness
* Toxicity
* Helpfulness
* Custom evaluators

---

# End-to-End Evaluation Pipeline

```text
                 Test Dataset

                      │

                      ▼

                 LangSmith

                      │

        ┌─────────────┼─────────────┐

        ▼             ▼             ▼

    Retrieval      Generation    Cost

        ▼             ▼             ▼

      Precision   Faithfulness  Tokens

                      │

                      ▼

                Human Review

                      │

                      ▼

                 Production
```

---

# Enterprise Architecture

```text
                     Users

                       │

                       ▼

                    FastAPI

                       │

                       ▼

                   LangGraph

          ┌───────────┼────────────┐

          ▼           ▼            ▼

      Retriever      LLM         Tools

          │           │            │

          ▼           ▼            ▼

              Evaluation Layer

        ┌──────────┼───────────┐

        ▼          ▼           ▼

   LangSmith   Prometheus   PostgreSQL

        ▼          ▼           ▼

              Grafana Dashboard
```

---

# What Should You Measure?

| Category        | Metrics                                                      |
| --------------- | ------------------------------------------------------------ |
| Quality         | Accuracy, Faithfulness, Answer Relevance, Hallucination Rate |
| Retrieval       | Context Precision, Context Recall, Hit Rate                  |
| Performance     | Latency (P50/P95/P99), Throughput                            |
| Cost            | Prompt Tokens, Completion Tokens, Cost per Request           |
| Reliability     | Error Rate, Retry Rate, Tool Success Rate                    |
| User Experience | CSAT, Thumbs Up/Down, Task Completion                        |
| Business        | Conversion Rate, Resolution Rate, Retention                  |

---

# Best Practices

### Evaluate retrieval separately

Poor retrieval often causes poor generation.

---

### Use automated and human evaluation

Offline metrics catch regressions, while human review captures nuances that automated metrics may miss.

---

### Continuously monitor production

Performance can change due to new prompts, new data, model updates, or user behavior.

---

### Track cost with quality

A cheaper model is only valuable if it still meets your quality targets.

---

### Version everything

Version:

* Models
* Prompts
* Embedding models
* Retrieval configurations
* Evaluation datasets

This makes regressions easier to identify.

---

# Interview Questions

### How do you evaluate an LLM?

I evaluate quality (correctness, faithfulness, relevance), performance (latency, throughput), cost (token usage), reliability (errors, retries), and user satisfaction using a combination of automated benchmarks, human review, and production metrics.

---

### How do you evaluate a RAG system?

I evaluate the retriever using context precision and recall, then evaluate the generated answer using faithfulness, answer relevance, and hallucination rate. Finally, I monitor latency, token usage, and user feedback in production.

---

### Why isn't accuracy enough?

LLMs generate free-form text. Two different answers can both be correct, so metrics based only on exact string matching often underestimate quality. Semantic metrics, LLM-based evaluation, and human review are better suited for many generative AI tasks.

---

# Senior AI Engineer Interview Answer

> **I use a layered evaluation strategy. Before deployment, I run offline benchmarks on curated datasets, measuring task-specific metrics such as accuracy, precision/recall, or semantic similarity, along with LLM-based evaluators for open-ended tasks. For RAG systems, I separately evaluate retrieval quality using context precision and recall, and generation quality using faithfulness and answer relevance. In production, I monitor latency, token usage, cost, error rates, tool success rates, and user feedback. I also perform A/B testing to compare model or prompt versions and use LangSmith together with Prometheus and Grafana to continuously track quality, performance, and cost over time.
