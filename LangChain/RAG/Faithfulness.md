# Faithfulness in RAG (Complete Guide with Production Code)

**Faithfulness** is one of the **most important evaluation metrics** for LLM and RAG systems.

It answers one question:

> **Is the generated answer completely supported by the retrieved context?**

Faithfulness is **not** about whether the answer is factually correct in the real world.

It is about whether the answer is **grounded in the provided evidence**.

---

# Where Faithfulness Fits

A RAG pipeline has two stages:

```text
                User Question
                      │
                      ▼
              Retriever
          (Vector/Hybrid Search)
                      │
              Retrieved Context
                      │
                      ▼
                    LLM
                      │
                Generated Answer
```

Retrieval metrics evaluate the retriever.

Faithfulness evaluates the **LLM output**.

---

# Example 1: Faithful Answer

User asks:

> How many maternity leave days are provided?

Retrieved Context:

```text
Employees are entitled to 180 days of maternity leave.
```

LLM Answer:

```text
Employees receive 180 days of maternity leave.
```

Everything in the answer comes directly from the context.

Faithfulness:

```text
100%
```

---

# Example 2: Unfaithful Answer

Retrieved Context:

```text
Employees receive 180 days of maternity leave.
```

LLM Answer:

```text
Employees receive 180 days of maternity leave and can extend it by another 30 days.
```

Where did **30 days** come from?

It does **not** exist in the retrieved context.

This is a hallucination.

Faithfulness is low.

---

# Another Example

Question

```text
What is the company's refund policy?
```

Retrieved Context

```text
Customers can return products within 30 days.
```

LLM Answer

```text
Customers can return products within 30 days with free shipping.
```

The phrase

```text
with free shipping
```

doesn't appear in the context.

Therefore the answer is **not fully faithful**.

---

# Faithfulness vs Correctness

Many engineers confuse these.

Suppose:

Context

```text
The Earth is flat.
```

LLM Answer

```text
The Earth is flat.
```

Faithfulness

```text
100%
```

because the answer exactly matches the context.

But factual correctness is:

```text
0%
```

because the context itself is wrong.

Faithfulness measures **agreement with retrieved evidence**, not objective truth.

---

# Faithfulness vs Context Recall

Question

```text
How many leave days?
```

Context

```text
Employees receive 180 days.
```

LLM Answer

```text
Employees receive 180 days.
```

Faithfulness:

```text
100%
```

Now suppose retrieval missed another document stating:

```text
Managers require approval.
```

Faithfulness is still 100% because the answer only uses retrieved evidence.

However:

Context Recall is low because important information was never retrieved.

---

# How Faithfulness is Measured

Modern evaluation frameworks break the generated answer into individual claims.

Example answer:

```text
Employees receive 180 days of maternity leave and free childcare.
```

Claims:

```text
Claim 1:
Employees receive 180 days.

Claim 2:
Employees receive free childcare.
```

Each claim is checked against the retrieved context.

---

Pipeline:

```text
Retrieved Context
        │
        ▼
Generated Answer
        │
        ▼
Split into Claims
        │
        ▼
Check Each Claim Against Context
        │
        ▼
Compute Score
```

---

# Simple Python Implementation

Suppose:

```python
context = """
Employees receive 180 days of maternity leave.
"""

answer = """
Employees receive 180 days of maternity leave
and free childcare.
"""
```

A simple illustrative implementation:

```python
import re

def split_into_claims(text):
    parts = re.split(r"\.\s+|\band\b", text)
    return [p.strip() for p in parts if p.strip()]

claims = split_into_claims(answer)

for claim in claims:
    print(claim)
```

Output

```text
Employees receive 180 days of maternity leave

free childcare
```

Now compare each claim with the retrieved context.

```python
def supported(claim, context):
    return claim.lower() in context.lower()

supported_claims = 0

for claim in claims:
    if supported(claim, context):
        supported_claims += 1

faithfulness = supported_claims / len(claims)

print(faithfulness)
```

Output

```text
0.5
```

Only half the claims are supported.

> **Note:** This is a simplified demonstration. Production systems use LLM-based or NLI (Natural Language Inference) models rather than exact string matching.

---

# Measuring Faithfulness with Ragas

Install

```bash
pip install ragas
```

Dataset

```python
from datasets import Dataset

dataset = Dataset.from_dict({
    "question": [
        "How many maternity leave days?"
    ],
    "contexts": [[
        "Employees receive 180 days of maternity leave."
    ]],
    "answer": [
        "Employees receive 180 days of maternity leave."
    ],
    "ground_truth": [
        "Employees receive 180 days of maternity leave."
    ]
})
```

Evaluate

```python
from ragas import evaluate
from ragas.metrics import faithfulness

results = evaluate(
    dataset,
    metrics=[faithfulness]
)

print(results)
```

Example output

```text
Faithfulness

0.98
```

A score close to 1 means the generated answer is well supported by the retrieved context.

---

# LLM-as-a-Judge

Many production systems use another LLM to judge whether claims are supported.

Prompt:

```text
Context:

Employees receive 180 days of maternity leave.

Answer:

Employees receive 180 days of maternity leave and free childcare.

Task:

Identify every statement that is unsupported by the context.
```

Judge Output

```text
Unsupported Claim:

Free childcare
```

Faithfulness

```text
1 supported claim
-------------------
2 total claims

= 0.5
```

---

# Production Evaluation Pipeline

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
LLM Answer
       │
       ▼
Faithfulness Evaluator
       │
       ▼
Claim Extraction
       │
       ▼
Evidence Verification
       │
       ▼
Faithfulness Score
```

---

# How to Improve Faithfulness

## 1. Better Retrieval

Poor retrieval leads to unsupported answers.

Improve with:

* Better chunking
* Hybrid search
* Better embeddings
* Metadata filtering
* Query rewriting

---

## 2. Reranking

Instead of:

```text
Top 5 Vector Results
```

Use:

```text
Top 50

↓

Cross-Encoder Reranker

↓

Best 5
```

The LLM receives more relevant evidence.

---

## 3. Prompt Grounding

Instead of:

```text
Answer the question.
```

Use:

```text
Answer ONLY using the supplied context.

If the answer is not in the context,
say:

"I don't have enough information."
```

Example:

```python
prompt = f"""
You are an enterprise assistant.

Use ONLY the provided context.

If the answer is not present,
respond:

"I don't know based on the provided documents."

Context:
{context}

Question:
{question}
"""
```

---

## 4. Reduce Context Noise

Low Context Precision can encourage hallucinations.

Improve:

* Semantic chunking
* Metadata filters
* Reranking
* Context compression

---

## 5. Lower Generation Creativity

For factual RAG systems:

```python
response = llm.invoke(
    prompt,
    temperature=0
)
```

Lower temperatures reduce unsupported additions.

---

# Production Monitoring

A production dashboard might track:

```text
Faithfulness
Answer Correctness
Context Precision
Context Recall
Latency
Token Cost
Hallucination Rate
```

If faithfulness suddenly drops, investigate:

* Retrieval quality
* Prompt changes
* Model upgrades
* Document ingestion issues

---

# Faithfulness vs Other Metrics

| Metric             | Evaluates    | Main Question                                                     |
| ------------------ | ------------ | ----------------------------------------------------------------- |
| Context Precision  | Retriever    | How much retrieved context is useful?                             |
| Context Recall     | Retriever    | Was all required evidence retrieved?                              |
| Faithfulness       | LLM output   | Is every claim supported by the retrieved context?                |
| Answer Correctness | Final answer | Is the answer factually correct relative to the reference answer? |

---

# Enterprise Evaluation Pipeline

```text
                  User Question
                        │
                        ▼
                Hybrid Retrieval
                        │
                        ▼
                 Top 50 Documents
                        │
                        ▼
              Cross-Encoder Reranker
                        │
                        ▼
                  Top 5 Contexts
                        │
                        ▼
                      LLM
                        │
                        ▼
                 Generated Answer
                        │
        ┌───────────────┴───────────────┐
        ▼                               ▼
 Context Precision/Recall          Faithfulness
        │                               │
        └───────────────┬───────────────┘
                        ▼
                 Evaluation Dashboard
```

---

# Common Interview Questions

### Why is faithfulness important?

Because enterprise RAG systems must avoid hallucinations. A faithful answer stays grounded in retrieved evidence instead of inventing facts.

---

### Can an answer be faithful but incorrect?

Yes.

If the retrieved documents contain incorrect information and the model faithfully repeats it, faithfulness is high while factual correctness is low.

---

### Can an answer be correct but unfaithful?

Yes.

Suppose the retrieved context does not mention the CEO's name, but the model knows it from pretraining and answers correctly. The answer is factually correct, but it is **not grounded in the retrieved context**, so faithfulness is low.

---

# Senior AI Engineer Interview Answer

> **Faithfulness measures whether every claim made by the LLM is supported by the retrieved context. It is a generation metric rather than a retrieval metric and is primarily used to detect hallucinations in RAG systems. Production evaluation frameworks such as Ragas typically decompose the generated answer into individual claims and verify whether each claim is supported by the retrieved evidence, often using an LLM or a natural language inference model. To improve faithfulness, I focus on stronger retrieval through hybrid search and reranking, better chunking, grounding prompts that instruct the model to answer only from the provided context, reducing irrelevant context, and using low-temperature generation for factual tasks. Continuous monitoring of faithfulness alongside Context Precision, Context Recall, and Answer Correctness provides a comprehensive view of RAG system quality.
