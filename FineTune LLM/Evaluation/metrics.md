# 135. What is Perplexity?

## Simple definition

**Perplexity (PPL)** measures how “surprised” a language model is when predicting the correct tokens.

It is derived from cross-entropy loss:

$$
\text{Perplexity} = e^{\text{Loss}}
$$

Generally:

```text
Lower perplexity → model predicts the text better
Higher perplexity → model is more uncertain
```

---

## Example

Suppose:

```text
Input: The capital of France is
Correct next token: Paris
```

### Model A

```text
P(Paris) = 0.90
```

The model is confident.

### Model B

```text
P(Paris) = 0.10
```

The model is uncertain.

Model A will have lower loss and lower perplexity.

---

## Python code

```python
import math

loss = 1.5

perplexity = math.exp(loss)

print(f"Loss: {loss}")
print(f"Perplexity: {perplexity:.2f}")
```

Output:

```text
Loss: 1.5
Perplexity: 4.48
```

---

## Perplexity from Hugging Face evaluation

```python
import math

metrics = trainer.evaluate()

eval_loss = metrics["eval_loss"]

perplexity = math.exp(eval_loss)

print("Validation loss:", eval_loss)
print("Perplexity:", perplexity)
```

A practical comparison:

```text
Model A
Loss = 2.0
PPL = 7.39

Model B
Loss = 1.0
PPL = 2.72
```

On the same evaluation setup, Model B is better at next-token prediction.

## Important limitation

Low perplexity does **not necessarily mean**:

* Better factual accuracy
* Better instruction following
* Less hallucination
* Better customer-support answers

So perplexity is a **modeling metric**, not a complete business-quality metric.

---

# 136. What is BLEU?

**BLEU (Bilingual Evaluation Understudy)** measures similarity between generated text and one or more reference texts.

It is mainly based on **n-gram precision**.

Originally designed for machine translation.

---

## Example

Reference:

```text
The cat is sitting on the mat
```

Prediction:

```text
The cat is on the mat
```

BLEU checks overlapping:

```text
1-grams:
The ✓
cat ✓
is ✓
on ✓
the ✓
mat ✓

2-grams:
The cat ✓
cat is ✓
is on ✓
on the ✓
the mat ✓
```

The more overlap, the higher the BLEU score.

---

## BLEU components

BLEU uses:

### 1. N-gram precision

It checks:

```text
BLEU-1 → single words
BLEU-2 → word pairs
BLEU-3 → three-word sequences
BLEU-4 → four-word sequences
```

### 2. Brevity penalty

Suppose:

Reference:

```text
The customer can reset the password from account settings
```

Prediction:

```text
Reset password
```

The prediction may contain correct words but is too short.

BLEU applies a **brevity penalty**.

Conceptually:

$$
BLEU = BP \times \exp\left(\sum_n w_n \log(p_n)\right)
$$

Where:

* \(BP\) = brevity penalty
* \(p_n\) = n-gram precision
* \(w_n\) = weight for each n-gram

---

## BLEU code

Install:

```bash
pip install evaluate sacrebleu
```

Using `evaluate`:

```python
import evaluate

bleu = evaluate.load("bleu")

predictions = [
    "The cat is on the mat"
]

references = [
    [
        "The cat is sitting on the mat"
    ]
]

results = bleu.compute(
    predictions=predictions,
    references=references
)

print(results)
```

Example output:

```python
{
    "bleu": 0.45,
    "precisions": [...],
    "brevity_penalty": ...,
    "length_ratio": ...
}
```

---

## Multiple reference answers

This is useful because LLM responses can have multiple valid answers.

```python
predictions = [
    "You can reset your password in Settings."
]

references = [
    [
        "Reset your password from Settings.",
        "Go to Settings and select Forgot Password."
    ]
]

results = bleu.compute(
    predictions=predictions,
    references=references
)

print(results["bleu"])
```

---

## BLEU limitation

Consider:

```text
Reference:
The user can reset their password in account settings.

Prediction:
Open your profile settings and choose Forgot Password.
```

The meaning may be correct.

But word overlap may be low.

Therefore:

```text
Low BLEU ≠ Bad answer
High BLEU ≠ Correct answer
```

BLEU is most useful for:

```text
✓ Machine translation
✓ Controlled text generation
✓ Standard benchmark comparison
```

Less useful alone for:

```text
✗ Open-ended chatbots
✗ Customer support
✗ Creative generation
```

---

# 137. What is ROUGE?

**ROUGE (Recall-Oriented Understudy for Gisting Evaluation)** measures overlap between generated text and reference text.

It is commonly used for:

```text
Summarization
```

The main difference:

```text
BLEU  → focuses more on precision
ROUGE → focuses more on recall
```

---

# ROUGE example

Reference:

```text
The customer can reset the password through account settings
```

Prediction:

```text
The customer can reset the password
```

The prediction captures many important reference words.

ROUGE measures how much of the reference content was recovered.

---

# Types of ROUGE

## ROUGE-1

Measures unigram overlap.

```text
Reference:
The customer can reset the password

Prediction:
The customer can reset password
```

---

## ROUGE-2

Measures bigram overlap.

Example:

```text
The customer ✓
customer can ✓
can reset ✓
reset the ✓
the password ✓
```

---

## ROUGE-L

Uses the **Longest Common Subsequence (LCS)**.

Example:

```text
Reference:
A B C D E

Prediction:
A B X C D
```

Longest common sequence:

```text
A B C D
```

ROUGE-L captures sequence similarity.

---

# ROUGE code

Install:

```bash
pip install evaluate rouge_score
```

Code:

```python
import evaluate

rouge = evaluate.load("rouge")

predictions = [
    "The customer can reset the password from settings."
]

references = [
    "Go to account settings to reset the customer password."
]

results = rouge.compute(
    predictions=predictions,
    references=references
)

print("ROUGE-1:", results["rouge1"])
print("ROUGE-2:", results["rouge2"])
print("ROUGE-L:", results["rougeL"])
```

---

# Manual ROUGE-1 intuition

```python
def rouge_1_recall(
    prediction,
    reference
):

    prediction_words = (
        prediction.lower().split()
    )

    reference_words = (
        reference.lower().split()
    )

    overlap = 0

    remaining_prediction = (
        prediction_words.copy()
    )

    for word in reference_words:

        if word in remaining_prediction:

            overlap += 1

            remaining_prediction.remove(
                word
            )

    return (
        overlap
        / len(reference_words)
    )
```

Usage:

```python
prediction = (
    "customer reset password"
)

reference = (
    "customer can reset password"
)

score = rouge_1_recall(
    prediction,
    reference
)

print(score)
```

---

# ROUGE limitation

ROUGE has the same major issue as BLEU:

```text
It depends on lexical overlap.
```

Two answers can have the same meaning but use different words.

Example:

```text
Reference:
The system rejected the payment.

Prediction:
Your transaction could not be completed.
```

Semantic meaning is similar.

Word overlap may be low.

This is where **BERTScore** is useful.

---

# 138. What is BERTScore?

## Simple definition

**BERTScore evaluates generated text using contextual embeddings instead of only exact word overlap.**

It compares:

```text
Generated Answer
       ↓
Transformer embeddings
       ↓
Semantic similarity
       ↑
Transformer embeddings
       ↑
Reference Answer
```

Unlike BLEU and ROUGE:

```text
BLEU / ROUGE:
"car" vs "automobile"
        ↓
Different words → low overlap

BERTScore:
"car" vs "automobile"
        ↓
Similar contextual embeddings → high similarity
```

---

# Example

Reference:

```text
The customer can cancel the subscription from account settings.
```

Prediction:

```text
You can end your membership through your account preferences.
```

Word overlap:

```text
Low
```

Semantic similarity:

```text
High
```

BERTScore can capture this.

---

# How BERTScore works

Suppose:

```text
Reference:
The cat sat on the mat

Prediction:
A kitten rested on the rug
```

Each token gets a contextual embedding:

```text
Reference embeddings:

The     → vector
cat     → vector
sat     → vector
mat     → vector

Prediction embeddings:

A       → vector
kitten  → vector
rested  → vector
rug     → vector
```

BERTScore computes cosine similarity between tokens.

Conceptually:

$$
Similarity(a,b)
=
\frac{a \cdot b}
{|a||b|}
$$

Then it finds the best matching semantic token relationships.

For example:

```text
cat    ↔ kitten
sat    ↔ rested
mat    ↔ rug
```

Then calculates:

```text
Precision
Recall
F1
```

---

# BERTScore code

Install:

```bash
pip install bert-score
```

Code:

```python
from bert_score import score

predictions = [
    "You can cancel your membership in account settings."
]

references = [
    "The customer can cancel the subscription from account settings."
]

P, R, F1 = score(
    predictions,
    references,
    lang="en"
)

print(
    "Precision:",
    P.mean().item()
)

print(
    "Recall:",
    R.mean().item()
)

print(
    "F1:",
    F1.mean().item()
)
```

Example:

```text
Precision: 0.91
Recall:    0.88
F1:        0.89
```

---

# BERTScore with a specific model

```python
P, R, F1 = score(
    predictions,
    references,
    model_type="microsoft/deberta-xlarge-mnli",
    lang="en"
)
```

The embedding model affects:

```text
Quality
Speed
GPU memory
Metric behavior
```

For reproducible benchmarking, pin:

```text
Model version
Metric implementation version
Preprocessing
Evaluation dataset
```

---

# Precision, Recall, and F1 in BERTScore

### Precision

> How well do generated tokens match the reference semantically?

### Recall

> How much of the reference meaning is captured?

### F1

Balances both:

$$
F1 =
\frac{2 \times Precision \times Recall}
{Precision + Recall}
$$

Code:

```python
precision = 0.9
recall = 0.8

f1 = (
    2 * precision * recall
    /
    (precision + recall)
)

print(f1)
```

Output:

```text
0.847
```

---

# Complete comparison: BLEU vs ROUGE vs BERTScore

```text
Reference:
The customer can cancel the subscription from account settings.

Prediction:
You can end your membership through account preferences.
```

| Metric    | Main approach                  | Expected behavior                   |
| --------- | ------------------------------ | ----------------------------------- |
| BLEU      | N-gram precision               | Low/moderate due to different words |
| ROUGE     | N-gram overlap/recall          | Low/moderate                        |
| BERTScore | Contextual semantic embeddings | Higher if meaning is similar        |

---

# Complete evaluation code

```python
import math
import evaluate
from bert_score import score


predictions = [
    "You can reset your password in account settings.",
    "Cancel your subscription from the billing page."
]

references = [
    "Go to account settings and reset your password.",
    "You can cancel the subscription through billing settings."
]


# =====================================================
# BLEU
# =====================================================

bleu_metric = evaluate.load(
    "bleu"
)

bleu_result = bleu_metric.compute(
    predictions=predictions,

    references=[
        [reference]
        for reference in references
    ]
)

print(
    "BLEU:",
    bleu_result["bleu"]
)


# =====================================================
# ROUGE
# =====================================================

rouge_metric = evaluate.load(
    "rouge"
)

rouge_result = rouge_metric.compute(
    predictions=predictions,
    references=references
)

print(
    "ROUGE-1:",
    rouge_result["rouge1"]
)

print(
    "ROUGE-2:",
    rouge_result["rouge2"]
)

print(
    "ROUGE-L:",
    rouge_result["rougeL"]
)


# =====================================================
# BERTScore
# =====================================================

P, R, F1 = score(
    predictions,
    references,
    lang="en"
)

print(
    "BERTScore Precision:",
    P.mean().item()
)

print(
    "BERTScore Recall:",
    R.mean().item()
)

print(
    "BERTScore F1:",
    F1.mean().item()
)
```

---

# How I would use these in a fine-tuning project

Suppose you fine-tuned an LLM for customer support.

```text
Fine-Tuned LLM
      │
      ▼
Customer Question
      │
      ▼
Generated Response
      │
      ├─────────── BLEU / ROUGE
      │
      ├─────────── BERTScore
      │
      ├─────────── LLM Judge
      │
      ├─────────── Policy Compliance
      │
      └─────────── Human Evaluation
```

I would calculate:

```python
evaluation_report = {

    "validation_loss": eval_loss,

    "perplexity": perplexity,

    "bleu": bleu_score,

    "rougeL": rouge_score,

    "bertscore_f1": bertscore_f1,

    "json_validity": json_validity,

    "hallucination_rate": hallucination_rate,

    "safety_violation_rate": safety_violation_rate
}
```

---

# When should you use each metric?

| Metric               | Best use                                      |
| -------------------- | --------------------------------------------- |
| Perplexity           | Language-model quality, checkpoint comparison |
| BLEU                 | Translation, controlled generation            |
| ROUGE                | Summarization                                 |
| BERTScore            | Semantic similarity                           |
| Exact Match          | QA/classification/deterministic output        |
| F1                   | Classification, QA                            |
| Execution Accuracy   | SQL                                           |
| Pass@k               | Code generation                               |
| JSON/Schema Validity | Structured output                             |
| LLM-as-a-Judge       | Open-ended generation                         |
| Human Evaluation     | Final quality validation                      |

---

# Interview-ready answer

> **Perplexity measures how well a language model predicts the next token and is calculated as the exponential of cross-entropy loss. Lower perplexity generally indicates better predictive performance.**
>
> **BLEU measures n-gram precision and includes a brevity penalty. It is commonly used for translation. ROUGE measures n-gram overlap with more emphasis on recall and is commonly used for summarization.**
>
> **BERTScore uses contextual embeddings from transformer models and cosine similarity to compare the semantic similarity between generated and reference text. Unlike BLEU and ROUGE, it can recognize semantically similar wording even when the exact words differ.**
>
> **In practice, I would not use any one of these metrics alone. For a fine-tuned LLM, I combine model-level metrics like validation loss and perplexity with task-specific metrics, semantic evaluation, safety checks, and human or LLM-based evaluation.**
