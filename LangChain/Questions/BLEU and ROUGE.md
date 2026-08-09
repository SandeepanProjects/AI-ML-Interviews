Absolutely. **BLEU and ROUGE are text-generation evaluation metrics**. They are commonly used for machine translation, summarization, question answering, and other NLP tasks.

The easiest way to remember them is:

> **BLEU asks: "How much of my generated text matches the reference?"**
> **ROUGE asks: "How much of the reference content did my generated text capture?"**

They are related, but their direction and emphasis are different.

---

# 1. First understand the setup

Suppose we have a **reference answer** written by a human:

```text
Reference:
The cat is sitting on the mat.
```

Our model generates:

```text
Candidate:
The cat is sitting on the mat.
```

Perfect match → BLEU and ROUGE will be high.

Now:

```text
Candidate:
The cat is sleeping on the sofa.
```

There is some overlap, but the meaning is different.

BLEU/ROUGE will detect some lexical overlap, but they don't truly understand the meaning like an LLM does.

---

# 2. BLEU

**BLEU = Bilingual Evaluation Understudy**

It was originally designed primarily for **machine translation**.

BLEU measures **n-gram precision**.

That means:

> How many pieces of the generated sentence appear in the reference?

---

# 3. What is an n-gram?

An n-gram is simply a sequence of `n` words.

Sentence:

```text
The cat is sitting
```

### Unigrams

Individual words:

```text
The
cat
is
sitting
```

### Bigrams

Two consecutive words:

```text
The cat
cat is
is sitting
```

### Trigrams

Three consecutive words:

```text
The cat is
cat is sitting
```

### 4-grams

```text
The cat is sitting
```

BLEU typically considers:

```text
1-gram
2-gram
3-gram
4-gram
```

---

# 4. BLEU calculation

Suppose:

```text
Reference:
The cat is sitting on the mat

Candidate:
The cat is sitting on the mat
```

Candidate has:

```text
7 words
```

Every unigram appears in the reference.

Therefore:

[
P_1 = \frac{7}{7}=1
]

Every bigram also matches:

```text
The cat
cat is
is sitting
sitting on
on the
the mat
```

So:

[
P_2 = 1
]

Similarly:

[
P_3=1
]

[
P_4=1
]

Therefore BLEU is:

[
BLEU=1
]

or:

```text
100%
```

---

# 5. Now let's make it interesting

Reference:

```text
The cat is sitting on the mat
```

Candidate:

```text
The cat is sleeping on the mat
```

Let's calculate unigram precision.

Candidate words:

```text
The
cat
is
sleeping
on
the
mat
```

Reference:

```text
The
cat
is
sitting
on
the
mat
```

Matching words:

```text
The
cat
is
on
the
mat
```

6 matches out of 7.

Therefore:

[
P_1=\frac{6}{7}=0.857
]

So unigram precision is approximately:

```text
85.7%
```

---

# 6. Bigram precision

Candidate bigrams:

```text
The cat          ✅
cat is           ✅
is sleeping      ❌
sleeping on      ❌
on the           ✅
the mat          ✅
```

4 of 6 match.

Therefore:

[
P_2=\frac{4}{6}=0.667
]

So:

```text
Bigram precision = 66.7%
```

As we increase n:

```text
Unigram → 85.7%
Bigram  → 66.7%
```

The metric becomes stricter.

---

# 7. Why BLEU uses geometric mean

BLEU combines the n-gram precisions using a **geometric mean**, rather than an arithmetic mean.

The simplified BLEU formula is:

[
BLEU =
BP \times
\exp
\left(
\sum_{n=1}^{N} w_n \log P_n
\right)
]

Usually:

[
w_1=w_2=w_3=w_4=0.25
]

So:

[
BLEU =
BP \times
(P_1P_2P_3P_4)^{1/4}
]

where:

* (P_1) = unigram precision
* (P_2) = bigram precision
* (P_3) = trigram precision
* (P_4) = 4-gram precision
* (BP) = brevity penalty

---

# 8. Why geometric mean?

Suppose:

```text
P1 = 0.9
P2 = 0.8
P3 = 0.7
P4 = 0.1
```

Arithmetic mean:

[
\frac{0.9+0.8+0.7+0.1}{4}=0.625
]

But geometric mean:

[
(0.9\times0.8\times0.7\times0.1)^{1/4}
]

≈

[
0.474
]

The low 4-gram score significantly hurts the final score.

That's intentional.

BLEU wants the generated text to have **consistent n-gram overlap**, not just good unigram overlap.

---

# 9. BLEU's Brevity Penalty

This is a very important part.

Imagine reference:

```text
The cat is sitting on the mat
```

Candidate:

```text
The cat
```

Everything in the candidate appears in the reference.

So precision might look excellent.

But the candidate is obviously too short.

BLEU therefore applies a **brevity penalty**.

The formula is:

[
BP =
\begin{cases}
1 & c > r\
e^{(1-r/c)} & c \le r
\end{cases}
]

where:

* (c) = candidate length
* (r) = reference length

Example:

```text
Candidate length = 2
Reference length = 7
```

Then:

[
BP=e^{1-7/2}
]

[
=e^{-2.5}
]

≈

[
0.082
]

So the score gets heavily penalized.

---

# 10. Why BLEU is good

BLEU is particularly useful for:

### Machine translation

```text
English → French
English → Hindi
English → German
```

because lexical/phrase overlap is important.

It is also useful for comparing:

```text
Model A
Model B
Model C
```

on the same translation benchmark.

---

# 11. Problems with BLEU

BLEU has an important weakness:

> **It doesn't understand meaning.**

Reference:

```text
The automobile is very fast.
```

Candidate:

```text
The car is extremely fast.
```

Humans would say:

> Very similar meaning.

BLEU may not give a particularly high score because:

```text
automobile ≠ car
very ≠ extremely
```

There isn't enough exact n-gram overlap.

So BLEU is a **lexical overlap metric**, not a true semantic metric.

---

# 12. ROUGE

Now let's look at ROUGE.

**ROUGE = Recall-Oriented Understudy for Gisting Evaluation**

ROUGE is particularly popular for:

> **Text summarization**

The fundamental idea is:

> **How much of the reference content did my generated text recover?**

This is why ROUGE is **recall-oriented**.

---

# 13. ROUGE-N

ROUGE-N measures n-gram overlap.

For example:

```text
Reference:
The cat is sitting on the mat
```

Candidate:

```text
The cat is sleeping on the mat
```

Reference unigrams:

```text
The
cat
is
sitting
on
the
mat
```

Candidate contains:

```text
The
cat
is
sleeping
on
the
mat
```

Matching reference words:

```text
The
cat
is
on
the
mat
```

6 matches.

Therefore:

[
ROUGE\text{-}1
==============

# \frac{6}{7}

0.857
]

So:

```text
ROUGE-1 = 85.7%
```

---

# 14. ROUGE-1 vs BLEU-1

This is an important distinction.

BLEU uses:

[
Precision =
\frac{\text{matching candidate n-grams}}
{\text{candidate n-grams}}
]

ROUGE-1 typically uses:

[
Recall =
\frac{\text{matching reference n-grams}}
{\text{reference n-grams}}
]

So:

### BLEU

> How much of what I generated was correct according to the reference?

### ROUGE

> How much of the reference did I capture?

---

# 15. ROUGE-2

ROUGE-2 uses bigrams.

Reference:

```text
The cat is sitting on the mat
```

Bigrams:

```text
The cat
cat is
is sitting
sitting on
on the
the mat
```

Candidate:

```text
The cat is sleeping on the mat
```

Bigrams:

```text
The cat       ✅
cat is        ✅
is sleeping   ❌
sleeping on   ❌
on the        ✅
the mat       ✅
```

4 matches.

Therefore:

[
ROUGE-2=\frac{4}{6}=0.667
]

approximately:

```text
66.7%
```

---

# 16. ROUGE-L

ROUGE-L is different.

It uses:

> **Longest Common Subsequence (LCS)**

Consider:

```text
Reference:
The cat is sitting on the mat
```

Candidate:

```text
The cat is sleeping on the mat
```

The longest common subsequence is:

```text
The cat is on the mat
```

Length:

```text
6
```

Reference length:

```text
7
```

Candidate length:

```text
7
```

So:

[
Recall_{LCS}=\frac{6}{7}
]

and:

[
Precision_{LCS}=\frac{6}{7}
]

ROUGE-L combines precision and recall using an F-score.

---

# 17. ROUGE F1

ROUGE can report:

```text
Precision
Recall
F1
```

F1 is:

[
F1 =
2\frac{Precision\times Recall}
{Precision+Recall}
]

Suppose:

```text
Precision = 0.8
Recall = 0.6
```

Then:

[
F1=
2\frac{0.8\times0.6}{0.8+0.6}
]

# [

\frac{0.96}{1.4}
]

[
=0.686
]

So:

```text
F1 ≈ 68.6%
```

---

# 18. BLEU vs ROUGE

This is the easiest comparison:

| Metric    | Main idea                         | Typical use   |
| --------- | --------------------------------- | ------------- |
| **BLEU**  | Precision-oriented n-gram overlap | Translation   |
| **ROUGE** | Recall-oriented overlap           | Summarization |
| ROUGE-1   | Unigram overlap                   | Summarization |
| ROUGE-2   | Bigram overlap                    | Summarization |
| ROUGE-L   | Longest common subsequence        | Summarization |

Think:

```text
BLEU
 ↓
"What did I generate that matches?"

ROUGE
 ↓
"What did I miss from the reference?"
```

---

# 19. A very important problem with both

Neither BLEU nor ROUGE really understands **factual correctness**.

Suppose reference:

```text
The company has 10 offices.
```

Generated:

```text
The company has 100 offices.
```

There is huge word overlap:

```text
The
company
has
offices
```

So BLEU/ROUGE could still be relatively high.

But the answer is factually wrong.

This is extremely important for **RAG evaluation**.

---

# 20. Why I wouldn't use BLEU/ROUGE as your primary RAG metric

For your RAG systems, I'd use:

```text
                    RAG Evaluation
                          │
        ┌─────────────────┼─────────────────┐
        ↓                 ↓                 ↓
    Retrieval         Grounding          Answer
        │                 │                 │
   Recall@K          Faithfulness       Correctness
   Precision@K       Citation          Relevance
   MRR               Accuracy
   NDCG
```

BLEU/ROUGE can be additional metrics when you have a **reference answer**, but they shouldn't be your main measure of RAG quality.

---

# 21. Example with RAG

Question:

```text
What is the company's refund policy?
```

Reference answer:

```text
Customers can request a refund within 30 days of purchase.
```

RAG answer:

```text
Customers have 30 days from purchase to request a refund.
```

BLEU/ROUGE:

```text
Likely high
```

Good.

But consider:

```text
RAG answer:
Customers have 60 days from purchase to request a refund.
```

There is still substantial word overlap.

BLEU/ROUGE might not be terrible.

But:

```text
Faithfulness = BAD
Answer correctness = BAD
```

Therefore, for RAG:

> **Semantic/factual evaluation is more important than lexical overlap.**

---

# 22. BLEU vs ROUGE vs modern LLM evaluation

Today, you will often see:

### Traditional metrics

```text
BLEU
ROUGE
METEOR
```

### Semantic metrics

```text
BERTScore
Embedding similarity
```

### LLM-based evaluation

```text
Faithfulness
Answer relevance
Correctness
LLM-as-a-judge
```

### RAG-specific evaluation

```text
Context precision
Context recall
Faithfulness
Answer correctness
Answer relevance
```

---

# 23. Interview answer

If an interviewer asks:

> **"What is the difference between BLEU and ROUGE?"**

A strong answer is:

> **"BLEU and ROUGE are reference-based text-generation metrics based primarily on lexical overlap. BLEU is precision-oriented and measures how much of the generated text's n-grams match the reference, while ROUGE is recall-oriented and measures how much of the reference content is captured by the generated text. BLEU is traditionally used for machine translation, while ROUGE is widely used for summarization. ROUGE-N measures n-gram overlap, such as ROUGE-1 and ROUGE-2, and ROUGE-L uses the longest common subsequence. Both have limitations because they don't reliably understand semantic equivalence or factual correctness, so for modern RAG systems I'd combine them with retrieval metrics, faithfulness, answer correctness, and semantic or LLM-based evaluation."**

### One-line memory trick

```text
BLEU  → Precision → "How much of my output matches?"
ROUGE → Recall    → "How much of the reference did I capture?"
```

And for **RAG**, remember:

```text
BLEU/ROUGE ≠ RAG accuracy
```

They are useful supporting metrics, but **faithfulness + context retrieval quality + answer correctness** are much more important for a production RAG system.

Yes. **BLEU and ROUGE are normally calculated by code** using evaluation libraries. You *can* calculate them manually to understand the mathematics, but in a real ML/RAG evaluation pipeline you would use libraries.

I'll show you both:

1. **Manual calculation** → understand what's happening.
2. **Python implementation** → how you actually calculate it.
3. **RAG evaluation pipeline** → where BLEU/ROUGE fit in production.

---

# 1. BLEU calculation with Python

The most common Python library is `sacrebleu`.

Install:

```bash
pip install sacrebleu
```

Then:

```python
import sacrebleu

reference = "The cat is sitting on the mat"
candidate = "The cat is sitting on the mat"

score = sacrebleu.corpus_bleu(
    [candidate],
    [[reference]]
)

print(score.score)
```

Output:

```text
100.0
```

So:

```text
BLEU = 100
```

Note that SacreBLEU reports BLEU on a **0–100 scale**, while the mathematical formulation is often represented as **0–1**.

So:

```text
100 → 1.0
80  → 0.8
50  → 0.5
```

---

# 2. Try a different candidate

```python
import sacrebleu

reference = "The cat is sitting on the mat"
candidate = "The cat is sleeping on the mat"

score = sacrebleu.corpus_bleu(
    [candidate],
    [[reference]]
)

print(score.score)
```

You'll get a substantially lower score than 100.

Why?

Because:

```text
Reference:
The cat is sitting on the mat

Candidate:
The cat is sleeping on the mat
```

Some n-grams match:

```text
The
cat
is

The cat
cat is

on
the
mat

on the
the mat
```

but:

```text
is sitting
sitting on
```

don't match.

---

# 3. See BLEU's internal components

SacreBLEU can show you more information.

```python
import sacrebleu

reference = "The cat is sitting on the mat"
candidate = "The cat is sleeping on the mat"

result = sacrebleu.corpus_bleu(
    [candidate],
    [[reference]]
)

print("BLEU:", result.score)
print("Precisions:", result.precisions)
print("Brevity penalty:", result.bp)
print("Candidate length:", result.sys_len)
print("Reference length:", result.ref_len)
```

Conceptually you'll see something like:

```text
BLEU:  ...
Precisions: [...]
Brevity penalty: ...
Candidate length: 7
Reference length: 7
```

The exact values depend on SacreBLEU's implementation/tokenization and smoothing details.

---

# 4. Manually calculate BLEU

Now let's understand what's happening underneath.

Take:

```text
Reference:
The cat is sitting on the mat

Candidate:
The cat is sleeping on the mat
```

Tokenize:

```python
reference = [
    "The",
    "cat",
    "is",
    "sitting",
    "on",
    "the",
    "mat"
]

candidate = [
    "The",
    "cat",
    "is",
    "sleeping",
    "on",
    "the",
    "mat"
]
```

---

# 5. Calculate unigram precision

Candidate unigrams:

```text
The
cat
is
sleeping
on
the
mat
```

Matching:

```text
The       ✅
cat       ✅
is        ✅
sleeping  ❌
on        ✅
the       ✅
mat       ✅
```

Therefore:

[
P_1 = \frac{6}{7}
]

```python
p1 = 6 / 7

print(p1)
```

Output:

```text
0.857142857
```

---

# 6. Calculate bigram precision

Candidate:

```text
The cat
cat is
is sleeping
sleeping on
on the
the mat
```

Reference:

```text
The cat
cat is
is sitting
sitting on
on the
the mat
```

Matching:

```text
The cat       ✅
cat is        ✅
is sleeping   ❌
sleeping on   ❌
on the        ✅
the mat       ✅
```

Therefore:

[
P_2 = \frac{4}{6}
]

```python
p2 = 4 / 6

print(p2)
```

Output:

```text
0.6666666667
```

---

# 7. Trigram precision

Candidate:

```text
The cat is
cat is sleeping
is sleeping on
sleeping on the
on the mat
```

Reference:

```text
The cat is
cat is sitting
is sitting on
sitting on the
on the mat
```

Matching:

```text
The cat is       ✅
cat is sleeping  ❌
is sleeping on   ❌
sleeping on the  ❌
on the mat       ✅
```

So:

[
P_3 = \frac{2}{5}
]

```python
p3 = 2 / 5
```

---

# 8. 4-gram precision

Candidate:

```text
The cat is sleeping
cat is sleeping on
is sleeping on the
sleeping on the mat
```

Reference:

```text
The cat is sitting
cat is sitting on
is sitting on the
sitting on the mat
```

None match.

Therefore:

[
P_4 = 0
]

This is where an important practical issue appears.

If you simply calculate:

[
(P_1P_2P_3P_4)^{1/4}
]

you get:

[
0
]

because:

[
P_4=0
]

This is why BLEU implementations use **smoothing and specific implementation conventions** in some settings, particularly for short sentences.

So don't try to reproduce a library's exact score with a simplistic formula unless you also reproduce its tokenization, smoothing, and BLEU variant.

---

# 9. The BLEU formula

The core formula is:

[
BLEU =
BP \times
\exp
\left(
\sum_{n=1}^{N}
w_n \log P_n
\right)
]

For standard 4-gram BLEU:

[
w_1=w_2=w_3=w_4=0.25
]

And:

[
BP =
\begin{cases}
1 & c>r\
e^{1-r/c} & c\leq r
\end{cases}
]

where:

```text
c = candidate length
r = reference length
```

So the code conceptually looks like:

```python
import math

precisions = [p1, p2, p3, p4]

weights = [0.25, 0.25, 0.25, 0.25]

score = 0

for p, w in zip(precisions, weights):
    score += w * math.log(p)

bleu = math.exp(score)
```

But remember: if `p4 == 0`, `log(0)` is undefined. Real implementations handle this using their specified smoothing/algorithm.

---

# 10. ROUGE calculation

For ROUGE, a very commonly used Python package is:

```bash
pip install rouge-score
```

Then:

```python
from rouge_score import rouge_scorer

reference = "The cat is sitting on the mat"
candidate = "The cat is sleeping on the mat"

scorer = rouge_scorer.RougeScorer(
    ["rouge1", "rouge2", "rougeL"],
    use_stemmer=True
)

scores = scorer.score(
    reference,
    candidate
)

print(scores)
```

You'll get objects containing:

```text
precision
recall
fmeasure
```

for:

```text
ROUGE-1
ROUGE-2
ROUGE-L
```

---

# 11. Extract the scores

```python
for metric, score in scores.items():
    print(metric)
    print("Precision:", score.precision)
    print("Recall:", score.recall)
    print("F1:", score.fmeasure)
```

Conceptually:

```text
rouge1
Precision: ...
Recall: ...
F1: ...

rouge2
Precision: ...
Recall: ...
F1: ...

rougeL
Precision: ...
Recall: ...
F1: ...
```

---

# 12. Manually calculate ROUGE-1

Reference:

```text
The cat is sitting on the mat
```

Candidate:

```text
The cat is sleeping on the mat
```

Reference has:

```text
7 words
```

Candidate has:

```text
7 words
```

Matching words:

```text
The
cat
is
on
the
mat
```

6 matches.

### Recall

[
Recall =
\frac{6}{7}
]

```text
Recall = 0.857
```

### Precision

[
Precision =
\frac{6}{7}
]

```text
Precision = 0.857
```

### F1

[
F1 =
2\frac{P\times R}{P+R}
]

Because:

[
P=R=0.857
]

F1 is also approximately:

```text
0.857
```

---

# 13. ROUGE-2 manually

Reference bigrams:

```text
The cat
cat is
is sitting
sitting on
on the
the mat
```

Candidate bigrams:

```text
The cat
cat is
is sleeping
sleeping on
on the
the mat
```

Matching:

```text
The cat
cat is
on the
the mat
```

4 matches.

Therefore:

[
Recall =
\frac{4}{6}
===========

0.667
]

And candidate has 6 bigrams:

[
Precision =
\frac{4}{6}
===========

0.667
]

Therefore:

[
F1=0.667
]

---

# 14. ROUGE-L manually

ROUGE-L uses the **Longest Common Subsequence**.

Reference:

```text
The cat is sitting on the mat
```

Candidate:

```text
The cat is sleeping on the mat
```

LCS:

```text
The cat is on the mat
```

Length:

```text
6
```

Reference length:

```text
7
```

Candidate length:

```text
7
```

Therefore:

[
Recall_{LCS}=\frac{6}{7}
]

[
Precision_{LCS}=\frac{6}{7}
]

And F1 is approximately:

[
0.857
]

---

# 15. Where are BLEU and ROUGE actually calculated?

This is important in a real project.

Suppose you have:

```text
                    RAG System
                       │
User Question ────────►│
                       ↓
                    Retrieval
                       ↓
                    Context
                       ↓
                      LLM
                       ↓
                Generated Answer
```

You need a **gold/reference answer**:

```text
Generated Answer
       │
       │
       ├──────────────┐
       ↓              ↓
Reference Answer   Retrieved Context
       │              │
       ↓              ↓
  BLEU / ROUGE     Faithfulness
```

For example:

```python
question = "What is the refund period?"

reference = """
Customers can request a refund within 30 days of purchase.
"""

generated = """
Customers have 30 days from purchase to request a refund.
"""
```

Then:

```python
bleu_score = calculate_bleu(reference, generated)
rouge_score = calculate_rouge(reference, generated)
```

---

# 16. Batch evaluation

In production, you don't evaluate just one question.

You create a dataset:

```python
evaluation_data = [
    {
        "question": "What is the refund period?",
        "reference": "Customers can request refunds within 30 days.",
    },
    {
        "question": "Who is eligible for insurance?",
        "reference": "Full-time employees are eligible.",
    },
    {
        "question": "What is the notice period?",
        "reference": "The notice period is 30 days.",
    },
]
```

Run your RAG system:

```python
for item in evaluation_data:

    answer = rag_pipeline(item["question"])

    bleu = calculate_bleu(
        item["reference"],
        answer
    )

    rouge = calculate_rouge(
        item["reference"],
        answer
    )
```

Then calculate averages:

```text
Average BLEU
Average ROUGE-1
Average ROUGE-2
Average ROUGE-L
```

---

# 17. But there is an important problem with RAG evaluation

Imagine:

```text
Reference:
The refund period is 30 days.
```

Generated:

```text
The customer has thirty days to request a refund.
```

A human says:

> Correct.

But BLEU/ROUGE may not give as high a score as you expect because:

```text
30 ≠ thirty
```

and word ordering may differ.

That's why modern RAG evaluation shouldn't depend exclusively on BLEU/ROUGE.

---

# 18. Better RAG evaluation

I'd structure your evaluation code like this:

```python
def evaluate_rag(
    question,
    retrieved_context,
    generated_answer,
    reference_answer
):
    return {
        "bleu": calculate_bleu(
            reference_answer,
            generated_answer
        ),

        "rouge": calculate_rouge(
            reference_answer,
            generated_answer
        ),

        "faithfulness": evaluate_faithfulness(
            retrieved_context,
            generated_answer
        ),

        "answer_correctness": evaluate_correctness(
            reference_answer,
            generated_answer
        ),

        "answer_relevance": evaluate_relevance(
            question,
            generated_answer
        ),

        "context_recall": evaluate_context_recall(
            reference_answer,
            retrieved_context
        )
    }
```

Then your final evaluation looks like:

```text
Question
   │
   ↓
Retriever
   │
   ├── Precision@K
   ├── Recall@K
   ├── MRR
   └── NDCG
   │
   ↓
Retrieved Context
   │
   ├── Context Recall
   └── Context Precision
   │
   ↓
LLM
   │
   ↓
Generated Answer
   │
   ├── BLEU
   ├── ROUGE
   ├── Faithfulness
   ├── Correctness
   └── Relevance
```

---

# 19. When would I actually use BLEU/ROUGE?

### BLEU

I'd primarily use it for:

```text
Machine Translation
```

For example:

```text
English
   ↓
Translation model
   ↓
Hindi
```

Compare generated translation against human translation.

---

### ROUGE

I'd primarily use it for:

```text
Summarization
```

Example:

```text
100-page document
       ↓
    LLM
       ↓
  1-page summary
```

Compare:

```text
LLM summary
     vs
Human summary
```

ROUGE-1, ROUGE-2 and ROUGE-L are useful here.

---

### RAG

I'd use them as **secondary metrics**.

More important:

```text
Retrieval
──────────────
Recall@K
Precision@K
MRR
NDCG

Generation
──────────────
Faithfulness
Answer correctness
Answer relevance
Citation accuracy
```

---

# 20. One important terminology point

You said **"ROGUE"** earlier.

The correct spelling is:

> **ROUGE**

Not "ROGUE".

It stands for:

**Recall-Oriented Understudy for Gisting Evaluation.**

And:

**BLEU = Bilingual Evaluation Understudy.**

---

## The interview-level answer

If someone asks you:

> **"How do you calculate BLEU and ROUGE in your project?"**

You can say:

> "I calculate them programmatically during offline evaluation using libraries such as SacreBLEU for BLEU and rouge-score for ROUGE. For each evaluation sample, I run the model, compare the generated answer against a human-written reference answer, and aggregate the scores across the evaluation dataset. BLEU calculates n-gram precision with a brevity penalty, while ROUGE calculates overlap between generated and reference text, commonly using ROUGE-1, ROUGE-2 and ROUGE-L with precision, recall and F1. However, for RAG I don't treat BLEU or ROUGE as the primary accuracy metric because they measure lexical overlap rather than factual grounding. I additionally evaluate retrieval recall/precision, context recall, faithfulness, answer correctness, and answer relevance."

That is the **production-level way to think about it**.
