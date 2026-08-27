# How Do You Evaluate Hallucinations in an LLM?

## 1. What is a hallucination?

A hallucination is when an LLM generates information that is:

* false
* unsupported by the provided context
* fabricated
* incorrectly attributed to a source
* stated with unjustified certainty

Example:

### Context

```text
Acme Corp was founded in 2015.
Its headquarters are in Bangalore.
```

### Question

```text
Who founded Acme Corp?
```

### Model answer

```text
Acme Corp was founded by John Smith.
```

This is a hallucination because **John Smith is not supported by the context**.

---

# 2. Types of hallucinations

## A. Factual hallucination

```text
Question:
When was Python created?

Answer:
Python was created in 1985.
```

Incorrect fact.

---

## B. Contextual hallucination

The answer contradicts or adds information not present in the supplied context.

```text
Context:
The refund takes 5–7 business days.

Answer:
The refund will arrive within 24 hours.
```

Unsupported and contradictory.

---

## C. Extrinsic hallucination

The model introduces information outside the provided context.

```text
Context:
The company has offices in India and the US.

Answer:
The company also has an office in Japan.
```

The Japan claim is unsupported.

---

## D. Intrinsic hallucination

The model distorts information already present.

```text
Context:
The product costs $100.

Answer:
The product costs $1,000.
```

The model used the context but changed the fact.

---

# 3. The most important idea: evaluate claims, not just answers

A long answer may contain:

```text
Answer:
1. Correct claim
2. Correct claim
3. Hallucinated claim
4. Correct claim
```

If we evaluate the entire answer as:

```text
Correct / Incorrect
```

we lose useful information.

Instead:

```text
Answer
   │
   ▼
Claim extraction
   │
   ├── Claim 1 → supported?
   ├── Claim 2 → supported?
   ├── Claim 3 → supported?
   └── Claim 4 → supported?
   │
   ▼
Hallucination metrics
```

This is the production-style approach.

---

# 4. Hallucination Rate

A simple metric is:

$$
Hallucination\ Rate =
\frac{Number\ of\ hallucinated\ claims}
{Total\ claims}
$$

Example:

```text
Total claims = 10
Hallucinated claims = 2
```

Then:

$$
Hallucination\ Rate = 20\%
$$

Code:

```python
def calculate_hallucination_rate(
    total_claims: int,
    hallucinated_claims: int
) -> float:

    if total_claims == 0:
        return 0.0

    return (
        hallucinated_claims
        / total_claims
    )
```

Usage:

```python
rate = calculate_hallucination_rate(
    total_claims=10,
    hallucinated_claims=2
)

print(f"Hallucination rate: {rate:.2%}")
```

Output:

```text
Hallucination rate: 20.00%
```

But the difficult part is determining whether a claim is supported.

---

# 5. Method 1: Exact fact checking with a ground-truth dataset

For structured tasks, create:

```python
evaluation_data = [
    {
        "question": "Where is Acme headquarters?",
        "ground_truth": "Bangalore",
        "answer": "Acme headquarters is in Bangalore."
    }
]
```

Simple evaluator:

```python
def evaluate_factual_answer(
    answer: str,
    ground_truth: str
):

    supported = (
        ground_truth.lower()
        in answer.lower()
    )

    return {
        "supported": supported,
        "hallucinated": not supported
    }
```

Usage:

```python
result = evaluate_factual_answer(
    answer="Acme headquarters is in Bangalore.",
    ground_truth="Bangalore"
)

print(result)
```

### Problem

This does not scale well for open-ended answers.

For example:

```text
Ground truth:
Bangalore

Answer:
The company is headquartered in Bengaluru, Karnataka.
```

Semantically correct but exact matching may fail.

---

# 6. Method 2: Context-grounded evaluation

This is especially important for **RAG systems**.

Suppose:

```python
context = """
Acme Corp was founded in 2015.
The company headquarters are in Bangalore.
Acme has 500 employees.
"""
```

Question:

```python
question = """
Where is Acme Corp headquartered?
"""
```

Answer:

```python
answer = """
Acme Corp is headquartered in Bangalore.
"""
```

We want to check:

```text
Answer claims
       │
       ▼
Are they supported by the context?
```

---

# 7. Simple sentence-level support checking

First split the answer into claims.

```python
import re


def split_into_claims(
    answer: str
):

    sentences = re.split(
        r"[.!?]+",
        answer
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]
```

Usage:

```python
answer = """
Acme was founded in 2015.
It has 500 employees.
Its CEO is John Smith.
"""

claims = split_into_claims(
    answer
)

print(claims)
```

Output:

```text
[
    "Acme was founded in 2015",
    "It has 500 employees",
    "Its CEO is John Smith"
]
```

Now each claim can be evaluated separately.

---

# 8. Semantic similarity evaluation

We can compare a claim against the context using embeddings.

Install:

```bash
pip install sentence-transformers
```

Code:

```python
from sentence_transformers import (
    SentenceTransformer,
    util
)


embedding_model = (
    SentenceTransformer(
        "all-MiniLM-L6-v2"
    )
)
```

Create embeddings:

```python
context = """
Acme Corp was founded in 2015.
The company headquarters are in Bangalore.
Acme has 500 employees.
"""

claim = """
Acme was founded in 2015.
"""


context_embedding = (
    embedding_model.encode(
        context,
        convert_to_tensor=True
    )
)

claim_embedding = (
    embedding_model.encode(
        claim,
        convert_to_tensor=True
    )
)
```

Calculate cosine similarity:

```python
similarity = util.cos_sim(
    claim_embedding,
    context_embedding
)

print(similarity.item())
```

### Better approach: compare against chunks

```python
context_chunks = [
    "Acme Corp was founded in 2015.",
    "The company headquarters are in Bangalore.",
    "Acme has 500 employees."
]
```

Code:

```python
def calculate_claim_support(
    claim,
    context_chunks,
    embedding_model
):

    claim_embedding = (
        embedding_model.encode(
            claim,
            convert_to_tensor=True
        )
    )

    context_embeddings = (
        embedding_model.encode(
            context_chunks,
            convert_to_tensor=True
        )
    )

    similarities = util.cos_sim(
        claim_embedding,
        context_embeddings
    )[0]

    best_score = similarities.max().item()

    best_index = (
        similarities.argmax().item()
    )

    return {
        "claim": claim,
        "best_similarity": best_score,
        "best_context": (
            context_chunks[
                best_index
            ]
        )
    }
```

Usage:

```python
result = calculate_claim_support(
    claim="Acme was founded in 2015.",
    context_chunks=context_chunks,
    embedding_model=embedding_model
)

print(result)
```

Example:

```text
{
    "claim": "Acme was founded in 2015.",
    "best_similarity": 0.92,
    "best_context": "Acme Corp was founded in 2015."
}
```

---

# Important limitation of embeddings

Consider:

```text
Context:
Acme has 500 employees.

Claim:
Acme has 5,000 employees.
```

Embedding similarity might still be very high because both sentences are semantically similar.

Therefore:

```text
Semantic similarity
≠
Factual correctness
```

This is a critical interview point.

---

# 9. Method 3: NLI-based hallucination detection

A better method is **Natural Language Inference (NLI)**.

We classify:

```text
Premise:
Acme has 500 employees.

Hypothesis:
Acme has 500 employees.

→ Entailment
```

Another:

```text
Premise:
Acme has 500 employees.

Hypothesis:
Acme has 5,000 employees.

→ Contradiction
```

Another:

```text
Premise:
Acme has 500 employees.

Hypothesis:
Acme has offices in Japan.

→ Neutral / Unknown
```

For grounded generation:

```text
Entailment     → Supported
Contradiction  → Hallucination
Neutral        → Unsupported
```

---

## NLI code

```bash
pip install transformers torch
```

```python
from transformers import pipeline


nli_model = pipeline(
    task="text-classification",
    model="facebook/bart-large-mnli"
)
```

Evaluate a claim:

```python
def check_entailment(
    context: str,
    claim: str
):

    result = nli_model(
        {
            "text": context,
            "text_pair": claim
        }
    )

    return result
```

Depending on the model/pipeline configuration, NLI labels can vary. Inspect and normalize labels before using them in production.

Conceptually:

```python
result = check_entailment(
    context="Acme has 500 employees.",

    claim="Acme has 5,000 employees."
)

print(result)
```

Expected interpretation:

```text
CONTRADICTION
```

---

# 10. Better RAG hallucination evaluator

For a RAG system:

```text
User Question
      │
      ▼
Retriever
      │
      ▼
Retrieved Documents
      │
      ▼
LLM
      │
      ▼
Answer
      │
      ▼
Claim Extraction
      │
      ▼
Support Verification
      │
      ├── Entailed
      ├── Contradicted
      └── Unsupported
```

Let's implement the evaluation architecture.

---

# 11. Claim extraction using an LLM

A robust system can ask an LLM to extract atomic factual claims.

Example answer:

```text
Acme was founded in 2015 and has 500 employees.
Its headquarters are in Bangalore.
```

Extract:

```python
[
    "Acme was founded in 2015.",
    "Acme has 500 employees.",
    "Acme headquarters are in Bangalore."
]
```

Prompt:

```python
CLAIM_EXTRACTION_PROMPT = """
Extract all atomic factual claims from the answer.

Rules:
- Split compound claims.
- Ignore opinions.
- Ignore greetings.
- Return only JSON.

ANSWER:
{answer}

Return:

{
    "claims": [
        "claim 1",
        "claim 2"
    ]
}
"""
```

Example implementation:

```python
import json


def extract_claims(
    llm_client,
    answer: str
):

    prompt = (
        CLAIM_EXTRACTION_PROMPT.format(
            answer=answer
        )
    )

    response = llm_client.generate(
        prompt,
        temperature=0
    )

    data = json.loads(
        response
    )

    return data["claims"]
```

For production, validate this JSON with Pydantic.

---

# 12. Evaluate each claim with an LLM judge

Sometimes NLI models struggle with:

* tables
* long documents
* numerical reasoning
* domain-specific terminology
* complex multi-hop facts

An LLM judge can evaluate support.

Prompt:

```python
GROUNDING_PROMPT = """
You are evaluating factual grounding.

Determine whether the CLAIM is supported by the CONTEXT.

Return one label only:

SUPPORTED
CONTRADICTED
UNSUPPORTED

Definitions:

SUPPORTED:
The context directly supports the claim.

CONTRADICTED:
The context contradicts the claim.

UNSUPPORTED:
The context does not contain enough information.

CONTEXT:
{context}

CLAIM:
{claim}
"""
```

Code:

```python
def evaluate_claim_with_llm(
    llm_client,
    context: str,
    claim: str
):

    prompt = (
        GROUNDING_PROMPT.format(
            context=context,
            claim=claim
        )
    )

    result = llm_client.generate(
        prompt,
        temperature=0
    )

    return result.strip().upper()
```

Example:

```python
label = evaluate_claim_with_llm(
    llm_client=judge_llm,

    context="""
    Acme was founded in 2015.
    Acme has 500 employees.
    """,

    claim="""
    Acme has 5,000 employees.
    """
)

print(label)
```

Expected:

```text
CONTRADICTED
```

---

# 13. Production-style claim evaluation

```python
def evaluate_answer_grounding(
    answer,
    context,
    claim_extractor,
    judge_llm
):

    claims = extract_claims(
        claim_extractor,
        answer
    )

    results = []

    for claim in claims:

        label = (
            evaluate_claim_with_llm(
                llm_client=judge_llm,
                context=context,
                claim=claim
            )
        )

        results.append(
            {
                "claim": claim,
                "label": label
            }
        )

    return results
```

Example:

```python
results = evaluate_answer_grounding(
    answer="""
    Acme was founded in 2015.
    It has 500 employees.
    Its CEO is John Smith.
    """,

    context="""
    Acme was founded in 2015.
    The company has 500 employees.
    """,

    claim_extractor=claim_llm,

    judge_llm=judge_llm
)
```

Expected:

```python
[
    {
        "claim": "Acme was founded in 2015.",
        "label": "SUPPORTED"
    },

    {
        "claim": "Acme has 500 employees.",
        "label": "SUPPORTED"
    },

    {
        "claim": "Acme CEO is John Smith.",
        "label": "UNSUPPORTED"
    }
]
```

---

# 14. Calculate hallucination metrics

```python
def calculate_grounding_metrics(
    claim_results
):

    total = len(
        claim_results
    )

    supported = sum(
        item["label"] == "SUPPORTED"
        for item in claim_results
    )

    contradicted = sum(
        item["label"] == "CONTRADICTED"
        for item in claim_results
    )

    unsupported = sum(
        item["label"] == "UNSUPPORTED"
        for item in claim_results
    )

    hallucinated = (
        contradicted
        + unsupported
    )

    return {

        "total_claims": total,

        "supported_claims": supported,

        "contradicted_claims": contradicted,

        "unsupported_claims": unsupported,

        "hallucination_rate": (
            hallucinated / total
            if total > 0
            else 0
        ),

        "groundedness_rate": (
            supported / total
            if total > 0
            else 0
        )
    }
```

Example:

```python
metrics = calculate_grounding_metrics(
    results
)

print(metrics)
```

Output:

```text
{
    "total_claims": 3,
    "supported_claims": 2,
    "contradicted_claims": 0,
    "unsupported_claims": 1,
    "hallucination_rate": 0.333,
    "groundedness_rate": 0.667
}
```

---

# 15. Faithfulness metric

For RAG, one of the most important metrics is **faithfulness**.

$$
Faithfulness =
\frac{Supported\ Claims}
{Total\ Claims}
$$

Example:

```text
Total claims = 5
Supported claims = 4
```

$$
Faithfulness = 0.8
$$

Code:

```python
def calculate_faithfulness(
    supported_claims,
    total_claims
):

    if total_claims == 0:
        return 1.0

    return (
        supported_claims
        / total_claims
    )
```

Relationship:

```text
High faithfulness
      ↓
Low hallucination
```

But they are not always perfect opposites because:

* some answers contain irrelevant claims
* some answers omit important facts
* unsupported claims may be harmless or critical

---

# 16. Evaluate citation correctness

If your LLM produces citations:

```text
Acme was founded in 2015 [Source 2].
```

You can check:

```text
Claim
   │
   ▼
Citation
   │
   ▼
Does the cited document support the claim?
```

Example:

```python
def evaluate_citation(
    claim,
    cited_context,
    judge_llm
):

    return evaluate_claim_with_llm(
        llm_client=judge_llm,
        context=cited_context,
        claim=claim
    )
```

Production metrics:

```text
Citation Precision:
Correct citations / Total citations

Citation Recall:
Claims with citations / Claims requiring citations

Citation Correctness:
Does cited evidence actually support the claim?
```

This is especially useful in enterprise RAG.

---

# 17. Detect hallucination using multiple methods

A robust system should not rely on only one evaluator.

```text
                   Generated Answer
                          │
                          ▼
                    Claim Extraction
                          │
          ┌───────────────┼────────────────┐
          ▼               ▼                ▼

      Exact Match      NLI Model       LLM Judge
          │               │                │
          └───────────────┼────────────────┘
                          ▼
                    Aggregation
                          │
                          ▼
                Hallucination Score
```

Example:

```python
def aggregate_judgments(
    nli_label,
    llm_label
):

    if (
        nli_label == "CONTRADICTION"
        or llm_label == "CONTRADICTED"
    ):
        return "HALLUCINATED"

    if (
        nli_label == "ENTAILMENT"
        and llm_label == "SUPPORTED"
    ):
        return "SUPPORTED"

    return "NEEDS_REVIEW"
```

---

# 18. Evaluate hallucinations in fine-tuned models

Suppose you fine-tuned:

```text
Base Model
      │
      ▼
Fine-Tuned Model
```

You should compare both models on the same held-out test set.

```python
def compare_hallucination_rates(
    base_results,
    fine_tuned_results
):

    base_metrics = (
        calculate_grounding_metrics(
            base_results
        )
    )

    fine_tuned_metrics = (
        calculate_grounding_metrics(
            fine_tuned_results
        )
    )

    return {

        "base_hallucination_rate":
            base_metrics[
                "hallucination_rate"
            ],

        "fine_tuned_hallucination_rate":
            fine_tuned_metrics[
                "hallucination_rate"
            ],

        "improvement": (
            base_metrics[
                "hallucination_rate"
            ]
            -
            fine_tuned_metrics[
                "hallucination_rate"
            ]
        )
    }
```

Example:

```text
Base model hallucination rate:        18%
Fine-tuned model hallucination rate:   7%
```

This gives evidence that the fine-tuned model improved.

---

# 19. Evaluate different hallucination categories

Do not use only one aggregate score.

```python
evaluation_categories = {

    "factual": [],

    "contextual": [],

    "numerical": [],

    "temporal": [],

    "citation": [],

    "domain_specific": []
}
```

For example:

```text
Overall Hallucination Rate: 8%

But:

Factual:         4%
Numerical:       18%
Temporal:        14%
Citation:         2%
Domain-specific:  6%
```

This tells you where the model needs improvement.

---

# 20. Evaluate abstention behavior

A very important test:

```text
Context:
Acme has 500 employees.

Question:
Who is the CEO?

Good answer:
The provided context does not specify who the CEO is.
```

Bad answer:

```text
The CEO is John Smith.
```

You should explicitly measure whether the model abstains when information is missing.

---

## Code

```python
def evaluate_abstention(
    response,
    expected_abstention_phrases
):

    response_lower = (
        response.lower()
    )

    abstained = any(

        phrase.lower()
        in response_lower

        for phrase
        in expected_abstention_phrases
    )

    return {
        "correctly_abstained": abstained
    }
```

Example:

```python
result = evaluate_abstention(

    response="""
    The provided information does not
    specify who the CEO is.
    """,

    expected_abstention_phrases=[
        "does not specify",
        "not provided",
        "do not have enough information"
    ]
)
```

For production, semantic evaluation is better than phrase matching.

---

# 21. A production-style evaluation dataset

```python
hallucination_dataset = [

    {
        "id": "supported_001",

        "context": """
        The refund takes between
        5 and 7 business days.
        """,

        "question": """
        How long does a refund take?
        """,

        "expected_behavior": "answer"
    },


    {
        "id": "unsupported_001",

        "context": """
        The refund takes between
        5 and 7 business days.
        """,

        "question": """
        Who processes the refund?
        """,

        "expected_behavior": "abstain"
    },


    {
        "id": "contradiction_001",

        "context": """
        The subscription costs
        $20 per month.
        """,

        "question": """
        What is the monthly cost?
        """,

        "expected_answer": "$20"
    }
]
```

This tests:

```text
✓ Correct answering
✓ Grounding
✓ Abstention
✓ Contradiction resistance
```

---

# 22. Complete evaluation pipeline

```python
def evaluate_hallucinations(
    model,
    tokenizer,
    dataset,
    claim_extractor,
    judge_llm
):

    all_results = []

    for example in dataset:

        # -----------------------------
        # Generate model response
        # -----------------------------

        response = generate_response(
            model=model,
            tokenizer=tokenizer,
            prompt=f"""
Context:
{example['context']}

Question:
{example['question']}
"""
        )


        # -----------------------------
        # Evaluate grounding
        # -----------------------------

        claim_results = (
            evaluate_answer_grounding(

                answer=response,

                context=example[
                    "context"
                ],

                claim_extractor=claim_extractor,

                judge_llm=judge_llm
            )
        )


        # -----------------------------
        # Calculate metrics
        # -----------------------------

        metrics = (
            calculate_grounding_metrics(
                claim_results
            )
        )


        all_results.append({

            "id": example["id"],

            "question":
                example["question"],

            "response":
                response,

            "claim_results":
                claim_results,

            "metrics":
                metrics
        })

    return all_results
```

---

# 23. Example production output

```json
{
  "total_examples": 1000,
  "total_claims": 4520,
  "supported_claims": 4180,
  "contradicted_claims": 120,
  "unsupported_claims": 220,
  "hallucination_rate": 0.075,
  "faithfulness": 0.925,
  "correct_abstention_rate": 0.91,
  "citation_precision": 0.96
}
```

This is much more meaningful than simply saying:

```text
The model looks good.
```

---

# 24. Important distinction: RAG hallucination vs model hallucination

## RAG hallucination

The question is:

> Did the answer stay faithful to the retrieved documents?

```text
Retrieved Context
       ↓
Answer
       ↓
Faithfulness evaluation
```

Metrics:

* groundedness
* faithfulness
* context precision
* citation correctness
* unsupported claim rate

---

## General model hallucination

The question is:

> Is the answer factually true in the real world?

```text
Question
    ↓
Answer
    ↓
Ground truth / trusted sources
    ↓
Fact verification
```

This is harder because the evaluator needs reliable external ground truth.

For a closed enterprise system, RAG hallucination evaluation is usually more controllable because you have a defined evidence set.

---

# Interview-ready answer

> **I evaluate hallucinations at the claim level rather than marking an entire response as simply correct or incorrect. First, I split the answer into atomic factual claims. For each claim, I check whether it is supported, contradicted, or unsupported by the ground-truth context.**
>
> **For deterministic facts, I use exact or structured validation. For semantic grounding, I use NLI models or an LLM-as-a-judge. I calculate metrics such as hallucination rate, faithfulness, groundedness, contradiction rate, unsupported claim rate, citation correctness, and correct abstention rate.**
>
> **For a RAG system, I specifically measure whether generated claims are entailed by retrieved documents. I also test adversarial cases where the answer is not present in the context, because a good model should abstain instead of inventing information. Finally, I compare the base model and fine-tuned model on the same frozen evaluation dataset.**

## The key production principle

```text
Don't ask:

"Is the answer generally good?"

Ask:

For every factual claim:

Does evidence support it?

        │
        ├── Yes → SUPPORTED
        ├── No, opposite evidence → CONTRADICTED
        └── No evidence → UNSUPPORTED / HALLUCINATION
```

For an enterprise RAG system, this claim-level approach is one of the strongest ways to measure and monitor hallucinations.
