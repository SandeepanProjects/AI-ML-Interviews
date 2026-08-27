# How Do You Perform Human Evaluation for an LLM?

Human evaluation means **real people review model outputs using a defined rubric**.

It is important because automatic metrics such as:

* BLEU
* ROUGE
* Perplexity
* BERTScore
* LLM-as-a-Judge

cannot perfectly evaluate qualities such as:

* usefulness
* naturalness
* instruction following
* safety
* domain correctness
* hallucinations
* user satisfaction

A production evaluation system usually looks like:

```text
                    Evaluation Dataset
                           │
                           ▼
                      Test Prompts
                           │
                           ▼
                ┌─────────────────────┐
                │    Model A          │
                │    Model B          │
                │    Model C          │
                └─────────────────────┘
                           │
                           ▼
                    Blind Evaluation
                           │
                           ▼
                  Human Annotators
                           │
             ┌─────────────┼─────────────┐
             ▼             ▼             ▼
       Correctness      Helpfulness   Safety
             │             │             │
             └─────────────┼─────────────┘
                           ▼
                    Quality Metrics
                           │
                           ▼
                   Model Selection
```

---

# 1. Why do we need human evaluation?

Consider this answer:

### Prompt

```text
Explain RAG to a beginner.
```

### Model A

```text
RAG is Retrieval-Augmented Generation.
It retrieves relevant information and gives it to an LLM.
```

### Model B

```text
Imagine an open-book exam. Before answering, the model first
looks up relevant pages and then uses those pages to answer.
```

Both might be factually correct.

But humans may prefer Model B because it is:

* easier to understand
* more engaging
* more appropriate for a beginner

Traditional automatic metrics may not reliably capture this.

---

# 2. Define what humans should evaluate

Do not give annotators only:

```text
Is this answer good?
```

That produces inconsistent results.

Instead define a rubric.

Example:

| Metric                | Question                                         |
| --------------------- | ------------------------------------------------ |
| Correctness           | Is the information factually correct?            |
| Groundedness          | Is the answer supported by the provided context? |
| Helpfulness           | Does the answer solve the user's problem?        |
| Instruction Following | Did the model follow the instructions?           |
| Relevance             | Is the answer relevant to the question?          |
| Clarity               | Is the answer easy to understand?                |
| Safety                | Does the answer avoid harmful behavior?          |

A typical 1–5 scoring system:

```text
5 = Excellent
4 = Good
3 = Acceptable
2 = Poor
1 = Very Poor
```

---

# 3. Build a human evaluation dataset

A good evaluation dataset should be **held out** from training.

```python
evaluation_dataset = [
    {
        "id": "eval_001",
        "prompt": "Explain RAG to a beginner.",
        "category": "general",
        "difficulty": "easy"
    },
    {
        "id": "eval_002",
        "prompt": """
Explain how to reset a password using the provided policy.
""",
        "category": "customer_support",
        "difficulty": "medium"
    }
]
```

For RAG evaluation:

```python
evaluation_dataset = [
    {
        "id": "rag_001",

        "context": """
The refund process takes 5 to 7 business days.
Refunds are returned to the original payment method.
""",

        "question": """
How long does a refund take?
"""
    }
]
```

---

# 4. Generate outputs from multiple models

Suppose we compare:

```text
Model A → Base model
Model B → Fine-tuned model
```

```python
def generate_model_outputs(
    models,
    dataset,
    tokenizer
):

    results = []

    for example in dataset:

        for model_name, model in models.items():

            prompt = example["prompt"]

            response = generate_response(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt
            )

            results.append({
                "example_id": example["id"],
                "model": model_name,
                "prompt": prompt,
                "response": response
            })

    return results
```

Example output:

```python
[
    {
        "example_id": "eval_001",
        "model": "base_model",
        "response": "..."
    },
    {
        "example_id": "eval_001",
        "model": "fine_tuned_model",
        "response": "..."
    }
]
```

---

# 5. Blind the evaluation

This is extremely important.

Do not show annotators:

```text
Model A: GPT-X
Model B: Fine-Tuned Model
```

Humans may be biased.

Instead:

```text
Response X
Response Y
```

Randomize which model is X or Y.

---

## Code for anonymization

```python
import random


def create_blind_evaluation(
    model_outputs
):

    grouped = {}

    for item in model_outputs:

        example_id = item["example_id"]

        if example_id not in grouped:
            grouped[example_id] = []

        grouped[example_id].append(item)


    blind_tasks = []

    for example_id, outputs in grouped.items():

        shuffled_outputs = outputs.copy()

        random.shuffle(
            shuffled_outputs
        )

        responses = []

        for index, item in enumerate(
            shuffled_outputs
        ):

            responses.append({
                "response_id":
                    f"response_{index + 1}",

                "response":
                    item["response"]
            })


        blind_tasks.append({

            "example_id": example_id,

            "prompt":
                shuffled_outputs[0]["prompt"],

            "responses":
                responses,

            # Keep hidden mapping internally
            "hidden_mapping": {
                f"response_{i + 1}":
                    item["model"]
                for i, item in enumerate(
                    shuffled_outputs
                )
            }
        })

    return blind_tasks
```

The evaluator sees:

```text
Prompt:
Explain RAG to a beginner.

Response A:
...

Response B:
...
```

But does **not** know which model generated which answer.

---

# 6. Pointwise evaluation

In pointwise evaluation, humans score one answer independently.

Example evaluation form:

```text
PROMPT:
Explain RAG.

RESPONSE:
RAG retrieves relevant information and provides it to the LLM.

Score:

Correctness:           1 2 3 4 5
Helpfulness:           1 2 3 4 5
Clarity:               1 2 3 4 5
Instruction Following: 1 2 3 4 5

Comments:
______________________
```

Data model:

```python
from pydantic import BaseModel
from typing import Optional


class HumanEvaluation(BaseModel):

    example_id: str

    response_id: str

    evaluator_id: str

    correctness: int

    helpfulness: int

    clarity: int

    instruction_following: int

    comments: Optional[str] = None
```

Validate scores:

```python
from pydantic import Field


class HumanEvaluation(BaseModel):

    example_id: str

    response_id: str

    evaluator_id: str

    correctness: int = Field(
        ge=1,
        le=5
    )

    helpfulness: int = Field(
        ge=1,
        le=5
    )

    clarity: int = Field(
        ge=1,
        le=5
    )

    instruction_following: int = Field(
        ge=1,
        le=5
    )
```

This prevents invalid data such as:

```text
correctness = 10
```

---

# 7. Calculate average human scores

Suppose:

```python
evaluations = [
    {
        "model": "base",
        "correctness": 4,
        "helpfulness": 3,
        "clarity": 4
    },
    {
        "model": "fine_tuned",
        "correctness": 5,
        "helpfulness": 5,
        "clarity": 5
    }
]
```

Using pandas:

```python
import pandas as pd


df = pd.DataFrame(
    evaluations
)


scores = (
    df
    .groupby("model")
    [
        [
            "correctness",
            "helpfulness",
            "clarity"
        ]
    ]
    .mean()
)

print(scores)
```

Example:

```text
             correctness  helpfulness  clarity
base                 4.1          3.5      4.0
fine_tuned           4.7          4.6      4.5
```

---

# 8. Pairwise human evaluation

Often, pairwise evaluation is easier and more reliable.

Instead of asking:

```text
Rate this answer from 1 to 5.
```

Ask:

```text
Which answer is better?
```

Example:

```text
Prompt:
Explain vector embeddings.

Response A:
Embeddings convert information into numerical vectors.

Response B:
An embedding represents meaning as numbers, allowing
semantically similar items to be located near each other.

Which is better?

[ ] A
[ ] B
[ ] Tie
```

Why pairwise evaluation is useful:

* easier for humans
* less scoring inconsistency
* useful for model comparisons
* used in preference datasets

---

## Data model

```python
class PairwiseEvaluation(BaseModel):

    example_id: str

    evaluator_id: str

    winner: str
    # "A", "B", or "tie"

    reason: Optional[str] = None
```

Calculate win rate:

```python
def calculate_win_rate(
    evaluations
):

    model_a_wins = 0
    model_b_wins = 0
    ties = 0

    for evaluation in evaluations:

        if evaluation["winner"] == "A":
            model_a_wins += 1

        elif evaluation["winner"] == "B":
            model_b_wins += 1

        else:
            ties += 1


    total = len(evaluations)

    return {

        "model_a_win_rate":
            model_a_wins / total,

        "model_b_win_rate":
            model_b_wins / total,

        "tie_rate":
            ties / total
    }
```

Example:

```text
Model A wins: 35%
Model B wins: 55%
Ties:         10%
```

Model B is preferred.

---

# 9. Human evaluation for hallucinations

Suppose the system is a RAG application.

Give the evaluator:

```text
QUESTION:
What is the refund period?

CONTEXT:
Refunds take between 5 and 7 business days.

MODEL ANSWER:
Refunds typically take 2 business days.
```

Ask:

```text
1. Is the answer supported by the context?

[ ] Fully supported
[ ] Partially supported
[ ] Unsupported
[ ] Contradicted

2. Does the answer contain hallucinated information?

[ ] Yes
[ ] No

3. Severity:

[ ] Low
[ ] Medium
[ ] High
```

Data model:

```python
class HallucinationEvaluation(BaseModel):

    example_id: str

    evaluator_id: str

    support_level: str

    hallucination_detected: bool

    severity: Optional[str] = None
```

Metrics:

```python
def calculate_human_hallucination_rate(
    evaluations
):

    hallucinations = sum(
        item["hallucination_detected"]
        for item in evaluations
    )

    total = len(evaluations)

    return (
        hallucinations / total
        if total > 0
        else 0
    )
```

---

# 10. Multiple human evaluators

One evaluator is not enough for important systems.

Suppose three annotators evaluate:

```text
Evaluator 1 → Score 5
Evaluator 2 → Score 4
Evaluator 3 → Score 5
```

Final score:

```python
import statistics


scores = [5, 4, 5]

average_score = (
    statistics.mean(scores)
)

print(average_score)
```

Output:

```text
4.67
```

But average score alone is not enough.

You must measure **inter-annotator agreement**.

---

# 11. What is inter-annotator agreement?

It measures:

> Do humans agree with each other?

Example:

```text
Response:

Evaluator 1 → Excellent
Evaluator 2 → Excellent
Evaluator 3 → Excellent
```

High agreement.

Another:

```text
Evaluator 1 → 5
Evaluator 2 → 1
Evaluator 3 → 3
```

Low agreement.

Low agreement could mean:

* ambiguous rubric
* poor evaluator training
* subjective task
* unclear instructions

---

# 12. Cohen's Kappa

For two evaluators, Cohen's Kappa is commonly used.

Install:

```bash
pip install scikit-learn
```

Code:

```python
from sklearn.metrics import cohen_kappa_score


annotator_1 = [
    5, 4, 3, 5, 2
]

annotator_2 = [
    5, 4, 4, 5, 2
]


score = cohen_kappa_score(
    annotator_1,
    annotator_2
)

print(score)
```

Interpretation roughly:

```text
1.0     Perfect agreement
0.8+    Strong agreement
0.6+    Moderate agreement
0.4+    Weak agreement
0.0     Agreement no better than chance
```

For ordinal ratings such as 1–5, use weighted kappa:

```python
score = cohen_kappa_score(
    annotator_1,
    annotator_2,
    weights="quadratic"
)
```

Weighted kappa understands that:

```text
4 vs 5
```

is a smaller disagreement than:

```text
1 vs 5
```

---

# 13. Fleiss' Kappa for multiple evaluators

If three or more people evaluate the same examples:

```text
Example 1:
Evaluator 1 → Good
Evaluator 2 → Good
Evaluator 3 → Excellent
```

You can use Fleiss' Kappa.

Conceptually:

```text
2 annotators → Cohen's Kappa

3+ annotators → Fleiss' Kappa
```

In practice, teams may also use:

* Krippendorff's alpha
* weighted agreement
* majority vote

depending on the annotation setup.

---

# 14. Human evaluation workflow in a real company

A production workflow could be:

```text
Step 1
Collect production-like prompts
       │
       ▼
Step 2
Create a frozen evaluation dataset
       │
       ▼
Step 3
Generate answers from candidate models
       │
       ▼
Step 4
Randomize and anonymize responses
       │
       ▼
Step 5
Give responses to 2–3 evaluators
       │
       ▼
Step 6
Collect rubric scores
       │
       ▼
Step 7
Calculate agreement
       │
       ▼
Step 8
Resolve disagreements
       │
       ▼
Step 9
Calculate model metrics
       │
       ▼
Step 10
Choose or reject model
```

---

# 15. Disagreement resolution

Suppose:

```text
Evaluator 1 → 5
Evaluator 2 → 1
Evaluator 3 → 2
```

You should not silently average.

A common approach:

```text
Evaluator disagreement
        │
        ▼
High disagreement?
        │
    ┌───┴────┐
    │        │
   No       Yes
    │        │
    ▼        ▼
 Aggregate  Senior reviewer
             │
             ▼
         Final decision
```

Code:

```python
import statistics


def needs_adjudication(
    scores,
    threshold=2
):

    score_range = (
        max(scores)
        - min(scores)
    )

    return (
        score_range >= threshold
    )
```

Example:

```python
scores = [5, 1, 2]

print(
    needs_adjudication(scores)
)
```

Output:

```text
True
```

---

# 16. Build a complete evaluation result

A useful evaluation record:

```python
evaluation_record = {

    "example_id": "001",

    "model_version": "v2",

    "response_id": "anonymous_A",

    "evaluations": [

        {
            "annotator_id": "annotator_1",

            "correctness": 5,

            "helpfulness": 4,

            "instruction_following": 5,

            "hallucination": False
        },

        {
            "annotator_id": "annotator_2",

            "correctness": 4,

            "helpfulness": 5,

            "instruction_following": 5,

            "hallucination": False
        }
    ]
}
```

Aggregate:

```python
def aggregate_human_scores(
    evaluations
):

    metrics = [
        "correctness",
        "helpfulness",
        "instruction_following"
    ]

    result = {}

    for metric in metrics:

        values = [
            evaluation[metric]
            for evaluation in evaluations
        ]

        result[metric] = {
            "mean": (
                sum(values)
                / len(values)
            ),

            "min": min(values),

            "max": max(values)
        }

    return result
```

Usage:

```python
result = aggregate_human_scores(
    evaluation_record["evaluations"]
)

print(result)
```

---

# 17. Use human evaluation to create a preference dataset

Human evaluation can later be used for DPO or preference tuning.

Suppose:

```text
Prompt:
How do I reset my password?

Response A:
Click Settings → Security → Reset Password.

Response B:
I am not sure.
```

Human chooses:

```text
Response A
```

Then create:

```json
{
  "prompt": "How do I reset my password?",
  "chosen": "Click Settings → Security → Reset Password.",
  "rejected": "I am not sure."
}
```

This becomes a preference dataset:

```text
Human evaluation
       │
       ▼
Chosen / Rejected pairs
       │
       ▼
DPO / Preference tuning
```

Example JSONL:

```json
{"prompt":"Explain RAG","chosen":"RAG retrieves relevant context before generation.","rejected":"RAG is an unknown technology."}
```

This is one of the practical connections between:

```text
Human Evaluation
       ↓
Preference Data
       ↓
DPO / RLHF
       ↓
Improved Model
```

---

# 18. Human evaluation vs LLM-as-a-Judge

| Aspect           | Human Evaluation         | LLM Judge              |
| ---------------- | ------------------------ | ---------------------- |
| Quality          | High                     | Variable               |
| Cost             | High                     | Low                    |
| Speed            | Slow                     | Fast                   |
| Scalability      | Limited                  | High                   |
| Nuanced judgment | Excellent                | Good                   |
| Bias             | Human bias               | Model bias             |
| Domain expertise | Excellent with experts   | Depends on model       |
| Best use         | Gold-standard evaluation | Large-scale evaluation |

The best production approach is:

```text
                    Human Evaluation
                         │
                         ▼
                  Gold Evaluation Set
                         │
                         ▼
                  Calibrate LLM Judge
                         │
                         ▼
                Large-scale Evaluation
                         │
                         ▼
              Periodic Human Re-evaluation
```

---

# 19. Production architecture example

For an enterprise RAG system:

```text
                 Production Queries
                        │
                        ▼
                  Sampling Layer
                        │
                        ▼
                Evaluation Dataset
                        │
             ┌──────────┴──────────┐
             ▼                     ▼
        Automated Eval         Human Eval
             │                     │
             │              Domain Experts
             │                     │
             └──────────┬──────────┘
                        ▼
                 Evaluation Store
                        │
                        ▼
              Metrics Dashboard
                        │
                        ├── Correctness
                        ├── Faithfulness
                        ├── Helpfulness
                        ├── Hallucination
                        ├── Safety
                        └── User preference
```

A record might be stored in PostgreSQL like:

```python
class EvaluationResult(BaseModel):

    prompt_id: str

    model_version: str

    evaluator_type: str

    correctness: float | None

    helpfulness: float | None

    groundedness: float | None

    hallucination_detected: bool | None

    comments: str | None
```

---

# 20. A complete practical example

Let's say you have a customer-support LLM.

## Step 1: Test dataset

```python
test_cases = [
    {
        "id": "001",

        "question": """
How long does a refund take?
""",

        "context": """
Refunds take between 5 and 7 business days.
"""
    }
]
```

## Step 2: Generate response

```python
response = """
Refunds usually take between
5 and 7 business days.
"""
```

## Step 3: Human evaluation form

```python
human_evaluation = {

    "example_id": "001",

    "correctness": 5,

    "groundedness": 5,

    "helpfulness": 5,

    "instruction_following": 5,

    "hallucination": False,

    "comments": """
Correctly used the provided policy.
"""
}
```

## Step 4: Store multiple evaluator results

```python
all_evaluations = [
    human_evaluation,

    {
        "example_id": "001",

        "correctness": 5,

        "groundedness": 5,

        "helpfulness": 4,

        "instruction_following": 5,

        "hallucination": False
    }
]
```

## Step 5: Aggregate

```python
def calculate_average_metric(
    evaluations,
    metric
):

    values = [
        item[metric]
        for item in evaluations
    ]

    return sum(values) / len(values)


for metric in [
    "correctness",
    "groundedness",
    "helpfulness",
    "instruction_following"
]:

    score = calculate_average_metric(
        all_evaluations,
        metric
    )

    print(
        f"{metric}: {score:.2f}"
    )
```

Output:

```text
correctness: 5.00
groundedness: 5.00
helpfulness: 4.50
instruction_following: 5.00
```

---

# Interview-ready answer

> **I perform human evaluation by first creating a representative held-out evaluation dataset and defining a detailed annotation rubric. I evaluate dimensions such as correctness, helpfulness, instruction following, groundedness, clarity, safety, and hallucination rate.**
>
> **I use blind evaluation so annotators don't know which model generated an answer. For absolute quality, I use pointwise 1–5 scoring. For comparing models, I prefer pairwise evaluation where humans select the better response or declare a tie.**
>
> **For reliability, I use multiple annotators and measure inter-annotator agreement using metrics such as Cohen's Kappa or Krippendorff's Alpha. Cases with high disagreement go through adjudication.**
>
> **In production, I treat human judgments as the gold standard, use them to validate automated metrics and LLM-as-a-Judge systems, and periodically re-evaluate production samples. The same chosen-versus-rejected human preferences can also be converted into datasets for DPO or other preference-tuning approaches.**

## Key takeaway

```text
Human evaluation is not:

"Ask someone whether the answer is good."

Human evaluation is:

Representative test set
        +
Clear rubric
        +
Blind evaluation
        +
Multiple annotators
        +
Agreement measurement
        +
Adjudication
        +
Statistical analysis
```

For production LLM systems, **human evaluation is usually the final gold standard for determining whether model quality actually improved**.
