# How do you compare a Base Model against a Fine-Tuned Model?

When you fine-tune an LLM, the key question is:

> **Did the fine-tuned model actually become better for the target task without becoming worse in important ways?**

You should **not compare only training loss**.

A proper comparison looks like this:

```text
                 Same held-out test dataset
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
         Base Model               Fine-tuned Model
              │                         │
              ▼                         ▼
          Responses                 Responses
              │                         │
              └────────────┬────────────┘
                           ▼
                  Evaluation Pipeline
                           │
         ┌─────────────────┼──────────────────┐
         ▼                 ▼                  ▼
   Automatic Metrics   LLM-as-Judge      Human Evaluation
         │                 │                  │
         └─────────────────┼──────────────────┘
                           ▼
                    Final Comparison
```

The important rule is:

> **Both models must be evaluated on exactly the same held-out examples and under comparable inference settings.**

---

# 1. What should you compare?

The metrics depend on the task.

## General instruction-following model

Compare:

* instruction-following rate
* correctness
* helpfulness
* relevance
* hallucination rate
* safety

## RAG model

Compare:

* groundedness
* faithfulness
* context utilization
* hallucination rate
* answer relevance

## Classification model

Compare:

* accuracy
* precision
* recall
* F1-score

## SQL generation

Compare:

* exact match
* SQL validity
* execution accuracy

## Code generation

Compare:

* compilation rate
* unit-test pass rate
* functional correctness

For a general fine-tuned LLM, I recommend:

```text
Task performance
      +
LLM Judge
      +
Human evaluation
      +
Latency
      +
Cost
      +
Regression testing
```

---

# 2. Create a held-out test dataset

Do **not** evaluate on training data.

Example:

```python
test_dataset = [
    {
        "id": "001",
        "prompt": "How do I reset my password?"
    },
    {
        "id": "002",
        "prompt": "Explain what RAG is in simple terms."
    },
    {
        "id": "003",
        "prompt": "Write exactly three bullet points about vector databases."
    }
]
```

For instruction tuning, the dataset can contain expected constraints:

```python
test_dataset = [
    {
        "id": "003",
        "prompt": "Write exactly three bullet points about vector databases.",
        "requirements": {
            "format": "bullets",
            "bullet_count": 3
        }
    }
]
```

### Important

The test dataset should be:

```text
Not used in training
        +
Not used for hyperparameter tuning
        +
Representative of production
```

---

# 3. Load the base and fine-tuned models

Suppose you fine-tuned using LoRA.

```text
Base Model
    +
LoRA Adapter
    =
Fine-tuned Model
```

## Load the base model

```python
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

MODEL_NAME = "your-base-model"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

base_model.eval()
```

---

## Load the fine-tuned LoRA model

```python
from peft import PeftModel


fine_tuned_model = PeftModel.from_pretrained(
    base_model,
    "path/to/lora_adapter"
)

fine_tuned_model.eval()
```

### Important warning

The above reuses `base_model`.

For a fair comparison, it is often clearer to load separate model instances:

```python
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

fine_tuned_base = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

fine_tuned_model = PeftModel.from_pretrained(
    fine_tuned_base,
    "path/to/lora_adapter"
)
```

Now:

```text
base_model          → Original model

fine_tuned_model    → Base + LoRA adapter
```

This is easier to reason about.

---

# 4. Use the same generation settings

This is critical.

If you use:

```python
max_new_tokens=100
temperature=0
```

for the base model, use the same settings for the fine-tuned model.

Create one function:

```python
def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256
):

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():

        outputs = model.generate(

            **inputs,

            max_new_tokens=max_new_tokens,

            do_sample=False,

            temperature=None
        )

    generated_tokens = outputs[
        0,
        inputs["input_ids"].shape[1]:
    ]

    return tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    )
```

We use:

```python
do_sample=False
```

for deterministic evaluation.

---

# 5. Generate responses from both models

```python
def compare_models(
    base_model,
    fine_tuned_model,
    tokenizer,
    dataset
):

    results = []

    for example in dataset:

        prompt = example["prompt"]

        base_answer = generate_response(
            model=base_model,
            tokenizer=tokenizer,
            prompt=prompt
        )

        fine_tuned_answer = generate_response(
            model=fine_tuned_model,
            tokenizer=tokenizer,
            prompt=prompt
        )

        results.append({
            "id": example["id"],
            "prompt": prompt,
            "base_answer": base_answer,
            "fine_tuned_answer": fine_tuned_answer
        })

    return results
```

Usage:

```python
results = compare_models(
    base_model=base_model,
    fine_tuned_model=fine_tuned_model,
    tokenizer=tokenizer,
    dataset=test_dataset
)
```

Example:

```python
{
    "id": "003",
    "prompt": "Write exactly three bullet points about vector databases.",

    "base_answer": """
Vector databases store embeddings.

They are useful for AI.

They support similarity search.
""",

    "fine_tuned_answer": """
- Store vector embeddings efficiently.
- Perform similarity and nearest-neighbor search.
- Support retrieval for AI applications.
"""
}
```

The fine-tuned model may better follow the desired format.

---

# 6. Compare exact instruction following

For deterministic requirements, use code instead of an LLM judge.

Example requirement:

```text
Exactly 3 bullet points
```

```python
def check_three_bullets(answer: str) -> bool:

    lines = [
        line.strip()
        for line in answer.splitlines()
        if line.strip()
    ]

    bullets = [
        line
        for line in lines
        if line.startswith(
            ("-", "*", "•")
        )
    ]

    return len(bullets) == 3
```

Evaluate:

```python
for item in results:

    base_pass = check_three_bullets(
        item["base_answer"]
    )

    fine_tuned_pass = check_three_bullets(
        item["fine_tuned_answer"]
    )

    print(
        item["id"],
        "Base:",
        base_pass,
        "Fine-tuned:",
        fine_tuned_pass
    )
```

Calculate pass rate:

```python
def calculate_pass_rate(
    results,
    answer_key,
    evaluator
):

    passed = 0

    for item in results:

        answer = item[answer_key]

        if evaluator(answer):
            passed += 1

    return passed / len(results)
```

Usage:

```python
base_pass_rate = calculate_pass_rate(
    results,
    answer_key="base_answer",
    evaluator=check_three_bullets
)

fine_tuned_pass_rate = calculate_pass_rate(
    results,
    answer_key="fine_tuned_answer",
    evaluator=check_three_bullets
)

print("Base:", base_pass_rate)
print("Fine-tuned:", fine_tuned_pass_rate)
```

Example:

```text
Base Model:       62%
Fine-tuned Model: 91%
```

---

# 7. Compare using task-specific metrics

Suppose the model generates SQL.

Test example:

```python
sql_test_data = [
    {
        "question":
            "Find all users older than 30.",

        "expected_sql":
            "SELECT * FROM users WHERE age > 30"
    }
]
```

A simple normalization function:

```python
def normalize_sql(sql: str):

    return (
        sql.lower()
        .strip()
        .rstrip(";")
    )
```

Exact-match accuracy:

```python
def sql_exact_match(
    prediction,
    expected
):

    return (
        normalize_sql(prediction)
        ==
        normalize_sql(expected)
    )
```

However, exact string matching is limited.

These can be equivalent:

```sql
SELECT * FROM users WHERE age > 30
```

and:

```sql
SELECT *
FROM users
WHERE age > 30;
```

A stronger evaluation is:

```text
Generated SQL
      │
      ▼
Execute on test database
      │
      ▼
Compare results
      │
      ▼
Execution Accuracy
```

That is usually more meaningful than BLEU/ROUGE for SQL.

---

# 8. Compare with LLM-as-a-Judge

For open-ended outputs, use a judge.

Pairwise evaluation:

```python
PAIRWISE_PROMPT = """
You are an impartial evaluator.

Compare two answers to the same user question.

QUESTION:
{question}

ANSWER A:
{answer_a}

ANSWER B:
{answer_b}

Evaluate:

1. Correctness
2. Helpfulness
3. Relevance
4. Instruction following
5. Clarity

Do not reward unnecessary verbosity.

Return JSON only:

{{
    "winner": "A | B | TIE",
    "reason": ""
}}
"""
```

Pydantic schema:

```python
from typing import Literal
from pydantic import BaseModel


class PairwiseResult(BaseModel):

    winner: Literal[
        "A",
        "B",
        "TIE"
    ]

    reason: str
```

Judge function:

```python
import json


def judge_pair(
    judge_client,
    question,
    answer_a,
    answer_b
):

    prompt = PAIRWISE_PROMPT.format(
        question=question,
        answer_a=answer_a,
        answer_b=answer_b
    )

    raw_response = judge_client.generate(
        prompt=prompt,
        temperature=0
    )

    data = json.loads(
        raw_response
    )

    return PairwiseResult.model_validate(
        data
    )
```

---

# 9. Run the comparison

For each test example:

```python
def evaluate_pairwise(
    results,
    judge_client
):

    evaluations = []

    for item in results:

        judgment = judge_pair(

            judge_client=judge_client,

            question=item["prompt"],

            answer_a=item["base_answer"],

            answer_b=item["fine_tuned_answer"]
        )

        evaluations.append({

            "id": item["id"],

            "winner":
                judgment.winner,

            "reason":
                judgment.reason
        })

    return evaluations
```

Remember:

```text
A = Base Model

B = Fine-tuned Model
```

---

# 10. Calculate win rate

```python
def calculate_win_rate(
    evaluations
):

    counts = {
        "base": 0,
        "fine_tuned": 0,
        "tie": 0
    }

    for item in evaluations:

        winner = item["winner"]

        if winner == "A":

            counts["base"] += 1

        elif winner == "B":

            counts["fine_tuned"] += 1

        else:

            counts["tie"] += 1


    total = len(evaluations)

    return {

        "base_win_rate":
            counts["base"] / total,

        "fine_tuned_win_rate":
            counts["fine_tuned"] / total,

        "tie_rate":
            counts["tie"] / total
    }
```

Example:

```text
Base Model wins:        20%
Fine-tuned Model wins:  70%
Tie:                    10%
```

This suggests the fine-tuned model improved.

But that is not the complete story.

---

# 11. Check for catastrophic forgetting

Fine-tuning can improve the target domain while damaging general capabilities.

Example:

```text
Before fine-tuning:

General QA Score:      85%
Customer Support:      60%

After fine-tuning:

General QA Score:      70%
Customer Support:      90%
```

The model improved at the target task but regressed significantly on general tasks.

Therefore use two evaluation sets:

```text
                 Fine-tuned Model
                       │
          ┌────────────┴─────────────┐
          ▼                          ▼
    Target Domain                General Tasks
          │                          │
          ▼                          ▼
    Task Improvement            Regression Check
```

Example:

```python
target_dataset = [
    {
        "prompt":
            "How do I reset my company password?"
    }
]

general_dataset = [
    {
        "prompt":
            "Explain photosynthesis."
    },
    {
        "prompt":
            "What is recursion?"
    }
]
```

Evaluate both models on both datasets.

```python
base_target_score = 0.60
fine_tuned_target_score = 0.90

base_general_score = 0.85
fine_tuned_general_score = 0.83
```

This is a good outcome:

```text
Target task:
+30%

General capability:
-2%
```

But:

```text
Target task:
+30%

General capability:
-25%
```

might indicate unacceptable catastrophic forgetting.

---

# 12. Compare latency

Model quality is not the only production metric.

Measure:

```python
import time


def generate_with_latency(
    model,
    tokenizer,
    prompt
):

    start = time.perf_counter()

    answer = generate_response(
        model,
        tokenizer,
        prompt
    )

    latency = (
        time.perf_counter()
        - start
    )

    return answer, latency
```

Run multiple times and calculate:

* average latency
* p50 latency
* p95 latency
* tokens per second

Example:

```python
import numpy as np


latencies = [
    0.9,
    1.1,
    1.0,
    2.0,
    1.2
]

p50 = np.percentile(
    latencies,
    50
)

p95 = np.percentile(
    latencies,
    95
)

print("P50:", p50)
print("P95:", p95)
```

Production comparison:

| Metric             | Base | Fine-tuned |
| ------------------ | ---: | ---------: |
| Task Quality       |  62% |        91% |
| General Quality    |  85% |        83% |
| Hallucination Rate |  12% |         5% |
| P50 Latency        | 1.1s |       1.2s |
| P95 Latency        | 2.3s |       2.5s |

This gives a much better picture than training loss alone.

---

# 13. Compare hallucination rate

For RAG or factual tasks:

```python
def calculate_hallucination_rate(
    evaluations
):

    hallucinations = sum(

        item[
            "hallucination_detected"
        ]

        for item in evaluations
    )

    return (
        hallucinations
        /
        len(evaluations)
    )
```

Example:

```text
Base Model:

12 hallucinations / 100 answers
= 12%

Fine-tuned Model:

5 hallucinations / 100 answers
= 5%
```

Fine-tuning improved factual behavior.

---

# 14. Compare training and validation loss

This is useful but not sufficient.

```text
Epoch

        Base fine-tuning
              │
              ▼

Training Loss:
3.2 → 2.0 → 1.2

Validation Loss:
3.0 → 2.1 → 2.3
```

This may indicate overfitting.

Code:

```python
def find_best_epoch(
    validation_losses
):

    best_epoch = min(
        range(
            len(validation_losses)
        ),
        key=lambda i:
            validation_losses[i]
    )

    return best_epoch + 1
```

Example:

```python
validation_losses = [
    2.8,
    2.1,
    1.9,
    2.0,
    2.3
]

best_epoch = find_best_epoch(
    validation_losses
)

print(best_epoch)
```

Output:

```text
3
```

But a lower validation loss does not automatically mean better user experience.

That is why we also use task evaluation.

---

# 15. A complete evaluation pipeline

Here is a simplified production-style architecture:

```python
def evaluate_model(
    model,
    tokenizer,
    dataset,
    answer_key
):

    outputs = []

    for example in dataset:

        answer = generate_response(
            model=model,
            tokenizer=tokenizer,
            prompt=example["prompt"]
        )

        outputs.append({

            "id": example["id"],

            "prompt":
                example["prompt"],

            answer_key:
                answer
        })

    return outputs
```

Generate outputs:

```python
base_outputs = evaluate_model(

    model=base_model,

    tokenizer=tokenizer,

    dataset=test_dataset,

    answer_key="base_answer"
)


fine_tuned_outputs = evaluate_model(

    model=fine_tuned_model,

    tokenizer=tokenizer,

    dataset=test_dataset,

    answer_key="fine_tuned_answer"
)
```

Merge results:

```python
def merge_outputs(
    base_outputs,
    fine_tuned_outputs
):

    merged = []

    for base, fine_tuned in zip(
        base_outputs,
        fine_tuned_outputs
    ):

        merged.append({

            "id":
                base["id"],

            "prompt":
                base["prompt"],

            "base_answer":
                base["base_answer"],

            "fine_tuned_answer":
                fine_tuned[
                    "fine_tuned_answer"
                ]
        })

    return merged
```

Now evaluate:

```text
merged results
       │
       ├── Rule-based evaluation
       │
       ├── Task-specific evaluation
       │
       ├── LLM-as-a-Judge
       │
       ├── Human evaluation
       │
       └── Regression testing
```

---

# 16. Production comparison report

I would create a final report like:

```python
comparison_report = {

    "base_model": {
        "target_task_score": 0.62,
        "general_score": 0.85,
        "hallucination_rate": 0.12,
        "avg_latency_seconds": 1.1
    },

    "fine_tuned_model": {
        "target_task_score": 0.91,
        "general_score": 0.83,
        "hallucination_rate": 0.05,
        "avg_latency_seconds": 1.2
    }
}
```

Then calculate improvement:

```python
def percentage_change(
    old,
    new
):

    return (
        (new - old)
        / old
    ) * 100
```

```python
improvement = percentage_change(
    0.62,
    0.91
)

print(
    f"Improvement: "
    f"{improvement:.2f}%"
)
```

---

# 17. How would I decide whether to deploy?

I would define acceptance criteria **before running the evaluation**.

Example:

```python
deployment_criteria = {

    "minimum_target_score":
        0.85,

    "maximum_general_regression":
        0.05,

    "maximum_hallucination_rate":
        0.05,

    "minimum_fine_tuned_win_rate":
        0.55,

    "maximum_latency_increase":
        0.20
}
```

Decision function:

```python
def should_deploy(
    metrics
):

    if (
        metrics["target_score"]
        < 0.85
    ):
        return False

    if (
        metrics["general_regression"]
        > 0.05
    ):
        return False

    if (
        metrics["hallucination_rate"]
        > 0.05
    ):
        return False

    if (
        metrics["fine_tuned_win_rate"]
        < 0.55
    ):
        return False

    return True
```

This prevents deploying a model simply because:

```text
Fine-tuned model looks better in a few examples
```

---

# 18. Best evaluation strategy for your fine-tuning project

Since you are learning **LoRA/QLoRA, SFT, DPO, RAG, and production LLM systems**, I would evaluate your fine-tuned model using this combination:

```text
                         Test Dataset
                              │
             ┌────────────────┴────────────────┐
             ▼                                 ▼
        Base Model                       Fine-tuned Model
             │                                 │
             └────────────────┬────────────────┘
                              ▼
                       Same Prompts
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
      Exact Checks       LLM-as-Judge       Human Review
          │                   │                   │
          ▼                   ▼                   ▼
 Instruction Rate       Pairwise Wins      Expert Score
 JSON Validity          Correctness        Usefulness
 SQL Execution          Helpfulness        Safety
 Test Pass Rate         Groundedness       Accuracy
                              │
                              ▼
                     Regression Evaluation
                              │
                              ▼
                       Deployment Decision
```

## Interview-ready answer

> **To compare a base model and a fine-tuned model, I evaluate both models on the exact same held-out test set using identical prompts and inference settings. I first use task-specific metrics—for example exact match or execution accuracy for SQL, test pass rate for code, and constraint checks for structured outputs.**
>
> **For open-ended responses, I use pairwise LLM-as-a-Judge evaluation, where the judge compares anonymized outputs from both models and calculates win rate. I reduce position bias by randomizing or swapping answer order.**
>
> **I also perform human evaluation on a representative sample and compare correctness, helpfulness, instruction following, and groundedness. Finally, I check regression on general capabilities to detect catastrophic forgetting, as well as hallucination rate, latency, throughput, and cost.**
>
> **I define deployment thresholds before evaluation and deploy only if the fine-tuned model provides a statistically meaningful improvement on the target task without unacceptable regressions.**

### The key idea

```text
A fine-tuned model is NOT better simply because:

Training loss decreased
```

A fine-tuned model is better when:

```text
Held-out task performance ↑
        +
Instruction following ↑
        +
Human/LLM preference ↑
        +
Hallucination acceptable
        +
General capability regression acceptable
        +
Latency/cost acceptable
```

That is the production-level approach to comparing a **base LLM vs a fine-tuned LLM**.
