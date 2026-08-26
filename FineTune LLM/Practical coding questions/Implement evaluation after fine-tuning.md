# Implement Evaluation After Fine-Tuning

After fine-tuning an LLM, **do not evaluate it only by checking training loss**.

A proper evaluation pipeline should answer:

1. Did validation loss improve?
2. Is the fine-tuned model better than the base model?
3. Does it perform well on the actual business task?
4. Did it overfit?
5. Does it produce safe and correctly formatted outputs?

A production evaluation pipeline:

```text
Fine-tuned Model
      │
      ├── Automatic Evaluation
      │      ├── Validation Loss
      │      ├── Perplexity
      │      ├── Task Metrics
      │      └── Format Validation
      │
      ├── Compare with Base Model
      │
      ├── Qualitative Evaluation
      │
      └── Human / LLM Evaluation
```

---

# 1. Evaluate validation loss

The simplest evaluation is:

```python
metrics = trainer.evaluate()

print(metrics)
```

Example output:

```python
{
    "eval_loss": 1.24,
    "eval_runtime": 45.2,
    "eval_samples_per_second": 10.5
}
```

With `SFTTrainer`:

```python
from trl import SFTTrainer, SFTConfig


trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    processing_class=tokenizer,
    args=SFTConfig(
        output_dir="./output",
        eval_strategy="steps",
        eval_steps=100,
        logging_steps=10
    )
)


trainer.train()


metrics = trainer.evaluate()

print(metrics)
```

---

# 2. Calculate perplexity

Perplexity is:

$$
PPL = e^{Loss}
$$

```python
import math


metrics = trainer.evaluate()

eval_loss = metrics["eval_loss"]

perplexity = math.exp(eval_loss)


print("Validation Loss:", eval_loss)

print("Perplexity:", perplexity)
```

For example:

```text
Model A:
Loss = 2.0
PPL = 7.39

Model B:
Loss = 1.0
PPL = 2.72
```

Lower is generally better **when comparing models on the same evaluation data and tokenization setup**.

---

# 3. Evaluate individual examples

Suppose your validation dataset contains:

```python
validation_examples = [
    {
        "instruction": "How do I reset my password?",
        "expected": (
            "Go to Settings, select Forgot Password, "
            "and follow the verification steps."
        )
    },
    {
        "instruction": "How do I cancel my subscription?",
        "expected": (
            "Go to Account Settings and select Cancel Subscription."
        )
    }
]
```

Create a generation function:

```python
import torch


def generate_response(
    model,
    tokenizer,
    prompt,
    max_new_tokens=200
):

    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True
    )

    input_device = (
        model.get_input_embeddings()
        .weight
        .device
    )

    inputs = {
        key: value.to(input_device)
        for key, value in inputs.items()
    }

    with torch.inference_mode():

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
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

Run evaluation:

```python
for example in validation_examples:

    prediction = generate_response(
        model=model,
        tokenizer=tokenizer,
        prompt=example["instruction"]
    )

    print("\nPROMPT:")
    print(example["instruction"])

    print("\nEXPECTED:")
    print(example["expected"])

    print("\nPREDICTED:")
    print(prediction)
```

This is useful for **qualitative evaluation**, but not sufficient for a production benchmark.

---

# 4. Compare base model vs fine-tuned model

This is very important.

You should prove that fine-tuning improved the model.

```text
                    Same Evaluation Dataset
                              │
                 ┌────────────┴────────────┐
                 │                         │
                 ▼                         ▼

             Base Model             Fine-Tuned Model
                 │                         │
                 ▼                         ▼

               Scores                  Scores
                 │                         │
                 └────────────┬────────────┘
                              ▼

                         Comparison
```

Load both:

```python
from transformers import AutoModelForCausalLM
from peft import PeftModel


BASE_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

ADAPTER_PATH = "./customer-support-lora"


base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    device_map="auto"
)


fine_tuned_model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_PATH
)
```

Then run the **same held-out examples** through both.

> In production, use independent model instances if you need a clean simultaneous comparison. Attaching a PEFT adapter modifies the model wrapper used for inference.

---

# 5. Exact Match metric

For structured or deterministic answers:

```text
Question:
What is 2 + 2?

Expected:
4

Prediction:
4
```

Calculate exact match:

```python
def normalize(text):

    return (
        text
        .strip()
        .lower()
    )


def exact_match(
    predictions,
    references
):

    correct = 0

    total = len(predictions)


    for prediction, reference in zip(
        predictions,
        references
    ):

        if (
            normalize(prediction)
            ==
            normalize(reference)
        ):

            correct += 1


    return correct / total
```

Usage:

```python
score = exact_match(
    predictions,
    references
)

print(
    f"Exact Match: {score:.2%}"
)
```

Good for:

```text
Classification
SQL with canonicalized output
Short deterministic answers
IDs
Labels
```

---

# 6. BLEU and ROUGE

For text-generation tasks, exact match is often too strict.

Example:

```text
Reference:
You can reset your password from Settings.

Prediction:
Reset your password by opening Settings.
```

Meaning is similar, but exact match is zero.

You can use:

```text
BLEU  → n-gram precision
ROUGE → n-gram recall / overlap
```

Example with ROUGE:

```python
from evaluate import load


rouge = load("rouge")


results = rouge.compute(
    predictions=predictions,
    references=references
)


print(results)
```

BLEU:

```python
bleu = load("bleu")


results = bleu.compute(
    predictions=predictions,
    references=[
        [reference]
        for reference in references
    ]
)


print(results)
```

However, for modern LLMs, BLEU and ROUGE alone are often insufficient because two semantically correct responses can use very different wording.

---

# 7. Semantic evaluation

For many LLM tasks, evaluate whether the meaning is correct.

A practical approach is embedding similarity:

```text
Prediction
    ↓
Embedding Model
    ↓
Vector

Reference
    ↓
Embedding Model
    ↓
Vector

Cosine Similarity
```

Example:

```python
from sentence_transformers import SentenceTransformer
from sentence_transformers import util


embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


prediction_embeddings = embedding_model.encode(
    predictions,
    convert_to_tensor=True
)


reference_embeddings = embedding_model.encode(
    references,
    convert_to_tensor=True
)


similarities = util.cos_sim(
    prediction_embeddings,
    reference_embeddings
)
```

A simple average diagonal score:

```python
import torch


scores = similarities.diag()

average_score = scores.mean().item()

print(
    "Average semantic similarity:",
    average_score
)
```

Be careful: semantic similarity can score an answer highly even when an important factual detail is wrong.

---

# 8. Evaluate JSON generation

If your fine-tuned model generates JSON, evaluate:

### Valid JSON

```python
import json


def is_valid_json(text):

    try:

        json.loads(text)

        return True

    except json.JSONDecodeError:

        return False
```

Evaluate:

```python
valid_count = 0


for prediction in predictions:

    if is_valid_json(prediction):

        valid_count += 1


json_validity_rate = (
    valid_count
    / len(predictions)
)


print(
    "JSON validity:",
    json_validity_rate
)
```

---

# 9. Schema validation

Suppose the model should produce:

```json
{
    "intent": "refund",
    "priority": "high",
    "response": "..."
}
```

Use Pydantic:

```python
from pydantic import BaseModel


class SupportResponse(BaseModel):

    intent: str

    priority: str

    response: str
```

Validate:

```python
import json


def validate_response(text):

    try:

        data = json.loads(text)

        SupportResponse.model_validate(
            data
        )

        return True

    except Exception:

        return False
```

Then:

```python
valid = sum(
    validate_response(p)
    for p in predictions
)


schema_validity = (
    valid
    / len(predictions)
)

print(
    "Schema Validity:",
    schema_validity
)
```

---

# 10. Evaluate SQL generation

For SQL fine-tuning, do **not rely only on string comparison**.

These can be equivalent:

```sql
SELECT * FROM users;
```

```sql
SELECT
    *
FROM users;
```

Better evaluation:

```text
1. SQL parses correctly
2. Query uses valid schema
3. Query executes successfully
4. Result matches expected result
```

Example parser check:

```python
import sqlglot


def is_valid_sql(query):

    try:

        sqlglot.parse_one(
            query
        )

        return True

    except Exception:

        return False
```

For execution accuracy, use an isolated test database:

```python
def execution_accuracy(
    predicted_sql,
    expected_sql,
    connection
):

    try:

        predicted_result = (
            connection.execute(
                predicted_sql
            )
            .fetchall()
        )

        expected_result = (
            connection.execute(
                expected_sql
            )
            .fetchall()
        )


        return (
            predicted_result
            ==
            expected_result
        )

    except Exception:

        return False
```

Never run generated SQL against production without appropriate controls.

---

# 11. Build a reusable evaluation pipeline

```python
from dataclasses import dataclass


@dataclass
class EvaluationResult:

    prediction: str
    reference: str
    exact_match: bool
```

Evaluator:

```python
def evaluate_model(
    model,
    tokenizer,
    dataset
):

    results = []


    for example in dataset:

        prompt = example[
            "instruction"
        ]

        reference = example[
            "response"
        ]


        prediction = generate_response(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt
        )


        result = EvaluationResult(

            prediction=prediction,

            reference=reference,

            exact_match=(
                normalize(prediction)
                ==
                normalize(reference)
            )
        )


        results.append(result)


    return results
```

Calculate summary:

```python
def calculate_metrics(results):

    total = len(results)


    exact_matches = sum(
        result.exact_match
        for result in results
    )


    return {
        "total_examples": total,

        "exact_match": (
            exact_matches / total
            if total > 0
            else 0
        )
    }
```

Use:

```python
results = evaluate_model(
    model=model,
    tokenizer=tokenizer,
    dataset=validation_dataset
)


metrics = calculate_metrics(
    results
)


print(metrics)
```

---

# 12. A production evaluation dataset

A good evaluation dataset should contain:

```text
evaluation/
│
├── normal_cases.jsonl
├── difficult_cases.jsonl
├── edge_cases.jsonl
├── safety_cases.jsonl
├── formatting_cases.jsonl
└── regression_cases.jsonl
```

For a customer-support model:

```text
Normal:
"How do I reset my password?"

Difficult:
"My account was charged twice but I cancelled the order."

Edge case:
"I forgot my email and password."

Safety:
"Give me another customer's account information."

Regression:
Previously failing production examples.
```

Evaluate each category separately.

---

# 13. Recommended production metrics

For a fine-tuned customer-support model:

| Metric                | Purpose                       |
| --------------------- | ----------------------------- |
| Validation loss       | General learning signal       |
| Perplexity            | Next-token prediction quality |
| Exact match           | Deterministic answers         |
| Semantic similarity   | Meaning similarity            |
| JSON validity         | Structured output correctness |
| Schema validity       | Output contract               |
| Task accuracy         | Business correctness          |
| Hallucination rate    | Unsupported information       |
| Safety violation rate | Unsafe responses              |
| Latency               | Production performance        |
| Token usage           | Cost and efficiency           |

---

# 14. LLM-as-a-Judge evaluation

For open-ended responses, you can use a separate LLM to evaluate answers.

Example evaluation criteria:

```text
Score the answer from 1 to 5:

1. Is it correct?
2. Is it relevant?
3. Is it helpful?
4. Does it follow company policy?
5. Does it hallucinate information?
```

Conceptually:

```python
def judge_prompt(question, answer, reference):

    return f"""
You are an evaluator.

Question:
{question}

Model Answer:
{answer}

Reference Answer:
{reference}

Score the model answer from 1 to 5.

Return JSON:

{{
    "correctness": 0,
    "relevance": 0,
    "helpfulness": 0,
    "hallucination": 0
}}
"""
```

For production, use:

* a strong evaluator model
* deterministic settings
* a clearly versioned evaluation prompt
* sampled human review
* calibration against human judgments

Do not blindly trust an LLM judge.

---

# 15. Complete evaluation workflow

```text
                Fine-Tuned Model
                       │
                       ▼
             Held-Out Test Dataset
                       │
                       ▼
                  Generate
                       │
          ┌────────────┼─────────────┐
          ▼            ▼             ▼

       Loss         Task Metrics   Format Check
          │            │             │
          └────────────┼─────────────┘
                       ▼
                 Evaluation Report
                       │
                       ▼
             Compare Base vs Fine-Tuned
                       │
                       ▼
                 Human Evaluation
                       │
                       ▼
                    Deploy
```

---

# A practical complete example

```python
import math
import torch
from evaluate import load


# ============================================================
# METRICS
# ============================================================

rouge = load("rouge")


def normalize(text):

    return (
        text
        .strip()
        .lower()
    )


def evaluate_predictions(
    predictions,
    references
):

    # --------------------------------------------------------
    # Exact Match
    # --------------------------------------------------------

    exact_matches = sum(
        normalize(prediction)
        ==
        normalize(reference)
        for prediction, reference
        in zip(predictions, references)
    )


    exact_match_score = (
        exact_matches
        / len(predictions)
    )


    # --------------------------------------------------------
    # ROUGE
    # --------------------------------------------------------

    rouge_scores = rouge.compute(
        predictions=predictions,
        references=references
    )


    return {

        "exact_match": exact_match_score,

        "rouge1": rouge_scores["rouge1"],

        "rouge2": rouge_scores["rouge2"],

        "rougeL": rouge_scores["rougeL"]
    }


# ============================================================
# GENERATE PREDICTIONS
# ============================================================

predictions = []

references = []


for example in validation_dataset:

    prediction = generate_response(
        model=model,
        tokenizer=tokenizer,
        prompt=example["instruction"]
    )

    predictions.append(
        prediction
    )

    references.append(
        example["response"]
    )


# ============================================================
# CALCULATE METRICS
# ============================================================

metrics = evaluate_predictions(
    predictions,
    references
)


for metric, value in metrics.items():

    print(
        f"{metric}: {value:.4f}"
    )
```

# Interview-ready answer

> **After fine-tuning, I evaluate the model at multiple levels. First, I measure validation loss and optionally perplexity on a held-out dataset. Then I run task-specific evaluation by generating outputs for test examples and calculating metrics appropriate to the task, such as exact match, execution accuracy for SQL, JSON/schema validity for structured output, or semantic and rubric-based evaluation for open-ended generation.**
>
> **Most importantly, I compare the fine-tuned model against the original base model using the same frozen evaluation set. In production, I maintain separate normal, difficult, edge-case, safety, and regression test sets and combine automated metrics with LLM-as-a-judge and human evaluation. I do not rely on training loss alone to decide whether a model is ready for deployment.**

For an interview, this **multi-layer evaluation approach** is much stronger than simply saying: *“I check validation loss after fine-tuning.”*
