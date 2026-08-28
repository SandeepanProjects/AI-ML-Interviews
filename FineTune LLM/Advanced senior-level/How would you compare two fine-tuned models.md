# How would you compare two fine-tuned models?

Suppose you have two models:

```text
Model A → Llama 3.1 8B + LoRA experiment A
Model B → Llama 3.1 8B + LoRA experiment B
```

You should **not compare them using only training loss**.

A production comparison should evaluate:

```text
                    Model A
                       │
                       ├── Accuracy / task metrics
                       ├── Instruction following
                       ├── Hallucination
                       ├── Safety
                       ├── Latency
                       ├── Throughput
                       └── Cost
                       
                    Model B
                       │
                       ├── Same evaluation dataset
                       ├── Same prompts
                       ├── Same decoding settings
                       └── Same hardware
```

The key principle is:

> **Compare both models under identical evaluation conditions using the same frozen dataset, prompts, generation parameters, and infrastructure.**

---

# 1. What should you compare?

I would usually compare models across these categories:

| Category              | Example metrics                  |
| --------------------- | -------------------------------- |
| Task quality          | Accuracy, F1, Exact Match        |
| Generation quality    | BLEU, ROUGE, BERTScore           |
| Language modeling     | Validation loss, perplexity      |
| Instruction following | Task success rate                |
| Factuality            | Groundedness, hallucination rate |
| Safety                | Unsafe response rate             |
| Human preference      | Win rate                         |
| LLM judge             | Judge score                      |
| Performance           | Latency, tokens/sec              |
| Cost                  | Cost/request, cost/1M tokens     |
| Model size            | GPU memory                       |

A model with the best quality score is not automatically the best production model.

Example:

```text
Model A

Accuracy:       92%
Latency:        5 seconds
Cost:           ₹10/request


Model B

Accuracy:       90%
Latency:        1 second
Cost:           ₹2/request
```

Model B might be better for production.

---

# 2. Create a fixed evaluation dataset

Never evaluate Model A and Model B using different datasets.

Example:

```python
evaluation_dataset = [
    {
        "id": "1",
        "prompt": "What is LoRA?",
        "reference": (
            "LoRA is a parameter-efficient "
            "fine-tuning technique."
        )
    },
    {
        "id": "2",
        "prompt": "What is RAG?",
        "reference": (
            "RAG retrieves relevant information "
            "and provides it to an LLM."
        )
    }
]
```

In production, this dataset should be:

```text
Versioned
Frozen
Access controlled
Separated from training data
Checked for leakage
```

For example:

```text
evaluation_dataset_v3
```

---

# 3. Load two fine-tuned models

Assume two LoRA adapters:

```text
models/
├── experiment_A/
│   └── adapter_model.safetensors
│
└── experiment_B/
    └── adapter_model.safetensors
```

Load the base model:

```python
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

from peft import PeftModel


BASE_MODEL = "meta-llama/Llama-3.1-8B"


tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL
)


base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
```

Load Model A:

```python
model_a = PeftModel.from_pretrained(
    base_model,
    "models/experiment_A"
)

model_a.eval()
```

For a fair isolated comparison, load Model B with a separate base model:

```python
base_model_b = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)


model_b = PeftModel.from_pretrained(
    base_model_b,
    "models/experiment_B"
)

model_b.eval()
```

This avoids accidentally sharing mutable adapter state.

---

# 4. Generate answers using identical settings

This is critical.

Bad comparison:

```text
Model A
temperature = 0.7

Model B
temperature = 0.2
```

That is not a fair comparison.

Use:

```python
GENERATION_CONFIG = {
    "max_new_tokens": 256,
    "temperature": 0.0,
    "do_sample": False
}
```

Generation function:

```python
def generate_response(
    model,
    prompt
):

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(model.device)

    with torch.inference_mode():

        output_ids = model.generate(
            **inputs,
            **GENERATION_CONFIG
        )

    generated_tokens = output_ids[
        0,
        inputs["input_ids"].shape[1]:
    ]

    return tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    )
```

Evaluate both:

```python
results = []


for example in evaluation_dataset:

    prompt = example["prompt"]

    response_a = generate_response(
        model_a,
        prompt
    )

    response_b = generate_response(
        model_b,
        prompt
    )

    results.append({

        "id":
            example["id"],

        "prompt":
            prompt,

        "reference":
            example["reference"],

        "model_a":
            response_a,

        "model_b":
            response_b
    })
```

Now both models have answered exactly the same questions.

---

# 5. Compare exact match

Useful for:

```text
Classification
Question answering
Structured output
Code output
```

Code:

```python
def exact_match(
    prediction,
    reference
):

    return int(
        prediction.strip().lower()
        ==
        reference.strip().lower()
    )
```

Calculate score:

```python
def calculate_exact_match(
    predictions,
    references
):

    scores = [

        exact_match(
            prediction,
            reference
        )

        for prediction, reference
        in zip(
            predictions,
            references
        )
    ]

    return sum(scores) / len(scores)
```

Usage:

```python
predictions_a = [
    item["model_a"]
    for item in results
]

predictions_b = [
    item["model_b"]
    for item in results
]

references = [
    item["reference"]
    for item in results
]


score_a = calculate_exact_match(
    predictions_a,
    references
)

score_b = calculate_exact_match(
    predictions_b,
    references
)
```

---

# 6. Compare ROUGE

ROUGE is useful when there may be multiple valid answers.

Example:

```text
Reference:
LoRA reduces trainable parameters.

Model:
LoRA trains a small number of additional parameters.
```

Exact match may fail:

```text
0
```

But ROUGE may show semantic overlap.

Using Hugging Face Evaluate:

```python
import evaluate


rouge = evaluate.load(
    "rouge"
)


rouge_a = rouge.compute(
    predictions=predictions_a,
    references=references
)


rouge_b = rouge.compute(
    predictions=predictions_b,
    references=references
)
```

Print:

```python
print(
    "Model A:",
    rouge_a
)

print(
    "Model B:",
    rouge_b
)
```

---

# 7. Compare BERTScore

BERTScore is often more useful for semantic similarity than pure word overlap.

```python
bertscore = evaluate.load(
    "bertscore"
)
```

Model A:

```python
score_a = bertscore.compute(
    predictions=predictions_a,
    references=references,
    lang="en"
)
```

Model B:

```python
score_b = bertscore.compute(
    predictions=predictions_b,
    references=references,
    lang="en"
)
```

Calculate average F1:

```python
avg_f1_a = sum(
    score_a["f1"]
) / len(
    score_a["f1"]
)


avg_f1_b = sum(
    score_b["f1"]
) / len(
    score_b["f1"]
)
```

---

# 8. Compare validation loss and perplexity

During fine-tuning, compare validation loss.

Suppose:

```text
Model A validation loss = 1.8
Model B validation loss = 1.4
```

Calculate perplexity:

$$
Perplexity = e^{loss}
$$

Code:

```python
import math


def perplexity(
    loss
):

    return math.exp(
        loss
    )
```

Example:

```python
model_a_loss = 1.8

model_b_loss = 1.4


print(
    perplexity(model_a_loss)
)

print(
    perplexity(model_b_loss)
)
```

But remember:

> **Lower validation loss does not guarantee a better instruction-following model.**

Always evaluate downstream behavior.

---

# 9. Compare instruction following

Create a task-specific rubric.

Example dataset:

```python
instruction_eval = [

    {
        "prompt":
            "Return exactly three Python programming tips.",

        "criteria": {
            "exact_count": 3,
            "language": "Python"
        }
    }
]
```

Rule-based evaluator:

```python
def evaluate_instruction_following(
    response
):

    lines = [

        line
        for line in response.split("\n")

        if line.strip()
    ]

    return {

        "passed":
            len(lines) >= 3
    }
```

A real production evaluator would be more task-specific.

---

# 10. Compare structured output quality

Suppose your models must generate JSON.

Example:

```text
Prompt:

Return:

{
  "name": string,
  "age": integer
}
```

Evaluate:

```python
import json


def is_valid_json(
    response
):

    try:

        json.loads(
            response
        )

        return True

    except json.JSONDecodeError:

        return False
```

Calculate success rate:

```python
def json_success_rate(
    responses
):

    valid = sum(

        is_valid_json(response)

        for response in responses
    )

    return valid / len(
        responses
    )
```

Model A:

```python
json_rate_a = json_success_rate(
    predictions_a
)
```

Model B:

```python
json_rate_b = json_success_rate(
    predictions_b
)
```

This is often more meaningful than BLEU for structured tasks.

---

# 11. Compare hallucination rates

Suppose you have a RAG or grounded evaluation dataset.

Example:

```python
evaluation_example = {

    "context": """
    LoRA freezes the base model and trains
    low-rank adapter matrices.
    """,

    "question":
        "How does LoRA work?",

    "response":
        "..."
}
```

You can evaluate whether claims are supported by the context.

Simple architecture:

```text
Context
   │
   ▼
Model Response
   │
   ▼
Claim Extraction
   │
   ▼
Evidence Check
   │
   ├── Supported
   └── Unsupported
```

A simple metric:

$$
Hallucination\ Rate =
\frac{Unsupported\ Responses}
{Total\ Responses}
$$

Code:

```python
def hallucination_rate(
    unsupported,
    total
):

    return (
        unsupported / total
    )
```

In production, I would use:

```text
Rule-based checks
Retrieval-based verification
Human evaluation
LLM-as-a-judge
```

rather than a simple keyword check.

---

# 12. Use LLM-as-a-judge

For open-ended answers, a judge can compare both models.

Prompt concept:

```text
Question:
{question}

Model A answer:
{answer_a}

Model B answer:
{answer_b}

Choose the better answer based on:

1. Correctness
2. Relevance
3. Completeness
4. Instruction following

Return only:

A
B
TIE
```

Python interface:

```python
def judge_answers(
    question,
    answer_a,
    answer_b,
    judge_llm
):

    prompt = f"""
Question:
{question}

Answer A:
{answer_a}

Answer B:
{answer_b}

Evaluate correctness, relevance,
completeness and instruction following.

Return:

A
B
or
TIE
"""

    result = judge_llm.invoke(
        prompt
    )

    return result.strip()
```

---

# 13. Avoid position bias

LLM judges may prefer the first answer.

Therefore run the comparison twice:

```text
Run 1

A → Model A
B → Model B


Run 2

A → Model B
B → Model A
```

Example:

```python
def compare_without_position_bias(
    question,
    model_a_answer,
    model_b_answer,
    judge_llm
):

    result_1 = judge_answers(
        question,
        model_a_answer,
        model_b_answer,
        judge_llm
    )

    result_2 = judge_answers(
        question,
        model_b_answer,
        model_a_answer,
        judge_llm
    )

    return result_1, result_2
```

If the judge consistently prefers the same model after swapping positions, confidence is higher.

---

# 14. Calculate win rate

Suppose the results are:

```text
Model A wins: 60
Model B wins: 30
Tie:          10
```

Model A win rate:

$$
Win\ Rate =
\frac{Model\ Wins}
{Total\ Comparisons}
$$

Code:

```python
def calculate_win_rate(
    wins,
    total
):

    return wins / total
```

Or excluding ties:

```python
def decisive_win_rate(
    model_a_wins,
    model_b_wins
):

    return (
        model_a_wins
        /
        (
            model_a_wins
            +
            model_b_wins
        )
    )
```

---

# 15. Human evaluation

For important production models, use blind evaluation.

Evaluators see:

```text
Question

Response A
Response B
```

They do **not** know which model generated each response.

Example data structure:

```python
human_evaluations = [

    {
        "prompt":
            "Explain LoRA",

        "response_a":
            "...",

        "response_b":
            "...",

        "winner":
            "A"
    }
]
```

Calculate:

```python
from collections import Counter


winners = [

    item["winner"]
    for item in human_evaluations
]


Counter(
    winners
)
```

This provides:

```text
Model A win rate
Model B win rate
Tie rate
```

---

# 16. Compare latency

Quality alone is not enough.

Measure:

```python
import time


def measure_latency(
    model,
    prompt
):

    start = time.perf_counter()

    response = generate_response(
        model,
        prompt
    )

    end = time.perf_counter()

    latency = end - start

    return response, latency
```

For every request:

```python
latencies_a = []

latencies_b = []


for example in evaluation_dataset:

    _, latency_a = measure_latency(
        model_a,
        example["prompt"]
    )

    _, latency_b = measure_latency(
        model_b,
        example["prompt"]
    )

    latencies_a.append(
        latency_a
    )

    latencies_b.append(
        latency_b
    )
```

Calculate average:

```python
def average(
    values
):

    return sum(values) / len(values)
```

But production teams should also measure:

```text
p50 latency
p95 latency
p99 latency
```

Example:

```python
import numpy as np


p50 = np.percentile(
    latencies_a,
    50
)

p95 = np.percentile(
    latencies_a,
    95
)

p99 = np.percentile(
    latencies_a,
    99
)
```

---

# 17. Compare throughput

Measure tokens generated per second.

```python
def tokens_per_second(
    generated_tokens,
    seconds
):

    return (
        generated_tokens
        /
        seconds
    )
```

Example:

```text
Model A = 45 tokens/sec

Model B = 120 tokens/sec
```

Model B may provide much better user experience.

---

# 18. Compare GPU memory

```python
torch.cuda.reset_peak_memory_stats()


response = generate_response(
    model_a,
    prompt
)


memory = (
    torch.cuda.max_memory_allocated()
    /
    1024**3
)


print(
    f"Peak GPU memory: {memory:.2f} GB"
)
```

Measure both models using:

```text
Same GPU
Same batch size
Same sequence length
Same generation parameters
```

---

# 19. Build a complete comparison result

A production comparison could generate:

```python
comparison = {

    "model_a": {

        "name":
            "experiment_A",

        "exact_match":
            0.82,

        "rouge_l":
            0.71,

        "bertscore_f1":
            0.91,

        "judge_win_rate":
            0.62,

        "hallucination_rate":
            0.04,

        "p95_latency_ms":
            850,

        "gpu_memory_gb":
            14.2
    },

    "model_b": {

        "name":
            "experiment_B",

        "exact_match":
            0.79,

        "rouge_l":
            0.74,

        "bertscore_f1":
            0.93,

        "judge_win_rate":
            0.38,

        "hallucination_rate":
            0.03,

        "p95_latency_ms":
            520,

        "gpu_memory_gb":
            11.8
    }
}
```

Now the decision becomes a trade-off.

---

# 20. Use a weighted model selection score

Sometimes you need a single decision score.

For example:

```text
Quality           50%
Latency           20%
Cost              20%
Hallucination     10%
```

Code:

```python
def model_score(
    quality,
    latency_score,
    cost_score,
    safety_score
):

    return (

        0.50 * quality
        +
        0.20 * latency_score
        +
        0.20 * cost_score
        +
        0.10 * safety_score
    )
```

But don't blindly use one score for every application.

For a medical application:

```text
Safety > Cost
```

For a customer support chatbot:

```text
Latency + Cost may matter more
```

---

# 21. Statistical significance

Suppose:

```text
Model A accuracy = 91.2%
Model B accuracy = 91.5%
```

The difference may be random.

Use paired evaluation because both models answer the same examples.

Example bootstrap approach:

```python
import numpy as np


def bootstrap_difference(
    scores_a,
    scores_b,
    n_iterations=1000
):

    scores_a = np.array(
        scores_a
    )

    scores_b = np.array(
        scores_b
    )

    differences = []

    n = len(
        scores_a
    )

    for _ in range(
        n_iterations
    ):

        indices = np.random.choice(
            n,
            n,
            replace=True
        )

        difference = (

            scores_a[indices].mean()
            -
            scores_b[indices].mean()
        )

        differences.append(
            difference
        )

    return np.percentile(
        differences,
        [2.5, 97.5]
    )
```

If the confidence interval strongly favors one model, the improvement is more convincing.

---

# 22. Production experiment tracking

I would store:

```text
Experiment ID
Base model
Dataset version
Dataset hash
Training configuration
LoRA rank
Learning rate
Epochs
Random seed
Evaluation dataset version
Metrics
Latency
Cost
Deployment environment
Git commit
```

Example:

```python
experiment = {

    "experiment_id":
        "llama-lora-v42",

    "base_model":
        "Llama",

    "dataset_version":
        "support_data_v5",

    "lora_rank":
        16,

    "learning_rate":
        2e-4,

    "epochs":
        3,

    "seed":
        42
}
```

This makes comparison reproducible.

---

# Recommended production architecture

```text
                   ┌──────────────┐
                   │ Model A      │
                   └──────┬───────┘
                          │
Evaluation Dataset ───────┼───────► Metrics
                          │
                   ┌──────▼───────┐
                   │ Model B      │
                   └──────┬───────┘
                          │
                          ▼
                  Automated Evaluation
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
        ▼                 ▼                 ▼
    Task Metrics      LLM Judge       Performance
        │                 │                 │
        └─────────────────┼─────────────────┘
                          ▼
                   Comparison Report
                          │
                          ▼
                    Human Review
                          │
                          ▼
                    Deploy Winner
```

# Strong interview answer

> **To compare two fine-tuned models, I would first freeze a held-out evaluation dataset that neither model trained on and ensure there is no data leakage. I would run both models with identical prompts, decoding parameters, hardware, and evaluation conditions. Then I would measure task-specific quality metrics such as accuracy, F1, exact match, ROUGE, or BERTScore depending on the task.**
>
> **For open-ended tasks, I would use pairwise evaluation with human evaluators or an LLM-as-a-judge and calculate win rate, while controlling for judge position bias. I would also compare hallucination, safety, structured-output correctness, latency, throughput, GPU memory, and inference cost.**
>
> **Finally, I would track dataset versions, model versions, training configurations, and random seeds so the comparison is reproducible. I would select the model based on statistically meaningful improvements and production trade-offs rather than training loss alone.**

## The key takeaway

```text
Don't ask:

"Which model has lower training loss?"

Ask:

"Which model performs better on the
real production task under the same
evaluation conditions and constraints?"
```
