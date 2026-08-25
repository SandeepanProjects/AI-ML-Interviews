# What is Preference Tuning?

## Simple definition

**Preference tuning** is the process of training an LLM using data that tells the model:

> **For the same prompt, which response is preferred over another response?**

Unlike normal supervised fine-tuning, where we provide one "correct" answer, preference tuning provides a **comparison**.

```text
Prompt:
Explain RAG.

Response A: RAG is a database.
Response B: RAG retrieves relevant information and gives it to an LLM.

Human preference:

Response B > Response A
```

The model learns:

```text
Increase probability of Response B
Decrease probability of Response A
```

---

# 1. Why do we need preference tuning?

Consider an instruction-tuned LLM.

Prompt:

```text
Explain LoRA.
```

The model can generate many technically correct answers.

### Response A

```text
LoRA is a fine-tuning technique.
```

### Response B

```text
LoRA is a parameter-efficient fine-tuning technique that keeps
the base model frozen and trains small low-rank matrices.
```

Both may be correct, but humans prefer B because it is:

* more accurate
* more useful
* more complete
* still concise

Traditional SFT struggles to directly express:

```text
B is better than A
```

Preference tuning explicitly captures that relationship.

---

# 2. SFT vs preference tuning

## Supervised Fine-Tuning (SFT)

Dataset:

```python
{
    "prompt": "What is RAG?",
    "response": "RAG retrieves relevant documents before generating an answer."
}
```

The objective is:

```text
Prompt → Expected Answer
```

The model learns:

[
P(response \mid prompt)
]

---

## Preference tuning

Dataset:

```python
{
    "prompt": "What is RAG?",

    "chosen": "RAG retrieves relevant documents and uses them as context before generating an answer.",

    "rejected": "RAG is a database."
}
```

The objective becomes:

[
P(chosen \mid prompt)

>

P(rejected \mid prompt)
]

---

# 3. High-level architecture

```text
                Prompt
                  │
         ┌────────┴────────┐
         │                 │
         ▼                 ▼
      Chosen ✓         Rejected ✗
         │                 │
         └────────┬────────┘
                  │
                  ▼
          Preference Algorithm
          (DPO / PPO / etc.)
                  │
                  ▼
              Update LLM
```

---

# 4. What is a preference dataset?

A **preference dataset** contains examples where multiple possible responses are compared and one response is marked as preferable.

The most common format is:

```text
Prompt
Chosen response
Rejected response
```

Example:

```python
example = {
    "prompt": "My payment failed.",

    "chosen": (
        "I'm sorry your payment failed. Please verify your "
        "payment method and available balance. If the issue "
        "continues, I can guide you through the next steps."
    ),

    "rejected": (
        "Payment failed."
    )
}
```

Here:

```text
chosen > rejected
```

---

# 5. Basic preference dataset format

A JSONL file might look like:

```json
{"prompt":"Explain RAG.","chosen":"RAG retrieves relevant documents and provides them as context to an LLM.","rejected":"RAG is a database."}
{"prompt":"Explain LoRA.","chosen":"LoRA is a parameter-efficient fine-tuning technique using low-rank adapters.","rejected":"LoRA modifies every parameter of the model."}
```

Each line is one training example.

Create it with Python:

```python
import json

data = [
    {
        "prompt": "Explain RAG.",

        "chosen": (
            "RAG retrieves relevant documents and provides "
            "them as context to an LLM before generating an answer."
        ),

        "rejected": (
            "RAG is a database."
        )
    },

    {
        "prompt": "Explain LoRA.",

        "chosen": (
            "LoRA is a parameter-efficient fine-tuning technique "
            "that trains low-rank adapter weights while keeping "
            "most base model parameters frozen."
        ),

        "rejected": (
            "LoRA trains every parameter in the model."
        )
    }
]

with open(
    "preferences.jsonl",
    "w"
) as file:

    for example in data:

        file.write(
            json.dumps(example)
            + "\n"
        )
```

---

# 6. Where does preference data come from?

There are several sources.

## A. Human annotators

You show multiple answers:

```text
Prompt:
Explain vector databases.
```

Responses:

```text
A → Short but incomplete
B → Accurate and useful
C → Incorrect
```

Annotator ranks:

```text
B > A > C
```

Convert rankings into pairs:

```text
B > A
B > C
A > C
```

Code:

```python
responses = [
    "Response B",
    "Response A",
    "Response C"
]

pairs = []

for i in range(len(responses)):

    for j in range(i + 1, len(responses)):

        pairs.append({
            "chosen": responses[i],
            "rejected": responses[j]
        })

print(pairs)
```

---

## B. Production feedback

For example, a support assistant gives two candidate responses:

```text
Customer question
       │
       ├── Response A
       │
       └── Response B
              │
              ▼
       User / Expert chooses
```

If the user selects B:

```python
preference = {
    "prompt": customer_question,
    "chosen": response_b,
    "rejected": response_a
}
```

In production, you should also apply privacy controls and review before training on customer data.

---

## C. Expert comparisons

For specialized domains:

* legal
* finance
* healthcare
* enterprise systems
* software engineering

Domain experts can label:

```text
Which answer is:

✓ Correct
✓ Safe
✓ Helpful
✓ Complete
```

Example:

```python
{
    "prompt": "How should an API handle retries?",

    "chosen": (
        "Use bounded retries with exponential backoff and jitter, "
        "and only retry idempotent or safely repeatable operations."
    ),

    "rejected": (
        "Retry every failed request forever."
    )
}
```

---

## D. AI-generated preference data

A stronger model can generate candidate responses and judge them.

```text
Prompt
   │
   ▼
Generate 3 responses
   │
   ▼
Judge model
   │
   ▼
Chosen / Rejected
```

Example:

```python
def create_preference(
    prompt,
    response_a,
    response_b,
    judge_model
):

    decision = judge_model(
        prompt=prompt,
        response_a=response_a,
        response_b=response_b
    )

    if decision == "A":

        return {
            "prompt": prompt,
            "chosen": response_a,
            "rejected": response_b
        }

    return {
        "prompt": prompt,
        "chosen": response_b,
        "rejected": response_a
    }
```

This can scale data generation, but it has a major risk:

> The judge can introduce its own biases and mistakes.

So important samples should still be human-reviewed.

---

# 7. What makes a good preference dataset?

A preference dataset should contain **meaningful differences**.

### Bad example

```text
Chosen:
RAG retrieves documents.

Rejected:
RAG retrieves relevant documents.
```

The difference is weak and subjective.

---

### Better example

```text
Chosen:
RAG retrieves relevant documents and provides them as context
to the LLM, helping ground answers in external information.

Rejected:
RAG permanently trains documents into the model.
```

This teaches a clear preference.

---

# 8. Important: rejected responses should be realistic

A common mistake is creating extremely bad rejected answers.

Example:

```text
Chosen:
RAG retrieves documents before generation.

Rejected:
The moon is made of cheese.
```

This is not very useful.

The model already knows the second answer is nonsense.

Better:

```text
Chosen:
RAG retrieves relevant documents at inference time and provides
them as context to the LLM.

Rejected:
RAG permanently stores all documents inside the LLM weights.
```

Now the rejected answer is:

* plausible
* related
* subtly wrong

This produces better learning.

---

# 9. Types of preference datasets

## Type 1: Pairwise preference

Most common.

```python
{
    "prompt": "...",
    "chosen": "...",
    "rejected": "..."
}
```

Used by:

* DPO
* IPO
* ORPO-style methods
* other preference optimization algorithms

---

## Type 2: Ranking data

```python
{
    "prompt": "Explain RAG",

    "responses": [
        {
            "text": "Best response",
            "rank": 1
        },
        {
            "text": "Okay response",
            "rank": 2
        },
        {
            "text": "Bad response",
            "rank": 3
        }
    ]
}
```

Convert ranking to preference pairs.

```python
def ranking_to_pairs(
    prompt,
    ranked_responses
):

    pairs = []

    ranked_responses = sorted(
        ranked_responses,
        key=lambda x: x["rank"]
    )

    for i in range(
        len(ranked_responses)
    ):

        for j in range(
            i + 1,
            len(ranked_responses)
        ):

            pairs.append({
                "prompt": prompt,

                "chosen":
                    ranked_responses[i]["text"],

                "rejected":
                    ranked_responses[j]["text"]
            })

    return pairs
```

Example:

```python
responses = [

    {
        "text": "Detailed correct answer",
        "rank": 1
    },

    {
        "text": "Short correct answer",
        "rank": 2
    },

    {
        "text": "Incorrect answer",
        "rank": 3
    }
]

pairs = ranking_to_pairs(
    "Explain RAG.",
    responses
)

for pair in pairs:
    print(pair)
```

---

# 10. Preference tuning with DPO

A common implementation uses:

```text
Preference Dataset
        +
DPO Algorithm
```

Dataset:

```python
from datasets import Dataset

dataset = Dataset.from_list([

    {
        "prompt":
            "Explain RAG.",

        "chosen":
            (
                "RAG retrieves relevant documents and provides "
                "them as context to an LLM."
            ),

        "rejected":
            (
                "RAG permanently trains documents into the model."
            )
    },

    {
        "prompt":
            "What is LoRA?",

        "chosen":
            (
                "LoRA is a parameter-efficient fine-tuning method "
                "that trains low-rank adapter matrices."
            ),

        "rejected":
            (
                "LoRA fine-tunes every model parameter."
            )
    }
])
```

---

# 11. Training with DPOTrainer

A practical implementation commonly uses [Hugging Face TRL DPOTrainer documentation](https://huggingface.co/docs/trl/dpo_trainer?utm_source=chatgpt.com).

```python
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)

from trl import (
    DPOTrainer,
    DPOConfig
)
```

Load the model:

```python
MODEL_NAME = "your-base-model"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

model = (
    AutoModelForCausalLM
    .from_pretrained(
        MODEL_NAME
    )
)
```

Create a reference model:

```python
reference_model = (
    AutoModelForCausalLM
    .from_pretrained(
        MODEL_NAME
    )
)
```

Training configuration:

```python
config = DPOConfig(

    output_dir="./preference_model",

    learning_rate=5e-7,

    per_device_train_batch_size=2,

    gradient_accumulation_steps=8,

    num_train_epochs=1,

    beta=0.1,

    logging_steps=10
)
```

Create trainer:

```python
trainer = DPOTrainer(

    model=model,

    ref_model=reference_model,

    args=config,

    train_dataset=dataset,

    processing_class=tokenizer
)
```

Train:

```python
trainer.train()
```

Save:

```python
trainer.save_model(
    "./aligned_model"
)
```

---

# 12. What happens internally during preference tuning?

Suppose:

```text
Prompt:
Explain RAG.
```

We have:

```text
Chosen:
Correct explanation

Rejected:
Incorrect explanation
```

The model calculates:

[
\log P_\theta(chosen|prompt)
]

and:

[
\log P_\theta(rejected|prompt)
]

The goal is:

```text
Chosen probability ↑

Rejected probability ↓
```

A simplified training loop:

```python
for example in dataset:

    prompt = example["prompt"]

    chosen = example["chosen"]

    rejected = example["rejected"]

    chosen_log_prob = calculate_log_probability(
        model,
        prompt,
        chosen
    )

    rejected_log_prob = calculate_log_probability(
        model,
        prompt,
        rejected
    )

    loss = preference_loss(
        chosen_log_prob,
        rejected_log_prob
    )

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()
```

The exact loss depends on the algorithm:

```text
DPO
IPO
ORPO
SimPO
PPO-based RLHF
```

But the general goal remains:

> Make preferred responses more likely than non-preferred responses.

---

# 13. A simple preference loss for understanding

For educational purposes:

```python
import torch
import torch.nn.functional as F


def simple_preference_loss(
    chosen_log_prob,
    rejected_log_prob
):
    """
    Encourage:

    chosen_log_prob > rejected_log_prob
    """

    preference_margin = (
        chosen_log_prob
        -
        rejected_log_prob
    )

    loss = (
        -F.logsigmoid(
            preference_margin
        )
    )

    return loss.mean()
```

Example:

```python
chosen_log_prob = torch.tensor([
    -1.0
])

rejected_log_prob = torch.tensor([
    -3.0
])

loss = simple_preference_loss(
    chosen_log_prob,
    rejected_log_prob
)

print(loss.item())
```

The model is rewarded when:

```text
-1 > -3
```

Remember:

```text
Higher log probability = more likely
```

---

# 14. How would you create a preference dataset in a real project?

Imagine you are building an enterprise customer-support assistant.

## Step 1: Collect prompts

```python
prompts = [

    "My payment failed.",

    "How do I reset my password?",

    "Why was my account locked?"
]
```

---

## Step 2: Generate multiple candidate answers

```python
def generate_candidates(
    model,
    prompt,
    n=3
):

    responses = []

    for _ in range(n):

        response = model.generate(
            prompt
        )

        responses.append(
            response
        )

    return responses
```

Example:

```text
Prompt
   │
   ▼
Model
   │
   ├── Candidate A
   ├── Candidate B
   └── Candidate C
```

---

## Step 3: Human review

An expert chooses:

```text
Candidate B ✓
```

and rejects:

```text
Candidate A ✗
Candidate C ✗
```

Store:

```python
preference_examples = [

    {
        "prompt":
            "My payment failed.",

        "chosen":
            "Helpful candidate B",

        "rejected":
            "Candidate A"
    },

    {
        "prompt":
            "My payment failed.",

        "chosen":
            "Helpful candidate B",

        "rejected":
            "Candidate C"
    }
]
```

---

# 15. How do you validate preference data?

This is extremely important.

Check:

### 1. Is the chosen answer actually better?

```python
def validate_preference(
    example
):

    if (
        example["chosen"]
        ==
        example["rejected"]
    ):
        return False

    return True
```

---

### 2. Is the chosen answer empty?

```python
def validate_example(
    example
):

    required_fields = [
        "prompt",
        "chosen",
        "rejected"
    ]

    for field in required_fields:

        if not example.get(field):

            return False

    return True
```

---

### 3. Is the rejected answer relevant?

Bad:

```text
Prompt:
How do I reset my password?

Chosen:
Click Forgot Password.

Rejected:
Python was created by Guido van Rossum.
```

This teaches very little.

The rejected response should usually be a **realistic alternative**.

---

# 16. Preference data quality is more important than quantity

Consider:

```text
100,000 noisy preference examples
```

versus:

```text
10,000 high-quality expert comparisons
```

The second dataset can often be much more valuable.

Why?

Because the model learns from the comparison signal:

```text
Chosen > Rejected
```

If the labels are wrong:

```text
Bad response
      ↓
Marked as chosen
```

the model learns the wrong behavior.

---

# 17. Real-world preference tuning pipeline

```text
                Production Prompts
                        │
                        ▼
                Generate Candidates
                        │
                        ▼
                Human / Expert Review
                        │
                        ▼
               Preference Dataset
                        │
                        ├── Data Cleaning
                        │
                        ├── Deduplication
                        │
                        ├── PII Removal
                        │
                        ├── Safety Review
                        │
                        ▼
                    Train / Validation
                        │
                        ▼
                  SFT / Base Model
                        │
                        ▼
              Preference Optimization
                  (DPO, etc.)
                        │
                        ▼
                    Evaluation
                        │
                        ├── Win Rate
                        ├── Human Preference
                        ├── Safety
                        ├── Task Accuracy
                        ▼
                  Production Model
```

---

# 18. Preference tuning vs instruction tuning

| Feature           | Instruction Tuning  | Preference Tuning          |
| ----------------- | ------------------- | -------------------------- |
| Training signal   | Desired answer      | Relative preference        |
| Dataset           | Input → Output      | Prompt → Chosen → Rejected |
| Goal              | Follow instructions | Produce preferred behavior |
| Example           | "Answer this"       | "Which answer is better?"  |
| Typical algorithm | SFT                 | DPO / PPO                  |
| Common stage      | Early alignment     | Later alignment            |

Example pipeline:

```text
Pretrained Model
       │
       ▼
Instruction Tuning
       │
       ▼
Can follow instructions
       │
       ▼
Preference Tuning
       │
       ▼
Better aligned with preferences
```

---

# 19. Preference tuning vs RLHF

This is another important distinction:

```text
Preference tuning
      │
      ├── DPO
      │
      ├── IPO
      │
      ├── ORPO
      │
      └── Other preference optimization methods
```

Traditional RLHF:

```text
Human Preferences
       │
       ▼
Reward Model
       │
       ▼
PPO
       │
       ▼
Aligned LLM
```

So:

> **Preference tuning is the broader idea of improving an LLM using comparative preferences. DPO is one method. Classical RLHF uses preference data to train a reward model and then uses reinforcement learning.**

---

# Interview-ready answer

> **Preference tuning is a post-training technique where an LLM learns which responses humans prefer. Instead of training on a single target response like supervised fine-tuning, the dataset usually contains a prompt, a chosen response, and a rejected response. The objective is to increase the relative probability of the preferred response.**
>
> **A preference dataset contains comparisons such as prompt, chosen answer, and rejected answer. The data can come from human annotators, domain experts, user feedback, or carefully validated synthetic comparisons. The rejected responses should ideally be realistic alternatives rather than obviously nonsensical answers.**
>
> **Algorithms such as DPO can directly optimize these preference pairs, while traditional RLHF first trains a reward model from preference data and then optimizes the LLM using reinforcement learning such as PPO.**

## The simplest mental model

```text
SFT:

Prompt
  ↓
Correct Answer
  ↓
Learn answer
```

```text
Preference tuning:

Prompt
  ↓
Chosen ✓ vs Rejected ✗
  ↓
Learn which behavior humans prefer
```

**In one sentence:**

> **A preference dataset teaches the model not just what an answer can be, but which of multiple possible answers is better.**
