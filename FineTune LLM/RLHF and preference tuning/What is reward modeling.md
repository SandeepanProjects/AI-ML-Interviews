# What is Reward Modeling?

**Reward modeling** is the process of training a model to predict a **numerical score representing human preference** for an LLM's output.

It is a major component of **traditional RLHF (Reinforcement Learning from Human Feedback)**.

The basic idea:

```text
Prompt + Model Response
          │
          ▼
     Reward Model
          │
          ▼
    Reward Score
```

For example:

```text
Prompt:
Explain RAG.

Response A:
"RAG uses documents."

Reward = 0.25
```

```text
Prompt:
Explain RAG clearly with an example.

Response B:
"RAG retrieves relevant documents at query time and provides
them as context to an LLM, helping the answer use external
knowledge."

Reward = 0.92
```

The reward model learns that:

```text
Response B > Response A
```

---

# 1. Why do we need reward modeling?

Humans cannot directly provide a mathematical reward function like:

```python
def good_answer(answer):
    if answer_is_helpful:
        return 10
```

Qualities such as:

* helpfulness
* clarity
* relevance
* harmlessness
* correctness
* tone

are difficult to encode manually.

Instead, humans provide feedback:

```text
Prompt
   │
   ├── Answer A
   │
   └── Answer B
         │
         ▼
Human says:
B is better
```

We convert that into training data:

```python
{
    "prompt": "What is RAG?",

    "chosen": "RAG retrieves relevant documents...",

    "rejected": "RAG uses a database."
}
```

The reward model learns:

[
R(prompt, chosen) > R(prompt, rejected)
]

---

# 2. Where does the reward model fit into RLHF?

The traditional RLHF pipeline:

```text
              Pretrained LLM
                    │
                    ▼
                   SFT
                    │
                    ▼
                SFT Model
                    │
             Generate responses
                    │
                    ▼
             Human preferences
                    │
                    ▼
              Reward Modeling
                    │
                    ▼
                Reward Model
                    │
                    ▼
              PPO Optimization
                    │
                    ▼
                Aligned LLM
```

The reward model acts as a **learned judge**.

Instead of asking humans to evaluate every answer during reinforcement learning:

```text
LLM → Answer → Human → Feedback
```

we train the reward model once:

```text
Human Feedback
      ↓
Reward Model
      ↓
Automatic scoring
```

Then:

```text
LLM → Answer → Reward Model → Reward
```

---

# 3. What does reward model training data look like?

The most common format is:

```text
Prompt
Chosen response
Rejected response
```

Example:

```python
preference_examples = [
    {
        "prompt": "Explain LoRA.",

        "chosen": (
            "LoRA is a parameter-efficient fine-tuning "
            "method that keeps the base model mostly frozen "
            "and trains small low-rank matrices."
        ),

        "rejected": (
            "LoRA is a method for making models bigger."
        )
    }
]
```

The human preference is:

```text
chosen > rejected
```

---

# 4. How does a reward model work internally?

Usually, we start with a pretrained language model.

For example:

```text
Base Transformer

Input:
Prompt + Response

        ↓

Transformer Layers

        ↓

Hidden Representation

        ↓

Reward Head

        ↓

Single Number
```

A normal language model predicts:

```text
Vocabulary probabilities

P(token_1)
P(token_2)
P(token_3)
...
```

A reward model predicts:

```text
Single scalar

R = 0.83
```

Conceptually:

[
R(x, y) \rightarrow \mathbb{R}
]

Where:

* (x) = prompt
* (y) = response
* (R) = reward score

---

# 5. Reward model architecture

Suppose the base transformer produces a hidden state:

```text
Prompt + Response
       │
       ▼
   Transformer
       │
       ▼
Hidden vector
[0.12, -0.45, 0.87, ...]
       │
       ▼
Linear Reward Head
       │
       ▼
Reward Score
```

Mathematically:

[
R = W h + b
]

Where:

* (h) = transformer representation
* (W) = learned weights
* (b) = bias

---

# 6. Building a reward model with Transformers

We can use a sequence-classification model with:

```python
num_labels=1
```

```python
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)
```

Load the model:

```python
MODEL_NAME = "your-base-model"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

reward_model = (
    AutoModelForSequenceClassification
    .from_pretrained(
        MODEL_NAME,
        num_labels=1
    )
)
```

The model architecture is conceptually:

```text
Base Transformer
       +
Reward Head
       ↓
One scalar score
```

---

# 7. Tokenizing chosen and rejected responses

We need two inputs for every training example.

### Chosen

```text
Prompt:
Explain RAG.

Response:
RAG retrieves relevant documents...
```

### Rejected

```text
Prompt:
Explain RAG.

Response:
RAG uses a database.
```

Code:

```python
def format_input(
    prompt: str,
    response: str
) -> str:

    return f"""
User:
{prompt}

Assistant:
{response}
"""
```

Create inputs:

```python
chosen_text = format_input(
    prompt="Explain RAG.",
    response=(
        "RAG retrieves relevant documents "
        "and provides them as context to "
        "an LLM."
    )
)

rejected_text = format_input(
    prompt="Explain RAG.",
    response=(
        "RAG uses a database."
    )
)
```

---

# 8. Tokenize both responses

```python
chosen_inputs = tokenizer(
    chosen_text,
    return_tensors="pt",
    truncation=True,
    padding=True,
    max_length=512
)

rejected_inputs = tokenizer(
    rejected_text,
    return_tensors="pt",
    truncation=True,
    padding=True,
    max_length=512
)
```

Now we have:

```text
Chosen tokens
      ↓
Reward Model
      ↓
Chosen reward
```

and:

```text
Rejected tokens
      ↓
Reward Model
      ↓
Rejected reward
```

---

# 9. Calculate reward scores

```python
chosen_output = reward_model(
    **chosen_inputs
)

rejected_output = reward_model(
    **rejected_inputs
)
```

Get the scores:

```python
chosen_reward = (
    chosen_output.logits
)

rejected_reward = (
    rejected_output.logits
)
```

Example:

```text
Chosen reward:

2.4
```

```text
Rejected reward:

0.7
```

Good.

We want:

[
R_{chosen} > R_{rejected}
]

---

# 10. The reward model loss

The key objective is:

> Make the chosen answer receive a higher score than the rejected answer.

A common pairwise preference loss is:

[
L =
-\log
\sigma(
R_{chosen}
----------

R_{rejected}
)
]

Where:

* (R_{chosen}) = reward of preferred answer
* (R_{rejected}) = reward of rejected answer
* (\sigma) = sigmoid

Let's implement it.

```python
import torch
import torch.nn.functional as F
```

```python
def reward_model_loss(
    chosen_rewards,
    rejected_rewards
):
    """
    Pairwise preference loss.

    We want:

    chosen_reward > rejected_reward
    """

    reward_difference = (
        chosen_rewards
        - rejected_rewards
    )

    loss = -F.logsigmoid(
        reward_difference
    ).mean()

    return loss
```

Example:

```python
chosen_reward = torch.tensor([
    [2.5]
])

rejected_reward = torch.tensor([
    [0.5]
])

loss = reward_model_loss(
    chosen_reward,
    rejected_reward
)

print(loss.item())
```

Because:

```text
2.5 > 0.5
```

the loss should be relatively low.

---

# 11. What if the reward model ranks incorrectly?

Suppose:

```text
Chosen reward:

0.3
```

```text
Rejected reward:

2.0
```

Then:

[
R_{chosen} - R_{rejected}
=========================

# 0.3 - 2.0

-1.7
]

The loss becomes high.

Backpropagation updates the model so that eventually:

```text
Before:

Chosen    → 0.3
Rejected  → 2.0
```

becomes:

```text
After training:

Chosen    → 2.1
Rejected  → 0.8
```

---

# 12. Complete simplified training loop

Here is a simplified PyTorch implementation.

```python
import torch

from torch.optim import AdamW

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)
```

Load:

```python
MODEL_NAME = "your-base-model"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

model = (
    AutoModelForSequenceClassification
    .from_pretrained(
        MODEL_NAME,
        num_labels=1
    )
)
```

Optimizer:

```python
optimizer = AdamW(
    model.parameters(),
    lr=1e-5
)
```

Preference data:

```python
training_data = [

    {
        "prompt":
            "What is RAG?",

        "chosen":
            (
                "RAG retrieves relevant documents "
                "and provides them as context to "
                "an LLM."
            ),

        "rejected":
            (
                "RAG is a database."
            )
    },

    {
        "prompt":
            "What is LoRA?",

        "chosen":
            (
                "LoRA is a parameter-efficient "
                "fine-tuning technique that trains "
                "small low-rank matrices."
            ),

        "rejected":
            (
                "LoRA increases the size of the model."
            )
    }
]
```

Formatting function:

```python
def create_input(
    prompt: str,
    response: str
):

    return (
        f"User: {prompt}\n"
        f"Assistant: {response}"
    )
```

Training:

```python
model.train()

for epoch in range(3):

    total_loss = 0

    for example in training_data:

        # ------------------------
        # Chosen input
        # ------------------------

        chosen_text = create_input(

            example["prompt"],

            example["chosen"]
        )


        # ------------------------
        # Rejected input
        # ------------------------

        rejected_text = create_input(

            example["prompt"],

            example["rejected"]
        )


        # ------------------------
        # Tokenize
        # ------------------------

        chosen_inputs = tokenizer(

            chosen_text,

            return_tensors="pt",

            truncation=True,

            max_length=512
        )


        rejected_inputs = tokenizer(

            rejected_text,

            return_tensors="pt",

            truncation=True,

            max_length=512
        )


        # ------------------------
        # Forward pass
        # ------------------------

        chosen_output = model(
            **chosen_inputs
        )

        rejected_output = model(
            **rejected_inputs
        )


        # ------------------------
        # Rewards
        # ------------------------

        chosen_reward = (
            chosen_output.logits
        )

        rejected_reward = (
            rejected_output.logits
        )


        # ------------------------
        # Preference loss
        # ------------------------

        loss = reward_model_loss(

            chosen_reward,

            rejected_reward
        )


        # ------------------------
        # Backpropagation
        # ------------------------

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()


        total_loss += loss.item()


    print(

        f"Epoch {epoch + 1}, "
        f"Loss: {total_loss:.4f}"
    )
```

Conceptually:

```text
                    PROMPT

                       │
              ┌────────┴────────┐
              ▼                 ▼

       CHOSEN ANSWER      REJECTED ANSWER
              │                 │
              ▼                 ▼

         Reward Model      Reward Model
              │                 │
              ▼                 ▼

        Reward = 2.5      Reward = 0.5
              │                 │
              └────────┬────────┘
                       ▼

              Preference Loss
                       │
                       ▼

               Backpropagation
```

---

# 13. Production approach using TRL

In real projects, you normally don't manually write the complete reward-model training loop.

Libraries such as [Hugging Face TRL](https://huggingface.co/docs/trl?utm_source=chatgpt.com) provide trainers for preference and reward-model training.

Conceptually, your dataset might look like:

```python
from datasets import Dataset


data = [
    {
        "prompt": "Explain RAG.",

        "chosen": (
            "RAG retrieves relevant documents "
            "and provides them as context to "
            "an LLM."
        ),

        "rejected": (
            "RAG is a database."
        )
    }
]

dataset = Dataset.from_list(data)
```

Depending on the installed TRL version, the API can vary, but the core workflow remains:

```text
Preference pairs
      ↓
Reward model
      ↓
Pairwise ranking loss
      ↓
Trained reward model
```

---

# 14. How do you evaluate a reward model?

You should not evaluate only using training loss.

## A. Pairwise accuracy

For each test example:

```text
Human preference:

Chosen > Rejected
```

Check whether the reward model predicts:

```text
Reward(chosen) > Reward(rejected)
```

Code:

```python
def calculate_pairwise_accuracy(
    chosen_rewards,
    rejected_rewards
):

    correct = (
        chosen_rewards
        > rejected_rewards
    )

    accuracy = (
        correct.float()
        .mean()
        .item()
    )

    return accuracy
```

Example:

```python
chosen = torch.tensor([
    2.0,
    1.5,
    0.4
])

rejected = torch.tensor([
    1.0,
    0.8,
    0.7
])

accuracy = calculate_pairwise_accuracy(
    chosen,
    rejected
)

print(accuracy)
```

The third example is wrong because:

```text
0.4 < 0.7
```

So:

```text
Accuracy = 2 / 3
         = 66.7%
```

---

# 15. Reward scores are relative

This is important.

If the reward model says:

```text
Response A = 2.4
Response B = 1.2
```

You should usually interpret:

```text
A is preferred over B
```

You should **not automatically interpret**:

```text
2.4 = 96% quality
```

Reward models are generally trained for **ranking**, not necessarily calibrated absolute quality.

So this:

```text
A > B
```

is often more meaningful than:

```text
A = 0.82 quality
```

---

# 16. Example with multiple quality dimensions

A reward model might implicitly learn:

```text
Reward
   =
   Helpfulness
   +
   Relevance
   +
   Safety
   +
   Clarity
```

Conceptually:

```text
                    Response
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼

      Helpful       Relevant       Safe
          │            │            │
          └────────────┼────────────┘
                       ▼

                  Reward Score
```

However, a single reward model can hide trade-offs.

For example:

```text
Very helpful but slightly unsafe
```

versus:

```text
Safe but not useful
```

This is why advanced systems may use:

* multiple reward models
* separate safety classifiers
* rule-based guardrails
* process reward models
* human review

---

# 17. Reward Model vs Critic

These terms are sometimes confused.

## Reward Model

```text
Prompt + Response
       ↓
Reward Model
       ↓
Reward Score
```

Usually trained from:

```text
Human preference data
```

---

## Critic / Value Model

In reinforcement learning:

```text
State
  ↓
Value Model
  ↓
Expected future reward
```

The critic estimates how good a state or action is expected to be.

A simplified distinction:

| Model              | Main purpose                                     |
| ------------------ | ------------------------------------------------ |
| Reward Model       | Scores generated outputs according to preference |
| Policy Model       | Generates responses                              |
| Value/Critic Model | Estimates expected future reward                 |

---

# 18. Reward hacking

One major challenge.

Suppose humans often prefer detailed answers.

The reward model may learn:

```text
Longer answer = better
```

Then the LLM may generate:

```text
Very long answers
```

even when a short answer is better.

Example:

```text
Question:
What is RAG?
```

Good:

```text
RAG retrieves relevant information and provides it as context to an LLM.
```

But a poorly optimized reward model might reward:

```text
RAG is a revolutionary and extremely important advanced artificial
intelligence architecture that represents a groundbreaking...
```

This is **reward hacking** or over-optimization against imperfections in the reward model.

---

# 19. How do you prevent reward hacking?

Common techniques:

### 1. Diverse preference data

Don't teach:

```text
Always longer = better
```

Include:

```text
Short answers when appropriate
Detailed answers when appropriate
```

---

### 2. Separate evaluation models

Use another evaluator:

```text
Policy Model
     │
     ▼
Generated Answer
     │
     ├── Reward Model
     │
     └── Independent Evaluator
```

---

### 3. Human evaluation

Periodically sample outputs:

```text
Production Outputs
       ↓
Human Review
       ↓
Detect reward hacking
```

---

### 4. KL penalty

During RLHF:

[
FinalReward =
RewardModelScore
----------------

\beta KL
]

This prevents the policy from changing too aggressively.

---

# 20. Reward Model vs DPO

Traditional RLHF:

```text
Preference Data
      │
      ▼
Reward Model
      │
      ▼
Reward Scores
      │
      ▼
PPO
      │
      ▼
Aligned LLM
```

DPO:

```text
Preference Data
      │
      ▼
Direct Preference Optimization
      │
      ▼
Aligned LLM
```

DPO often avoids explicitly training a separate reward model.

---

# 21. Where would reward modeling be used in a real project?

Suppose you are building an enterprise AI assistant.

You collect:

```text
User question
      │
      ├── Response A
      │
      └── Response B
             │
             ▼
       Domain Expert
             │
             ▼
        Better response
```

Example:

```python
{
    "prompt":
        "How should we handle a payment dispute?",

    "chosen":
        (
            "Follow the approved dispute workflow, "
            "verify the transaction details, document "
            "the case, and escalate when required."
        ),

    "rejected":
        (
            "Tell the customer to contact someone."
        )
}
```

Train a reward model.

Then:

```text
Fine-tuned LLM
       │
       ▼
Generate candidate response
       │
       ▼
Reward Model
       │
       ▼
Quality score
```

You could use it for:

* RLHF
* ranking multiple generated answers
* rejection sampling
* offline evaluation
* model comparison

---

# Interview-ready answer

> **Reward modeling is the process of training a model to approximate human preferences by assigning a scalar reward score to an LLM response. The reward model is usually trained on preference pairs consisting of a prompt, a chosen response, and a rejected response.**
>
> **The objective is to assign a higher reward to the human-preferred response than the rejected response. A common pairwise loss is negative log sigmoid of the difference between the chosen and rejected rewards.**
>
> **In traditional RLHF, the reward model replaces continuous human evaluation during reinforcement learning. The policy model generates responses, the reward model scores them, and an algorithm such as PPO updates the policy to maximize reward while a KL penalty prevents the model from drifting too far from a reference model.**
>
> **I would evaluate a reward model primarily using pairwise ranking accuracy and human evaluation, while also monitoring for reward hacking and distribution shift.**

## Final mental model

```text
Humans say:

Response B is better than Response A
                │
                ▼
         Preference Dataset
                │
                ▼
           Reward Model
                │
                ▼
Learn:

Reward(B) > Reward(A)
                │
                ▼
Use reward to improve the LLM
```

**One-line definition:**

> **A reward model is a learned scoring model that converts human preferences about LLM outputs into numerical rewards.**
