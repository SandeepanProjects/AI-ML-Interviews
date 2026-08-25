# What is a **Chosen vs Rejected Response**?

This is a fundamental concept in **preference tuning**, especially algorithms such as **DPO** and traditional **RLHF**.

## Simple definition

For the **same prompt**, we have two or more possible model responses:

* **Chosen response** → the response preferred by a human, expert, user, or evaluator.
* **Rejected response** → a less preferred response.

```text
                    Same Prompt
                        │
             ┌──────────┴──────────┐
             │                     │
             ▼                     ▼
         Response A             Response B
             │                     │
             │ Human evaluates     │
             └──────────┬──────────┘
                        │
                 Which is better?
                        │
             ┌──────────┴──────────┐
             │                     │
             ▼                     ▼
          CHOSEN ✓              REJECTED ✗
```

The important idea is:

> A **rejected response does not necessarily mean a completely wrong response**. It means it is **less preferred than the chosen response**.

---

# 1. Simple example

Prompt:

```text
What is RAG?
```

Two responses:

### Response A

```text
RAG is a technique used with language models.
```

### Response B

```text
RAG retrieves relevant external information and provides it
to an LLM as context before generating an answer.
```

An evaluator prefers Response B.

So:

```python
example = {
    "prompt": "What is RAG?",

    "chosen": (
        "RAG retrieves relevant external information and provides "
        "it to an LLM as context before generating an answer."
    ),

    "rejected": (
        "RAG is a technique used with language models."
    )
}
```

```text
Chosen   → Response B ✓
Rejected → Response A ✗
```

Notice: Response A is not entirely false. It is simply:

* vague
* less useful
* less informative

---

# 2. Why use chosen and rejected responses?

Suppose we only use normal SFT:

```python
{
    "prompt": "Explain RAG",
    "answer": "RAG retrieves relevant documents before generation."
}
```

The model learns:

```text
Prompt → Desired answer
```

But with preference data:

```python
{
    "prompt": "Explain RAG",

    "chosen": "Detailed, accurate answer",

    "rejected": "Vague or lower-quality answer"
}
```

The model learns:

```text
                Prompt
                  │
         ┌────────┴────────┐
         │                 │
         ▼                 ▼
      Chosen ✓         Rejected ✗
         │                 │
         ▼                 ▼
 Increase probability  Decrease relative
                       probability
         │                 │
         └────────┬────────┘
                  ▼
            Better behavior
```

Mathematically:

[
P(\text{chosen} \mid \text{prompt})

>

P(\text{rejected} \mid \text{prompt})
]

---

# 3. The most important point: same prompt

A valid preference pair normally compares responses to the **same context/instruction**.

Correct:

```python
{
    "prompt": "Explain LoRA",

    "chosen": (
        "LoRA is a parameter-efficient fine-tuning technique "
        "that trains low-rank adapters while the base model "
        "remains mostly frozen."
    ),

    "rejected": (
        "LoRA is a technique for fine-tuning language models."
    )
}
```

Both responses answer:

```text
Explain LoRA
```

Therefore, the comparison is meaningful.

---

## Bad example

```python
{
    "prompt": "Explain LoRA",

    "chosen": (
        "LoRA uses low-rank adapters."
    ),

    "rejected": (
        "The capital of India is New Delhi."
    )
}
```

The rejected answer is irrelevant.

The model learns very little from this comparison because the difference is too obvious.

---

# 4. Rejected does NOT always mean incorrect

This is very important for interviews.

Consider:

```text
Prompt:
How do I handle API retries?
```

### Response A

```text
Retry failed requests with exponential backoff.
```

### Response B

```text
Use bounded retries with exponential backoff and jitter.
Only retry operations that are safe to repeat, such as idempotent
requests, and stop retrying after a configured maximum number.
```

Both answers are generally reasonable.

But an expert might choose B.

So:

```python
{
    "prompt": "How do I handle API retries?",

    "chosen": (
        "Use bounded retries with exponential backoff and jitter. "
        "Retry only operations that are safe to repeat, and stop "
        "after a configured maximum number of attempts."
    ),

    "rejected": (
        "Retry failed requests with exponential backoff."
    )
}
```

The rejected answer is:

```text
✓ Related
✓ Mostly correct
✗ Less complete
```

These are often **high-quality preference examples**.

---

# 5. How are chosen and rejected responses created?

## Method 1: Generate multiple responses

```text
Prompt
   │
   ▼
LLM generates 3 responses
   │
   ├── Response A
   ├── Response B
   └── Response C
           │
           ▼
     Human evaluation
           │
           ▼
       B > A > C
```

Then create pairs:

```python
data = [
    {
        "prompt": prompt,
        "chosen": response_b,
        "rejected": response_a
    },
    {
        "prompt": prompt,
        "chosen": response_b,
        "rejected": response_c
    },
    {
        "prompt": prompt,
        "chosen": response_a,
        "rejected": response_c
    }
]
```

---

# 6. Code: create preference pairs from rankings

Suppose human evaluators rank responses:

```python
ranked_responses = [
    {
        "text": "Detailed and accurate response",
        "rank": 1
    },
    {
        "text": "Short but correct response",
        "rank": 2
    },
    {
        "text": "Incorrect response",
        "rank": 3
    }
]
```

Lower rank means better:

```text
Rank 1 → Best
Rank 2 → Second
Rank 3 → Worst
```

Convert to chosen/rejected pairs:

```python
def create_preference_pairs(
    prompt: str,
    ranked_responses: list[dict]
) -> list[dict]:

    # Sort best to worst
    ranked_responses = sorted(
        ranked_responses,
        key=lambda x: x["rank"]
    )

    pairs = []

    # Compare every better response
    # against every worse response
    for i in range(len(ranked_responses)):

        for j in range(
            i + 1,
            len(ranked_responses)
        ):

            chosen = (
                ranked_responses[i]["text"]
            )

            rejected = (
                ranked_responses[j]["text"]
            )

            pairs.append({
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected
            })

    return pairs
```

Use it:

```python
prompt = "Explain RAG."

pairs = create_preference_pairs(
    prompt,
    ranked_responses
)

for pair in pairs:
    print(pair)
```

Output conceptually:

```text
Chosen: Detailed and accurate
Rejected: Short but correct

Chosen: Detailed and accurate
Rejected: Incorrect

Chosen: Short but correct
Rejected: Incorrect
```

---

# 7. How DPO uses chosen vs rejected

Let's see what happens internally.

Dataset:

```python
example = {
    "prompt": "What is RAG?",

    "chosen": (
        "RAG retrieves relevant information and provides it "
        "as context to the LLM."
    ),

    "rejected": (
        "RAG permanently stores all documents in the model."
    )
}
```

The model calculates:

```text
log probability of chosen response

P(chosen | prompt)
```

and:

```text
log probability of rejected response

P(rejected | prompt)
```

DPO tries to make:

[
P(chosen|prompt)

>

P(rejected|prompt)
]

---

## Simplified code

```python
import torch
import torch.nn.functional as F


def preference_loss(
    chosen_log_prob: torch.Tensor,
    rejected_log_prob: torch.Tensor
):

    # How much more likely is the
    # chosen response than rejected?
    preference_margin = (
        chosen_log_prob
        -
        rejected_log_prob
    )

    # Larger margin = better
    loss = (
        -F.logsigmoid(
            preference_margin
        )
    )

    return loss.mean()
```

---

## Example

```python
chosen_log_prob = torch.tensor([
    -1.0
])

rejected_log_prob = torch.tensor([
    -3.0
])

loss = preference_loss(
    chosen_log_prob,
    rejected_log_prob
)

print(loss.item())
```

Why is this good?

```text
Chosen log probability   = -1
Rejected log probability = -3
```

Since:

```text
-1 > -3
```

the chosen response is more likely.

---

# 8. What happens when the model prefers the wrong answer?

Suppose:

```python
chosen_log_prob = torch.tensor([
    -4.0
])

rejected_log_prob = torch.tensor([
    -1.0
])
```

Now:

```text
Chosen   = -4
Rejected = -1
```

The model currently prefers the rejected response.

```text
Rejected probability ↑
Chosen probability   ↓
```

The loss becomes larger.

During backpropagation:

```text
Gradient
   │
   ▼
Increase chosen probability
        +
Decrease rejected probability
```

Simplified training:

```python
for example in dataset:

    chosen_log_prob = get_log_probability(
        model=model,
        prompt=example["prompt"],
        response=example["chosen"]
    )

    rejected_log_prob = get_log_probability(
        model=model,
        prompt=example["prompt"],
        response=example["rejected"]
    )

    loss = preference_loss(
        chosen_log_prob,
        rejected_log_prob
    )

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()
```

---

# 9. What does `get_log_probability()` do?

For an LLM, the model generates tokens.

Example:

```text
Chosen response:

"RAG retrieves documents"
```

Tokenized:

```text
[RAG] [retrieves] [documents]
```

The model calculates probabilities:

```text
P(RAG | prompt)

P(retrieves | prompt, RAG)

P(documents | prompt, RAG, retrieves)
```

The total sequence probability is:

[
P(response|prompt)
==================

\prod_i P(token_i|previous\ tokens)
]

We usually use log probabilities:

[
\log P(response|prompt)
=======================

\sum_i \log P(token_i|previous\ tokens)
]

Simplified code:

```python
def get_sequence_log_probability(
    model,
    input_ids,
    labels
):

    outputs = model(
        input_ids=input_ids
    )

    logits = outputs.logits

    log_probs = torch.log_softmax(
        logits,
        dim=-1
    )

    token_log_probs = log_probs.gather(
        dim=-1,
        index=labels.unsqueeze(-1)
    ).squeeze(-1)

    return token_log_probs.sum()
```

Real implementations must handle:

* shifted labels
* attention masks
* padding
* masking prompt tokens
* variable sequence lengths

Libraries such as TRL handle these details.

---

# 10. Real DPO dataset example

For a conversational model, the dataset may look like:

```python
from datasets import Dataset

dataset = Dataset.from_list([
    {
        "prompt": (
            "User: My payment failed.\n"
            "Assistant:"
        ),

        "chosen": (
            "I'm sorry your payment failed. Please check whether "
            "your payment method is valid and has sufficient "
            "available balance. If the problem continues, I can "
            "guide you through the next steps."
        ),

        "rejected": (
            "Your payment failed. Contact support."
        )
    },

    {
        "prompt": (
            "User: How do I reset my password?\n"
            "Assistant:"
        ),

        "chosen": (
            "Use the 'Forgot Password' option on the sign-in page. "
            "You will receive instructions to verify your identity "
            "and create a new password."
        ),

        "rejected": (
            "Reset your password."
        )
    }
])
```

Then a preference optimization algorithm learns:

```text
For each prompt:

Chosen response → more likely
Rejected response → less likely
```

---

# 11. Chosen/rejected examples by quality

## Example A: Correct vs incorrect

```text
Prompt:
What is LoRA?

CHOSEN:
LoRA trains small low-rank adapter matrices while keeping most
base model weights frozen.

REJECTED:
LoRA trains every parameter in the model.
```

This teaches **correctness**.

---

## Example B: Helpful vs unhelpful

```text
Prompt:
My API request is timing out.

CHOSEN:
Check the timeout configuration, downstream latency, and network
errors. Use bounded retries with exponential backoff and instrument
latency before increasing timeouts blindly.

REJECTED:
Increase the timeout.
```

This teaches **helpfulness**.

---

## Example C: Safe vs unsafe

```text
Prompt:
How should I handle sensitive customer data?

CHOSEN:
Minimize collection, restrict access, protect data in transit and
at rest, and follow the applicable security and privacy policies.

REJECTED:
Store all customer data in logs so debugging is easier.
```

This teaches **safety and good engineering practice**.

---

## Example D: Concise vs unnecessarily verbose

```text
Prompt:
What is Redis?

CHOSEN:
Redis is an in-memory data store commonly used for caching,
sessions, queues, and fast data access.

REJECTED:
Redis is a revolutionary and extraordinarily powerful technology...
```

This can teach the model your preferred style.

---

# 12. How should you choose the rejected response?

A good rejected response should ideally be:

### Relevant

It should answer the same prompt.

### Realistic

It should look like something an LLM might actually generate.

### Meaningfully worse

It should be worse because of:

* incorrectness
* missing information
* poor reasoning
* unsafe advice
* bad formatting
* hallucination
* unnecessary verbosity
* poor tone

---

## Bad rejected example

```text
Prompt:
What is Kubernetes?

Rejected:
Banana airplane purple elephant.
```

Too easy.

---

## Better rejected example

```text
Prompt:
What is Kubernetes?

Chosen:
Kubernetes is an orchestration platform for deploying, scaling,
and managing containerized applications.

Rejected:
Kubernetes is a Docker container that automatically runs your
application on one server.
```

The rejected response is plausible but wrong.

That teaches the model a more useful distinction.

---

# 13. Important: chosen does not mean "perfect"

This is a subtle but important point.

Suppose:

```text
Response A:
Good, but slightly verbose.

Response B:
Correct, concise, and actionable.
```

You choose B.

Does that mean A is garbage?

No.

It means:

[
B > A
]

Preference tuning is **relative**.

```text
A can be good

B can be better
```

The dataset teaches the ranking.

---

# 14. Common data quality problem

Imagine two annotators:

```text
Annotator 1:
A > B

Annotator 2:
B > A
```

This creates inconsistent labels.

Example:

```python
bad_dataset = [

    {
        "prompt": "Explain RAG",
        "chosen": "Response A",
        "rejected": "Response B"
    },

    {
        "prompt": "Explain RAG",
        "chosen": "Response B",
        "rejected": "Response A"
    }
]
```

The model receives conflicting signals.

A basic validation:

```python
def find_conflicting_pairs(dataset):

    preferences = set()

    conflicts = []

    for example in dataset:

        key = (
            example["prompt"],
            example["chosen"],
            example["rejected"]
        )

        reverse_key = (
            example["prompt"],
            example["rejected"],
            example["chosen"]
        )

        if reverse_key in preferences:

            conflicts.append(
                example
            )

        preferences.add(key)

    return conflicts
```

In production, you would also use:

* multiple annotators
* agreement metrics
* expert review
* majority voting
* annotation guidelines

---

# 15. Chosen vs rejected in the complete LLM training pipeline

```text
                     PRETRAINING
                          │
                          ▼
                    Base LLM
                          │
                          ▼
                 INSTRUCTION TUNING
                          │
                          ▼
                  Helpful LLM
                          │
                          ▼
                PREFERENCE DATA
                          │
              ┌───────────┴───────────┐
              │                       │
              ▼                       ▼
         CHOSEN ✓                 REJECTED ✗
              │                       │
              └───────────┬───────────┘
                          │
                          ▼
                   DPO / RLHF
                          │
                          ▼
                    Aligned LLM
```

---

# Interview-ready answer

> **A chosen and rejected response are two candidate answers to the same prompt used in preference learning. The chosen response is the one preferred by a human or evaluator, while the rejected response is less preferred. The rejected response does not necessarily have to be completely incorrect—it may simply be less helpful, less accurate, less safe, or less clear.**
>
> **During algorithms such as DPO, the model computes the probability of both responses and is trained to increase the relative probability of the chosen response compared with the rejected response. High-quality preference datasets use realistic rejected answers so the model learns subtle distinctions in quality rather than just obvious nonsense.**

## One-line mental model

```text
Same Prompt

Chosen ✓  → "Generate behavior like this more often"

Rejected ✗ → "Generate behavior like this less often"
```

That is the core idea behind **chosen vs rejected responses in preference tuning**.
