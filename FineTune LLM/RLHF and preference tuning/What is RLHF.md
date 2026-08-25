# What is RLHF?

**RLHF = Reinforcement Learning from Human Feedback.**

It is a technique used to make an LLM's behavior better aligned with what humans consider:

* helpful
* accurate
* safe
* polite
* relevant
* instruction-following

The core idea is:

> Instead of training the model only on the “correct answer,” humans provide feedback about **which answers are better**.

---

# 1. Why isn't supervised fine-tuning enough?

Suppose a user asks:

> Explain what RAG is.

With **Supervised Fine-Tuning (SFT)**, the training data might be:

```text
Question:
What is RAG?

Target answer:
RAG retrieves relevant external information and provides it
to an LLM to generate a grounded answer.
```

The model learns:

```text
Input → Expected Output
```

But in the real world, there can be multiple good answers.

### Response A

```text
RAG retrieves relevant documents and gives them to the LLM.
```

### Response B

```text
RAG retrieves relevant external information, places it into
the model's context, and helps generate more grounded answers.
```

Both may be correct.

But humans might prefer **B** because it is:

* clearer
* more complete
* more helpful

Traditional SFT does not naturally capture all such preferences.

RLHF tries to learn:

```text
Human preference:
Answer B > Answer A
```

---

# 2. High-level RLHF pipeline

A traditional RLHF pipeline looks like this:

```text
                 ┌───────────────────────┐
                 │  Pretrained Base LLM  │
                 └───────────┬───────────┘
                             │
                             ▼
                    Supervised Fine-Tuning
                             │
                             ▼
                        SFT Model
                             │
                             ▼
                Generate Multiple Answers
                             │
                    ┌────────┴────────┐
                    ▼                 ▼
                Answer A          Answer B
                    │                 │
                    └────────┬────────┘
                             ▼
                       Human Feedback
                             │
                     Which is better?
                             │
                             ▼
                        Reward Model
                             │
                             ▼
                  Reinforcement Learning
                       (PPO etc.)
                             │
                             ▼
                      Aligned LLM
```

The main stages are:

1. **Pre-training**
2. **Supervised Fine-Tuning (SFT)**
3. **Collect human preferences**
4. **Train a Reward Model**
5. **Optimize the LLM using Reinforcement Learning**

---

# 3. Stage 1: Pre-training

The base model learns from huge amounts of text.

```text
Internet
Books
Code
Articles
Documents
      │
      ▼
Next Token Prediction
      │
      ▼
Pretrained LLM
```

For example:

```text
Input:

The capital of France is

Target:

Paris
```

The objective is generally:

```text
Predict the next token
```

After pre-training, the model has broad language knowledge.

But it may not reliably behave like an assistant.

---

# 4. Stage 2: Supervised Fine-Tuning

We then train the model on demonstrations.

Example dataset:

```python
sft_data = [
    {
        "prompt": "What is RAG?",
        "response": (
            "RAG stands for Retrieval-Augmented Generation. "
            "It retrieves relevant external information and "
            "provides it to an LLM to generate a grounded answer."
        )
    },
    {
        "prompt": "What is LoRA?",
        "response": (
            "LoRA is a parameter-efficient fine-tuning technique "
            "that trains small low-rank matrices while keeping "
            "most base model weights frozen."
        )
    }
]
```

The model learns:

```text
User instruction
       ↓
Helpful assistant response
```

But SFT still depends on demonstrations.

RLHF adds **comparative human preferences**.

---

# 5. Stage 3: Collect human preferences

Give the same prompt to the model multiple times.

```text
Prompt:

Explain RAG.
```

Generate answers:

### Answer A

```text
RAG uses a database.
```

### Answer B

```text
RAG retrieves relevant documents and supplies them as context
to an LLM before it generates an answer.
```

A human annotator selects:

```text
B > A
```

The preference dataset becomes:

```python
preference_data = [
    {
        "prompt": "Explain RAG.",

        "chosen": (
            "RAG retrieves relevant documents and supplies "
            "them as context to an LLM before it generates "
            "an answer."
        ),

        "rejected": (
            "RAG uses a database."
        )
    }
]
```

This is the key RLHF data structure:

```text
Prompt
   │
   ├── Chosen Answer ✓
   │
   └── Rejected Answer ✗
```

---

# 6. What is a Reward Model?

A Reward Model learns to predict:

> How much would a human prefer this answer?

Conceptually:

```text
Prompt + Response
        │
        ▼
    Reward Model
        │
        ▼
   Reward Score
```

Example:

```text
Prompt:
Explain RAG

Response A:
RAG uses a database.

Reward:
0.2
```

```text
Response B:
RAG retrieves relevant documents and provides them
to the LLM as context.

Reward:
0.95
```

The reward model learns:

```text
Human preferences → numerical reward
```

---

# 7. How is the reward model trained?

Suppose:

```text
Prompt: What is LoRA?

Chosen:
LoRA trains small low-rank matrices while keeping the
base model mostly frozen.

Rejected:
LoRA changes all model weights.
```

We want:

```text
Reward(chosen) > Reward(rejected)
```

Mathematically:

```text
r(chosen) > r(rejected)
```

A common preference loss is:

[
L = -\log(\sigma(r_{chosen} - r_{rejected}))
]

Where:

* (r_{chosen}) = reward for preferred response
* (r_{rejected}) = reward for rejected response
* (\sigma) = sigmoid function

The loss encourages:

```text
reward(chosen)
        >
reward(rejected)
```

---

# 8. Simplified reward model code

Using a Hugging Face sequence classification model:

```python
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

import torch
```

Load:

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

The reward model outputs one number:

```text
Prompt + Answer
       ↓
Reward Model
       ↓
0.83
```

---

# 9. Preference loss in code

A simplified implementation:

```python
import torch.nn.functional as F
```

```python
def preference_loss(
    chosen_reward,
    rejected_reward
):
    """
    Train the reward model so that
    chosen answers receive a higher reward.
    """

    difference = (
        chosen_reward
        - rejected_reward
    )

    loss = -F.logsigmoid(
        difference
    ).mean()

    return loss
```

Example:

```python
chosen_reward = torch.tensor([
    2.5
])

rejected_reward = torch.tensor([
    0.7
])
```

Calculate:

```python
loss = preference_loss(
    chosen_reward,
    rejected_reward
)

print(loss)
```

If:

```text
chosen_reward = 2.5
rejected_reward = 0.7
```

The model is doing well because:

```text
2.5 > 0.7
```

But if:

```text
chosen_reward = 0.5
rejected_reward = 2.0
```

The loss becomes larger, pushing the model to correct the ranking.

---

# 10. What happens after training the reward model?

We now have:

```text
Fine-tuned LLM
      +
Reward Model
```

The LLM generates an answer:

```text
Question
    ↓
LLM
    ↓
Generated Answer
    ↓
Reward Model
    ↓
Reward Score
```

Then reinforcement learning updates the LLM to generate responses with higher reward.

---

# 11. PPO in RLHF

Traditionally, RLHF often uses:

> **PPO = Proximal Policy Optimization**

The LLM is treated as a policy.

```text
LLM
 ↓
Generates tokens
 ↓
Response
 ↓
Reward Model
 ↓
Reward
 ↓
PPO updates LLM
```

Conceptually:

```python
for prompt in prompts:

    # 1. Model generates answer
    response = policy_model.generate(
        prompt
    )

    # 2. Reward model evaluates answer
    reward = reward_model(
        prompt,
        response
    )

    # 3. Update the policy
    update_policy_with_ppo(
        prompt,
        response,
        reward
    )
```

This is simplified pseudocode, but the concept is correct.

---

# 12. Why do we need PPO?

Suppose the model finds a response that gets a high reward.

If we aggressively update the model:

```text
Old model
    ↓↓↓↓↓
Massive update
    ↓
New model
```

The model may:

* become unstable
* forget useful behavior
* exploit the reward model
* generate repetitive answers

PPO tries to limit overly large updates.

Conceptually:

```text
Old Policy
     │
     │ small controlled update
     ▼
New Policy
```

The word **Proximal** means:

> Keep the new model reasonably close to the previous model.

---

# 13. The KL penalty

RLHF often includes a **KL divergence penalty**.

The objective conceptually looks like:

[
Reward = RewardModelScore - \beta \cdot KL
]

Where:

* `RewardModelScore` = how much humans would prefer the answer
* `KL` = how far the model moved from the reference model
* `β` = penalty strength

Example:

```text
Good answer:
Reward Model Score = 0.95
KL penalty         = 0.10

Final reward:
0.85
```

The KL penalty says:

> Don't completely change the model just to maximize the reward.

Simplified:

```python
def final_reward(
    reward_score,
    kl_divergence,
    beta=0.1
):

    return (
        reward_score
        - beta * kl_divergence
    )
```

---

# 14. The complete RLHF loop

```text
                   ┌─────────────────┐
                   │    SFT Model    │
                   └────────┬────────┘
                            │
                            ▼
                        Prompt
                            │
                            ▼
                    Generate Response
                            │
                            ▼
                   ┌────────────────┐
                   │  Reward Model  │
                   └────────┬───────┘
                            │
                         Reward
                            │
                            ▼
                       PPO Update
                            │
                            ▼
                   Updated Policy Model
                            │
                            └───────┐
                                    │
                                    ▼
                              Repeat Loop
```

---

# 15. Modern alternative: DPO

In practice, you should also know **DPO (Direct Preference Optimization)**.

DPO was introduced because traditional RLHF with:

```text
Reward Model
    +
PPO
```

can be:

* complex
* expensive
* unstable

DPO directly learns from:

```text
Prompt
Chosen response
Rejected response
```

Instead of:

```text
Train Reward Model
       ↓
Run PPO
```

DPO does:

```text
Preference Dataset
       ↓
DPO Training
       ↓
Aligned Model
```

Example dataset:

```python
dpo_data = [
    {
        "prompt": (
            "Explain RAG."
        ),

        "chosen": (
            "RAG retrieves relevant external information "
            "and provides it as context to an LLM."
        ),

        "rejected": (
            "RAG is an AI model."
        )
    }
]
```

A simplified training setup using TRL looks like:

```python
from datasets import Dataset

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)

from trl import (
    DPOTrainer,
    DPOConfig
)
```

Load data:

```python
dataset = Dataset.from_list(
    dpo_data
)
```

Load model:

```python
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME
)

reference_model = (
    AutoModelForCausalLM
    .from_pretrained(
        MODEL_NAME
    )
)
```

Configure:

```python
training_args = DPOConfig(

    output_dir="./dpo_output",

    learning_rate=5e-7,

    per_device_train_batch_size=2,

    num_train_epochs=1,

    logging_steps=10
)
```

Create trainer:

```python
trainer = DPOTrainer(

    model=model,

    ref_model=reference_model,

    args=training_args,

    train_dataset=dataset,

    processing_class=tokenizer
)
```

Train:

```python
trainer.train()
```

Conceptually:

```text
Chosen Answer
      ↑
      │ Increase probability
      │
Prompt
      │
      │ Decrease probability
      ↓
Rejected Answer
```

---

# 16. RLHF vs SFT

| Feature                     | SFT            | RLHF                        |
| --------------------------- | -------------- | --------------------------- |
| Training data               | Input → answer | Chosen vs rejected          |
| Learns exact demonstrations | Yes            | Indirectly                  |
| Learns human preferences    | Limited        | Yes                         |
| Reward model                | No             | Usually yes in classic RLHF |
| Reinforcement learning      | No             | Yes in classic RLHF         |
| Complexity                  | Lower          | Higher                      |
| Cost                        | Lower          | Higher                      |

---

# 17. Example: Customer support assistant

Suppose the user says:

> My payment failed.

### Response A

```text
Contact support.
```

### Response B

```text
I'm sorry your payment failed. Please check whether your
payment method has sufficient funds and try again. If the
problem continues, I can help you identify the next steps.
```

Humans prefer:

```text
B > A
```

Why?

Because B is:

```text
✓ Helpful
✓ Polite
✓ Actionable
✓ Clear
```

RLHF can teach the model this preference.

---

# 18. Why is RLHF used?

RLHF is primarily used to improve **alignment**.

## A. Better instruction following

Without alignment:

```text
User:
Explain Python in 3 sentences.

Model:
[Writes 20 paragraphs]
```

With better preference alignment:

```text
Model:
[Writes approximately 3 useful sentences]
```

---

## B. More helpful answers

Humans can rank:

```text
Unhelpful
    <
Helpful
    <
Highly useful
```

The model learns which response patterns humans prefer.

---

## C. Better safety behavior

Humans can identify:

```text
Safe answer ✓
Unsafe answer ✗
```

The preference dataset can reward safer responses.

---

## D. Better style

For example, users might prefer:

```text
Clear
Structured
Concise
Professional
```

over:

```text
Confusing
Verbose
Unstructured
```

RLHF can optimize toward those preferences.

---

## E. Learn subjective quality

Many qualities are difficult to represent with a single "correct answer."

For example:

> Which explanation is better?

The answer may depend on:

* clarity
* tone
* completeness
* relevance
* usefulness

Human preference data captures this better than ordinary labels.

---

# 19. Real-world training pipeline

A production pipeline might look like:

```text
                    BASE MODEL
                        │
                        ▼
              Supervised Fine-Tuning
                        │
                        ▼
                   SFT MODEL
                        │
                        ▼
             Generate candidate answers
                        │
                        ▼
              Human preference ranking
                        │
                        ▼
                Preference Dataset
                        │
              ┌─────────┴──────────┐
              ▼                    ▼
         Classic RLHF             DPO
              │                    │
              ▼                    ▼
         Reward Model         Direct Optimization
              │
              ▼
             PPO
              │
              ▼
          Aligned Model
```

---

# 20. Practical enterprise example

Suppose you're building an internal financial-support assistant.

You have:

```text
Base Model
    │
    ▼
SFT
    │
    ├── Company terminology
    ├── Internal workflows
    └── Response format
```

Then collect expert feedback:

```text
Question:
How should we handle an unauthorized transaction?

Answer A:
Tell the customer to contact the bank.

Answer B:
Verify the customer's identity, follow the unauthorized
transaction workflow, document the case, and escalate
according to the approved policy.
```

Domain experts select:

```text
B > A
```

You can use those preferences to improve the model.

The full system might be:

```text
SFT
 +
Preference Optimization
 +
RAG
 +
Guardrails
 +
Evaluation
```

---

# 21. Important limitations of RLHF

RLHF is not magic.

## Problem 1: Expensive human feedback

You need:

* annotators
* domain experts
* quality control
* agreement checks

---

## Problem 2: Human disagreement

Example:

```text
Reviewer A:
Answer A is better.

Reviewer B:
Answer B is better.
```

Preferences can be subjective.

---

## Problem 3: Reward hacking

The model may learn to maximize reward without actually being better.

For example:

```text
Always sound confident
Always write long answers
Always use polite language
```

Even when those behaviors are not useful.

---

## Problem 4: Reward model errors

The pipeline is:

```text
Human preferences
       ↓
Reward model
       ↓
Approximation of preferences
```

The reward model can make mistakes.

The policy may optimize those mistakes.

---

# 22. Interview-ready answer

> **RLHF stands for Reinforcement Learning from Human Feedback. It is used to align an LLM with human preferences such as helpfulness, safety, relevance, and instruction following.**
>
> **The traditional pipeline starts with a pretrained model, performs supervised fine-tuning using high-quality demonstrations, then generates multiple candidate responses for prompts. Humans rank or compare the responses, creating chosen and rejected pairs. These preferences are used to train a reward model that predicts how preferable a response is. Finally, reinforcement learning, traditionally PPO with a KL penalty, updates the language model to generate higher-reward responses while staying reasonably close to the original model.**
>
> **Modern approaches such as DPO can directly optimize preference pairs without separately training a reward model and running PPO.**

# Final mental model

```text
SFT teaches:

"What should a good answer look like?"

                +

Human preferences teach:

"Which good answer is better?"

                ↓

RLHF / Preference Optimization

                ↓

More aligned LLM
```

**One-line answer:**

> **SFT teaches an LLM from demonstrations; RLHF teaches it from human preferences about which outputs are better.**
