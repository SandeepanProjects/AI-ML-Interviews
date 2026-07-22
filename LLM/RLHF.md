# Reinforcement Learning from Human Feedback (RLHF) Explained End-to-End

RLHF (Reinforcement Learning from Human Feedback) is the training process used to make Large Language Models (LLMs) **more helpful, honest, and aligned with human preferences**.

It is used in models like **ChatGPT, Claude, Gemini, and Llama** (or closely related preference-optimization methods such as DPO).

The overall idea is:

> **First teach the model language, then teach it what humans prefer.**

---

# Why RLHF?

Suppose we train an LLM only with next-token prediction.

Prompt:

```text
How do I become rich?
```

Possible outputs:

```text
Response A:
Work hard, invest regularly, and build valuable skills.
```

```text
Response B:
Rob a bank.
```

Both are grammatically correct.

A language model trained only on internet text may assign a non-zero probability to either response because it predicts what text could come next—not what humans want.

RLHF teaches the model:

* Helpful responses
* Safe responses
* Honest responses
* Polite responses

---

# Three Training Stages

```text
                Internet Data
                      │
                      ▼
      Stage 1: Pretraining (Next Token Prediction)
                      │
                      ▼
             Base Language Model
                      │
                      ▼
      Stage 2: Supervised Fine-Tuning (SFT)
                      │
                      ▼
             Instruction Model
                      │
                      ▼
      Stage 3: RLHF
         Human Preferences
                │
                ▼
          Reward Model
                │
                ▼
      PPO (or another RL optimizer)
                │
                ▼
          Final Chat Model
```

---

# Stage 1 — Pretraining

The model learns language from massive text corpora.

Training objective:

```text
Predict the next token.
```

Example:

```text
The capital of France is ___
```

Target:

```text
Paris
```

Loss:

```python
loss = CrossEntropy(logits, labels)
```

This stage gives the model general language understanding but not conversational behavior.

---

# Stage 2 — Supervised Fine-Tuning (SFT)

Now we use high-quality prompt–response pairs written by humans.

Example dataset:

```text
Prompt:
Explain recursion.

Answer:
Recursion is a function calling itself...
```

PyTorch example:

```python
for batch in dataloader:

    outputs = model(
        input_ids=batch["input_ids"],
        labels=batch["labels"]
    )

    loss = outputs.loss

    loss.backward()

    optimizer.step()
```

The model learns to follow instructions.

---

# Stage 3 — Human Preference Collection

Now we ask humans to compare responses.

Prompt:

```text
Explain machine learning.
```

Model generates:

### Response A

```text
Machine learning is a branch of AI...
```

### Response B

```text
Machine learning is computer magic.
```

Human chooses:

```text
A is better.
```

Dataset:

| Prompt         | Chosen | Rejected |
| -------------- | ------ | -------- |
| Explain ML     | A      | B        |
| Write email    | A      | B        |
| Explain Python | A      | B        |

Notice that humans **rank** responses rather than assigning numerical scores.

---

# Stage 4 — Reward Model

Instead of asking humans every time, train a model that predicts human preferences.

Input:

```text
Prompt

+

Response
```

Output:

```text
Reward Score

0.95
```

Architecture:

```text
Prompt
      │
      ▼
LLM Encoder
      │
      ▼
Hidden State
      │
      ▼
Linear Layer
      │
      ▼
Reward Score
```

---

## Reward Model Code

```python
import torch
import torch.nn as nn

class RewardModel(nn.Module):

    def __init__(self, hidden_size):
        super().__init__()

        self.reward_head = nn.Linear(hidden_size, 1)

    def forward(self, hidden):

        return self.reward_head(hidden)
```

Output:

```text
Response A

0.95
```

```text
Response B

0.12
```

Higher is better.

---

# Training the Reward Model

Suppose:

Chosen response:

```text
Score = 3.2
```

Rejected response:

```text
Score = 1.8
```

Loss:

[
L = -\log\left(\sigma(r_{\text{chosen}} - r_{\text{rejected}})\right)
]

where:

* (r_{\text{chosen}}) = reward for preferred answer
* (r_{\text{rejected}}) = reward for rejected answer
* (\sigma) = sigmoid function

PyTorch:

```python
import torch

chosen = torch.tensor([3.2])

rejected = torch.tensor([1.8])

loss = -torch.log(torch.sigmoid(chosen - rejected))

print(loss)
```

This pushes the chosen response to receive a higher reward than the rejected one.

---

# Stage 5 — Reinforcement Learning

Now the LLM generates responses.

Example:

```text
Prompt

Explain AI.
```

Generated response:

```text
Artificial Intelligence...
```

Reward model:

```text
Reward = 9.8
```

The RL algorithm updates the LLM so it becomes more likely to generate responses with higher predicted rewards.

---

# PPO (Proximal Policy Optimization)

The original RLHF implementations (such as InstructGPT) used PPO.

Pipeline:

```text
Prompt
      │
      ▼
Policy Model
      │
      ▼
Generated Response
      │
      ▼
Reward Model
      │
      ▼
Reward
      │
      ▼
PPO Update
      │
      ▼
Improved Policy
```

---

## Simplified PPO Example

```python
reward = reward_model(response)

loss = -reward.mean()

loss.backward()

optimizer.step()
```

In a real PPO implementation, the objective also includes:

* Importance sampling ratios
* Clipping
* Value function loss
* Entropy bonus
* KL penalty (often used to keep the policy close to the reference model)

The above snippet is only an intuition, not a complete PPO algorithm.

---

# Why PPO?

Without constraints, the model may exploit weaknesses in the reward model.

Example:

```text
Reward model likes:

"Thank you."
```

The LLM could generate:

```text
Thank you.

Thank you.

Thank you.

Thank you.
```

to maximize reward.

PPO uses clipping and regularization (often including a KL penalty to the reference model) to reduce destructive updates and help maintain stable learning.

---

# Complete RLHF Pipeline

```text
          Internet Text
                │
                ▼
      Next Token Prediction
                │
                ▼
         Base Language Model
                │
                ▼
      Supervised Fine-Tuning
                │
                ▼
     Instruction Following Model
                │
                ▼
     Generate Multiple Answers
                │
                ▼
        Human Preference Ranking
                │
                ▼
         Reward Model Training
                │
                ▼
      Reinforcement Learning (PPO)
                │
                ▼
         Final Aligned Model
```

---

# Example

Prompt:

```text
How should I prepare for interviews?
```

Model outputs:

### Response A

```text
Practice coding, review fundamentals,
and do mock interviews.
```

Reward:

```text
9.8
```

### Response B

```text
Just memorize random answers.
```

Reward:

```text
2.1
```

RLHF increases the probability of generating Response A in the future.

---

# RLHF Challenges

### 1. Reward Hacking

The model may exploit flaws in the reward model instead of producing genuinely better responses.

### 2. Expensive Human Labeling

Millions of preference comparisons require significant human effort.

### 3. PPO Complexity

PPO is computationally expensive and can be unstable to tune.

### 4. Reward Model Errors

If the reward model is biased or inaccurate, the policy learns those mistakes.

---

# Modern Alternatives

Many recent LLMs use preference optimization methods that avoid full reinforcement learning.

| Method                               | Uses RL? | Notes                                                             |
| ------------------------------------ | -------- | ----------------------------------------------------------------- |
| RLHF + PPO                           | Yes      | Original InstructGPT approach                                     |
| DPO (Direct Preference Optimization) | No       | Learns directly from preference pairs; simpler and widely adopted |
| ORPO                                 | No       | Combines supervised and preference learning in one objective      |
| KTO                                  | No       | Another preference-based optimization method                      |

Today, **DPO** has become a popular alternative because it is simpler to train while achieving strong alignment performance.

---

# Interview Questions

1. Why is pretraining alone insufficient for building a chatbot?
2. What is the purpose of Supervised Fine-Tuning (SFT)?
3. How are human preference datasets collected?
4. What is a reward model, and how is it trained?
5. Why did InstructGPT use PPO for RLHF?
6. What is reward hacking, and how can it be mitigated?
7. Why is a KL penalty often included during RLHF?
8. How does DPO differ from RLHF with PPO?
9. What are the advantages and disadvantages of RLHF?
10. Why have many modern LLMs shifted from PPO-based RLHF to methods like DPO?
