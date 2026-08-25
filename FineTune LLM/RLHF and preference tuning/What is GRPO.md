# What is GRPO?

**GRPO = Group Relative Policy Optimization.**

It is a reinforcement-learning algorithm for training LLMs, introduced in the **DeepSeekMath** work. It is closely related to PPO, but its major idea is:

> Instead of using a separate value/critic model to calculate advantage, GRPO generates multiple responses for the same prompt and compares their rewards **within the group**.

This makes it more memory-efficient than traditional PPO. ([arXiv][1])

---

# 1. The simplest intuition

Suppose the prompt is:

```text
What is 12 × 13?
```

The model generates **4 responses**:

```text
Prompt: What is 12 × 13?

Response 1 → 156  ✓
Response 2 → 154  ✗
Response 3 → 156  ✓
Response 4 → 169  ✗
```

Assign rewards:

```text
Response 1 → 1
Response 2 → 0
Response 3 → 1
Response 4 → 0
```

Now GRPO compares the responses **relative to each other**.

```text
                Same Prompt
                    │
         Generate a group of answers
                    │
       ┌────────────┼────────────┐
       ▼            ▼            ▼
      R1           R2           R3 ...
       │            │            │
    Reward=1     Reward=0     Reward=1
       │            │            │
       └──────── Group Comparison
                       │
                       ▼
              Calculate Advantage
                       │
                       ▼
            Increase good responses
            Decrease bad responses
```

That is why it is called:

> **Group Relative Policy Optimization**

---

# 2. Why was GRPO introduced?

Traditional PPO for LLM alignment often involves multiple components:

```text
Policy Model
     +
Reference Model
     +
Reward Model
     +
Value / Critic Model
```

The value model estimates:

```text
How good was this response expected to be?
```

This adds memory and complexity.

GRPO removes the separate critic/value-model requirement by estimating the advantage from the rewards of multiple responses generated for the same prompt. The original DeepSeekMath paper introduced GRPO as a PPO variant that improves memory usage. ([arXiv][1])

---

# 3. PPO vs GRPO

## PPO

```text
Prompt
   │
   ▼
Generate response
   │
   ▼
Calculate reward
   │
   ▼
Value Model estimates expected reward
   │
   ▼
Advantage = Reward - Expected Value
   │
   ▼
Update policy
```

The advantage is conceptually:

[
A = R - V
]

Where:

* (R) = actual reward
* (V) = estimated value from critic

---

## GRPO

```text
Prompt
   │
   ▼
Generate GROUP of responses
   │
   ├── Response 1
   ├── Response 2
   ├── Response 3
   └── Response 4
          │
          ▼
    Calculate rewards
          │
          ▼
 Compare rewards within group
          │
          ▼
 Calculate relative advantage
          │
          ▼
     Update policy
```

No separate value model is needed.

---

# 4. The key idea: group-relative advantage

Suppose:

```text
Prompt:
Solve 12 × 13
```

The model generates 4 responses.

```python
rewards = [1.0, 0.0, 1.0, 0.0]
```

Calculate the group mean:

```python
import torch

rewards = torch.tensor([
    1.0,
    0.0,
    1.0,
    0.0
])

mean_reward = rewards.mean()

print(mean_reward)
```

Output:

```text
0.5
```

Now calculate standard deviation:

```python
std_reward = rewards.std(
    unbiased=False
)

print(std_reward)
```

Then calculate normalized advantages:

[
A_i =
\frac{
R_i - \mu_{group}
}{
\sigma_{group}
}
]

Code:

```python
advantages = (
    rewards - mean_reward
) / (
    std_reward + 1e-8
)

print(advantages)
```

Conceptually:

```text
Rewards:

[1, 0, 1, 0]

Mean = 0.5

Advantages:

[+1, -1, +1, -1]
```

Therefore:

```text
Correct responses   → Positive advantage
Wrong responses     → Negative advantage
```

This is the core of GRPO.

Current TRL documentation describes GRPO as generating a group of completions per prompt, scoring them, and computing normalized group-relative advantages. ([GitHub][2])

---

# 5. Why generate multiple responses?

Suppose we only generate one response:

```text
Prompt
   │
   ▼
Response
   │
   ▼
Reward = 0.8
```

Is `0.8` good?

Maybe.

But compare four responses:

```text
Response A → 0.2
Response B → 0.4
Response C → 0.8
Response D → 0.9
```

Now we know:

```text
D is best
C is good
A is poor
```

GRPO uses this relative comparison.

```text
                  GROUP
        ┌──────────┼──────────┐
        │          │          │
        ▼          ▼          ▼
        A          B          C
      0.2        0.4        0.9
        │          │          │
        └──────── Compare ────┘
                    │
                    ▼
             Relative ranking
```

---

# 6. Example with an LLM

Suppose we want to train mathematical reasoning.

Prompt:

```text
What is 25 × 16?
```

Generate:

```text
Response 1:
25 × 16 = 400

Response 2:
25 × 16 = 350

Response 3:
25 × 16 = 400

Response 4:
25 × 16 = 420
```

Reward function:

```python
def reward_function(
    response: str,
    correct_answer: str
) -> float:

    if correct_answer in response:
        return 1.0

    return 0.0
```

Evaluate:

```python
responses = [
    "25 × 16 = 400",
    "25 × 16 = 350",
    "25 × 16 = 400",
    "25 × 16 = 420"
]

correct_answer = "400"

rewards = []

for response in responses:

    reward = reward_function(
        response,
        correct_answer
    )

    rewards.append(reward)

print(rewards)
```

Output:

```text
[1.0, 0.0, 1.0, 0.0]
```

Then:

```python
import torch

rewards = torch.tensor(
    rewards,
    dtype=torch.float32
)

advantages = (
    rewards - rewards.mean()
) / (
    rewards.std(unbiased=False)
    + 1e-8
)

print(advantages)
```

Conceptually:

```text
Response 1 → Positive advantage
Response 2 → Negative advantage
Response 3 → Positive advantage
Response 4 → Negative advantage
```

The model is trained to make responses like 1 and 3 more likely.

---

# 7. GRPO training pipeline

The complete process looks like this:

```text
                     TRAINING PROMPT
                            │
                            ▼
                     Current LLM Policy
                            │
                            ▼
                 Generate G Responses
                            │
          ┌─────────────────┼─────────────────┐
          ▼                 ▼                 ▼
      Response 1        Response 2        Response G
          │                 │                 │
          └─────────────────┼─────────────────┘
                            ▼
                     Reward Function
                            │
                            ▼
                    [R1, R2, ... RG]
                            │
                            ▼
                Group Mean + Std Deviation
                            │
                            ▼
                  Relative Advantages
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
       Positive advantage          Negative advantage
              │                           │
              ▼                           ▼
       Increase probability       Decrease probability
              │                           │
              └─────────────┬─────────────┘
                            ▼
                       Update LLM
```

---

# 8. Simplified GRPO advantage calculation

Here is a reusable function.

```python
import torch


def compute_group_advantages(
    rewards: torch.Tensor
) -> torch.Tensor:
    """
    Compute relative advantages for one group.

    Args:
        rewards:
            Rewards for multiple responses
            generated for the SAME prompt.

    Returns:
        Normalized group-relative advantages.
    """

    mean = rewards.mean()

    std = rewards.std(
        unbiased=False
    )

    advantages = (
        rewards - mean
    ) / (
        std + 1e-8
    )

    return advantages
```

Example:

```python
rewards = torch.tensor([
    0.9,
    0.2,
    1.0,
    0.4
])

advantages = compute_group_advantages(
    rewards
)

print(advantages)
```

Interpretation:

```text
Reward above group average
        ↓
Positive advantage
        ↓
Increase probability


Reward below group average
        ↓
Negative advantage
        ↓
Decrease probability
```

---

# 9. What happens when all responses have the same reward?

Example:

```python
rewards = torch.tensor([
    1.0,
    1.0,
    1.0,
    1.0
])
```

Mean:

```text
1.0
```

Standard deviation:

```text
0
```

All responses are equally good.

Therefore, GRPO has little relative information:

```text
Response A = Response B = Response C = Response D
```

We can handle numerical stability:

```python
def compute_group_advantages(
    rewards,
    epsilon=1e-8
):

    mean = rewards.mean()

    std = rewards.std(
        unbiased=False
    )

    if std < epsilon:

        return torch.zeros_like(
            rewards
        )

    return (
        rewards - mean
    ) / std
```

Output:

```text
[0, 0, 0, 0]
```

This is important.

> GRPO learns best when the generated group provides useful variation.

---

# 10. How does GRPO update the model?

Suppose:

```text
Prompt
  ↓
Generate Response A
  ↓
Reward = 1
  ↓
Positive advantage
```

We want:

[
P(Response\ A|Prompt) \uparrow
]

For:

```text
Response B
Reward = 0
Negative advantage
```

We want:

[
P(Response\ B|Prompt) \downarrow
]

Conceptually, a policy-gradient loss looks like:

[
L =
-\mathbb{E}
[
A \log P_\theta(response|prompt)
]
]

Simplified code:

```python
def simplified_policy_loss(
    log_probabilities,
    advantages
):

    loss = (
        -advantages
        *
        log_probabilities
    ).mean()

    return loss
```

Example:

```python
log_probs = torch.tensor([
    -1.2,
    -0.8,
    -1.0,
    -0.6
])

advantages = torch.tensor([
    1.0,
    -1.0,
    1.0,
    -1.0
])

loss = simplified_policy_loss(
    log_probs,
    advantages
)

print(loss)
```

During gradient descent:

```text
Positive advantage
      ↓
Increase probability of response


Negative advantage
      ↓
Decrease probability of response
```

---

# 11. GRPO uses policy ratios like PPO

GRPO is related to PPO, so practical formulations can use a clipped policy-ratio objective.

The ratio is:

[
r =
\frac{
\pi_\theta(response)
}{
\pi_{old}(response)
}
]

Code:

```python
def policy_ratio(
    new_log_prob,
    old_log_prob
):

    return torch.exp(
        new_log_prob
        -
        old_log_prob
    )
```

Example:

```python
new_log_prob = torch.tensor(
    -0.8
)

old_log_prob = torch.tensor(
    -1.0
)

ratio = policy_ratio(
    new_log_prob,
    old_log_prob
)

print(ratio)
```

Then clipping prevents excessively large policy updates:

```python
def clipped_grpo_objective(
    ratio,
    advantage,
    epsilon=0.2
):

    unclipped = (
        ratio
        *
        advantage
    )

    clipped_ratio = torch.clamp(
        ratio,
        1 - epsilon,
        1 + epsilon
    )

    clipped = (
        clipped_ratio
        *
        advantage
    )

    objective = torch.minimum(
        unclipped,
        clipped
    )

    return objective
```

The important principle is:

```text
Do not allow the model to change too aggressively
in one update.
```

Current GRPO implementations may use different objective variants and settings; TRL documents both the group-relative advantage calculation and optional clipped updates for multiple iterations. ([GitHub][2])

---

# 12. Simplified end-to-end GRPO pseudocode

```python
for prompts in dataloader:

    # ----------------------------------
    # 1. Generate multiple responses
    # ----------------------------------

    responses = []

    for prompt in prompts:

        group = []

        for _ in range(GROUP_SIZE):

            response = model.generate(
                prompt
            )

            group.append(
                response
            )

        responses.append(
            group
        )

    # ----------------------------------
    # 2. Calculate reward
    # ----------------------------------

    rewards = []

    for group in responses:

        group_rewards = []

        for response in group:

            reward = reward_function(
                response
            )

            group_rewards.append(
                reward
            )

        rewards.append(
            group_rewards
        )

    # ----------------------------------
    # 3. Calculate group advantages
    # ----------------------------------

    advantages = []

    for group_rewards in rewards:

        group_rewards = torch.tensor(
            group_rewards
        )

        group_advantages = (
            group_rewards
            -
            group_rewards.mean()
        ) / (
            group_rewards.std(
                unbiased=False
            )
            + 1e-8
        )

        advantages.append(
            group_advantages
        )

    # ----------------------------------
    # 4. Calculate policy loss
    # ----------------------------------

    loss = calculate_grpo_loss(
        model=model,
        prompts=prompts,
        responses=responses,
        advantages=advantages
    )

    # ----------------------------------
    # 5. Update model
    # ----------------------------------

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()
```

Again, this is for understanding. Production implementations should use a tested trainer because correct token-level log-probabilities, distributed rollouts, masking, padding, old-policy ratios, and numerical stability are non-trivial.

---

# 13. Practical GRPO with TRL

A practical current implementation uses [Hugging Face TRL GRPOTrainer documentation](https://github.com/huggingface/trl/blob/main/docs/source/grpo_trainer.md?utm_source=chatgpt.com).

For example, a dataset:

```python
from datasets import Dataset

dataset = Dataset.from_list([
    {
        "prompt": "What is 2 + 2?",
        "answer": "4"
    },
    {
        "prompt": "What is 12 × 13?",
        "answer": "156"
    }
])
```

Define a reward function:

```python
def accuracy_reward(
    prompts,
    completions,
    answer,
    **kwargs
):
    rewards = []

    for completion, expected in zip(
        completions,
        answer
    ):

        completion = str(
            completion
        ).strip()

        if expected in completion:
            rewards.append(1.0)
        else:
            rewards.append(0.0)

    return rewards
```

Then configure GRPO conceptually:

```python
from trl import (
    GRPOConfig,
    GRPOTrainer
)

training_args = GRPOConfig(

    output_dir="./grpo-output",

    learning_rate=1e-6,

    num_train_epochs=1,

    per_device_train_batch_size=2,

    num_generations=4,

    max_completion_length=256,

    logging_steps=10
)
```

Create the trainer:

```python
trainer = GRPOTrainer(

    model="your-base-model",

    reward_funcs=accuracy_reward,

    args=training_args,

    train_dataset=dataset
)
```

Train:

```python
trainer.train()
```

The important parameter is conceptually:

```python
num_generations=4
```

Meaning:

```text
For each prompt:

Generate 4 responses

        ↓

Score all 4

        ↓

Compare rewards

        ↓

Calculate relative advantages

        ↓

Update model
```

TRL currently provides `GRPOTrainer` for this online RL workflow and supports reward functions or reward models. ([GitHub][2])

---

# 14. A better reward function

For reasoning models, rewards often have multiple components.

For example:

```text
Reward =
Correctness
+
Formatting
+
Reasoning quality
```

Code:

```python
import re


def reward_function(
    response,
    expected_answer
):

    reward = 0.0

    # -------------------------
    # 1. Correctness
    # -------------------------

    if expected_answer in response:

        reward += 1.0

    # -------------------------
    # 2. Structured answer
    # -------------------------

    if "<answer>" in response:

        reward += 0.2

    # -------------------------
    # 3. Penalize empty answers
    # -------------------------

    if len(response.strip()) == 0:

        reward -= 1.0

    return reward
```

For a group:

```text
Response A → 1.2
Response B → 0.2
Response C → 1.0
Response D → 0.0
```

GRPO calculates relative advantages.

```python
rewards = torch.tensor([
    1.2,
    0.2,
    1.0,
    0.0
])

advantages = (
    rewards
    -
    rewards.mean()
) / (
    rewards.std(
        unbiased=False
    )
    + 1e-8
)

print(advantages)
```

The model learns:

```text
A → Strongly preferred
C → Preferred
B → Less preferred
D → Poor
```

---

# 15. GRPO vs DPO

This is very important.

## DPO

You already have:

```text
Prompt

Chosen ✓
Rejected ✗
```

Dataset:

```python
{
    "prompt": "Explain RAG",

    "chosen": "Correct answer",

    "rejected": "Poor answer"
}
```

Training is offline.

```text
Existing preference data
        ↓
      DPO
        ↓
Update model
```

---

## GRPO

You have:

```text
Prompt
```

During training:

```text
Prompt
   ↓
Generate multiple NEW responses
   ↓
Score them with a reward
   ↓
Compare within group
   ↓
Update model
```

So:

| Feature                          | DPO                             | GRPO               |
| -------------------------------- | ------------------------------- | ------------------ |
| Training type                    | Offline preference optimization | Online RL          |
| Requires chosen/rejected dataset | Yes                             | No                 |
| Generates during training        | Not for each update             | Yes                |
| Reward function                  | No explicit reward required     | Yes                |
| Multiple responses per prompt    | No                              | Yes                |
| Critic/value model               | No                              | No separate critic |
| Complexity                       | Medium                          | Higher than DPO    |
| Reasoning optimization           | Good                            | Especially useful  |

---

# 16. GRPO vs PPO

| Feature                         | PPO                     | GRPO               |
| ------------------------------- | ----------------------- | ------------------ |
| Reinforcement learning          | Yes                     | Yes                |
| Generates responses             | Yes                     | Yes                |
| Reward signal                   | Yes                     | Yes                |
| Value/Critic model              | Typically yes           | No separate critic |
| Advantage calculation           | Reward − Value estimate | Relative to group  |
| Memory                          | Higher                  | Lower              |
| Complexity                      | High                    | Lower than PPO     |
| Multiple completions per prompt | Optional                | Core idea          |

The original GRPO paper describes it as a PPO variant that reduces memory use by replacing the critic with group-relative estimates. ([arXiv][1])

---

# 17. Why is GRPO useful for reasoning?

Reasoning tasks often have **verifiable rewards**.

Example:

```text
Question:
Solve:

37 × 18
```

The model can generate multiple reasoning paths:

```text
Path A:
Correct reasoning → 666 ✓

Path B:
Arithmetic mistake → 656 ✗

Path C:
Correct reasoning → 666 ✓

Path D:
Wrong reasoning → 676 ✗
```

A reward function can automatically verify:

```python
def math_reward(
    generated_answer,
    correct_answer
):

    return float(
        generated_answer
        ==
        correct_answer
    )
```

Then:

```text
Correct paths
      ↓
Positive relative advantage
      ↓
Become more probable


Incorrect paths
      ↓
Negative relative advantage
      ↓
Become less probable
```

This makes GRPO particularly attractive for domains with automatically verifiable outcomes:

* mathematics
* programming
* unit tests
* SQL execution
* theorem proving
* tool-using agents
* structured tasks

---

# 18. Example: GRPO for code generation

Prompt:

```text
Write a Python function that returns the factorial of n.
```

Generate:

```text
Response 1 → Code A
Response 2 → Code B
Response 3 → Code C
Response 4 → Code D
```

Run unit tests:

```python
def code_reward(
    code
):

    try:

        namespace = {}

        exec(
            code,
            namespace
        )

        function = namespace[
            "factorial"
        ]

        tests = [

            function(0) == 1,

            function(3) == 6,

            function(5) == 120
        ]

        if all(tests):
            return 1.0

        return 0.0

    except Exception:

        return 0.0
```

Then:

```text
Code A → Tests pass → Reward 1
Code B → Tests fail → Reward 0
Code C → Tests pass → Reward 1
Code D → Error      → Reward 0
```

GRPO compares these responses within the group.

**Important production note:** never execute untrusted model-generated code directly on your production machine. Use a sandbox/container with strict CPU, memory, filesystem, and network restrictions.

---

# 19. GRPO + LoRA

Training the entire LLM can be expensive.

You can combine:

```text
Base LLM
    +
LoRA
    +
GRPO
```

Conceptually:

```python
from peft import LoraConfig

peft_config = LoraConfig(

    r=16,

    lora_alpha=32,

    lora_dropout=0.05,

    target_modules=[
        "q_proj",
        "v_proj"
    ],

    task_type="CAUSAL_LM"
)
```

Architecture:

```text
                 Base Model
                (mostly frozen)
                       │
                       ▼
                 LoRA Adapters
                  (trainable)
                       │
                       ▼
                    GRPO
                       │
                       ▼
                 Reward Signal
                       │
                       ▼
                 Update Adapters
```

This can reduce trainable parameters, though GRPO still has significant generation/rollout costs.

---

# 20. Real production architecture

For a reasoning model:

```text
                   Training Dataset
                          │
                          ▼
                       Prompts
                          │
                          ▼
                     GRPO Engine
                          │
                          ▼
              Generate G completions
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
      Completion 1    Completion 2    Completion G
          │               │               │
          └───────────────┼───────────────┘
                          ▼
                    Reward System
                          │
             ┌────────────┼────────────┐
             │            │            │
             ▼            ▼            ▼
         Accuracy       Format       Safety
             │            │            │
             └────────────┼────────────┘
                          ▼
                    Total Reward
                          │
                          ▼
                  Group Advantages
                          │
                          ▼
                    GRPO Update
                          │
                          ▼
                     New Policy
                          │
                          └──────→ Generate Again
```

---

# 21. Advantages of GRPO

### 1. No separate critic/value model

```text
Less memory
```

### 2. Relative comparison

The model learns:

```text
For this exact problem,
which generated solutions were better?
```

### 3. Good for verifiable rewards

```text
Math answer correct?
Code passes tests?
SQL executes?
```

### 4. Scales naturally to reasoning

Multiple candidate reasoning paths provide learning signals.

### 5. Lower memory pressure than PPO

The original GRPO design was explicitly motivated by PPO memory costs. ([arXiv][1])

---

# 22. Limitations

GRPO is not "free" or simple like SFT.

### Expensive generation

You must generate multiple completions:

```text
1 prompt × G responses
```

If:

```text
100,000 prompts
G = 8
```

You may need roughly:

```text
800,000 generated rollouts
```

---

### Reward design is critical

Bad reward:

```text
Reward = 1 if response is long
```

The model may learn:

```text
Generate extremely long answers
```

This is a form of reward hacking.

---

### Group size matters

Too small:

```text
G = 1
```

No meaningful group comparison.

Too large:

```text
G = 32
```

More compute and memory.

You must balance:

```text
Signal quality
vs
Training cost
```

---

# 23. Interview-ready answer

> **GRPO, or Group Relative Policy Optimization, is a reinforcement-learning algorithm for LLM post-training introduced in DeepSeekMath. It is related to PPO but avoids maintaining a separate critic/value model.**
>
> **For each prompt, GRPO generates a group of multiple candidate responses. Each response receives a reward from a reward function or reward model. Instead of calculating advantage using a value model, GRPO calculates a relative advantage by comparing each response's reward with the mean and variation of rewards within that group.**
>
> **Responses that perform better than the group receive positive advantage, while worse responses receive negative advantage. The policy is then updated to increase the probability of higher-advantage responses.**
>
> **GRPO is especially useful for reasoning tasks where rewards can be automatically verified, such as mathematics, code generation, unit-test execution, SQL execution, and agent tasks.**

---

# Final mental model

```text
PPO:

One response
    ↓
Reward
    ↓
Compare with Value Model
    ↓
Advantage
```

```text
GRPO:

Multiple responses for SAME prompt
           ↓
         Rewards
           ↓
   Compare within group
           ↓
   Relative Advantages
           ↓
     Update LLM
```

## One-line summary

> **GRPO trains an LLM by generating multiple answers to the same prompt, rewarding them, and learning from which answers perform better relative to the other answers in that group.**

[1]: https://arxiv.org/abs/2402.03300?utm_source=chatgpt.com "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models"
[2]: https://github.com/huggingface/trl/blob/main/docs/source/grpo_trainer.md?utm_source=chatgpt.com "trl/docs/source/grpo_trainer.md at main · huggingface/trl · GitHub"
