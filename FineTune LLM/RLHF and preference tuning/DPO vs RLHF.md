# DPO vs RLHF — explained properly with code

First, an important clarification:

> **DPO and RLHF are not exactly the same category of thing.**

* **RLHF** = a broader approach: *Reinforcement Learning from Human Feedback*
* **DPO** = a specific preference-optimization algorithm that can use human preference data **without a separate reward-model + PPO training loop**

In interviews, people often compare:

```text
Traditional RLHF = SFT → Reward Model → PPO

vs

DPO = SFT → Preference Pairs → Direct Optimization
```

---

# 1. The core problem both are solving

Suppose we ask an LLM:

```text
User: Explain RAG.
```

The model produces two answers.

### Answer A

```text
RAG uses a database.
```

### Answer B

```text
RAG retrieves relevant documents and provides them as context
to an LLM so it can generate a more grounded response.
```

A human says:

```text
Answer B > Answer A
```

This creates preference data:

```python
example = {
    "prompt": "Explain RAG.",

    "chosen": (
        "RAG retrieves relevant documents and provides them "
        "as context to an LLM so it can generate a more "
        "grounded response."
    ),

    "rejected": (
        "RAG uses a database."
    )
}
```

The question is:

> How do we train the LLM from this human preference?

Traditional RLHF and DPO solve this differently.

---

# 2. Traditional RLHF architecture

The classical RLHF pipeline looks like:

```text
                   Base Model
                       │
                       ▼
                      SFT
                       │
                       ▼
                   SFT Model
                       │
             Generate Responses
                       │
                       ▼
               Human Preferences
                       │
                       ▼
                 Reward Model
                       │
                       ▼
                 Reward Score
                       │
                       ▼
                       PPO
                       │
                       ▼
                 Aligned Model
```

There are two major training phases after SFT:

### Phase 1

Train a **Reward Model**.

### Phase 2

Use **PPO** to optimize the LLM using that reward.

---

# 3. DPO architecture

DPO is simpler:

```text
                 Base Model
                     │
                     ▼
                    SFT
                     │
                     ▼
                 SFT Model
                     │
                     ▼
              Preference Dataset
                     │
          ┌──────────┴──────────┐
          │                     │
       Chosen ✓             Rejected ✗
          │                     │
          └──────────┬──────────┘
                     │
                     ▼
                    DPO
                     │
                     ▼
                Aligned Model
```

DPO directly learns:

[
P(chosen|prompt) > P(rejected|prompt)
]

without explicitly creating:

```text
Reward Model → PPO loop
```

---

# 4. Traditional RLHF step by step

Let's build a simplified version.

---

## Step 1: Start with an SFT model

```text
Pretrained LLM
       │
       ▼
Supervised Fine-Tuning
       │
       ▼
Instruction-following model
```

Example SFT data:

```python
sft_data = [
    {
        "prompt": "What is RAG?",
        "response": (
            "RAG stands for Retrieval-Augmented Generation. "
            "It retrieves relevant information and gives it "
            "to an LLM as context."
        )
    }
]
```

After SFT, we get:

```text
π_SFT
```

This becomes the starting point for alignment.

---

# 5. RLHF Step 2: Collect preference data

Generate multiple responses:

```text
Prompt
   │
   ▼
SFT Model
   │
   ├── Response A
   │
   ├── Response B
   │
   └── Response C
```

Humans rank them:

```text
B > A > C
```

For reward-model training, we often convert this into pairs:

```python
preference_data = [
    {
        "prompt": "Explain RAG.",

        "chosen": (
            "RAG retrieves relevant documents and supplies "
            "them to the LLM as context."
        ),

        "rejected": (
            "RAG is a database."
        )
    }
]
```

---

# 6. RLHF Step 3: Train the Reward Model

The reward model receives:

```text
Prompt + Response
       │
       ▼
Transformer
       │
       ▼
Reward Head
       │
       ▼
Scalar Reward
```

Example:

```text
Prompt:
Explain RAG.

Response A
Reward = 0.2

Response B
Reward = 0.9
```

The goal is:

[
R(chosen) > R(rejected)
]

---

## Reward model code

```python
import torch
import torch.nn.functional as F
```

The pairwise reward-model loss:

```python
def reward_model_loss(
    chosen_reward,
    rejected_reward
):
    """
    Train the reward model so that:

    Reward(chosen) > Reward(rejected)
    """

    reward_difference = (
        chosen_reward
        -
        rejected_reward
    )

    loss = (
        -F.logsigmoid(
            reward_difference
        ).mean()
    )

    return loss
```

Example:

```python
chosen_reward = torch.tensor([
    2.5
])

rejected_reward = torch.tensor([
    0.5
])

loss = reward_model_loss(
    chosen_reward,
    rejected_reward
)

print(loss.item())
```

We want:

```text
2.5 > 0.5
```

So the model correctly ranks the chosen answer higher.

---

# 7. RLHF Step 4: Use PPO

Now the reward model is trained.

We generate a response:

```text
Prompt
   │
   ▼
Policy LLM
   │
   ▼
Response
   │
   ▼
Reward Model
   │
   ▼
Reward = 0.87
```

PPO updates the LLM.

Conceptually:

```python
for prompt in prompts:

    # Generate response
    response = policy.generate(
        prompt
    )

    # Score response
    reward = reward_model(
        prompt,
        response
    )

    # Calculate PPO loss
    loss = calculate_ppo_loss(
        policy,
        prompt,
        response,
        reward
    )

    # Update LLM
    optimizer.zero_grad()

    loss.backward()

    optimizer.step()
```

---

# 8. Why does PPO need a reference model?

Suppose the reward model mistakenly rewards:

```text
Very long answers
```

The LLM could start generating:

```text
Extremely long answers to everything
```

This is called **reward hacking**.

So RLHF typically uses:

```text
                Trainable Policy
                       │
                       ▼
                    Response
                       │
                       ▼
                 Reward Model
                       │
                       ▼
                     Reward

                       +

               Reference Model
                       │
                       ▼
                   KL Penalty
```

Conceptually:

[
FinalReward =
Reward -
\beta \times KL
]

Code:

```python
def final_reward(
    reward,
    kl_divergence,
    beta=0.1
):

    return (
        reward
        -
        beta * kl_divergence
    )
```

Example:

```python
reward = 1.0
kl = 0.3

adjusted_reward = final_reward(
    reward,
    kl,
    beta=0.1
)

print(adjusted_reward)
```

Output:

```text
0.97
```

The model is rewarded for good behavior but penalized for moving too far from the original model.

---

# 9. DPO solves the same problem differently

DPO starts with exactly this data:

```python
preference_data = [
    {
        "prompt": "Explain RAG.",

        "chosen": (
            "RAG retrieves relevant documents and provides "
            "them as context to an LLM."
        ),

        "rejected": (
            "RAG uses a database."
        )
    }
]
```

Instead of:

```text
Preference Data
      │
      ▼
Train Reward Model
      │
      ▼
PPO
```

DPO does:

```text
Preference Data
      │
      ▼
DPO Loss
      │
      ▼
Update LLM
```

---

# 10. How DPO works internally

DPO has:

```text
1. Trainable Policy Model

2. Frozen Reference Model
```

For each example:

```text
Prompt
   │
   ├───────────────┐
   │               │
   ▼               ▼
Chosen          Rejected
   │               │
   ▼               ▼
Trainable Model
   │               │
   ▼               ▼
Log P(chosen)  Log P(rejected)
```

The same is evaluated using the reference model:

```text
Reference Model
     │
     ├── Log P(chosen)
     │
     └── Log P(rejected)
```

DPO asks:

> Has the trainable model increased the relative probability of the chosen answer compared with the rejected answer, relative to the reference model?

---

# 11. DPO loss

Simplified mathematically:

[
L =
-\log
\sigma
\left(
\beta
[
(
\log P_\theta(chosen)
---------------------

\log P_\theta(rejected)
)
-

(
\log P_{ref}(chosen)
--------------------

\log P_{ref}(rejected)
)
]
\right)
]

Let's implement it.

---

## DPO loss code

```python
import torch
import torch.nn.functional as F
```

```python
def dpo_loss(
    policy_chosen_logps,
    policy_rejected_logps,
    reference_chosen_logps,
    reference_rejected_logps,
    beta=0.1
):

    # How strongly the trainable model
    # prefers chosen over rejected
    policy_preference = (
        policy_chosen_logps
        -
        policy_rejected_logps
    )

    # How strongly the reference model
    # prefers chosen over rejected
    reference_preference = (
        reference_chosen_logps
        -
        reference_rejected_logps
    )

    # Improvement over reference
    preference_margin = (
        policy_preference
        -
        reference_preference
    )

    # DPO loss
    loss = (
        -F.logsigmoid(
            beta * preference_margin
        ).mean()
    )

    return loss
```

---

# 12. Example DPO calculation

Suppose:

```python
policy_chosen = torch.tensor([
    -1.0
])

policy_rejected = torch.tensor([
    -3.0
])
```

The policy preference:

[
-1 - (-3) = 2
]

Now reference:

```python
reference_chosen = torch.tensor([
    -1.5
])

reference_rejected = torch.tensor([
    -2.0
])
```

Reference preference:

[
-1.5 - (-2) = 0.5
]

DPO improvement:

[
2 - 0.5 = 1.5
]

Code:

```python
loss = dpo_loss(

    policy_chosen,

    policy_rejected,

    reference_chosen,

    reference_rejected,

    beta=0.1
)

print(loss.item())
```

The model has improved the preference for:

```text
Chosen > Rejected
```

compared with the reference model.

---

# 13. The biggest conceptual difference

## Traditional RLHF

First learn:

```text
What humans like
       │
       ▼
Reward Model
```

Then optimize:

```text
Maximize Reward
       │
       ▼
PPO
```

So:

```text
Human Preferences
        ↓
Learned Reward Function
        ↓
Reinforcement Learning
        ↓
Aligned LLM
```

---

## DPO

Directly learn:

```text
Human says:

Chosen > Rejected

        ↓

Increase probability of Chosen

Decrease probability of Rejected
```

So:

```text
Human Preferences
        ↓
Direct Preference Optimization
        ↓
Aligned LLM
```

---

# 14. Full comparison table

| Feature                 | Traditional RLHF | DPO                      |
| ----------------------- | ---------------- | ------------------------ |
| Human feedback          | Yes              | Yes                      |
| Preference pairs        | Yes              | Yes                      |
| Reward model            | Yes              | No explicit reward model |
| PPO / RL                | Yes              | No explicit RL loop      |
| Policy model            | Yes              | Yes                      |
| Reference model         | Usually yes      | Yes                      |
| Value/Critic model      | Often yes        | No                       |
| Training complexity     | High             | Lower                    |
| GPU memory              | Higher           | Lower                    |
| Training stability      | Can be difficult | Generally simpler        |
| Dynamic rewards         | Excellent        | Limited                  |
| Online learning         | More natural     | Less natural             |
| Preference optimization | Indirect         | Direct                   |

---

# 15. Infrastructure comparison

## RLHF + PPO

You may need several models simultaneously:

```text
                ┌────────────────┐
                │ Policy Model   │
                └────────────────┘

                ┌────────────────┐
                │ Reference Model│
                └────────────────┘

                ┌────────────────┐
                │ Reward Model   │
                └────────────────┘

                ┌────────────────┐
                │ Value Model    │
                └────────────────┘
```

This increases GPU memory.

---

## DPO

Usually conceptually:

```text
                ┌────────────────┐
                │ Trainable Model│
                └────────────────┘

                ┌────────────────┐
                │ Reference Model│
                └────────────────┘
```

Some implementations can reduce memory further by avoiding a separately loaded reference model.

For example, with PEFT/LoRA, the base model can sometimes serve as the reference behavior while switching adapters.

---

# 16. Production DPO with LoRA

In real projects, you often combine:

```text
DPO + LoRA
```

This makes alignment cheaper.

Install:

```bash
pip install transformers datasets trl peft accelerate
```

Example dataset:

```python
from datasets import Dataset

dataset = Dataset.from_list([
    {
        "prompt":
            "Explain RAG simply.",

        "chosen":
            (
                "RAG retrieves relevant information "
                "and gives it to an LLM as context."
            ),

        "rejected":
            (
                "RAG is a database."
            )
    },

    {
        "prompt":
            "Explain LoRA.",

        "chosen":
            (
                "LoRA is a parameter-efficient fine-tuning "
                "technique that trains small low-rank updates "
                "while keeping the base model mostly frozen."
            ),

        "rejected":
            (
                "LoRA changes every parameter in the model."
            )
    }
])
```

Load model:

```python
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)

MODEL_NAME = "your-base-model"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

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

LoRA configuration:

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

DPO configuration:

```python
from trl import (
    DPOTrainer,
    DPOConfig
)

training_args = DPOConfig(

    output_dir="./dpo_output",

    learning_rate=5e-7,

    per_device_train_batch_size=2,

    gradient_accumulation_steps=4,

    num_train_epochs=1,

    beta=0.1,

    logging_steps=10
)
```

Trainer:

```python
trainer = DPOTrainer(

    model=model,

    ref_model=reference_model,

    args=training_args,

    train_dataset=dataset,

    processing_class=tokenizer,

    peft_config=peft_config
)
```

Train:

```python
trainer.train()

trainer.save_model(
    "./dpo_lora_model"
)
```

The exact `TRL` arguments can vary by version, but the architecture remains:

```text
Base Model
     │
     ▼
LoRA Adapters
     │
     ▼
DPO Training
     │
     ▼
Aligned Adapter
```

---

# 17. A simplified side-by-side code comparison

## Traditional RLHF

```python
# ---------------------------------
# STEP 1
# Train reward model
# ---------------------------------

reward_model = train_reward_model(
    preference_dataset
)


# ---------------------------------
# STEP 2
# RLHF training
# ---------------------------------

for prompt in prompts:

    # Policy generates response
    response = policy_model.generate(
        prompt
    )

    # Reward model scores response
    reward = reward_model.score(
        prompt,
        response
    )

    # Calculate PPO objective
    loss = ppo_loss(
        policy_model,
        prompt,
        response,
        reward
    )

    # Update policy
    optimizer.zero_grad()

    loss.backward()

    optimizer.step()
```

---

## DPO

```python
for example in preference_dataset:

    prompt = example["prompt"]

    chosen = example["chosen"]

    rejected = example["rejected"]


    # Get sequence log probabilities
    policy_chosen_logp = get_log_probability(
        policy_model,
        prompt,
        chosen
    )

    policy_rejected_logp = get_log_probability(
        policy_model,
        prompt,
        rejected
    )


    # Reference probabilities
    with torch.no_grad():

        ref_chosen_logp = (
            get_log_probability(
                reference_model,
                prompt,
                chosen
            )
        )

        ref_rejected_logp = (
            get_log_probability(
                reference_model,
                prompt,
                rejected
            )
        )


    # DPO loss
    loss = dpo_loss(

        policy_chosen_logp,

        policy_rejected_logp,

        ref_chosen_logp,

        ref_rejected_logp
    )


    # Update model
    optimizer.zero_grad()

    loss.backward()

    optimizer.step()
```

Notice the difference:

### RLHF

```text
Generate → Score → PPO
```

### DPO

```text
Chosen + Rejected → Compare probabilities → Update
```

---

# 18. When should you use traditional RLHF?

I would consider traditional RLHF/PPO when:

### 1. You have a real reward signal

For example, SQL generation:

```text
LLM generates SQL
       │
       ▼
Execute SQL
       │
       ▼
Correct result?
       │
       ├── Yes → Reward +1
       │
       └── No  → Reward 0
```

This reward can be directly optimized.

---

### 2. Agent environments

```text
Agent
  │
  ▼
Action
  │
  ▼
Environment
  │
  ▼
Reward
  │
  ▼
Next Action
```

Examples:

* robotics
* games
* browser agents
* tool-using agents
* multi-step planning

---

### 3. Complex reward functions

For example:

[
Reward =
Correctness
+
Helpfulness
+
Safety
------

## Cost

Latency
]

A PPO-style online optimization framework can work naturally with such reward signals.

---

# 19. When should you use DPO?

I would consider DPO when:

### 1. You have preference pairs

```text
Prompt
Chosen
Rejected
```

### 2. You want simpler infrastructure

You avoid the classical:

```text
Reward Model
+
Value Model
+
PPO loop
```

### 3. You want offline alignment

For example:

```text
Customer support preferences
Coding-answer preferences
Enterprise response preferences
Writing-style preferences
```

### 4. You have limited compute

DPO can be easier to train and operationalize than full PPO-based RLHF.

---

# 20. Real-world example: customer-support LLM

Suppose your company has 100,000 conversations.

You ask support experts to compare model responses.

```text
Customer:
My payment failed.
```

### Response A

```text
Contact support.
```

### Response B

```text
I'm sorry your payment failed. Please check whether your
payment method is valid and has sufficient available balance.
If the problem continues, I can guide you through the next
appropriate support step.
```

Experts select:

```text
B > A
```

You create:

```python
{
    "prompt": "My payment failed.",

    "chosen": "Helpful, clear, actionable response...",

    "rejected": "Contact support."
}
```

For this use case, I would usually start with:

```text
SFT
 +
DPO
```

Why?

Because the problem is primarily:

```text
Which response does a human prefer?
```

You already have the preference signal directly.

---

# 21. Interview answer: Which would you choose?

A strong answer is:

> **I would first distinguish classical RLHF from DPO. Classical RLHF typically trains a reward model on human preference data and then uses a reinforcement-learning algorithm such as PPO to optimize the policy model. DPO directly optimizes the language model using chosen and rejected response pairs, avoiding a separately trained reward model and an explicit PPO loop.**
>
> **For offline preference alignment of an instruction-following or enterprise assistant, I would generally start with DPO because it has fewer moving parts and is easier to train and operate.**
>
> **For an agent that interacts with an environment and receives dynamic or execution-based rewards, I would consider PPO or another RL algorithm because it can optimize directly from those rewards.**

---

# Final mental model

```text
CLASSICAL RLHF

Human Preferences
       │
       ▼
  Reward Model
       │
       ▼
 Reward Signal
       │
       ▼
      PPO
       │
       ▼
Aligned LLM
```

```text
DPO

Human Preferences
       │
       ▼
Chosen > Rejected
       │
       ▼
 Direct Probability
   Optimization
       │
       ▼
Aligned LLM
```

## Simplest possible difference

> **Traditional RLHF learns a reward function first and then uses reinforcement learning to maximize that reward. DPO skips the explicit reward-model and RL loop and directly teaches the model to prefer human-chosen responses over rejected ones.**
