# How would you preserve the model's general capabilities during fine-tuning?

When fine-tuning an LLM for a specialized task, I don't want this:

```text
                    Base LLM
                       │
                       ▼
               Fine-tune on
             narrow domain data
                       │
                       ▼
          ┌──────────────────────┐
          │ New task:     +++++  │
          │ General ability: --- │
          └──────────────────────┘
```

I want:

```text
                    Base LLM
                       │
             ┌─────────┴─────────┐
             │                   │
       General abilities     New domain
             │                   │
             └─────────┬─────────┘
                       ▼
               Fine-tuned model
```

The main techniques I would use are:

1. **PEFT/LoRA instead of full fine-tuning**
2. **Freeze the base model**
3. **Mix new-domain data with general/replay data**
4. **Use a conservative learning rate**
5. **Limit training epochs**
6. **Use early stopping**
7. **Use regularization**
8. **Evaluate old capabilities before and after fine-tuning**
9. **Use knowledge distillation when preservation is especially important**

---

# 1. First establish a baseline

Before fine-tuning, evaluate the base model on the capabilities you want to preserve.

For example:

```text
Base model

General QA       90%
Python           88%
Reasoning        85%
Summarization    92%

New domain       65%
```

After fine-tuning, you want something like:

```text
                 Before    After

General QA         90%       88%   ✓
Python             88%       86%   ✓
Reasoning          85%       83%   ✓
Summarization      92%       90%   ✓

New domain         65%       94%   ✓
```

But this would be concerning:

```text
                 Before    After

General QA         90%       70%   ✗
Python             88%       61%   ✗
Reasoning          85%       68%   ✗

New domain         65%       95%   ✓
```

The new capability improved, but the model suffered significant regression.

---

# 2. Create a baseline evaluation

Let's create a simple evaluation set.

```python
general_eval = [
    {
        "task": "python",
        "prompt": "Write a Python function to reverse a string."
    },
    {
        "task": "reasoning",
        "prompt": "If 5 apples cost $10, how much do 10 apples cost?"
    },
    {
        "task": "general_knowledge",
        "prompt": "What is the capital of France?"
    }
]
```

And our new domain:

```python
domain_eval = [
    {
        "task": "customer_support",
        "prompt": "How can I reset my account password?"
    },
    {
        "task": "customer_support",
        "prompt": "How do I request a refund?"
    }
]
```

---

# 3. Evaluate the base model

A simple generation function:

```python
import torch


def generate_answer(
    model,
    tokenizer,
    prompt,
    max_new_tokens=100
):

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(model.device)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False
        )

    return tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )
```

Run the baseline:

```python
baseline_results = []

for example in general_eval:

    answer = generate_answer(
        model,
        tokenizer,
        example["prompt"]
    )

    baseline_results.append({
        "task": example["task"],
        "prompt": example["prompt"],
        "answer": answer
    })
```

Store these results.

This becomes your **regression baseline**.

---

# 4. Strategy #1 — Use LoRA/PEFT

This is one of the most important techniques.

With full fine-tuning:

```text
7B parameter model

7B parameters
      │
      ▼
All potentially updated
```

With LoRA:

```text
7B parameter model
       │
       ▼
Base weights
FROZEN
       +
LoRA adapters
TRAINABLE
```

Code:

```python
from peft import (
    LoraConfig,
    get_peft_model
)


lora_config = LoraConfig(

    r=16,

    lora_alpha=32,

    lora_dropout=0.05,

    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj"
    ],

    bias="none",

    task_type="CAUSAL_LM"
)


model = get_peft_model(
    model,
    lora_config
)
```

Check:

```python
model.print_trainable_parameters()
```

You should see something conceptually like:

```text
trainable params:      0.1%
all params:          100%
```

The exact percentage depends on the model architecture and LoRA target modules.

---

# 5. Why LoRA helps preserve capabilities

LoRA represents the adaptation approximately as:

[
W' = W + \Delta W
]

where:

[
\Delta W = BA
]

The original:

[
W
]

remains frozen.

So:

```text
Original knowledge
       │
       ▼
      W
    FROZEN
       │
       +
       │
       ▼
Domain adaptation
       │
       ▼
      BA
   TRAINABLE
```

This is preferable to modifying every parameter when your goal is to specialize the model while keeping the base model intact.

**Important:** LoRA reduces the risk of destructive modification, but it does not guarantee that the resulting model preserves every capability. The adapter can still change behavior significantly.

---

# 6. Strategy #2 — Mix general data with domain data

This is called **replay**, **rehearsal**, or **mixed fine-tuning**.

Suppose your specialized dataset contains:

```text
100,000 customer-support examples
```

Don't necessarily train on:

```text
100% customer-support
```

Instead, include representative general examples:

```text
80% customer support
20% general/replay data
```

Conceptually:

```text
Customer support data
        │
        ├───────────────┐
        │               │
        ▼               ▼
   New capability   General capability
        │               │
        └───────┬───────┘
                ▼
          Mixed dataset
```

---

# 7. Create replay data

```python
domain_data = [
    {
        "text": "Customer: How do I reset my password?\n"
                "Agent: Use the Forgot Password option."
    },
    {
        "text": "Customer: How do I request a refund?\n"
                "Agent: Submit a refund request through the portal."
    }
]
```

General data:

```python
general_data = [
    {
        "text": "Question: What is Python?\n"
                "Answer: Python is a high-level programming language."
    },
    {
        "text": "Question: What is photosynthesis?\n"
                "Answer: Photosynthesis converts light energy into chemical energy."
    }
]
```

Mix them:

```python
mixed_data = (
    domain_data * 4
    +
    general_data
)
```

Shuffle:

```python
import random

random.shuffle(mixed_data)
```

Now:

```text
4 × domain examples
+
1 × general examples
```

approximately gives an 80/20 mixture.

In production, you would normally use much larger and carefully curated replay sets.

---

# 8. Why replay works

Without replay:

```text
Fine-tuning data:

100% domain
████████████████████

Gradient signal:
"Learn this domain"
```

With replay:

```text
Fine-tuning data:

80% domain
████████████████

20% general
████

Gradient signal:

Learn domain
+
Don't completely forget general behavior
```

The optimization objective now contains signals from both distributions.

Conceptually:

[
L =
\alpha L_{domain}
+
(1-\alpha)L_{general}
]

For example:

[
L =
0.8L_{domain}
+
0.2L_{general}
]

---

# 9. Strategy #3 — Freeze the base model

With LoRA this happens naturally for the base parameters.

You can also explicitly freeze parameters:

```python
for parameter in model.parameters():

    parameter.requires_grad = False
```

Then make only selected parameters trainable.

For example, with PEFT, the adapters are enabled as trainable parameters.

```python
model = get_peft_model(
    model,
    lora_config
)
```

Check:

```python
for name, parameter in model.named_parameters():

    if parameter.requires_grad:

        print(
            "TRAINABLE:",
            name
        )
```

You'll see the adapter parameters rather than the entire base model.

---

# 10. Strategy #4 — Use a conservative learning rate

A large learning rate can move the model away from its pretrained solution too aggressively.

For example, avoid blindly doing:

```python
learning_rate = 1e-3
```

for full LLM fine-tuning.

A more conservative starting point might be:

```python
learning_rate = 2e-5
```

for full fine-tuning, while LoRA commonly uses a higher learning rate such as:

```python
learning_rate = 1e-4
```

These are starting points, **not universal values**.

Always validate experimentally.

---

# 11. Strategy #5 — Limit the number of epochs

Suppose you have:

```text
50,000 domain examples
```

You don't necessarily need:

```python
num_train_epochs = 20
```

Start with something like:

```python
num_train_epochs = 2
```

or:

```python
num_train_epochs = 3
```

and evaluate.

The key is:

```text
Don't ask:
"How many epochs should I always use?"

Ask:
"At what point does validation/general capability stop improving?"
```

---

# 12. Strategy #6 — Early stopping

Use a validation dataset that measures both the new task and important old capabilities.

For example:

```python
from transformers import EarlyStoppingCallback
```

Then:

```python
trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=validation_dataset,

    callbacks=[
        EarlyStoppingCallback(
            early_stopping_patience=2
        )
    ]
)
```

If validation performance stops improving:

```text
Epoch 1 → improves
Epoch 2 → improves
Epoch 3 → improves
Epoch 4 → no improvement
Epoch 5 → no improvement

STOP
```

---

# 13. Strategy #7 — Weight decay

You can add regularization:

```python
from transformers import TrainingArguments


training_args = TrainingArguments(

    output_dir="./model",

    learning_rate=1e-4,

    num_train_epochs=3,

    weight_decay=0.01
)
```

Weight decay discourages unnecessarily large parameter updates.

It doesn't specifically solve catastrophic forgetting, but it can help generalization.

---

# 14. Strategy #8 — Knowledge distillation

For a more advanced production approach, keep the original model as a **teacher**.

```text
                  Original model
                     Teacher
                        │
                        │ outputs
                        ▼
                   Fine-tuned model
                      Student
                        │
                        ▼
                  New capabilities
```

The student learns:

```text
New domain behavior
        +
Original model behavior
```

---

# 15. Distillation loss

Suppose:

```text
Teacher logits
      ↓
Original model predictions

Student logits
      ↓
Fine-tuned model predictions
```

We can use:

[
L =
\alpha L_{task}
+
(1-\alpha)L_{distill}
]

Code:

```python
import torch.nn.functional as F


def distillation_loss(
    student_logits,
    teacher_logits,
    labels,
    temperature=2.0,
    alpha=0.7
):

    task_loss = F.cross_entropy(
        student_logits.view(
            -1,
            student_logits.size(-1)
        ),
        labels.view(-1)
    )

    student_log_probs = F.log_softmax(
        student_logits / temperature,
        dim=-1
    )

    teacher_probs = F.softmax(
        teacher_logits / temperature,
        dim=-1
    )

    distill_loss = F.kl_div(
        student_log_probs,
        teacher_probs,
        reduction="batchmean"
    ) * temperature**2

    total_loss = (
        alpha * task_loss
        +
        (1 - alpha) * distill_loss
    )

    return total_loss
```

The exact implementation for a causal LLM needs careful token masking and teacher/student alignment, but the idea is:

```text
Task loss
    ↓
Learn new domain

Distillation loss
    ↓
Stay close to old model behavior
```

---

# 16. Strategy #9 — Evaluate old capabilities after every experiment

This is probably the **most important production practice**.

Create an evaluation matrix:

```python
baseline = {
    "general_qa": 0.90,
    "python": 0.88,
    "reasoning": 0.85,
    "summarization": 0.92,
    "customer_support": 0.65
}
```

After fine-tuning:

```python
fine_tuned = {
    "general_qa": 0.87,
    "python": 0.86,
    "reasoning": 0.84,
    "summarization": 0.90,
    "customer_support": 0.94
}
```

Compare:

```python
for task in baseline:

    change = (
        fine_tuned[task]
        -
        baseline[task]
    )

    print(
        f"{task}: {change:+.2f}"
    )
```

Output:

```text
general_qa: -0.03
python: -0.02
reasoning: -0.01
summarization: -0.02
customer_support: +0.29
```

This looks reasonably healthy.

---

# 17. Set regression thresholds

In production, define acceptable degradation.

For example:

```python
MAX_ALLOWED_DROP = 0.05
```

Then:

```python
for task in baseline:

    before = baseline[task]
    after = fine_tuned[task]

    drop = before - after

    if drop > MAX_ALLOWED_DROP:

        print(
            f"REGRESSION: {task} "
            f"dropped by {drop:.2%}"
        )
```

This gives you a simple model acceptance gate.

Example:

```text
general_qa       -3%  ✓
python           -2%  ✓
reasoning        -1%  ✓
summarization    -2%  ✓
customer_support +29% ✓

MODEL ACCEPTED
```

But:

```text
python          -18% ✗
```

would trigger rejection.

---

# 18. Complete practical LoRA example

Here's the architecture I'd recommend for many domain-adaptation projects:

```python
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)

from peft import (
    LoraConfig,
    get_peft_model
)


MODEL_NAME = "your-base-model"


# -------------------------
# 1. Load base model
# -------------------------

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME
)


# -------------------------
# 2. Configure LoRA
# -------------------------

lora_config = LoraConfig(

    r=16,

    lora_alpha=32,

    lora_dropout=0.05,

    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj"
    ],

    bias="none",

    task_type="CAUSAL_LM"
)


# -------------------------
# 3. Add adapters
# -------------------------

model = get_peft_model(
    model,
    lora_config
)


model.print_trainable_parameters()


# -------------------------
# 4. Training configuration
# -------------------------

training_args = TrainingArguments(

    output_dir="./adapter",

    num_train_epochs=3,

    learning_rate=1e-4,

    per_device_train_batch_size=4,

    per_device_eval_batch_size=4,

    gradient_accumulation_steps=4,

    eval_strategy="steps",

    eval_steps=100,

    save_strategy="steps",

    save_steps=100,

    logging_steps=20,

    load_best_model_at_end=True,

    metric_for_best_model="eval_loss",

    greater_is_better=False,

    weight_decay=0.01,

    warmup_ratio=0.05,

    lr_scheduler_type="cosine"
)


# -------------------------
# 5. Train
# -------------------------

trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=mixed_train_dataset,

    eval_dataset=validation_dataset,

    processing_class=tokenizer,

    callbacks=[
        EarlyStoppingCallback(
            early_stopping_patience=2
        )
    ]
)


trainer.train()
```

Notice the important pieces:

```text
LoRA
 ↓
Base model remains frozen

Mixed dataset
 ↓
New domain + general/replay data

Conservative training
 ↓
Controlled adaptation

Validation
 ↓
Monitor performance

Early stopping
 ↓
Avoid unnecessary updates

Best checkpoint
 ↓
Select best model
```

---

# 19. What I would do in a real enterprise project

Suppose I have:

```text
Base LLM
      ↓
Enterprise customer-support model
```

I would use:

### Step 1 — Baseline

Evaluate:

```text
General QA
Coding
Reasoning
Summarization
Safety
Domain task
```

### Step 2 — Data preparation

```text
Customer conversations
        ↓
Clean
        ↓
Deduplicate
        ↓
Remove sensitive/unwanted data
        ↓
Quality filtering
        ↓
Split by conversation/customer/document
```

### Step 3 — Build replay set

Keep a representative sample of:

```text
General instruction following
Reasoning
Coding
Summarization
Other important capabilities
```

### Step 4 — Use LoRA

```text
Frozen base
+
Trainable adapter
```

### Step 5 — Train conservatively

```text
Small number of epochs
Appropriate LR
Warmup
Weight decay/dropout
```

### Step 6 — Evaluate continuously

```text
New capability ↑

AND

General capabilities
must not fall beyond
acceptable thresholds
```

### Step 7 — Reject bad checkpoints

```text
New task +20%
General QA -15%

→ REJECT
```

Even if the new task looks excellent.

---

# 20. The most important distinction

There are two different goals:

### Goal A — Preserve the base model itself

Use:

```text
LoRA / PEFT
```

because the base model remains frozen.

```text
Base model
████████████████
FROZEN

Adapter
██
TRAINED
```

### Goal B — Preserve the model's behavior

Use:

```text
LoRA
+
Replay data
+
Regularization
+
Baseline evaluation
+
Regression testing
```

Because simply freezing the base model does **not** guarantee that the combined model behaves exactly like the original model.

---

# Interview answer

If the interviewer asks:

> **"How would you preserve the model's general capabilities while fine-tuning it?"**

A strong answer is:

> **"First, I would establish a baseline evaluation of the pretrained model across the capabilities I need to preserve. For specialization, I would generally prefer PEFT such as LoRA so that the base model remains frozen and the domain-specific knowledge is stored in adapters. I would also mix the domain dataset with a representative replay set of general examples so the optimization continues to receive signals for existing capabilities. I would use a conservative learning rate, limited training steps, appropriate regularization and early stopping. After each experiment, I would evaluate both the new domain capability and the original capability suite. If the new task improves but an important existing capability drops beyond an agreed threshold, I would reject that checkpoint. For high-stakes continual learning, I could additionally use knowledge distillation from the original model."**

### Remember this architecture:

```text
                  BASE MODEL
                      │
             ┌────────┴────────┐
             │                 │
        General data       Domain data
             │                 │
             └────────┬────────┘
                      │
                 Mixed training
                      │
                      ▼
                 LoRA adapter
                      │
             ┌────────┴─────────┐
             ▼                  ▼
       Preserve old        Learn new
       capabilities        capability
             │                  │
             └────────┬─────────┘
                      ▼
             Regression testing
                      │
             ┌────────┴────────┐
             ▼                 ▼
        No regression       Regression
             │                 │
             ▼                 ▼
          ACCEPT             REJECT
```

**The key principle is:**

> **Don't measure fine-tuning success only by how much the new task improves. Measure the trade-off between new capability gained and old capability lost.**
