# What is Learning Rate?

The **learning rate (LR)** controls **how much the model's weights are changed after each training step**.

In simple terms:

> **Learning rate = step size used by the optimizer when updating model parameters.**

Suppose a parameter currently has:

```text
weight = 0.80
```

The model calculates a gradient:

```text
gradient = 0.20
```

A simplified gradient-descent update is:

[
w_{new} = w_{old} - \eta \times gradient
]

where:

* (w) = model parameter
* (\eta) = learning rate
* gradient = direction/magnitude of error

If:

```text
learning_rate = 0.01
```

then:

```text
new_weight
= 0.80 - (0.01 × 0.20)
= 0.798
```

If:

```text
learning_rate = 0.1
```

then:

```text
new_weight
= 0.80 - (0.1 × 0.20)
= 0.78
```

So the second update is much larger.

---

# 1. What happens during LLM training?

The basic training loop is:

```text
Training data
     ↓
Tokenization
     ↓
LLM
     ↓
Predictions
     ↓
Loss
     ↓
Backpropagation
     ↓
Gradients
     ↓
Optimizer
     ↓
Weight update
```

The learning rate controls the last step:

```text
Gradient
   ↓
Optimizer
   ↓
Learning rate
   ↓
How much weights change
```

For example:

```python
learning_rate = 1e-5
```

means:

```text
0.00001
```

---

# 2. Simple Python example

Let's implement gradient descent ourselves.

```python
weight = 10.0
learning_rate = 0.1

gradient = 2.0

weight = weight - learning_rate * gradient

print(weight)
```

Output:

```text
9.8
```

Now increase LR:

```python
weight = 10.0
learning_rate = 1.0

gradient = 2.0

weight = weight - learning_rate * gradient

print(weight)
```

Output:

```text
8.0
```

The larger LR makes a larger update.

---

# 3. Learning rate that is too small

Suppose:

```python
learning_rate = 1e-7
```

Updates may become extremely small:

```text
weight
 ↓
0.800000
 ↓
0.799999
 ↓
0.799998
 ↓
...
```

Training can become:

```text
Very slow
```

You may need many more training steps.

---

# 4. Learning rate that is too large

Suppose:

```python
learning_rate = 1.0
```

The model can make huge updates:

```text
Correct region
     ↓
overshoot
     ↓
overshoot again
     ↓
unstable training
```

You might observe:

```text
loss:
2.1
1.8
2.5
4.2
10.7
NaN
```

This can lead to:

```text
Training instability
Loss explosion
Catastrophic forgetting
NaNs
Poor final model
```

---

# 5. Visual intuition

Imagine you are walking down a mountain trying to reach the lowest point.

### Learning rate too small

```text
        ●
         \
          ●
           \
            ●
             \
              ●
```

You move very slowly.

### Good learning rate

```text
        ●
          \
            ●
              \
                ●
                  ★
```

You reach the minimum efficiently.

### Learning rate too large

```text
       ●
          \
             ●
                \
                   ●
                /
             ●
```

You keep jumping over the minimum.

---

# 6. Learning rate in PyTorch

Normally you don't manually perform:

```python
weight = weight - learning_rate * gradient
```

You use an optimizer.

```python
import torch

model = torch.nn.Linear(10, 1)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-5
)
```

Here:

```python
lr=1e-5
```

is the learning rate.

Training:

```python
for batch in dataloader:

    optimizer.zero_grad()

    output = model(
        batch["input"]
    )

    loss = loss_fn(
        output,
        batch["target"]
    )

    loss.backward()

    optimizer.step()
```

Conceptually:

```text
loss.backward()
     ↓
gradients calculated
     ↓
optimizer.step()
     ↓
weights updated using LR
```

---

# 7. Learning rate is not the same as loss

This is a common interview question.

### Loss

Measures:

> How wrong is the model?

Example:

```text
loss = 2.5
```

### Learning rate

Controls:

> How aggressively should the optimizer change the model?

Example:

```text
learning_rate = 1e-5
```

So:

```text
Loss       → measurement of error
LR         → size of parameter update
```

---

# 8. Why learning rate is especially important for LLM fine-tuning

This is where fine-tuning differs from training an LLM from scratch.

Suppose you start with:

```text
Pretrained LLM
     ↓
Already knows:
     language
     grammar
     reasoning patterns
     general knowledge
     syntax
```

You don't want to completely destroy those capabilities.

Fine-tuning should generally make:

```text
Small controlled changes
```

rather than:

```text
Huge changes to millions/billions of parameters
```

Therefore, fine-tuning often uses a **much smaller learning rate** than pre-training.

---

# 9. Typical learning-rate ranges

There is no universal correct LR, but these are useful starting points.

### Full fine-tuning

Often roughly:

```text
1e-6 → 5e-5
```

Common starting points:

```text
1e-5
2e-5
5e-6
```

### LoRA / PEFT

Often somewhat higher:

```text
1e-5 → 5e-4
```

Common starting points:

```text
1e-4
2e-4
5e-5
```

### QLoRA

A common starting range is:

```text
5e-5 → 2e-4
```

with:

```text
1e-4
```

being a reasonable initial experiment.

**These are starting ranges, not rules.** The optimal value depends on the model, dataset, LoRA configuration, batch size, optimizer, scheduler, and task.

---

# 10. Why LoRA can use a higher learning rate

Remember what LoRA does.

The base model:

```text
Base model
30B parameters
        │
        ├── FROZEN
        │
        └── LoRA adapters
              ↓
         Trainable
```

Instead of changing:

```text
30 billion parameters
```

you may train:

```text
10 million parameters
```

The adapters start from a small/near-zero update and must learn the task-specific transformation.

Therefore, LoRA often uses a larger LR than full fine-tuning.

For example:

```python
# Full fine-tuning
learning_rate = 2e-5

# LoRA
learning_rate = 2e-4
```

But don't interpret that as:

> "LoRA always needs 10× larger LR."

It doesn't.

---

# 11. Full fine-tuning vs LoRA

Example:

```text
Full fine-tuning

Base weights
     ↓
W → W + ΔW

Both W and ΔW are effectively trained.
```

LoRA:

```text
Base weights
     ↓
W frozen

LoRA:
A × B
     ↓
trainable
```

The learning rate applies to the **trainable parameters**.

With LoRA:

```python
optimizer = AdamW(
    model.parameters(),
    lr=1e-4
)
```

the optimizer should effectively update the LoRA parameters, not the frozen base parameters.

---

# 12. How do you choose learning rate for fine-tuning?

This is the important interview question.

I would use this process:

```text
1. Identify fine-tuning method
        ↓
2. Start from a reasonable baseline
        ↓
3. Check dataset size/quality
        ↓
4. Consider batch size
        ↓
5. Use warmup
        ↓
6. Use a scheduler
        ↓
7. Run small experiments
        ↓
8. Compare validation loss + task metrics
        ↓
9. Select stable LR
```

Let's examine this properly.

---

# 13. Factor 1 — Full fine-tuning vs LoRA

First determine what you're training.

### Full fine-tuning

Start conservatively:

```python
learning_rate = 1e-5
```

Try:

```text
5e-6
1e-5
2e-5
5e-5
```

### LoRA

Start around:

```python
learning_rate = 1e-4
```

Try:

```text
5e-5
1e-4
2e-4
5e-4
```

Again, these are candidate experiments, not fixed rules.

---

# 14. Factor 2 — Dataset size

Suppose you have:

```text
1,000 examples
```

versus:

```text
1,000,000 examples
```

You should not automatically use the same training configuration.

With a small dataset:

```text
Small data
   ↓
High overfitting risk
   ↓
Conservative LR
+
Fewer epochs
+
Strong validation
```

With a very large, high-quality dataset:

```text
Large data
   ↓
More optimization steps
   ↓
Can explore LR more systematically
```

---

# 15. Factor 3 — How far is the task from pretraining?

Suppose the base model is:

```text
General-purpose LLM
```

and you want:

```text
Customer support assistant
```

This is a relatively specialized behavioral adaptation.

But suppose you want:

```text
Train the model heavily toward a completely different distribution
```

You need to be more careful about:

```text
Catastrophic forgetting
```

A very large LR can move the model too far away from its pretrained solution.

---

# 16. Factor 4 — Dataset quality

Imagine:

```text
Dataset A:
100,000 high-quality conversations
```

versus:

```text
Dataset B:
1,000,000 noisy conversations
```

More data doesn't automatically mean better training.

With noisy labels:

```text
High LR
   +
Noisy labels
   ↓
Model learns noise aggressively
```

So I would clean the data before trying to solve the problem with hyperparameters.

---

# 17. Factor 5 — Effective batch size

This is important.

Suppose:

```python
per_device_batch_size = 4
```

and:

```python
gradient_accumulation_steps = 8
```

On one GPU:

[
\text{effective batch size}
===========================

# 4 \times 8

32
]

Code:

```python
effective_batch_size = (
    per_device_batch_size
    * gradient_accumulation_steps
)
```

With multiple GPUs:

```python
effective_batch_size = (
    per_device_batch_size
    * gradient_accumulation_steps
    * num_gpus
)
```

Larger batch sizes can change optimization behavior, so when you substantially change batch size, you should re-evaluate the LR rather than assuming the old value remains optimal.

---

# 18. Learning-rate warmup

Don't necessarily start immediately with your full LR.

Instead:

```text
Step 0
LR = 0

       ↓

Step 100
LR = 2e-5

       ↓

Step 500
LR = 1e-4
```

This is called **warmup**.

Why?

At the beginning:

```text
Model
+
New gradients
+
Random-ish adapter weights
```

The gradients can be unstable.

Warmup allows the optimizer to gradually reach the target learning rate.

---

# 19. Code: learning-rate warmup

With Hugging Face:

```python
from transformers import TrainingArguments


training_args = TrainingArguments(
    output_dir="./output",

    learning_rate=1e-4,

    warmup_ratio=0.05,

    num_train_epochs=3,

    per_device_train_batch_size=2,

    gradient_accumulation_steps=8,

    bf16=True,

    logging_steps=10
)
```

Here:

```python
learning_rate=1e-4
```

is the maximum/base LR.

And:

```python
warmup_ratio=0.05
```

means approximately the first 5% of training steps are used for warmup.

---

# 20. Learning-rate scheduler

Usually you don't keep the LR constant.

Example:

```text
Learning rate

1e-4 ──────╮
           │\
           │ \
           │  \
           │   \
           │    \
           │     \____
           │
           └──────────────
             training steps
```

The LR decreases during training.

Common schedulers include:

```text
Linear
Cosine
Constant
Constant with warmup
```

For LLM fine-tuning, **cosine decay + warmup** is a common starting choice.

---

# 21. Code: cosine scheduler

```python
from transformers import TrainingArguments


training_args = TrainingArguments(
    output_dir="./output",

    learning_rate=1e-4,

    lr_scheduler_type="cosine",

    warmup_ratio=0.05,

    num_train_epochs=3,

    per_device_train_batch_size=2,

    gradient_accumulation_steps=8,

    bf16=True
)
```

Conceptually:

```text
Warmup:

0
│
│     /
│    /
│   /
│  /
│ /
└──────────────

Then cosine decay:

       ──────╮
              \
               \
                \
                 ╲
                  ╲
```

---

# 22. How do you know your LR is too high?

Watch the training curves.

Suppose:

```text
Training loss:

2.1
1.8
1.5
1.3
1.1
```

Good.

But:

```text
2.1
1.5
2.8
1.2
4.5
8.2
```

Potentially too high or otherwise unstable.

Other symptoms:

```text
NaN loss
Exploding gradients
Validation loss increases sharply
Training becomes unstable
Model quality gets worse
```

---

# 23. How do you know LR is too low?

Example:

```text
Step       Loss

0          2.10
1000       2.09
2000       2.08
3000       2.07
4000       2.06
```

Very little progress.

Possible causes:

```text
Learning rate too low
+
Bad optimizer configuration
+
Poor data
+
Wrong labels
+
Insufficient training
```

Don't automatically conclude LR is the problem.

---

# 24. Important: training loss alone isn't enough

Suppose:

```text
LR = 5e-4
```

Training loss:

```text
0.2
```

Looks fantastic.

But validation:

```text
3.5
```

That's a warning.

The model may be overfitting.

Compare:

```text
              Train Loss    Val Loss

LR = 5e-5       1.0          1.1
LR = 1e-4       0.8          0.9
LR = 5e-4       0.2          3.5
```

I'd probably prefer:

```text
1e-4
```

because the validation behavior is much healthier.

---

# 25. Hyperparameter search

Instead of guessing one LR, test a small set.

For LoRA:

```python
learning_rates = [
    5e-5,
    1e-4,
    2e-4
]
```

For full fine-tuning:

```python
learning_rates = [
    5e-6,
    1e-5,
    2e-5
]
```

Train small pilot runs.

```python
for lr in learning_rates:

    train_model(
        learning_rate=lr,
        max_steps=500
    )

    evaluate_model()
```

Compare:

```text
LR       Train Loss    Val Loss    Task Score
------------------------------------------------
5e-5       1.10          1.20        0.72
1e-4       0.98          1.05        0.78
2e-4       0.85          1.40        0.70
```

Choose:

```text
1e-4
```

rather than:

```text
2e-4
```

because lower training loss isn't the only objective.

---

# 26. Better experiment tracking

For a real project, log:

```python
experiment = {
    "learning_rate": 1e-4,
    "batch_size": 2,
    "gradient_accumulation": 8,
    "epochs": 3,
    "warmup_ratio": 0.05,
    "scheduler": "cosine",
    "lora_rank": 16
}
```

Then track:

```text
Training loss
Validation loss
Task-specific accuracy
F1
ROUGE / BLEU where appropriate
LLM-as-judge metrics where justified
Human evaluation
Latency
Cost
```

For customer support specifically, I would also evaluate:

```text
Resolution quality
Correctness
Policy adherence
Hallucination rate
Tone
Safety
Escalation correctness
```

---

# 27. Complete LoRA example

Suppose we are fine-tuning an LLM using LoRA.

```python
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments
)

from peft import (
    LoraConfig,
    get_peft_model
)


MODEL_NAME = "your-model"
```

Load model:

```python
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)
```

LoRA:

```python
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

    task_type="CAUSAL_LM"
)
```

Attach adapters:

```python
model = get_peft_model(
    model,
    lora_config
)

model.print_trainable_parameters()
```

Now training configuration:

```python
training_args = TrainingArguments(

    output_dir="./support-model",

    # LR
    learning_rate=1e-4,

    # Scheduler
    lr_scheduler_type="cosine",

    # Warmup
    warmup_ratio=0.05,

    # Training
    num_train_epochs=3,

    # Memory
    per_device_train_batch_size=2,

    gradient_accumulation_steps=8,

    # Precision
    bf16=True,

    # Logging
    logging_steps=10,

    # Evaluation
    eval_strategy="steps",

    eval_steps=100,

    # Save
    save_steps=100,

    save_total_limit=2
)
```

The important part is:

```python
learning_rate=1e-4
```

combined with:

```python
lr_scheduler_type="cosine"
warmup_ratio=0.05
```

---

# 28. QLoRA example

For QLoRA:

```text
4-bit quantized base model
          │
          │ frozen
          ▼
     LoRA adapters
          │
          │ trainable
          ▼
       optimizer
```

Example:

```python
from transformers import BitsAndBytesConfig
import torch


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)
```

Then:

```python
training_args = TrainingArguments(
    output_dir="./qlora-output",

    learning_rate=1e-4,

    lr_scheduler_type="cosine",

    warmup_ratio=0.05,

    per_device_train_batch_size=1,

    gradient_accumulation_steps=16,

    num_train_epochs=3,

    bf16=True,

    gradient_checkpointing=True
)
```

Again, `1e-4` is a **starting experiment**, not a guaranteed optimum.

---

# 29. Learning rate and LoRA rank

LR should also be considered together with LoRA rank.

Suppose:

```text
Experiment A:
r = 8
LR = 1e-4

Experiment B:
r = 16
LR = 1e-4

Experiment C:
r = 32
LR = 1e-4
```

Increasing rank gives the adapter more capacity:

```text
r = 8
 ↓
less capacity

r = 16
 ↓
more capacity

r = 32
 ↓
more capacity
```

But don't simultaneously change:

```text
LR
rank
alpha
dropout
batch size
epochs
```

unless you have a reason.

Otherwise you won't know which change caused the result.

---

# 30. A practical LR selection strategy

For a new LLM fine-tuning project, I would start with:

### Full fine-tuning

```text
5e-6
1e-5
2e-5
```

### LoRA / QLoRA

```text
5e-5
1e-4
2e-4
```

Then:

```text
Run short pilot
       ↓
Compare validation loss
       ↓
Compare task metrics
       ↓
Check stability
       ↓
Check overfitting
       ↓
Select LR
       ↓
Run full training
```

---

# 31. One important interview nuance: learning rate is not the only "step size"

With AdamW, the actual parameter update isn't simply:

[
w = w - \eta g
]

Adam maintains moving estimates of gradients and squared gradients.

Conceptually:

[
m_t = \beta_1m_{t-1} + (1-\beta_1)g_t
]

[
v_t = \beta_2v_{t-1} + (1-\beta_2)g_t^2
]

and approximately:

[
w_{t+1}
=======

w_t -
\eta
\frac{\hat m_t}
{\sqrt{\hat v_t}+\epsilon}
]

So `learning_rate` controls the **overall scale of the optimizer update**, while Adam's normalization, weight decay, scheduler, warmup, and other optimizer settings also influence the effective update.

---

# 32. Learning-rate schedule vs learning rate

These are different.

### Learning rate

```python
learning_rate = 1e-4
```

The target/base LR.

### Scheduler

```python
lr_scheduler_type = "cosine"
```

Controls how LR changes over training.

So:

```text
Base LR = 1e-4

Warmup
  ↓
0 → 1e-4

Cosine decay
  ↓
1e-4 → near 0
```

---

# 33. The answer I'd give in an interview

### "What is learning rate?"

> Learning rate is the hyperparameter that controls the magnitude of parameter updates made by the optimizer during training. A very small learning rate makes training slow and may prevent the model from adapting sufficiently, while a very large learning rate can cause unstable training, loss explosion, or catastrophic forgetting.

### "How do you choose learning rate for LLM fine-tuning?"

> I first consider whether I'm doing full fine-tuning or PEFT such as LoRA/QLoRA. Full fine-tuning generally requires a smaller learning rate because I'm modifying the pretrained model weights directly. LoRA often allows a somewhat higher learning rate because only the adapter parameters are trained. I then consider dataset size and quality, effective batch size, model size, LoRA rank, and the distance between the pretrained task and target task. I normally start with a small LR sweep—for example, around `5e-6` to `2e-5` for full fine-tuning and `5e-5` to `2e-4` for LoRA/QLoRA—using warmup and cosine decay. I compare validation loss and, more importantly, task-specific metrics and human evaluation. I choose the highest learning rate that gives fast, stable improvement without overfitting or degrading general model capabilities.

### The key mental model

```text
                 LEARNING RATE
                       │
          ┌────────────┴────────────┐
          │                         │
       Too Low                    Too High
          │                         │
       Slow learning             Unstable
       Under-adaptation          Loss explosion
                                  Overfitting
                                  Forgetting
          │                         │
          └────────────┬────────────┘
                       │
                 Good LR range
                       │
                Fast + stable
                 adaptation
```

For **LLM fine-tuning**, don't ask *"What is the correct learning rate?"* in isolation. The better engineering question is:

> **"What learning-rate range is appropriate for this model, fine-tuning method, dataset, effective batch size, and objective, and what evidence from validation tells me which one is best?"**
