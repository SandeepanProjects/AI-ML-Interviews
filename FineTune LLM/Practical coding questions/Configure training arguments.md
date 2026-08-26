# Configure Training Arguments for LoRA / QLoRA

After:

1. Preparing the instruction dataset
2. Tokenizing the dataset
3. Loading the base model
4. Configuring `LoraConfig`

the next step is configuring **training arguments**.

Training arguments control:

```text
How long to train
How fast to learn
Batch size
GPU memory usage
Evaluation
Checkpointing
Logging
Precision
```

For Hugging Face + TRL, you will commonly use:

```python
from trl import SFTConfig
```

For the standard Hugging Face trainer:

```python
from transformers import TrainingArguments
```

For QLoRA instruction tuning, `SFTConfig` is often convenient.

---

# 1. Basic training configuration

```python
from trl import SFTConfig

training_args = SFTConfig(
    output_dir="./outputs",

    num_train_epochs=3,

    per_device_train_batch_size=2,

    gradient_accumulation_steps=8,

    learning_rate=2e-4,

    logging_steps=10,

    save_steps=100,
)
```

---

# 2. Full production-style configuration

```python
import torch
from trl import SFTConfig


# --------------------------------------------------
# Detect precision
# --------------------------------------------------

use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

use_fp16 = torch.cuda.is_available() and not use_bf16


training_args = SFTConfig(

    # ==============================================
    # OUTPUT
    # ==============================================

    output_dir="./outputs/qlora-customer-support",

    overwrite_output_dir=False,


    # ==============================================
    # TRAINING
    # ==============================================

    num_train_epochs=3,

    per_device_train_batch_size=2,

    per_device_eval_batch_size=2,

    gradient_accumulation_steps=8,


    # ==============================================
    # LEARNING RATE
    # ==============================================

    learning_rate=2e-4,

    lr_scheduler_type="cosine",

    warmup_ratio=0.03,

    weight_decay=0.01,


    # ==============================================
    # OPTIMIZER
    # ==============================================

    optim="paged_adamw_8bit",


    # ==============================================
    # GRADIENTS
    # ==============================================

    max_grad_norm=1.0,

    gradient_checkpointing=True,


    # ==============================================
    # PRECISION
    # ==============================================

    bf16=use_bf16,

    fp16=use_fp16,


    # ==============================================
    # SEQUENCE LENGTH
    # ==============================================

    max_length=2048,


    # ==============================================
    # LOGGING
    # ==============================================

    logging_strategy="steps",

    logging_steps=10,

    logging_first_step=True,

    report_to="tensorboard",


    # ==============================================
    # EVALUATION
    # ==============================================

    eval_strategy="steps",

    eval_steps=100,


    # ==============================================
    # CHECKPOINTING
    # ==============================================

    save_strategy="steps",

    save_steps=100,

    save_total_limit=2,


    # ==============================================
    # BEST MODEL
    # ==============================================

    load_best_model_at_end=True,

    metric_for_best_model="eval_loss",

    greater_is_better=False,


    # ==============================================
    # DATALOADER
    # ==============================================

    dataloader_num_workers=4,

    dataloader_pin_memory=True,


    # ==============================================
    # REPRODUCIBILITY
    # ==============================================

    seed=42,
)
```

---

# 3. Understanding each important argument

## `output_dir`

```python
output_dir="./outputs/qlora-customer-support"
```

This stores:

```text
outputs/
└── qlora-customer-support/
    ├── checkpoint-100/
    ├── checkpoint-200/
    ├── trainer_state.json
    └── ...
```

---

# 4. Number of epochs

```python
num_train_epochs=3
```

An epoch means the model sees the complete training dataset once.

```text
Epoch 1
Dataset → Model

Epoch 2
Dataset → Model

Epoch 3
Dataset → Model
```

For LoRA/QLoRA, a common starting point is:

```text
1–3 epochs → Large dataset
2–5 epochs → Medium dataset
More epochs → Only if validation metrics justify it
```

Do not simply train for many epochs.

Monitor:

```text
Train loss ↓
Validation loss ↓   → Good

Train loss ↓
Validation loss ↑   → Overfitting
```

---

# 5. Batch size

```python
per_device_train_batch_size=2
```

This means each GPU processes:

```text
2 examples per forward/backward pass
```

With multiple GPUs:

```text
2 GPUs × batch size 2
= 4 examples per step
```

But gradient accumulation also matters.

---

# 6. Gradient accumulation

```python
gradient_accumulation_steps=8
```

Instead of immediately updating after each batch:

```text
Batch 1 → Gradient
Batch 2 → Gradient
Batch 3 → Gradient
...
Batch 8 → Gradient
          ↓
      Update weights
```

Effective batch size is approximately:

$$
\text{Effective Batch Size}
=
\text{Per Device Batch Size}
\times
\text{Number of GPUs}
\times
\text{Gradient Accumulation Steps}
$$

For example:

```text
Batch size = 2
GPUs = 1
Accumulation = 8
```

$$
2 \times 1 \times 8 = 16
$$

So:

```text
GPU memory behaves like batch size 2
Training behaves approximately like batch size 16
```

---

# 7. Learning rate

For LoRA/QLoRA:

```python
learning_rate=2e-4
```

is a common starting point.

Why is it often higher than full fine-tuning?

Because:

```text
Full Fine-tuning:
Billions of parameters updated
→ Smaller LR often required

LoRA:
Small number of adapter parameters updated
→ Can often use higher LR
```

Common experiments:

```text
1e-5
5e-5
1e-4
2e-4
```

Example:

```python
learning_rates = [
    5e-5,
    1e-4,
    2e-4
]
```

Choose based on validation performance.

---

# 8. Learning rate scheduler

```python
lr_scheduler_type="cosine"
```

Conceptually:

```text
Learning Rate

2e-4      ●
          │\
          │ \
          │  \
          │   \
          │    \
0         └─────── Training Steps
```

Cosine scheduling generally:

```text
Start high
    ↓
Gradually reduce
    ↓
Smaller updates near the end
```

This can help stable convergence.

---

# 9. Warmup

```python
warmup_ratio=0.03
```

Suppose:

```text
Total steps = 10,000
```

Then:

```text
Warmup = 3%
       = 300 steps
```

Learning rate:

```text
0
│
│    /
│   /
│  /
│ /
│/________
       300 steps
```

Then the scheduler takes over.

Warmup helps avoid unstable updates at the beginning.

---

# 10. Weight decay

```python
weight_decay=0.01
```

Weight decay is a regularization technique.

Conceptually:

```text
Loss

=
Training Loss
+
Regularization Penalty
```

It discourages unnecessarily large weights.

For LoRA:

```text
0.0
0.01
```

are common values to experiment with.

---

# 11. Optimizer

For QLoRA:

```python
optim="paged_adamw_8bit"
```

Why?

Normal AdamW stores optimizer states.

```text
Parameters
+
Gradients
+
Momentum
+
Variance
```

This consumes memory.

An 8-bit optimizer reduces optimizer-state memory.

The "paged" variant helps manage GPU memory pressure.

For normal LoRA or full precision training, you might use:

```python
optim="adamw_torch"
```

---

# 12. Gradient clipping

```python
max_grad_norm=1.0
```

Sometimes gradients become very large:

```text
Gradient norm

0.5
1.2
4.0
1000 ← Problem
```

Gradient clipping limits them.

Conceptually:

```text
Gradient norm = 10

Maximum = 1

↓ scale gradient

Gradient norm = 1
```

This improves training stability.

---

# 13. Gradient checkpointing

```python
gradient_checkpointing=True
```

Without it:

```text
Forward pass
    ↓
Save activations
    ↓
Backward pass
```

With checkpointing:

```text
Forward pass
    ↓
Save fewer activations
    ↓
Recompute some during backward pass
```

Result:

```text
GPU memory ↓
Training time ↑
```

Very useful for:

```text
7B+
13B+
70B models
```

Especially with QLoRA.

---

# 14. BF16 vs FP16

```python
bf16=True
```

if your GPU supports BF16.

Otherwise:

```python
fp16=True
```

Typical logic:

```python
use_bf16 = torch.cuda.is_bf16_supported()

training_args = SFTConfig(
    bf16=use_bf16,
    fp16=not use_bf16
)
```

General preference:

```text
Modern NVIDIA GPU
        ↓
BF16 preferred

Older GPU
        ↓
FP16
```

---

# 15. Sequence length

```python
max_length=2048
```

This defines the maximum number of tokens used for training.

Example:

```text
Example length = 500 tokens
→ Keep 500

Example length = 1500 tokens
→ Keep 1500

Example length = 3000 tokens
→ Truncate to 2048
```

Memory grows significantly as sequence length increases.

Conceptually:

```text
Sequence Length ↑
GPU Memory ↑↑
```

Choose based on:

```text
Dataset token distribution
Model context window
GPU memory
Actual production use case
```

---

# 16. Logging

```python
logging_strategy="steps"
logging_steps=10
```

Logs:

```text
Step 10
Loss = 2.34

Step 20
Loss = 1.82

Step 30
Loss = 1.45
```

For TensorBoard:

```python
report_to="tensorboard"
```

Start it with:

```bash
tensorboard --logdir ./outputs
```

You can monitor:

```text
Training loss
Validation loss
Learning rate
Gradient norm
```

---

# 17. Evaluation

```python
eval_strategy="steps"
eval_steps=100
```

This means:

```text
Train 100 steps
      ↓
Evaluate
      ↓
Continue training
```

Example:

```text
Step 100
Train loss = 1.4
Eval loss  = 1.5

Step 200
Train loss = 1.1
Eval loss  = 1.2

Step 300
Train loss = 0.8
Eval loss  = 1.7
```

At step 300:

```text
Train loss ↓
Eval loss ↑
```

Possible overfitting.

---

# 18. Checkpointing

```python
save_strategy="steps"
save_steps=100
save_total_limit=2
```

Output:

```text
checkpoint-100
checkpoint-200
checkpoint-300
checkpoint-400
```

But:

```python
save_total_limit=2
```

keeps only a limited number of recent/best checkpoints according to trainer behavior.

This prevents storage from filling up.

---

# 19. Load the best model

```python
load_best_model_at_end=True

metric_for_best_model="eval_loss"

greater_is_better=False
```

The trainer chooses:

```text
Checkpoint A → eval loss = 1.5
Checkpoint B → eval loss = 1.2 ✓
Checkpoint C → eval loss = 1.8
```

Checkpoint B is selected.

---

# 20. Complete example with `SFTTrainer`

```python
from trl import SFTTrainer


trainer = SFTTrainer(

    model=model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=validation_dataset,

    processing_class=tokenizer,

    data_collator=data_collator,
)
```

Then:

```python
trainer.train()
```

Save:

```python
trainer.save_model(
    "./outputs/final-adapter"
)

tokenizer.save_pretrained(
    "./outputs/final-adapter"
)
```

---

# 21. Training configuration for a small GPU

Suppose you have limited GPU memory.

```python
training_args = SFTConfig(

    output_dir="./output",

    num_train_epochs=3,

    per_device_train_batch_size=1,

    per_device_eval_batch_size=1,

    gradient_accumulation_steps=16,

    learning_rate=2e-4,

    gradient_checkpointing=True,

    optim="paged_adamw_8bit",

    max_length=1024,

    bf16=True,

    logging_steps=10,

    eval_strategy="steps",

    eval_steps=100,

    save_strategy="steps",

    save_steps=100,

    report_to="tensorboard"
)
```

Effective batch size:

```text
1 × 16 = 16
```

This is a common strategy when GPU memory is limited.

---

# 22. Recommended starting configuration

For your instruction-tuning + QLoRA pipeline, I would start with:

```python
training_args = SFTConfig(

    output_dir="./outputs",

    num_train_epochs=3,

    per_device_train_batch_size=2,

    per_device_eval_batch_size=2,

    gradient_accumulation_steps=8,

    learning_rate=2e-4,

    lr_scheduler_type="cosine",

    warmup_ratio=0.03,

    weight_decay=0.01,

    optim="paged_adamw_8bit",

    max_grad_norm=1.0,

    gradient_checkpointing=True,

    bf16=True,

    logging_steps=10,

    eval_strategy="steps",

    eval_steps=100,

    save_strategy="steps",

    save_steps=100,

    save_total_limit=2,

    load_best_model_at_end=True,

    metric_for_best_model="eval_loss",

    greater_is_better=False,

    max_length=2048,

    report_to="tensorboard",

    seed=42
)
```

---

# 23. Full pipeline so far

You have now built:

```text
1. Raw Dataset
       ↓
2. Clean Dataset
       ↓
3. Instruction Dataset
       ↓
4. Convert to Chat Format
       ↓
5. Tokenization
       ↓
6. Load 4-bit Base Model
       ↓
7. Configure LoraConfig
       ↓
8. Configure Training Arguments
       ↓
9. Create SFTTrainer
       ↓
10. Train QLoRA
```

---

# Interview-ready answer

> **Training arguments define the optimization and execution strategy for fine-tuning. For QLoRA, I configure the number of epochs, per-device batch size, gradient accumulation, learning rate, scheduler, warmup, weight decay, optimizer, mixed precision, evaluation, checkpointing, and logging.**
>
> **Because GPU memory is often limited, I typically use a small per-device batch size with gradient accumulation, gradient checkpointing, and a memory-efficient optimizer such as `paged_adamw_8bit`. For LoRA/QLoRA, I often start with a learning rate around `1e-4` to `2e-4`, use a warmup ratio, and select the best checkpoint based on validation loss. I then tune these hyperparameters based on validation metrics and domain-specific evaluation.**

A strong practical configuration is:

```text
LoRA rank:                16
LoRA alpha:               32
LoRA dropout:             0.05
Learning rate:            2e-4
Epochs:                   3
Batch size:               2
Gradient accumulation:    8
Effective batch size:     16
Scheduler:                cosine
Warmup:                   3%
Weight decay:             0.01
Gradient clipping:        1.0
Optimizer:                paged_adamw_8bit
Precision:                BF16
Gradient checkpointing:   enabled
```

The next step is to **create `SFTTrainer` and start the QLoRA fine-tuning loop**.
