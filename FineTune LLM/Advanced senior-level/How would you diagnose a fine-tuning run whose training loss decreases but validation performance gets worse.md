# How would you diagnose a fine-tuning run whose training loss decreases but validation performance gets worse?

This is a very common **LLM fine-tuning interview question**.

The short answer is:

> **If training loss keeps decreasing while validation performance gets worse, I first suspect overfitting, but I don't immediately assume that is the only cause. I check the train/validation split, data leakage, distribution mismatch, evaluation pipeline, prompt formatting, tokenization, training hyperparameters, and catastrophic forgetting. Then I compare learning curves and run controlled experiments to isolate the cause.**

---

# 1. What does this pattern mean?

Suppose:

```text
Epoch       Train Loss       Validation Loss

1              1.20              1.15
2              0.85              0.90
3              0.55              0.88
4              0.30              1.02
5              0.15              1.35
```

The pattern is:

```text
Training loss
     ↓
     ↓
     ↓

Validation loss
     ↓ initially
     ↑
     ↑
```

This is usually:

```text
OVERFITTING
```

The model is becoming better at:

```text
Training data
```

but worse at:

```text
Unseen data
```

Conceptually:

```text
Training examples
       │
       ▼
   Model memorizes
       │
       ▼
Training loss ↓↓↓
       │
       ▼
Generalization ↓
       │
       ▼
Validation performance ↓
```

---

# 2. First, verify that it is actually overfitting

Before changing the model, I verify the evaluation.

Check:

```text
1. Same tokenizer?
2. Same chat template?
3. Same preprocessing?
4. Same max sequence length?
5. Correct labels?
6. Model in eval mode?
7. Correct generation parameters?
8. Same task metric?
```

A validation problem can look like overfitting even when the real issue is:

```text
Training pipeline ≠ Evaluation pipeline
```

---

# 3. Compare train and validation curves

I log:

* Training loss
* Validation loss
* Task-specific metric

Example:

```python
import matplotlib.pyplot as plt

epochs = [1, 2, 3, 4, 5]

train_loss = [
    1.2,
    0.85,
    0.55,
    0.30,
    0.15
]

eval_loss = [
    1.15,
    0.90,
    0.88,
    1.02,
    1.35
]

plt.plot(
    epochs,
    train_loss,
    label="Training Loss"
)

plt.plot(
    epochs,
    eval_loss,
    label="Validation Loss"
)

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()

plt.show()
```

The important diagnostic pattern is:

```text
Train loss ↓
Eval loss  ↑
```

That is strong evidence of overfitting.

But for LLMs, I would **not rely only on perplexity or loss**. I would also evaluate the actual task.

---

# 4. Check task-specific validation performance

For example:

```python
metrics = trainer.evaluate()

print(metrics)
```

You might see:

```text
Epoch 1
eval_loss = 0.85
task_accuracy = 0.81

Epoch 2
eval_loss = 0.78
task_accuracy = 0.86

Epoch 3
eval_loss = 0.95
task_accuracy = 0.82

Epoch 4
eval_loss = 1.20
task_accuracy = 0.74
```

Best checkpoint:

```text
Epoch 2
```

Continuing training made the model worse.

---

# 5. Use early stopping

If validation stops improving, stop training.

A simple conceptual callback:

```python
from transformers import EarlyStoppingCallback

training_args = TrainingArguments(
    output_dir="./output",

    num_train_epochs=10,

    eval_strategy="steps",

    eval_steps=100,

    save_strategy="steps",

    save_steps=100,

    load_best_model_at_end=True,

    metric_for_best_model="eval_loss",

    greater_is_better=False
)
```

Then:

```python
trainer = SFTTrainer(
    model=model,
    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=eval_dataset,

    callbacks=[
        EarlyStoppingCallback(
            early_stopping_patience=3
        )
    ]
)
```

This means:

```text
No validation improvement
        │
        ▼
Wait 3 evaluations
        │
        ▼
Still no improvement?
        │
        ▼
Stop training
        │
        ▼
Restore best checkpoint
```

---

# 6. Check whether the dataset split is correct

A bad split can produce misleading validation results.

Bad:

```text
Train:
Financial documents from 2024

Validation:
Medical documents from 2025
```

The problem may be:

```text
Distribution mismatch
```

The validation data isn't representative of production or training data.

Check:

```python
from collections import Counter


def get_domains(dataset):

    return Counter(
        row["domain"]
        for row in dataset
    )


print(
    get_domains(train_dataset)
)

print(
    get_domains(eval_dataset)
)
```

Example:

```text
Train:
finance: 90%
legal: 10%

Validation:
finance: 30%
legal: 70%
```

This can cause validation performance to worsen even if training is correct.

---

# 7. Check for data leakage

Usually we worry about leakage improving validation artificially, but duplicates and near-duplicates can also make metrics unstable.

Example:

```text
Training:
"Explain compound interest"

Validation:
"Explain compound interest with an example"
```

Use duplicate detection.

```python
import hashlib


def hash_text(text):

    return hashlib.sha256(
        text.strip()
        .lower()
        .encode()
    ).hexdigest()


train_hashes = set(
    hash_text(
        row["text"]
    )
    for row in train_data
)


duplicates = []

for row in eval_data:

    text_hash = hash_text(
        row["text"]
    )

    if text_hash in train_hashes:

        duplicates.append(
            row["text"]
        )


print(
    "Duplicates:",
    len(duplicates)
)
```

For semantic duplicates, use embeddings rather than only exact hashes.

---

# 8. Check prompt formatting and chat templates

This is extremely important for instruction fine-tuning.

Training format:

```text
<user>
Explain LoRA
</user>

<assistant>
LoRA is...
</assistant>
```

But validation:

```text
User: Explain LoRA

Assistant:
```

The model may behave differently.

Use the same formatting pipeline.

```python
def format_example(example):

    messages = [
        {
            "role": "user",
            "content": example["question"]
        },
        {
            "role": "assistant",
            "content": example["answer"]
        }
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False
    )
```

Use it consistently for:

```text
Training
Validation
Inference
```

A common production bug is:

```text
Fine-tuning template
        ≠
Production inference template
```

---

# 9. Check tokenization and truncation

Suppose training data is:

```text
Prompt + Answer
```

with:

```python
max_length = 2048
```

But many examples are:

```text
5000 tokens
```

The answer may get truncated.

Check sequence lengths.

```python
def token_length(example):

    tokens = tokenizer(
        example["text"],
        truncation=False
    )

    return {
        "length": len(
            tokens["input_ids"]
        )
    }


dataset_with_lengths = (
    dataset.map(
        token_length
    )
)
```

Inspect:

```python
lengths = (
    dataset_with_lengths["length"]
)

print(
    max(lengths)
)
```

Also check:

```text
Are assistant responses being truncated?
```

If the target answer is cut off, the model may optimize a broken objective.

---

# 10. Check the label masking

For causal instruction tuning, often you want:

```text
User prompt → ignored for loss
Assistant response → used for loss
```

Conceptually:

```text
User:
Explain LoRA

Assistant:
LoRA is a PEFT technique
```

Labels should look like:

```text
Prompt tokens:
-100
-100
-100
-100

Response tokens:
token_id
token_id
token_id
```

Example:

```python
def create_labels(input_ids, prompt_length):

    labels = input_ids.copy()

    labels[:prompt_length] = [
        -100
    ] * prompt_length

    return labels
```

If the model is trained to predict the entire prompt, training loss can decrease while actual instruction-following quality does not improve.

---

# 11. Check whether the validation metric matches the business goal

Suppose your task is:

```text
Generate valid JSON
```

But you only monitor:

```text
Cross-entropy loss
```

The loss may improve while:

```text
JSON validity ↓
```

Evaluate the real requirement.

Example:

```python
import json


def is_valid_json(text):

    try:

        json.loads(text)

        return True

    except json.JSONDecodeError:

        return False
```

Evaluate:

```python
valid_count = sum(
    is_valid_json(output)
    for output in predictions
)

json_validity = (
    valid_count
    / len(predictions)
)

print(
    f"JSON validity: "
    f"{json_validity:.2%}"
)
```

So I would monitor:

```text
Training loss
+
Validation loss
+
Business metric
```

---

# 12. Learning rate may be too high

A high learning rate can cause the model to:

```text
Forget useful pretrained behavior
```

This is especially important in LoRA and full fine-tuning.

Bad:

```python
learning_rate = 1e-3
```

Potentially better starting points for LoRA experiments:

```python
learning_rate = 1e-4
```

or:

```python
learning_rate = 2e-4
```

But the correct value depends on the model and training setup.

I would run controlled experiments:

```python
learning_rates = [
    5e-5,
    1e-4,
    2e-4
]
```

Keep everything else constant.

---

# 13. Too many training epochs

Example:

```text
Epoch 1 → Good
Epoch 2 → Better
Epoch 3 → Best
Epoch 4 → Worse
Epoch 5 → Worse
Epoch 6 → Much worse
```

The fix may simply be:

```text
Train fewer epochs
```

For example:

```python
TrainingArguments(
    num_train_epochs=3
)
```

instead of:

```python
TrainingArguments(
    num_train_epochs=10
)
```

Always keep the best checkpoint.

---

# 14. LoRA rank may be too high

Suppose:

```text
Dataset = 2,000 examples

Rank = 128
```

The adapter may have excessive capacity.

Try:

```text
r = 64
r = 32
r = 16
r = 8
```

Example:

```python
LoraConfig(
    r=16,

    lora_alpha=32,

    target_modules=[
        "q_proj",
        "v_proj"
    ],

    lora_dropout=0.05,

    task_type="CAUSAL_LM"
)
```

However:

> **Never diagnose rank in isolation.**

Also test:

```text
Dataset quality
Learning rate
Epochs
Target modules
```

---

# 15. Too many target modules

Suppose:

```python
target_modules=[
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj"
]
```

For a small dataset, this may provide too much adaptation capacity.

Try a smaller configuration:

```python
target_modules=[
    "q_proj",
    "v_proj"
]
```

Compare validation results.

---

# 16. Add LoRA dropout

If overfitting is confirmed:

```python
LoraConfig(
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

You can experiment with:

```text
0.0
0.05
0.1
```

But too much dropout can cause underfitting.

---

# 17. Check dataset quality

A decreasing loss doesn't mean your dataset is good.

Examples of bad training data:

```text
Incorrect answers
Contradictory answers
Duplicate examples
Inconsistent formatting
Poor instructions
Low-quality synthetic data
```

Example:

```text
Question:
What is LoRA?

Answer 1:
LoRA reduces trainable parameters.

Answer 2:
LoRA is a database.

Answer 3:
LoRA is used for image compression.
```

The model can learn inconsistent behavior while training loss still decreases.

I would sample and manually inspect:

```python
import random


samples = random.sample(
    train_data,
    20
)

for sample in samples:

    print(
        sample
    )
```

For production:

```text
Dataset validation
+
Schema validation
+
Deduplication
+
PII removal
+
Quality scoring
+
Human review
```

---

# 18. Check catastrophic forgetting

This is particularly important when:

```text
Fine-tuning data is narrow
```

Example:

```text
Base model:
General-purpose assistant
```

Fine-tuning data:

```text
100% financial documents
```

After aggressive training:

```text
Finance performance ↑

General reasoning ↓
General instruction following ↓
```

This is:

```text
Catastrophic forgetting
```

Test both:

```text
Domain benchmark
+
General benchmark
```

Example:

```python
evaluation_results = {

    "domain_score":
        evaluate_domain_model(),

    "general_score":
        evaluate_general_capability()
}

print(
    evaluation_results
)
```

If:

```text
Domain score ↑
General score ↓↓↓
```

then the fine-tuning may be too aggressive.

Possible fixes:

```text
Lower learning rate
Fewer epochs
Reduce LoRA rank
Reduce target modules
Add general instruction data
Use data mixing
```

---

# 19. Compare against the base model

Never evaluate only the fine-tuned model.

Create:

```text
                    Same benchmark
                         │
             ┌───────────┴───────────┐
             │                       │
        Base model              Fine-tuned
             │                       │
             └───────────┬───────────┘
                         │
                         ▼
                    Compare
```

Example:

```python
results = {
    "base": {
        "domain_accuracy": 0.72,
        "general_accuracy": 0.88
    },

    "fine_tuned": {
        "domain_accuracy": 0.91,
        "general_accuracy": 0.79
    }
}
```

This reveals whether you're trading:

```text
General capability
```

for:

```text
Narrow specialization
```

---

# 20. Reproduce the problem with controlled experiments

Change one thing at a time.

Bad experiment:

```text
Change:
Rank
Learning rate
Batch size
Epochs
Dataset
Target modules

All at once
```

Then you don't know what fixed the problem.

Instead:

### Experiment 1

```text
Baseline
```

### Experiment 2

```text
Only reduce learning rate
```

### Experiment 3

```text
Only reduce epochs
```

### Experiment 4

```text
Only reduce rank
```

### Experiment 5

```text
Only change dataset
```

Track experiments.

---

# 21. Production experiment tracking

I would log:

```python
experiment = {

    "model": "llama",

    "dataset_version": "v3",

    "lora_rank": 16,

    "lora_alpha": 32,

    "target_modules": [
        "q_proj",
        "v_proj"
    ],

    "learning_rate": 1e-4,

    "epochs": 3,

    "train_loss": 0.32,

    "eval_loss": 0.41,

    "domain_score": 0.89,

    "general_score": 0.86
}
```

This makes the diagnosis reproducible.

---

# 22. My production debugging workflow

This is how I would approach it:

```text
Training loss ↓
Validation performance ↓
          │
          ▼
   Verify evaluation pipeline
          │
          ▼
Same tokenizer/template?
          │
          ▼
Correct labels/masking?
          │
          ▼
Correct validation data?
          │
          ▼
Check train/eval distribution
          │
          ▼
Inspect data quality
          │
          ▼
Compare train/eval curves
          │
          ▼
Check base model benchmark
          │
          ▼
Tune:
  - epochs
  - learning rate
  - LoRA rank
  - target modules
  - dropout
          │
          ▼
Use early stopping
          │
          ▼
Select best checkpoint
```

---

# 23. A practical diagnosis function

Here's a simplified implementation:

```python
def diagnose_training(
    train_loss,
    eval_loss,
    task_score
):

    diagnosis = []

    # ---------------------------------
    # Check overfitting
    # ---------------------------------

    if (
        train_loss[-1] < train_loss[0]
        and
        eval_loss[-1] > min(eval_loss)
    ):

        diagnosis.append(
            "Possible overfitting"
        )

    # ---------------------------------
    # Check task degradation
    # ---------------------------------

    if (
        task_score[-1]
        <
        max(task_score)
    ):

        diagnosis.append(
            "Task performance degraded"
        )

    # ---------------------------------
    # Recommend actions
    # ---------------------------------

    recommendations = [

        "Verify validation pipeline",

        "Check train/eval distribution",

        "Check data leakage and duplicates",

        "Try early stopping",

        "Reduce epochs",

        "Reduce learning rate",

        "Try smaller LoRA rank",

        "Reduce target modules",

        "Increase LoRA dropout",

        "Compare against base model"
    ]

    return {
        "diagnosis": diagnosis,
        "recommendations": recommendations
    }
```

Usage:

```python
result = diagnose_training(

    train_loss=[
        1.2,
        0.8,
        0.5,
        0.2
    ],

    eval_loss=[
        1.1,
        0.9,
        1.0,
        1.3
    ],

    task_score=[
        0.70,
        0.82,
        0.78,
        0.69
    ]
)

print(result)
```

---

# 24. Strong interview answer

If asked:

**"How would you diagnose a fine-tuning run where training loss decreases but validation performance gets worse?"**

A strong answer is:

> **I would first suspect overfitting, but I would verify the evaluation pipeline before changing hyperparameters. I would confirm that training and validation use the same tokenizer, chat template, preprocessing, sequence length, and label masking. Then I would inspect training and validation curves and compare task-specific metrics rather than relying only on loss. I would validate the dataset split for distribution mismatch, duplicates, leakage, and data quality. I would also compare the fine-tuned model against the base model to detect catastrophic forgetting.**
>
> **If overfitting is confirmed, I would use early stopping and the best validation checkpoint, then run controlled experiments by reducing epochs or learning rate, reducing LoRA rank or target modules, and potentially adding LoRA dropout or more diverse training data. I would change one variable at a time and track every experiment so the root cause can be isolated.**

## The key mental model

```text
Train loss ↓ + Validation loss ↓
        = learning and generalizing

Train loss ↓ + Validation loss ↑
        = usually overfitting

Train loss ↓ + Task metric ↓
        = check evaluation pipeline,
          objective mismatch,
          data quality,
          and catastrophic forgetting
```

**The biggest interview takeaway:** a lower training loss does **not** automatically mean a better fine-tuned model. The model should ultimately be selected using **held-out, task-relevant validation metrics**, not training loss alone.
