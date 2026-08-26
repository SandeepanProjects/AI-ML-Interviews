# Calculate Training Loss in LLM Fine-Tuning

Training loss tells us **how different the model's predictions are from the correct target tokens**.

For an LLM doing next-token prediction:

```text
Input:
The capital of France is

Target:
Paris
```

The model predicts probabilities:

```text
London  → 0.10
Berlin  → 0.05
Paris   → 0.70  ✓
Rome    → 0.15
```

The loss is usually **cross-entropy loss**.

---

# 1. The basic idea

During causal language model training:

```text
Input tokens
    ↓
Model
    ↓
Predicted probability distribution
    ↓
Compare with correct next token
    ↓
Cross-Entropy Loss
    ↓
Backpropagation
```

Mathematically:

$$
Loss = -\frac{1}{N}\sum_{i=1}^{N}\log(P(y_i))
$$

Where:

* \(N\) = number of target tokens
* \(y_i\) = correct token
* \(P(y_i)\) = probability assigned to the correct token

If the model gives high probability to the correct token:

```text
Correct token probability = 0.90
Loss = -log(0.90)
     ≈ 0.105
```

Low loss.

If:

```text
Correct token probability = 0.01
Loss = -log(0.01)
     ≈ 4.605
```

High loss.

---

# 2. PyTorch example: calculate cross-entropy manually

```python
import torch
import torch.nn.functional as F


# Model predictions (logits)
logits = torch.tensor([
    [2.0, 1.0, 0.1],
    [0.5, 3.0, 0.2]
])


# Correct class/token IDs
labels = torch.tensor([
    0,
    1
])


loss = F.cross_entropy(
    logits,
    labels
)

print("Loss:", loss.item())
```

Here:

```text
Example 1:
Correct class = 0

Example 2:
Correct class = 1
```

`cross_entropy()` internally:

1. Applies softmax to logits
2. Calculates negative log probability of the correct class
3. Averages the losses

---

# 3. How loss works in an LLM

Suppose the sentence is:

```text
"I love machine learning"
```

After tokenization:

```text
[10, 25, 87, 300]
```

For causal language modeling:

```text
Input:
I       → predict love
love    → predict machine
machine → predict learning
```

Conceptually:

```text
Input IDs:

[I] [love] [machine] [learning]

Labels:

    [love] [machine] [learning]
```

The labels are shifted relative to the predictions.

---

# 4. Hugging Face automatically calculates training loss

The easiest approach is to pass `labels`.

```python
outputs = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    labels=labels
)

loss = outputs.loss

print(loss.item())
```

Example:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME
)


text = "The capital of France is Paris."

inputs = tokenizer(
    text,
    return_tensors="pt"
)


input_ids = inputs["input_ids"]

labels = input_ids.clone()


outputs = model(
    input_ids=input_ids,
    attention_mask=inputs["attention_mask"],
    labels=labels
)


loss = outputs.loss

print("Training loss:", loss.item())
```

For causal language models, the model implementation handles the token shifting internally when you pass `labels`.

---

# 5. What happens internally?

Suppose:

```text
Input IDs:

[A, B, C, D]
```

The model produces logits:

```text
Position 1 → Predict next token
Position 2 → Predict next token
Position 3 → Predict next token
Position 4 → Predict next token
```

Internally, the loss calculation aligns:

```text
Logits:

Predict B
Predict C
Predict D

Labels:

B
C
D
```

Conceptually:

```python
shift_logits = logits[:, :-1, :]
shift_labels = labels[:, 1:]
```

Then:

```python
loss = F.cross_entropy(
    shift_logits.reshape(-1, vocab_size),
    shift_labels.reshape(-1)
)
```

But usually you **do not need to implement this yourself** because Hugging Face causal LM models do it when `labels` are provided.

---

# 6. Manual LLM loss calculation

Here is a simplified example:

```python
import torch
import torch.nn.functional as F


batch_size = 2
sequence_length = 5
vocab_size = 1000


# Simulated model output
logits = torch.randn(
    batch_size,
    sequence_length,
    vocab_size
)


# Correct token IDs
labels = torch.randint(
    0,
    vocab_size,
    (batch_size, sequence_length)
)


# Shift for next-token prediction
shift_logits = logits[:, :-1, :].contiguous()

shift_labels = labels[:, 1:].contiguous()


# Flatten
shift_logits = shift_logits.view(
    -1,
    vocab_size
)

shift_labels = shift_labels.view(
    -1
)


# Cross entropy
loss = F.cross_entropy(
    shift_logits,
    shift_labels
)

print(loss.item())
```

This demonstrates the underlying calculation.

---

# 7. Training loss with instruction tuning

Suppose the training example is:

```text
User:
What is the capital of France?

Assistant:
Paris is the capital of France.
```

During instruction tuning, we often want the model to learn mainly from the **assistant response**, not necessarily from the user prompt.

Conceptually:

```text
User tokens:
[What] [is] [the] [capital] ...

Assistant tokens:
[Paris] [is] [the] [capital] ...
```

Labels:

```text
Prompt tokens      → -100
Assistant tokens   → actual token IDs
```

Example:

```python
labels = input_ids.clone()

labels[
    :prompt_length
] = -100
```

`-100` is the default ignore index used by PyTorch cross-entropy.

Those tokens do not contribute to the loss.

---

# 8. Example: mask prompt tokens

```python
def tokenize_instruction(example):

    prompt = (
        f"### Instruction:\n"
        f"{example['instruction']}\n\n"
        f"### Response:\n"
    )

    response = example["response"]


    prompt_tokens = tokenizer(
        prompt,
        add_special_tokens=False
    )

    full_text = prompt + response


    full_tokens = tokenizer(
        full_text,
        truncation=True,
        max_length=512,
        padding="max_length"
    )


    input_ids = full_tokens["input_ids"]

    labels = input_ids.copy()


    prompt_length = len(
        prompt_tokens["input_ids"]
    )


    # Ignore prompt tokens
    labels[:prompt_length] = [-100] * prompt_length


    # Ignore padding tokens
    labels = [
        label if token != tokenizer.pad_token_id
        else -100
        for token, label in zip(
            input_ids,
            labels
        )
    ]


    full_tokens["labels"] = labels

    return full_tokens
```

Now the loss is calculated only on:

```text
Assistant response tokens
```

This is often called **completion-only loss**.

---

# 9. Training loss in a custom PyTorch loop

```python
from torch.optim import AdamW


optimizer = AdamW(
    model.parameters(),
    lr=2e-4
)


model.train()


for batch in train_dataloader:

    input_ids = batch[
        "input_ids"
    ].to(device)

    attention_mask = batch[
        "attention_mask"
    ].to(device)

    labels = batch[
        "labels"
    ].to(device)


    # ---------------------------------------------
    # Forward pass
    # ---------------------------------------------

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels
    )


    loss = outputs.loss


    # ---------------------------------------------
    # Backward pass
    # ---------------------------------------------

    loss.backward()


    # ---------------------------------------------
    # Update weights
    # ---------------------------------------------

    optimizer.step()


    # ---------------------------------------------
    # Reset gradients
    # ---------------------------------------------

    optimizer.zero_grad()


    print(
        f"Loss: {loss.item():.4f}"
    )
```

For LoRA, only LoRA parameters receive gradients and are updated.

---

# 10. Training loss with gradient accumulation

In your QLoRA configuration, suppose:

```python
gradient_accumulation_steps = 8
```

The correct training pattern is:

```python
optimizer.zero_grad()

for step, batch in enumerate(train_dataloader):

    outputs = model(
        input_ids=batch["input_ids"].to(device),
        attention_mask=batch["attention_mask"].to(device),
        labels=batch["labels"].to(device)
    )

    loss = outputs.loss


    # Scale loss
    loss = loss / gradient_accumulation_steps


    # Accumulate gradients
    loss.backward()


    # Update after N steps
    if (
        (step + 1)
        % gradient_accumulation_steps
        == 0
    ):

        optimizer.step()

        optimizer.zero_grad()


        print(
            f"Loss: {loss.item() * gradient_accumulation_steps:.4f}"
        )
```

Important:

```python
loss = loss / gradient_accumulation_steps
```

This prevents gradients from becoming approximately `N` times larger when accumulating.

For logging, multiply back:

```python
actual_loss = (
    loss.item()
    * gradient_accumulation_steps
)
```

---

# 11. Calculate average training loss

Do not look at only one batch.

```python
total_loss = 0
num_batches = 0


for batch in train_dataloader:

    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"]
    )

    loss = outputs.loss

    total_loss += loss.item()

    num_batches += 1


average_loss = (
    total_loss
    / num_batches
)

print(
    f"Average loss: {average_loss:.4f}"
)
```

Mathematically:

$$
\text{Average Training Loss}
=
\frac{\sum \text{Batch Loss}}{\text{Number of Batches}}
$$

---

# 12. Token-weighted loss is more accurate

If batches have different numbers of valid target tokens, a simple average of batch losses can be misleading.

For example:

```text
Batch 1 → 100 target tokens
Loss = 1.0

Batch 2 → 10 target tokens
Loss = 3.0
```

Simple average:

$$
(1.0 + 3.0)/2 = 2.0
$$

But batch 2 has far fewer target tokens.

A token-weighted average is better:

```python
total_nll = 0.0
total_tokens = 0


for batch in train_dataloader:

    outputs = model(
        input_ids=batch["input_ids"].to(device),
        attention_mask=batch["attention_mask"].to(device),
        labels=batch["labels"].to(device)
    )

    loss = outputs.loss

    valid_tokens = (
        batch["labels"] != -100
    ).sum().item()


    total_nll += (
        loss.item()
        * valid_tokens
    )

    total_tokens += valid_tokens


average_loss = (
    total_nll
    / total_tokens
)

print(
    f"Token-weighted loss: {average_loss:.4f}"
)
```

---

# 13. Training loss with Hugging Face `Trainer`

You usually don't manually calculate loss.

```python
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=training_args,
    processing_class=tokenizer
)
```

Train:

```python
train_result = trainer.train()
```

Get metrics:

```python
print(
    train_result.metrics
)
```

Example:

```text
{
    'train_runtime': 1250,
    'train_samples_per_second': 10.2,
    'train_loss': 1.24
}
```

The trainer automatically handles:

```text
Forward pass
Loss calculation
Backward pass
Gradient accumulation
Optimizer step
Scheduler
Logging
```

---

# 14. Monitor training loss

With TensorBoard:

```python
training_args = SFTConfig(
    output_dir="./output",
    logging_steps=10,
    report_to="tensorboard"
)
```

Then:

```bash
tensorboard --logdir ./output
```

You can monitor:

```text
Loss
  │\
  │ \
  │  \
  │   \
  │    \____
  └────────────
       Steps
```

A healthy pattern is generally:

```text
Training Loss ↓
Validation Loss ↓
```

But loss alone does not guarantee good model behavior.

---

# 15. Training vs validation loss

Suppose:

```text
Epoch 1
Train = 2.5
Validation = 2.6

Epoch 2
Train = 1.8
Validation = 1.9

Epoch 3
Train = 1.2
Validation = 1.3
```

This generally indicates learning.

But:

```text
Epoch 4
Train = 0.7
Validation = 2.0
```

This suggests possible:

```text
Overfitting
```

The model is memorizing training data instead of generalizing.

---

# 16. Loss vs accuracy

For generative LLMs:

```text
Loss ≠ Accuracy
```

A model predicts a probability distribution over a large vocabulary.

Example:

```text
Correct token: Paris

Model A:
Paris = 0.51

Model B:
Paris = 0.99
```

Both might have the same top-1 prediction:

```text
Accuracy = correct
```

But Model B is much more confident.

Cross-entropy captures this difference.

---

# 17. Perplexity

Perplexity is derived from loss:

$$
Perplexity = e^{Loss}
$$

Code:

```python
import math

loss = 1.5

perplexity = math.exp(loss)

print(perplexity)
```

Example:

```text
Loss = 1.5
Perplexity ≈ 4.48
```

Generally:

```text
Lower loss
↓
Lower perplexity
↓
Better next-token prediction
```

Perplexity is meaningful mainly when comparing comparable tokenization/objectives.

---

# 18. Complete custom training loop with loss

```python
import torch

from torch.optim import AdamW


model.train()

optimizer = AdamW(
    model.parameters(),
    lr=2e-4
)


gradient_accumulation_steps = 8

optimizer.zero_grad()

running_loss = 0.0


for step, batch in enumerate(train_dataloader):

    # ---------------------------------------------
    # Move batch to device
    # ---------------------------------------------

    input_ids = batch[
        "input_ids"
    ].to(device)

    attention_mask = batch[
        "attention_mask"
    ].to(device)

    labels = batch[
        "labels"
    ].to(device)


    # ---------------------------------------------
    # Forward pass
    # ---------------------------------------------

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels
    )


    original_loss = outputs.loss


    # ---------------------------------------------
    # Scale for gradient accumulation
    # ---------------------------------------------

    loss = (
        original_loss
        / gradient_accumulation_steps
    )


    # ---------------------------------------------
    # Backpropagation
    # ---------------------------------------------

    loss.backward()


    running_loss += (
        original_loss.item()
    )


    # ---------------------------------------------
    # Optimizer step
    # ---------------------------------------------

    if (
        (step + 1)
        % gradient_accumulation_steps
        == 0
    ):

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )


        optimizer.step()

        optimizer.zero_grad()


        average_loss = (
            running_loss
            / gradient_accumulation_steps
        )


        print(
            f"Step {step + 1} | "
            f"Training Loss: "
            f"{average_loss:.4f}"
        )


        running_loss = 0.0
```

---

# Interview-ready answer

> **For LLM fine-tuning, training loss is typically token-level cross-entropy loss. The model produces logits for each vocabulary token, and we compare the predicted distribution with the correct next token. In Hugging Face, when we pass `labels` to a causal language model, the model automatically shifts the logits and labels and returns `outputs.loss`. During instruction tuning, I often mask prompt tokens with `-100` so that loss is calculated only on assistant response tokens.**
>
> **During training, I monitor both training and validation loss. Training loss should generally decrease, but if training loss continues decreasing while validation loss increases, that indicates possible overfitting. With gradient accumulation, I divide the loss by the accumulation steps before backpropagation.**

## Most important code

```python
outputs = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    labels=labels
)

loss = outputs.loss

loss.backward()
```

For a Hugging Face fine-tuning pipeline, this is the core mechanism used to calculate and optimize **LLM training loss**.
