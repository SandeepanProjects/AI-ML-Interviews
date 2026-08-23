# What happens internally during LLM fine-tuning?

Fine-tuning is basically:

> **Take a pre-trained model → show it task-specific examples → calculate how wrong its predictions are → use backpropagation to update its parameters → repeat many times.**

The important thing is that **fine-tuning does not start learning from zero**. It starts from a model that already understands language and has billions of learned parameters.

---

# 1. Start with a pre-trained model

Suppose we start with a model:

```text
Pre-trained LLM
    │
    ├── Token embeddings
    ├── Transformer layers
    ├── Attention weights
    ├── Feed-forward layers
    └── Output layer
```

Conceptually:

```text
Text
 ↓
Tokenizer
 ↓
Tokens
 ↓
Embeddings
 ↓
Transformer layers
 ↓
Output probabilities
```

The model already knows general language patterns.

For example, it has learned relationships such as:

```text
"Python" → programming language
"Paris" → France
"FastAPI" → Python web framework
```

Fine-tuning modifies the existing model so that it performs a particular task better.

---

# 2. Give it a training example

Suppose we want to fine-tune a model for customer-support classification.

Training example:

```text
Input:
"I was charged twice for my subscription."

Expected output:
"billing"
```

The tokenizer converts the text into tokens:

```text
"I was charged twice for my subscription"
                  ↓
        [token IDs]
```

The exact token IDs depend on the tokenizer.

---

# 3. Tokenization

The text is converted into numerical IDs.

Conceptually:

```text
"I was charged twice"
        ↓
[40, 128, 5732, 923, ...]
```

The model cannot directly process English strings.

It processes tensors containing token IDs/embeddings.

The training batch might look conceptually like:

```python
input_ids = [
    [101, 2045, 567, 892, ...],
    [101, 2034, 789, 123, ...],
]
```

---

# 4. Forward pass

The tokens go through the Transformer.

```text
Tokens
   ↓
Embedding
   ↓
Transformer Layer 1
   ↓
Transformer Layer 2
   ↓
Transformer Layer 3
   ↓
...
   ↓
Transformer Layer N
   ↓
Logits
```

Inside each Transformer block, you have components such as:

```text
Self-Attention
      ↓
Residual connection
      ↓
Layer Normalization
      ↓
Feed Forward Network
      ↓
Residual connection
      ↓
Layer Normalization
```

The model eventually produces **logits**.

For example:

```text
billing       → 4.8
technical     → 1.2
account       → 0.7
shipping      → 0.4
```

These are not probabilities yet.

---

# 5. Convert logits into probabilities

A softmax function converts logits into probabilities.

For example:

```text
billing       → 0.94
technical     → 0.03
account       → 0.02
shipping      → 0.01
```

Suppose the correct answer is:

```text
billing
```

Then the model is doing well.

But imagine it predicted:

```text
billing       → 0.40
technical     → 0.50
account       → 0.07
shipping      → 0.03
```

Now the model is wrong.

We need a way to measure how wrong it is.

---

# 6. Calculate the loss

The training process calculates a **loss**.

For classification-like objectives, cross-entropy is commonly used.

Conceptually:

```text
Expected:
billing

Model:
billing = 0.40

                 ↓

              Loss
```

If the model gives high probability to the correct answer:

```text
Probability(correct) = 0.95

       ↓

Small loss
```

If it gives low probability:

```text
Probability(correct) = 0.05

       ↓

Large loss
```

For language-model fine-tuning, the exact loss formulation depends on the training setup, but next-token cross-entropy is very common.

---

# 7. Backpropagation

Now comes the most important part.

The model asks:

> **Which parameters contributed to this error, and in what direction should they change?**

Backpropagation calculates gradients.

Conceptually:

```text
Loss
 ↓
Gradient calculation
 ↓
Gradients for model parameters
```

You can think of a gradient as telling the optimizer:

```text
"This parameter should move slightly in this direction
to reduce the error."
```

---

# 8. Optimizer updates the weights

Suppose a parameter is:

```text
weight = 0.72
```

The gradient tells us how it should change.

The optimizer updates it:

```text
Old weight
   ↓
Optimizer
   ↓
New weight
```

A simplified gradient-descent equation is:

```text
W_new = W_old - learning_rate × gradient
```

For example:

```text
W_old       = 0.72
gradient    = 0.04
learning rate = 0.01

W_new = 0.72 - (0.01 × 0.04)
      = 0.7196
```

Real LLM training normally uses optimizers such as AdamW and additional mechanisms such as learning-rate schedules, mixed precision, gradient accumulation, etc.

---

# 9. Repeat this process

This happens over and over:

```text
Training example
      ↓
Tokenization
      ↓
Forward pass
      ↓
Prediction
      ↓
Loss calculation
      ↓
Backpropagation
      ↓
Gradient calculation
      ↓
Optimizer
      ↓
Update weights
      ↓
Next batch
```

Eventually:

```text
Millions of tokens
        ↓
Many batches
        ↓
Multiple epochs/steps
        ↓
Specialized model
```

---

# 10. What is an epoch?

An **epoch** means the training process has gone through the training dataset once.

Suppose you have:

```text
10,000 training examples
```

One epoch:

```text
Example 1
Example 2
...
Example 10,000
```

Two epochs:

```text
Dataset processed
       ↓
Again
       ↓
Dataset processed again
```

So:

```text
Epoch 1 → entire dataset
Epoch 2 → entire dataset
Epoch 3 → entire dataset
```

More epochs are not automatically better.

Too many can cause **overfitting**.

---

# 11. What is a batch?

You generally don't process every training example simultaneously.

Suppose:

```text
Dataset = 10,000 examples
Batch size = 8
```

The model processes:

```text
Batch 1 → examples 1–8
Batch 2 → examples 9–16
Batch 3 → examples 17–24
...
```

After processing a batch, gradients can be accumulated and the optimizer updates the parameters.

A simplified loop looks like:

```python
for epoch in range(num_epochs):

    for batch in dataloader:

        outputs = model(batch)

        loss = calculate_loss(
            outputs,
            batch["labels"]
        )

        loss.backward()

        optimizer.step()

        optimizer.zero_grad()
```

That's the core idea behind fine-tuning.

---

# 12. What exactly changes inside the model?

This is a very important interview question.

Suppose your model has:

```text
7 billion parameters
```

With **full fine-tuning**, many/all trainable parameters can be updated:

```text
Before:

Transformer weights
      ↓
[W1, W2, W3, ... billions of parameters]

Fine-tuning
      ↓

After:

[W1', W2', W3', ... updated parameters]
```

The model doesn't simply store:

```text
"billing = customer charged twice"
```

as a database entry.

Instead, training adjusts distributed numerical representations and weights so that the desired behavior becomes more likely.

---

# 13. What happens with LoRA?

This is especially important for modern LLM interviews.

Full fine-tuning can require enormous GPU memory.

LoRA uses a different strategy.

Instead of updating the entire model:

```text
Base Model
   │
   ├── Frozen weights ❄️
   │
   └── Small trainable matrices
```

The base model remains frozen.

LoRA introduces trainable low-rank matrices.

Conceptually:

```text
Original weight:

W

Instead of directly changing W:

W' = W + ΔW

LoRA represents:

ΔW ≈ A × B
```

where `A` and `B` are much smaller matrices.

So instead of training billions of parameters:

```text
7B base parameters
       ↓
Frozen

Small LoRA parameters
       ↓
Trainable
```

This dramatically reduces trainable parameter count and often memory requirements.

---

# 14. What about QLoRA?

QLoRA goes one step further.

Conceptually:

```text
Base model
    ↓
Quantize model
    ↓
Keep base model in lower precision
    ↓
Add LoRA adapters
    ↓
Train adapters
```

So:

```text
QLoRA =
Quantized base model
+
LoRA adapters
```

The base model is typically frozen while the adapters are trained.

This allows fine-tuning relatively large models with substantially less GPU memory than full-precision full fine-tuning.

---

# 15. What happens during instruction fine-tuning?

Suppose the dataset contains:

```text
User:
Explain RAG.

Assistant:
RAG stands for Retrieval-Augmented Generation...
```

During supervised fine-tuning, the model is trained to predict the desired assistant response given the conversation context.

Conceptually:

```text
System instruction
        +
User message
        ↓
      Model
        ↓
Expected assistant tokens
        ↓
Loss
        ↓
Backpropagation
        ↓
Weight updates
```

For many chat fine-tuning setups, the loss is focused on the assistant/output tokens rather than treating every conversational token equally. The exact masking depends on the training implementation.

---

# 16. Why does fine-tuning change model behavior?

Imagine before fine-tuning:

```text
User:
Classify this ticket:
"I was charged twice."

Model:
It appears that the customer may have a billing issue...
```

After fine-tuning:

```text
User:
Classify this ticket:
"I was charged twice."

Model:
billing
```

Why?

Because during training the model repeatedly sees:

```text
Input → Desired output
```

and its parameters are adjusted so that the desired output becomes more probable.

After enough good examples:

```text
Pattern learned
      ↓
Higher probability of desired behavior
```

---

# 17. Fine-tuning does NOT work like adding rows to a database

This distinction is extremely important.

Suppose you fine-tune with:

```text
Company policy:
Employees get 25 vacation days.
```

The model doesn't simply create a database record:

```text
policy = 25 days
```

Instead, the training process changes model parameters.

That's why fine-tuning is generally **not the ideal mechanism for frequently changing factual knowledge**.

For frequently changing enterprise information:

```text
Documents
   ↓
RAG
   ↓
LLM at inference time
```

is generally more appropriate.

---

# 18. Fine-tuning vs RAG internally

This is a common interview comparison.

### Fine-tuning

```text
Training time:

Dataset
   ↓
Loss
   ↓
Backpropagation
   ↓
Weight updates
   ↓
Model behavior changes
```

### RAG

```text
Inference time:

User question
      ↓
Retriever
      ↓
Relevant documents
      ↓
Prompt + documents
      ↓
LLM
      ↓
Answer
```

So:

> **Fine-tuning changes the model. RAG changes the context given to the model.**

---

# 19. What happens to the loss during training?

Ideally:

```text
Loss
 │\
 │ \
 │  \
 │   \__
 │      \___
 └──────────────
       Steps
```

The loss generally decreases as the model learns the training examples.

But you should monitor **validation loss** too.

You might see:

```text
Training loss:
↓ ↓ ↓ ↓ ↓

Validation loss:
↓ ↓ ↓ ↑ ↑
```

This can indicate:

> **Overfitting**

The model is becoming better at the training examples but worse at generalizing to unseen examples.

---

# 20. Production fine-tuning pipeline

In a real AI engineering project, I would think about the pipeline like this:

```text
                    Raw Data
                       │
                       ▼
                Data Cleaning
                       │
                       ▼
               PII/Security Check
                       │
                       ▼
                 Data Labeling
                       │
                       ▼
              Quality Validation
                       │
                       ▼
              Train / Val / Test
                       │
                       ▼
                Base LLM
                       │
                       ▼
                 Fine-tuning
                       │
             ┌─────────┴─────────┐
             ▼                   ▼
       Training Metrics    Validation Metrics
             │                   │
             └─────────┬─────────┘
                       ▼
                  Evaluation
                       │
                       ▼
                 Model Registry
                       │
                       ▼
                  Deployment
                       │
                       ▼
                   Inference
                       │
                       ▼
                  Monitoring
```

---

# 21. What should you monitor?

During fine-tuning, I would monitor:

### Training metrics

```text
Training loss
Learning rate
Gradient norms
Training steps
```

### Validation metrics

```text
Validation loss
Task-specific accuracy
F1
Precision
Recall
```

### For LLM generation

Also evaluate:

```text
Instruction following
Output format compliance
Factuality / faithfulness
Safety
Human preference
Task success rate
```

And compare against the **original base model**.

---

# 22. A concrete example

Suppose you have:

```text
Base model: 7B
Dataset: 10,000 customer-support examples
Method: LoRA
Epochs: 3
Batch size: 8
```

The simplified process is:

```text
10,000 examples
       ↓
Tokenization
       ↓
Create batches of 8
       ↓
Forward pass
       ↓
Calculate loss
       ↓
Backpropagation
       ↓
Update LoRA parameters
       ↓
Next batch
       ↓
...
       ↓
Epoch 1
       ↓
Epoch 2
       ↓
Epoch 3
       ↓
Evaluate
```

The original 7B model remains frozen, while the LoRA adapter learns the task.

---

# 23. Interview answer

If the interviewer asks:

> **What happens internally during fine-tuning?**

A strong answer is:

> "Fine-tuning starts with a pre-trained model and a task-specific dataset. The training examples are tokenized and passed through the Transformer in a forward pass. The model generates logits, and a training loss such as cross-entropy measures the difference between the model's predictions and the expected outputs. Backpropagation calculates gradients with respect to the trainable parameters, and an optimizer such as AdamW updates those parameters. This process is repeated over batches and multiple training steps or epochs. In full fine-tuning, many or all model parameters can be updated, whereas approaches such as LoRA or QLoRA freeze the base model and train a much smaller set of adapter parameters. Finally, we evaluate the fine-tuned model on held-out data to check generalization and compare it with the original model."

## The mental model to remember

```text
              FINE-TUNING

Training example
      ↓
Tokenization
      ↓
Forward pass
      ↓
Model prediction
      ↓
Calculate loss
      ↓
Backpropagation
      ↓
Calculate gradients
      ↓
Optimizer
      ↓
Update weights / adapters
      ↓
Next batch
      ↓
Repeat
      ↓
Specialized model
```

**One sentence:**

> **Fine-tuning is continued training of a pre-trained model where task-specific examples produce a loss, gradients are computed through backpropagation, and model weights or adapter parameters are updated to make the desired behavior more likely.**
