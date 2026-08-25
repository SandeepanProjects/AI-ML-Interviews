# What happens if the learning rate is too high or too low?

The **learning rate (LR)** controls how large a step the model takes when updating its parameters.

The basic gradient descent update is:

[
\theta_{new}
============

## \theta_{old}

\eta \nabla_{\theta}L
]

Where:

* (\theta) = model parameters
* (L) = loss
* (\nabla_{\theta}L) = gradient
* (\eta) = learning rate

The learning rate determines **how aggressively the model changes its weights**.

---

# 1. Simple intuition

Imagine you are trying to reach the bottom of a valley.

### Good learning rate

```text
        \        /
         \      /
          \    /
           \  /
            \/

        ↓ ↓ ↓
      small enough
      useful steps
```

You eventually reach the bottom.

### Learning rate too high

```text
        \        /
         \      /
    X     \    /     X
      \    \  /
       \    \/
       overshooting
```

You keep jumping over the optimal point.

### Learning rate too low

```text
Start
  ↓
 . . . . . . . . . . . .
                    optimum
```

You move so slowly that training takes a very long time.

---

# 2. Code: simple gradient descent example

Let's use a simple mathematical function:

[
f(x)=x^2
]

The minimum is:

[
x=0
]

The gradient is:

[
\frac{df}{dx}=2x
]

## Gradient descent code

```python
def gradient_descent(
    learning_rate,
    epochs,
    start_x=10
):

    x = start_x

    history = []

    for epoch in range(epochs):

        # Loss
        loss = x ** 2

        # Gradient of x²
        gradient = 2 * x

        # Gradient descent update
        x = (
            x
            -
            learning_rate * gradient
        )

        history.append(
            {
                "epoch": epoch,
                "x": x,
                "loss": loss
            }
        )

    return history
```

---

# 3. Good learning rate

Let's try:

```python
history = gradient_descent(
    learning_rate=0.1,
    epochs=20
)

for item in history[:10]:

    print(
        item["epoch"],
        round(item["x"], 4),
        round(item["loss"], 4)
    )
```

Conceptually:

```text
Epoch     x        Loss

0         8.0      100
1         6.4       64
2         5.12      40.96
3         4.09      26.21
...
```

Eventually:

```text
x → 0
Loss → 0
```

This is healthy convergence.

---

# 4. What happens if the learning rate is too high?

Suppose:

```python
history = gradient_descent(
    learning_rate=1.1,
    epochs=10
)
```

The update is:

[
x_{new}
=======

## x

1.1(2x)
]

[
x_{new}
=======

-1.2x
]

Starting from:

[
x=10
]

We get approximately:

```text
10
↓
-12
↓
14.4
↓
-17.28
↓
20.73
↓
-24.88
```

The magnitude keeps increasing.

Loss:

```text
100
↓
144
↓
207
↓
298
↓
...
```

The training **diverges**.

---

# 5. Too-high learning rate causes overshooting

Suppose the loss landscape looks like this:

```text
Loss
 │
 │       \        /
 │        \      /
 │         \    /
 │          \  /
 │           \/
 │
 └──────────────────── Parameter

      Start

        BIG STEP
            ─────────────►
                       Overshoot

              ◄─────────────
                 BIG STEP

          ◄─────────────
```

Instead of approaching the minimum:

```text
Minimum
   ↓

Start → → → → ●
```

the model keeps jumping back and forth.

---

# 6. PyTorch example: learning rate too high

Let's create a simple regression model.

```python
import torch
import torch.nn as nn
```

Create data:

```python
torch.manual_seed(42)

X = torch.randn(
    100,
    1
)

y = (
    3 * X
    +
    2
)
```

Create a model:

```python
model = nn.Linear(
    1,
    1
)
```

Loss:

```python
criterion = nn.MSELoss()
```

Use an extremely high learning rate:

```python
optimizer = torch.optim.SGD(

    model.parameters(),

    lr=10.0
)
```

Training:

```python
for epoch in range(20):

    optimizer.zero_grad()

    predictions = model(X)

    loss = criterion(
        predictions,
        y
    )

    loss.backward()

    optimizer.step()

    print(
        f"Epoch {epoch}: "
        f"Loss = {loss.item():.4f}"
    )
```

You may see something like:

```text
Epoch 0: Loss = 12.4
Epoch 1: Loss = 850.2
Epoch 2: Loss = 125000.5
Epoch 3: Loss = 18000000
...
Epoch 10: Loss = inf
Epoch 11: Loss = nan
```

Exact numbers vary.

The pattern is:

```text
Loss
 ↑
 ↑
 ↑
 ↑
BOOM
 ↓
NaN
```

This happens because parameter updates are too large.

---

# 7. Symptoms of a learning rate that is too high

During LLM fine-tuning, you might see:

### 1. Loss explodes

```text
Step 100   Loss = 2.1
Step 200   Loss = 1.9
Step 300   Loss = 5.0
Step 400   Loss = 25.0
Step 500   Loss = NaN
```

---

### 2. Loss oscillates

```text
Loss

5 ──●     ●
    │ \   │
3   │  ●──│
    │
1   │
    └──────────
```

Example:

```text
2.1
1.5
2.8
1.4
3.2
1.8
```

The model is bouncing around the optimum.

---

### 3. Gradients become unstable

You might see:

```text
gradient norm:

0.5
0.8
2.1
50
1000
inf
```

---

### 4. Model quality suddenly degrades

For an LLM:

```text
Before training:

User: Explain Python decorators

Model:
Correct explanation
```

After aggressive fine-tuning:

```text
User: Explain Python decorators

Model:
Random domain-specific response
```

The model may become unstable or suffer significant behavioral regression.

---

# 8. What happens if the learning rate is too low?

Now try:

```python
history = gradient_descent(
    learning_rate=0.00001,
    epochs=20
)
```

The update is tiny.

Starting from:

```text
x = 10
```

After one step:

[
x_{new}
=======

## 10

0.00001(20)
]

[
x_{new}=9.9998
]

The model barely moves.

After many epochs:

```text
Epoch 0     x = 9.9998
Epoch 1     x = 9.9996
Epoch 2     x = 9.9994
Epoch 100   x ≈ 9.98
```

The model needs an enormous number of updates.

---

# 9. PyTorch example: learning rate too low

```python
import torch
import torch.nn as nn
```

Create model:

```python
model = nn.Linear(
    1,
    1
)
```

Loss:

```python
criterion = nn.MSELoss()
```

Use a tiny learning rate:

```python
optimizer = torch.optim.SGD(

    model.parameters(),

    lr=1e-7
)
```

Train:

```python
for epoch in range(20):

    optimizer.zero_grad()

    predictions = model(X)

    loss = criterion(
        predictions,
        y
    )

    loss.backward()

    optimizer.step()

    print(
        f"Epoch {epoch}: "
        f"Loss = {loss.item():.6f}"
    )
```

You may see:

```text
Epoch 0: Loss = 12.40
Epoch 1: Loss = 12.3998
Epoch 2: Loss = 12.3996
Epoch 10: Loss = 12.3985
Epoch 19: Loss = 12.3972
```

The model is technically learning, but **extremely slowly**.

---

# 10. Symptoms of a learning rate that is too low

## 1. Loss decreases extremely slowly

```text
Step 100     2.500
Step 1000    2.490
Step 5000    2.470
```

Training is wasting compute.

---

## 2. Model underfits because training finishes before convergence

Suppose:

```text
Learning rate = 1e-7
Epochs = 3
```

The model may barely learn anything.

```text
Before fine-tuning:
Domain score = 60%

After:
Domain score = 61%
```

Not enough adaptation happened.

---

## 3. Training takes too long

```text
Correct LR:

10,000 steps
↓
Good performance


Too low LR:

1,000,000 steps
↓
Similar performance
```

That means:

```text
More GPU cost
More time
More electricity
```

---

# 11. Comparison

| Learning Rate | Behavior                                |
| ------------- | --------------------------------------- |
| Too high      | Overshoots optimum                      |
| Too high      | Loss oscillates                         |
| Too high      | Loss can diverge                        |
| Too high      | Gradients can explode                   |
| Too high      | `NaN` / `inf` possible                  |
| Good          | Stable convergence                      |
| Good          | Loss decreases steadily                 |
| Too low       | Very slow learning                      |
| Too low       | Wastes compute                          |
| Too low       | May underfit                            |
| Too low       | May not converge within training budget |

---

# 12. Visualizing all three

```text
GOOD LEARNING RATE

Loss
│\
│ \
│  \
│   \
│    \____
│
└──────────── Steps
```

```text
LEARNING RATE TOO HIGH

Loss
│   /\      /\
│  /  \    /  \
│ /    \__/    \
│
└──────────── Steps
```

```text
LEARNING RATE TOO LOW

Loss
│\
│ \
│  \
│   \
│    \
│     \
└──────────── Steps
```

Very slow progress.

---

# 13. Compare multiple learning rates with code

This is a useful experiment.

```python
import torch
import torch.nn as nn
```

Create a function:

```python
def train_model(
    learning_rate,
    epochs=100
):

    # New model for each experiment
    model = nn.Linear(
        1,
        1
    )

    criterion = nn.MSELoss()

    optimizer = torch.optim.SGD(

        model.parameters(),

        lr=learning_rate
    )

    losses = []

    for epoch in range(epochs):

        optimizer.zero_grad()

        predictions = model(X)

        loss = criterion(
            predictions,
            y
        )

        loss.backward()

        optimizer.step()

        losses.append(
            loss.item()
        )

    return losses
```

Try different learning rates:

```python
low_lr_losses = train_model(
    learning_rate=1e-5
)

good_lr_losses = train_model(
    learning_rate=0.01
)

high_lr_losses = train_model(
    learning_rate=1.0
)
```

Plot them:

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))

plt.plot(
    low_lr_losses,
    label="Low LR = 1e-5"
)

plt.plot(
    good_lr_losses,
    label="Good LR = 0.01"
)

plt.plot(
    high_lr_losses,
    label="High LR = 1.0"
)

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.title(
    "Effect of Learning Rate"
)

plt.legend()

plt.show()
```

Typically:

```text
Low LR:
Slow decrease

Good LR:
Fast stable decrease

High LR:
Oscillation or explosion
```

---

# 14. How does this apply to LLM fine-tuning?

LLMs are particularly sensitive because they contain billions of parameters.

A full fine-tuning example:

```python
training_args = TrainingArguments(

    output_dir="./model",

    learning_rate=1e-5,

    num_train_epochs=3,

    warmup_ratio=0.03
)
```

A LoRA example might start with something like:

```python
training_args = TrainingArguments(

    output_dir="./lora_model",

    learning_rate=1e-4,

    num_train_epochs=3,

    warmup_ratio=0.05
)
```

Why might LoRA use a higher LR?

Because:

```text
Full fine-tuning:

Billions of parameters
↓
Need careful small updates
```

LoRA:

```text
Base model = frozen

Small adapter parameters = trainable
↓
Often needs to learn adaptation more quickly
```

But there is no universal rule. The correct LR depends on:

* model
* dataset size
* dataset quality
* batch size
* optimizer
* full fine-tuning vs LoRA/QLoRA
* sequence length
* effective batch size
* target modules
* number of training steps

---

# 15. A practical LLM learning-rate experiment

Don't choose a learning rate randomly.

For example, test:

```text
1e-5
3e-5
5e-5
1e-4
2e-4
```

Pseudo experiment:

```python
learning_rates = [

    1e-5,

    3e-5,

    5e-5,

    1e-4,

    2e-4
]
```

For each LR:

```python
for lr in learning_rates:

    training_args = TrainingArguments(

        output_dir=f"./experiment_lr_{lr}",

        learning_rate=lr,

        num_train_epochs=1,

        per_device_train_batch_size=4,

        gradient_accumulation_steps=8,

        eval_strategy="steps",

        eval_steps=100,

        report_to="none"
    )

    print(
        f"Training with LR = {lr}"
    )

    # Create fresh model here
    # Train
    # Evaluate
```

Important: each experiment should start from the **same original model checkpoint**, otherwise the comparison is invalid.

Track:

```text
Learning rate
Train loss
Validation loss
Task score
General capability score
Training stability
```

Example:

| LR   | Train Loss | Val Loss | Domain Score | Result        |
| ---- | ---------: | -------: | -----------: | ------------- |
| 1e-5 |        1.9 |      2.0 |          75% | Too slow      |
| 3e-5 |        1.5 |      1.6 |          84% | Good          |
| 5e-5 |        1.2 |      1.3 |          91% | Best          |
| 1e-4 |        0.9 |      1.7 |          85% | Over-adapting |
| 2e-4 |        NaN |      NaN |            — | Unstable      |

These are illustrative numbers, not universal thresholds.

---

# 16. Warmup helps prevent high-LR instability

Instead of immediately using:

```text
Learning rate = 1e-4
```

start small:

```text
Step 0:

0

Step 100:

2e-5

Step 500:

1e-4
```

Then:

```text
Peak LR
    │
1e-4│          ──────────
    │        /
    │      /
    │    /
0   └───────────────────
        Warmup
```

Code:

```python
training_args = TrainingArguments(

    output_dir="./model",

    learning_rate=1e-4,

    warmup_ratio=0.05,

    lr_scheduler_type="cosine"
)
```

Warmup is especially useful because early in training:

```text
Optimizer state is new
Gradients may be unstable
Model starts adapting
```

Gradually increasing the learning rate can make training more stable.

---

# 17. How to detect a bad learning rate during training

Log the loss:

```python
for epoch in range(epochs):

    optimizer.zero_grad()

    output = model(inputs)

    loss = loss_function(
        output,
        labels
    )

    loss.backward()

    optimizer.step()

    print(
        f"Loss: {loss.item()}"
    )
```

### Too high

```text
2.1
1.9
2.5
8.4
100
NaN
```

Action:

```text
Reduce LR
Check gradients
Use gradient clipping
Use warmup
Restart from a clean checkpoint
```

---

### Too low

```text
2.1000
2.0999
2.0998
2.0997
2.0996
```

Action:

```text
Increase LR
Train longer
Check whether gradients exist
Check dataset quality
```

---

### Good

```text
2.5
2.1
1.8
1.5
1.3
1.1
```

Validation:

```text
2.6
2.2
1.9
1.6
1.4
```

Stable improvement.

---

# 18. Add gradient clipping for protection

Gradient clipping does not fix a bad learning rate, but it can help prevent extremely large gradients from destabilizing training.

PyTorch:

```python
for batch in dataloader:

    optimizer.zero_grad()

    outputs = model(
        **batch
    )

    loss = outputs.loss

    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=1.0
    )

    optimizer.step()
```

In Hugging Face:

```python
training_args = TrainingArguments(

    output_dir="./model",

    learning_rate=5e-5,

    max_grad_norm=1.0,

    warmup_ratio=0.05
)
```

But remember:

> **Gradient clipping can control gradient spikes; it cannot make an excessively high learning rate fundamentally correct.**

---

# 19. Interview answer

### What happens if the learning rate is too high?

> **If the learning rate is too high, parameter updates become too large. The optimizer can overshoot the minimum, causing the loss to oscillate or diverge. In severe cases, gradients and weights become numerically unstable, producing `inf` or `NaN`. During LLM fine-tuning, an excessively high learning rate can cause rapid behavioral degradation or over-specialization. I monitor training and validation loss, gradient norms, and evaluation metrics, and use LR warmup and gradient clipping as stability measures.**

### What happens if the learning rate is too low?

> **If the learning rate is too low, parameter updates are extremely small. Training converges very slowly and may not reach a good solution within the available training budget. This wastes GPU compute and can lead to apparent underfitting because the model has not adapted sufficiently. I would detect this through very slow loss reduction and weak improvements in validation and task metrics.**

---

# Final mental model

```text
TOO LOW LR

Tiny updates

W
↓
W + 0.000001
↓
W + 0.000002

Slow learning
```

```text
GOOD LR

Controlled updates

W
↓
W₁
↓
W₂
↓
W₃

Stable convergence
```

```text
TOO HIGH LR

Huge updates

W
↓
W₁ ───────────────►
      overshoot
◄──────────────── W₂
       overshoot
              ↓
       instability / divergence
```

## One-line answer to remember

> **A high learning rate makes the model learn too aggressively and can cause instability or divergence; a low learning rate makes the model learn too slowly and may prevent it from converging within the training budget.**
