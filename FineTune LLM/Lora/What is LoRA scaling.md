# What is LoRA Scaling?

**LoRA scaling controls the strength of the LoRA update before it is added to the original model output.**

The common LoRA formula is:

[
\boxed{
W' = W_0 + \frac{\alpha}{r}BA
}
]

Where:

* (W_0) = original frozen pre-trained weight
* (A) and (B) = trainable LoRA matrices
* (r) = LoRA rank
* (\alpha) = LoRA alpha
* (\frac{\alpha}{r}) = **LoRA scaling factor**

So:

[
\boxed{\text{scaling} = \frac{\alpha}{r}}
]

---

# 1. Why do we need scaling?

Without scaling, LoRA would be:

[
W' = W_0 + BA
]

But the magnitude of (BA) can vary when you change the rank.

For example:

```text
Rank r = 4

ΔW = B₄A₄
```

versus:

```text
Rank r = 64

ΔW = B₆₄A₆₄
```

These updates can behave at different scales.

So LoRA uses:

[
\frac{\alpha}{r}
]

to control the effective strength of the adaptation:

[
W' = W_0 + \underbrace{\frac{\alpha}{r}}_{\text{scaling}}BA
]

---

# 2. Simple example

Suppose:

```python
r = 8
alpha = 16
```

Then:

[
scaling = \frac{16}{8} = 2
]

So:

[
W' = W_0 + 2BA
]

The LoRA update is multiplied by `2`.

---

Another example:

```python
r = 8
alpha = 8
```

Then:

[
scaling = \frac{8}{8} = 1
]

Therefore:

[
W' = W_0 + BA
]

No additional amplification.

---

Another example:

```python
r = 16
alpha = 8
```

Then:

[
scaling = \frac{8}{16} = 0.5
]

Therefore:

[
W' = W_0 + 0.5BA
]

The LoRA update is reduced.

---

# 3. Visual intuition

Imagine the base model output is:

```text
Base model output = 100
```

Suppose LoRA produces:

```text
LoRA update = 10
```

### Scaling = 1

```text
Final = 100 + 1 × 10
      = 110
```

### Scaling = 2

```text
Final = 100 + 2 × 10
      = 120
```

### Scaling = 0.5

```text
Final = 100 + 0.5 × 10
      = 105
```

Therefore:

```text
Scaling ↑
    ↓
LoRA influence ↑
```

and:

```text
Scaling ↓
    ↓
LoRA influence ↓
```

---

# 4. LoRA scaling in code

Here is a simplified implementation.

```python
import torch
import torch.nn as nn


class LoRALinear(nn.Module):

    def __init__(
        self,
        in_features,
        out_features,
        r=8,
        alpha=16
    ):
        super().__init__()

        # -------------------------
        # Base pretrained layer
        # -------------------------

        self.base = nn.Linear(
            in_features,
            out_features,
            bias=False
        )

        # Freeze original weights
        for param in self.base.parameters():
            param.requires_grad = False

        # -------------------------
        # LoRA configuration
        # -------------------------

        self.r = r
        self.alpha = alpha

        # LoRA scaling factor
        self.scaling = alpha / r

        # -------------------------
        # LoRA A and B matrices
        # -------------------------

        # A: input → low rank
        self.A = nn.Linear(
            in_features,
            r,
            bias=False
        )

        # B: low rank → output
        self.B = nn.Linear(
            r,
            out_features,
            bias=False
        )

        # Initialize B to zero
        # So initial LoRA update is zero
        nn.init.zeros_(self.B.weight)


    def forward(self, x):

        # -------------------------
        # Original model output
        # -------------------------

        base_output = self.base(x)

        # -------------------------
        # LoRA update
        # -------------------------

        lora_output = self.B(
            self.A(x)
        )

        # -------------------------
        # Apply scaling
        # -------------------------

        return (
            base_output
            +
            self.scaling * lora_output
        )
```

The important line is:

```python
self.scaling = alpha / r
```

And during the forward pass:

```python
output = (
    base_output
    +
    self.scaling * lora_output
)
```

---

# 5. Step-by-step example

Create:

```python
layer = LoRALinear(
    in_features=4,
    out_features=2,
    r=2,
    alpha=4
)
```

The scaling is:

```python
print(layer.scaling)
```

Output:

```text
2.0
```

Because:

[
\frac{4}{2} = 2
]

The forward pass effectively does:

```text
final output
=
base output
+
2 × LoRA output
```

---

# 6. Manual PyTorch example

Let's see scaling numerically.

```python
import torch

# Base output
base_output = torch.tensor([
    [10.0, 20.0]
])

# LoRA output
lora_output = torch.tensor([
    [2.0, 3.0]
])
```

### Scaling = 1

```python
scaling = 1

final = (
    base_output
    +
    scaling * lora_output
)

print(final)
```

Output:

```text
tensor([[12., 23.]])
```

---

### Scaling = 2

```python
scaling = 2

final = (
    base_output
    +
    scaling * lora_output
)

print(final)
```

Output:

```text
tensor([[14., 26.]])
```

---

### Scaling = 0.5

```python
scaling = 0.5

final = (
    base_output
    +
    scaling * lora_output
)

print(final)
```

Output:

```text
tensor([[11.0, 21.5]])
```

This clearly shows what LoRA scaling does.

---

# 7. Why divide by rank?

This is the key concept.

Recall:

[
\Delta W = BA
]

If we increase the rank:

```text
r = 4 → 4 low-rank components
r = 64 → 64 low-rank components
```

Without normalization, increasing rank can change the magnitude/behavior of the aggregate LoRA update.

Therefore the original LoRA formulation uses:

[
\frac{\alpha}{r}
]

to make the effective update scale more controllable across different ranks.

So instead of:

[
BA
]

we use:

[
\boxed{
\frac{\alpha}{r}BA
}
]

---

# 8. Relationship between `r` and `alpha`

Suppose:

### Configuration A

```python
r = 8
alpha = 16
```

Scaling:

[
16 / 8 = 2
]

---

### Configuration B

```python
r = 16
alpha = 32
```

Scaling:

[
32 / 16 = 2
]

The scaling factor stays the same:

```text
Configuration A → scaling = 2
Configuration B → scaling = 2
```

But configuration B has a higher-rank adapter:

```text
r = 16
```

So:

```text
Rank
↓
Adaptation capacity

Alpha / Rank
↓
Initial scaling coefficient for the LoRA branch
```

These are related but control different aspects.

---

# 9. Important distinction: scaling is not exactly "model strength"

A common oversimplification is:

> Higher alpha always means a stronger or better model.

Not necessarily.

During training, the optimizer learns the values of (A) and (B). The final behavior depends on:

* `r`
* `alpha`
* learning rate
* initialization
* dataset
* optimizer
* training steps
* target modules

So a better statement is:

> **LoRA scaling is a multiplicative coefficient that changes the contribution of the learned adapter update in the forward pass.**

---

# 10. LoRA scaling in PEFT

With the Hugging Face PEFT library:

```python
from peft import LoraConfig, TaskType

config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,

    target_modules=[
        "q_proj",
        "v_proj"
    ],

    task_type=TaskType.CAUSAL_LM
)
```

Conceptually:

```text
r = 8

alpha = 16

scaling = alpha / r

        16
       ────
         8

         = 2
```

The LoRA update is conceptually:

[
Output =
BaseOutput
+
2 \times LoRAUpdate
]

---

# 11. Training flow with scaling

```text
                    Frozen Base Model
Input ───────────────────────► W₀
   │                           │
   │                           ▼
   │                       Base Output
   │
   │
   └────► A ─────► B ─────► × (alpha / r)
                                    │
                                    ▼
                               LoRA Update
                                    │
                                    ▼
                        Base Output + LoRA Update
                                    │
                                    ▼
                                  Output
```

The base model remains frozen.

Only the LoRA path is trained:

```text
W₀      ❄️ Frozen

A       ✓ Trainable

B       ✓ Trainable

alpha   Hyperparameter

r       Hyperparameter
```

---

# 12. A small experiment with different scaling

```python
import torch

base = torch.tensor([10.0])

lora_update = torch.tensor([5.0])


configs = [
    {"r": 8, "alpha": 8},
    {"r": 8, "alpha": 16},
    {"r": 8, "alpha": 32}
]


for config in configs:

    r = config["r"]
    alpha = config["alpha"]

    scaling = alpha / r

    output = (
        base
        +
        scaling * lora_update
    )

    print(
        f"r={r}, "
        f"alpha={alpha}, "
        f"scaling={scaling}, "
        f"output={output.item()}"
    )
```

Conceptual output:

```text
r=8, alpha=8, scaling=1.0, output=15.0

r=8, alpha=16, scaling=2.0, output=20.0

r=8, alpha=32, scaling=4.0, output=30.0
```

---

# 13. Scaling vs Learning Rate

These are different.

### Learning rate

Controls:

> How much do the trainable weights change during optimization?

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=2e-4
)
```

### LoRA scaling

Controls:

> How is the LoRA update scaled when applied in the forward computation?

```python
scaling = alpha / r
```

So:

```text
Learning rate
      ↓
How A and B are updated

LoRA scaling
      ↓
How the learned A×B update contributes to output
```

---

# 14. Interview answer

> **LoRA scaling is the factor applied to the low-rank weight update before adding it to the frozen base weights. In the standard LoRA formulation, the update is scaled by alpha divided by rank: (W' = W_0 + (\alpha/r)BA). The rank controls the capacity of the low-rank adapter, while alpha controls the scaling coefficient of that update. Dividing by rank helps make the update scale more manageable when different ranks are used. In practice, I tune rank and alpha together and evaluate the configuration on a validation set.**

## One-line memory trick

```text
r
= how much LoRA can learn

alpha
= scaling hyperparameter

alpha / r
= LoRA scaling factor
```

The core formula to remember is:

[
\boxed{
W' = W_0 + \frac{\alpha}{r}BA
}
]
