# What is Adapter-Based Fine-Tuning?

**Adapter-based fine-tuning** is a **parameter-efficient fine-tuning (PEFT)** technique where:

1. You start with a pre-trained model.
2. You **freeze most or all original model weights**.
3. You insert small trainable neural network modules called **adapters** into the model.
4. You train only the adapters (and sometimes a few other parameters).

The original model remains unchanged.

```text
                    PRETRAINED MODEL

          ┌───────────────────────────────┐
Input ──► │ Transformer Layer 1  ❄️       │
          ├───────────────────────────────┤
          │       Adapter ✓               │
          ├───────────────────────────────┤
          │ Transformer Layer 2  ❄️       │
          ├───────────────────────────────┤
          │       Adapter ✓               │
          ├───────────────────────────────┤
          │ Transformer Layer 3  ❄️       │
          ├───────────────────────────────┤
          │       Adapter ✓               │
          └───────────────────────────────┘

❄️ = frozen
✓ = trainable
```

---

# 1. Why do we need adapters?

Suppose you have a 7B parameter LLM.

With full fine-tuning:

```text
7B parameters
      ↓
All parameters trainable
      ↓
Gradients for 7B
      +
Optimizer states for 7B
      ↓
High GPU memory and cost
```

With adapter-based fine-tuning:

```text
7B Base Model
     ↓
Frozen ❄️

Small Adapter Modules
     ↓
Trainable ✓
```

Instead of creating multiple copies:

```text
Finance model → 7GB+
Legal model   → 7GB+
Support model → 7GB+
```

You can use:

```text
                Base LLM
                   │
          ┌────────┼────────┐
          ▼        ▼        ▼
       Finance    Legal   Support
       Adapter    Adapter Adapter
```

So the same base model can support multiple specialized tasks.

---

# 2. What does an adapter actually look like?

A typical adapter is a small **bottleneck neural network**.

```text
Input hidden state

      h
      │
      ▼

Down Projection
Large dimension → Small dimension

      │

      ▼

   Activation

      │

      ▼

Up Projection
Small dimension → Large dimension

      │

      ▼

   Residual Add

      │

      ▼

   Output
```

Suppose the Transformer hidden size is:

```text
4096
```

The adapter bottleneck might be:

```text
64
```

So:

```text
4096
  ↓
 64
  ↓
4096
```

This is much smaller than training a full `4096 × 4096` layer.

---

# 3. Adapter mathematics

Suppose the output of a Transformer layer is:

[
h
]

An adapter performs:

[
z = W_{down}h
]

Then an activation:

[
z = f(z)
]

Then projects back:

[
a = W_{up}z
]

Finally, use a residual connection:

[
h_{output} = h + a
]

Or, combined:

[
h_{output}
==========

h
+
W_{up}
f(W_{down}h)
]

Where:

```text
h              = Transformer hidden state
Wdown          = down-projection matrix
Wup            = up-projection matrix
f              = activation function
```

The trainable parameters are mainly:

```text
Wdown ✓
Wup   ✓
```

The base Transformer weights remain:

```text
Frozen ❄️
```

---

# 4. Adapter vs LoRA

This is a common interview question.

## Adapter

Adds an actual small neural network layer:

```text
Transformer Output
        │
        ▼
    Down Projection
        │
        ▼
    Activation
        │
        ▼
    Up Projection
        │
        ▼
    Add Residual
```

## LoRA

Modifies a linear layer's effective output:

```text
Original:

Y = XW

LoRA:

Y = XW + XBA
```

Comparison:

| Feature                | Adapter                              | LoRA                               |
| ---------------------- | ------------------------------------ | ---------------------------------- |
| Adds new modules       | Yes                                  | Yes, low-rank branches             |
| Main location          | Between/around Transformer sublayers | Inside selected linear projections |
| Base model             | Frozen                               | Frozen                             |
| Trainable parameters   | Small                                | Usually very small                 |
| Extra inference layers | Yes                                  | Can often be merged into weights   |
| PEFT technique         | Yes                                  | Yes                                |

---

# 5. Adapter implementation from scratch

Let's build one in PyTorch.

```python
import torch
import torch.nn as nn
```

## Step 1: Create the adapter

```python
class Adapter(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        bottleneck_size: int
    ):
        super().__init__()

        # 4096 → 64
        self.down_project = nn.Linear(
            hidden_size,
            bottleneck_size
        )

        # Non-linear transformation
        self.activation = nn.ReLU()

        # 64 → 4096
        self.up_project = nn.Linear(
            bottleneck_size,
            hidden_size
        )

    def forward(self, hidden_states):

        # Save original representation
        residual = hidden_states

        # Down projection
        x = self.down_project(hidden_states)

        # Activation
        x = self.activation(x)

        # Up projection
        x = self.up_project(x)

        # Residual connection
        return residual + x
```

---

# 6. Test the adapter

```python
adapter = Adapter(
    hidden_size=768,
    bottleneck_size=64
)

x = torch.randn(
    2,      # batch size
    10,     # sequence length
    768     # hidden size
)

output = adapter(x)

print(output.shape)
```

Output:

```text
torch.Size([2, 10, 768])
```

The adapter preserves the hidden size.

```text
Input:

[2, 10, 768]

      ↓

Adapter

768 → 64 → 768

      ↓

Output:

[2, 10, 768]
```

---

# 7. How many parameters does an adapter have?

Let's calculate.

```python
def count_parameters(model):

    return sum(
        parameter.numel()
        for parameter in model.parameters()
    )


print(
    count_parameters(adapter)
)
```

For:

```text
hidden_size = 768
bottleneck = 64
```

Approximately:

```text
Down projection:
768 × 64

Up projection:
64 × 768
```

Total is roughly:

```text
98,304
```

plus biases.

Compare that with a full model containing hundreds of millions or billions of parameters.

---

# 8. Simulating a Transformer + Adapter

Let's create a simplified Transformer block.

```python
class TransformerBlockWithAdapter(nn.Module):

    def __init__(
        self,
        hidden_size=768,
        bottleneck_size=64
    ):
        super().__init__()

        # Simulated Transformer layer
        self.transformer_layer = nn.Sequential(
            nn.Linear(
                hidden_size,
                hidden_size
            ),
            nn.ReLU(),
            nn.Linear(
                hidden_size,
                hidden_size
            )
        )

        # Adapter
        self.adapter = Adapter(
            hidden_size=hidden_size,
            bottleneck_size=bottleneck_size
        )

    def forward(self, x):

        # Base transformer
        x = self.transformer_layer(x)

        # Adapter
        x = self.adapter(x)

        return x
```

---

# 9. Freeze the base model

This is the critical step.

```python
model = TransformerBlockWithAdapter(
    hidden_size=768,
    bottleneck_size=64
)
```

Freeze everything except the adapter:

```python
for name, parameter in model.named_parameters():

    if "adapter" not in name:
        parameter.requires_grad = False
```

Now inspect:

```python
for name, parameter in model.named_parameters():

    print(
        name,
        parameter.requires_grad
    )
```

Expected:

```text
transformer_layer.0.weight   False
transformer_layer.0.bias     False
transformer_layer.2.weight   False
transformer_layer.2.bias     False

adapter.down_project.weight  True
adapter.down_project.bias    True
adapter.up_project.weight    True
adapter.up_project.bias      True
```

Exactly what we want:

```text
Base Transformer → Frozen ❄️
Adapter          → Trainable ✓
```

---

# 10. Train only adapter parameters

```python
import torch.optim as optim
```

Create optimizer:

```python
optimizer = optim.AdamW(
    filter(
        lambda p: p.requires_grad,
        model.parameters()
    ),
    lr=1e-3
)
```

Check:

```python
trainable_params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

total_params = sum(
    p.numel()
    for p in model.parameters()
)

print("Total:", total_params)
print("Trainable:", trainable_params)
```

Training loop:

```python
loss_fn = nn.MSELoss()

model.train()

for epoch in range(10):

    optimizer.zero_grad()

    # Dummy input
    x = torch.randn(
        4,
        10,
        768
    )

    # Dummy target
    target = torch.randn(
        4,
        10,
        768
    )

    # Forward pass
    output = model(x)

    # Loss
    loss = loss_fn(
        output,
        target
    )

    # Backpropagation
    loss.backward()

    # Update adapter only
    optimizer.step()

    print(
        f"Epoch {epoch}, "
        f"Loss: {loss.item():.4f}"
    )
```

Internally:

```text
Loss
 │
 ▼
Backward
 │
 ├── Base Transformer
 │       Frozen ❄️
 │       No weight update
 │
 └── Adapter
         Gradients ✓
         Updated ✓
```

---

# 11. Real LLM adapter fine-tuning

In production, you usually don't manually rewrite every Transformer layer.

Libraries such as Hugging Face's PEFT ecosystem provide parameter-efficient methods for supported architectures.

Conceptually, the workflow is:

```python
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

MODEL_NAME = "some-base-model"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME
)
```

Then you:

```text
Load model
    ↓
Freeze base model
    ↓
Insert/configure adapters
    ↓
Train adapter parameters
    ↓
Save adapters
```

The exact API and supported adapter method depend on the PEFT library version and model architecture.

---

# 12. A complete conceptual adapter training example

Below is a complete PyTorch example using a small custom model so you can clearly understand what is happening.

```python
import torch
import torch.nn as nn
from torch.optim import AdamW


# ==========================================
# 1. ADAPTER MODULE
# ==========================================

class Adapter(nn.Module):

    def __init__(
        self,
        hidden_size,
        bottleneck_size
    ):
        super().__init__()

        self.down_project = nn.Linear(
            hidden_size,
            bottleneck_size
        )

        self.activation = nn.GELU()

        self.up_project = nn.Linear(
            bottleneck_size,
            hidden_size
        )

    def forward(self, x):

        residual = x

        x = self.down_project(x)

        x = self.activation(x)

        x = self.up_project(x)

        return residual + x


# ==========================================
# 2. BASE TRANSFORMER-LIKE BLOCK
# ==========================================

class BaseBlock(nn.Module):

    def __init__(self, hidden_size):

        super().__init__()

        self.linear1 = nn.Linear(
            hidden_size,
            hidden_size
        )

        self.activation = nn.GELU()

        self.linear2 = nn.Linear(
            hidden_size,
            hidden_size
        )

    def forward(self, x):

        x = self.linear1(x)

        x = self.activation(x)

        x = self.linear2(x)

        return x


# ==========================================
# 3. MODEL WITH BASE + ADAPTER
# ==========================================

class AdapterModel(nn.Module):

    def __init__(
        self,
        hidden_size=128,
        bottleneck_size=16
    ):

        super().__init__()

        # Pretend this is a pretrained model
        self.base_model = BaseBlock(
            hidden_size
        )

        # Task-specific adapter
        self.adapter = Adapter(
            hidden_size,
            bottleneck_size
        )

        # Classification head
        self.classifier = nn.Linear(
            hidden_size,
            3
        )

    def forward(self, x):

        # Frozen base model
        x = self.base_model(x)

        # Trainable adapter
        x = self.adapter(x)

        # Use final token/representation
        x = x.mean(dim=1)

        return self.classifier(x)


# ==========================================
# 4. CREATE MODEL
# ==========================================

model = AdapterModel()


# ==========================================
# 5. FREEZE BASE MODEL
# ==========================================

for parameter in model.base_model.parameters():

    parameter.requires_grad = False


# ==========================================
# 6. COUNT PARAMETERS
# ==========================================

total = sum(
    p.numel()
    for p in model.parameters()
)

trainable = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print(f"Total parameters: {total:,}")
print(f"Trainable parameters: {trainable:,}")


# ==========================================
# 7. OPTIMIZER
# ==========================================

optimizer = AdamW(
    filter(
        lambda p: p.requires_grad,
        model.parameters()
    ),
    lr=1e-3
)

loss_fn = nn.CrossEntropyLoss()


# ==========================================
# 8. TRAINING LOOP
# ==========================================

model.train()

for epoch in range(5):

    # Fake batch
    x = torch.randn(
        8,    # batch size
        20,   # sequence length
        128   # hidden size
    )

    # Fake labels
    labels = torch.randint(
        0,
        3,
        (8,)
    )


    # Forward pass
    logits = model(x)


    # Calculate loss
    loss = loss_fn(
        logits,
        labels
    )


    # Clear old gradients
    optimizer.zero_grad()


    # Backpropagation
    loss.backward()


    # Update only trainable parameters
    optimizer.step()


    print(
        f"Epoch: {epoch + 1}, "
        f"Loss: {loss.item():.4f}"
    )
```

---

# 13. What happens during backpropagation?

Consider:

```text
                    INPUT
                       │
                       ▼
               Base Transformer
                       │
                       │ Frozen ❄️
                       ▼
                   Adapter
                       │
                       │ Trainable ✓
                       ▼
                Classification Head
                       │
                       ▼
                     Loss
```

During:

```python
loss.backward()
```

PyTorch computes gradients through the computational graph.

But:

```python
base_parameter.requires_grad = False
```

means those base parameters are not updated by the optimizer.

Then:

```python
optimizer.step()
```

updates:

```text
Adapter weights
       +
Classification head
```

but not:

```text
Base Transformer weights
```

---

# 14. Multiple adapters for multiple domains

This is one of the biggest advantages.

```text
                  Base LLM
                     │
       ┌─────────────┼──────────────┐
       ▼             ▼              ▼
 Finance Adapter  Legal Adapter  Support Adapter
```

A simple implementation:

```python
class MultiAdapterModel(nn.Module):

    def __init__(
        self,
        hidden_size=128
    ):

        super().__init__()

        self.base_model = BaseBlock(
            hidden_size
        )

        self.adapters = nn.ModuleDict({

            "finance": Adapter(
                hidden_size,
                bottleneck_size=16
            ),

            "legal": Adapter(
                hidden_size,
                bottleneck_size=16
            ),

            "support": Adapter(
                hidden_size,
                bottleneck_size=16
            )
        })


    def forward(
        self,
        x,
        domain
    ):

        # Shared frozen model
        x = self.base_model(x)

        # Select adapter
        x = self.adapters[domain](x)

        return x
```

Use it:

```python
model = MultiAdapterModel()

output = model(
    x,
    domain="finance"
)
```

This gives:

```text
One Base Model
      +
Many Small Specialized Adapters
```

---

# 15. Adapter vs Full Fine-Tuning vs LoRA vs QLoRA

| Feature                     | Full Fine-Tuning  | Adapter            | LoRA                   | QLoRA                                  |
| --------------------------- | ----------------- | ------------------ | ---------------------- | -------------------------------------- |
| Base model weights          | Updated           | Frozen             | Frozen                 | Quantized + frozen                     |
| New trainable components    | No                | Bottleneck modules | Low-rank matrices      | Low-rank matrices                      |
| Where adaptation happens    | Entire model      | Inserted modules   | Selected linear layers | Selected linear layers                 |
| Trainable parameters        | Very high         | Low                | Usually lower          | Usually lower                          |
| GPU memory                  | Highest           | Lower              | Lower                  | Usually lowest                         |
| Can have multiple tasks     | Expensive         | Excellent          | Excellent              | Excellent                              |
| Extra inference computation | No extra adapters | Yes                | Can often be merged    | Usually uses quantized base + adapters |

---

# 16. When should you use adapter-based fine-tuning?

Use adapters when:

### 1. Multiple domains need specialization

```text
One base model
+
Finance adapter
+
Legal adapter
+
HR adapter
```

### 2. Storage is important

Instead of storing:

```text
10 × full 7B models
```

store:

```text
1 × base model
+
10 × small adapters
```

### 3. You want easy rollback

```text
Adapter v1
   ↓
Bad performance
   ↓
Switch to Adapter v0
```

The base model is untouched.

### 4. You want tenant-specific models

For a multi-tenant SaaS:

```text
                  Shared Base LLM
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
       Tenant A       Tenant B       Tenant C
       Adapter        Adapter        Adapter
```

---

# 17. Adapter-based fine-tuning vs LoRA — important difference

A strong interview answer is:

> **Both are PEFT methods. Traditional adapter tuning inserts small bottleneck neural network modules into Transformer layers and trains those modules while freezing the base model. LoRA instead keeps the architecture largely unchanged and represents weight updates in selected linear layers using low-rank matrices. LoRA often has lower parameter overhead and can be merged into the base weights for inference, while traditional adapters introduce additional modules in the forward path.**

---

# 18. Interview answer

> **Adapter-based fine-tuning is a parameter-efficient fine-tuning approach where a pre-trained model is mostly frozen and small trainable adapter modules are inserted into or alongside its layers. A typical adapter uses a bottleneck architecture: it projects the hidden representation to a smaller dimension, applies a non-linearity, projects it back to the original dimension, and adds the result through a residual connection. During training, only the adapter parameters—and optionally task-specific heads—are updated. This reduces memory, training cost, and storage compared with full fine-tuning and allows one base model to support multiple domain-specific adapters.**

## Remember

```text
FULL FINE-TUNING
All model weights → Train

ADAPTER TUNING
Base model → Freeze
Small neural adapters → Train

LoRA
Base model → Freeze
Low-rank weight updates → Train

QLoRA
Quantized base model → Freeze
Low-rank adapters → Train
```

The easiest mental model is:

```text
Adapter = add small neural modules

LoRA = add low-rank weight updates

QLoRA = quantize base model + LoRA

Full fine-tuning = update everything
```
