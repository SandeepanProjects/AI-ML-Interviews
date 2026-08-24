# Difference Between LoRA Adapters and the Base Model

The easiest way to understand this is:

> **The base model contains the original pre-trained knowledge. A LoRA adapter contains a small set of learned changes that specialize the base model for a particular task or domain.**

---

# 1. High-level architecture

```text
                 Pre-trained Base Model
                 ┌─────────────────────┐
Input ──────────►│  LLM Weights (W₀)   │
                 │                     │
                 │  Usually Frozen     │
                 └──────────┬──────────┘
                            │
                            │
                 ┌──────────▼──────────┐
                 │    LoRA Adapter     │
                 │                     │
                 │  Low-rank updates   │
                 │      A and B        │
                 └──────────┬──────────┘
                            │
                            ▼
                      Specialized Output
```

Mathematically:

[
W_{\text{final}} =
W_0 + \Delta W
]

For LoRA:

[
\Delta W =
\frac{\alpha}{r}BA
]

Therefore:

[
\boxed{
W_{\text{final}}
================

W_0
+
\frac{\alpha}{r}BA
}
]

---

# 2. What is the base model?

The **base model** is the original pre-trained LLM.

Examples include models from organizations such as [Meta AI](https://ai.meta.com/?utm_source=chatgpt.com), [Mistral AI](https://mistral.ai/?utm_source=chatgpt.com), or [Google DeepMind](https://deepmind.google/?utm_source=chatgpt.com).

Conceptually, the base model contains billions of parameters:

```text
Base Model

W₁
W₂
W₃
W₄
...
W billions
```

For example:

```text
Base model
7 billion parameters
```

Those weights were learned during:

```text
Pre-training
```

The model may already know:

* language
* grammar
* reasoning patterns
* programming concepts
* general knowledge
* relationships between concepts

During LoRA fine-tuning:

```text
Base Model Weights

W₀

❄️ Frozen
```

The base weights normally do not change.

---

# 3. What is a LoRA adapter?

A **LoRA adapter** is a small set of additional trainable matrices attached to selected layers of the base model.

Suppose a base model has:

```text
q_proj

Wq = 4096 × 4096
```

Normally:

```text
Q = XWq
```

With LoRA:

[
Q =
XW_q
+
X
\left(
\frac{\alpha}{r}BA
\right)
]

The adapter adds:

```text
A
B
```

Example:

```text
Base:

Wq
4096 × 4096

16,777,216 parameters
```

LoRA with:

```text
r = 8
```

might add:

```text
A: 8 × 4096
B: 4096 × 8
```

Total:

[
65,536
]

trainable parameters.

So:

```text
Base model
──────────────────
Billions of parameters
Frozen


LoRA adapter
──────────────────
Small number of parameters
Trainable
```

---

# 4. Code example

Here is a simplified base layer:

```python
import torch
import torch.nn as nn

base_layer = nn.Linear(
    in_features=4096,
    out_features=4096
)
```

Normally:

```python
output = base_layer(x)
```

---

## Freeze the base model

```python
for parameter in base_layer.parameters():
    parameter.requires_grad = False
```

Now:

```text
Base model
❄️ Frozen
```

---

## Add a LoRA adapter

```python
class LoRAAdapter(nn.Module):

    def __init__(
        self,
        input_size,
        output_size,
        r=8,
        alpha=16
    ):
        super().__init__()

        self.scaling = alpha / r

        # Down projection
        self.A = nn.Linear(
            input_size,
            r,
            bias=False
        )

        # Up projection
        self.B = nn.Linear(
            r,
            output_size,
            bias=False
        )

        # Start with zero LoRA contribution
        nn.init.zeros_(self.B.weight)

    def forward(self, x):

        return (
            self.B(
                self.A(x)
            )
            * self.scaling
        )
```

---

## Combine base model + adapter

```python
class ModelWithLoRA(nn.Module):

    def __init__(
        self,
        input_size=4096,
        output_size=4096,
        r=8,
        alpha=16
    ):
        super().__init__()

        # Base model layer
        self.base = nn.Linear(
            input_size,
            output_size,
            bias=False
        )

        # Freeze base model
        for param in self.base.parameters():
            param.requires_grad = False

        # Attach adapter
        self.lora = LoRAAdapter(
            input_size,
            output_size,
            r,
            alpha
        )

    def forward(self, x):

        base_output = self.base(x)

        adapter_output = self.lora(x)

        return (
            base_output
            +
            adapter_output
        )
```

The important line is:

```python
return base_output + adapter_output
```

Conceptually:

[
\boxed{
Output =
BaseModel(X)
+
LoRAAdapter(X)
}
]

---

# 5. What is actually trained?

Let's inspect it:

```python
model = ModelWithLoRA()

for name, param in model.named_parameters():

    print(
        name,
        "Trainable:"
        if param.requires_grad
        else "Frozen"
    )
```

Conceptually:

```text
base.weight    Frozen ❄️

lora.A.weight  Trainable ✓
lora.B.weight  Trainable ✓
```

This is the key difference.

---

# 6. Base model vs LoRA adapter

| Feature              | Base Model                         | LoRA Adapter                   |
| -------------------- | ---------------------------------- | ------------------------------ |
| Purpose              | General knowledge and capabilities | Task/domain specialization     |
| Size                 | Usually GBs                        | Usually much smaller           |
| Parameters           | Billions                           | Relatively few                 |
| During LoRA training | Frozen                             | Updated                        |
| Contains             | Original learned weights           | Low-rank updates               |
| Storage              | Large                              | Small                          |
| Can be reused        | Yes                                | Requires compatible base model |
| Per-task copy        | Expensive                          | Cheap                          |
| Multiple tasks       | One base model                     | Multiple adapters              |

---

# 7. Why are adapters useful?

Suppose your company has one base model:

```text
Llama Base Model
```

You need specialized models for:

```text
Customer Support
Financial Analysis
Legal Assistant
Code Generation
```

Without LoRA:

```text
Base Model
   │
   ├── Full fine-tuned model
   │      Customer Support
   │
   ├── Full fine-tuned model
   │      Finance
   │
   ├── Full fine-tuned model
   │      Legal
   │
   └── Full fine-tuned model
          Coding
```

You potentially store multiple complete copies.

With LoRA:

```text
                 Base Model
                 70 GB
                    │
        ┌───────────┼───────────┐
        │           │           │
        ▼           ▼           ▼

Customer      Finance       Legal
Adapter       Adapter       Adapter

200 MB        300 MB        250 MB
```

One large base model can potentially support many specialized adapters.

---

# 8. Loading the base model and adapter

Using Hugging Face PEFT conceptually:

```python
from transformers import AutoModelForCausalLM
from peft import PeftModel
```

Load the base model:

```python
base_model = AutoModelForCausalLM.from_pretrained(
    "your-base-model"
)
```

Then load the adapter:

```python
model = PeftModel.from_pretrained(
    base_model,
    "./finance-lora-adapter"
)
```

Now:

```text
Base Model
+
Finance LoRA Adapter
=
Finance-specialized LLM
```

---

# 9. Multiple adapters

This is one of the biggest advantages.

```text
                     Base Model
                          │
          ┌───────────────┼───────────────┐
          │               │               │
          ▼               ▼               ▼

      Adapter A       Adapter B       Adapter C

       Finance         Legal           Medical
```

You can conceptually load different adapters for different tasks.

For example:

```python
model.load_adapter(
    "./finance_adapter",
    adapter_name="finance"
)

model.load_adapter(
    "./legal_adapter",
    adapter_name="legal"
)
```

Then activate one:

```python
model.set_adapter(
    "finance"
)
```

Conceptually:

```text
Request:
"Analyze this company's revenue"

        │
        ▼

Base Model
        +
Finance Adapter
        │
        ▼

Financial Answer
```

Another request:

```text
"Analyze this contract"

        │
        ▼

Base Model
        +
Legal Adapter
        │
        ▼

Legal-focused Answer
```

---

# 10. What does a LoRA adapter file contain?

Conceptually, an adapter contains:

```text
adapter/

├── adapter_model.safetensors
│
└── adapter_config.json
```

### `adapter_model.safetensors`

Contains the learned LoRA weights:

```text
A matrices
B matrices
```

for selected model layers.

For example:

```text
Layer 1
├── q_proj.lora_A
└── q_proj.lora_B

Layer 2
├── q_proj.lora_A
└── q_proj.lora_B

...
```

### `adapter_config.json`

Contains information such as:

```json
{
  "r": 8,
  "lora_alpha": 16,
  "lora_dropout": 0.05,
  "target_modules": [
    "q_proj",
    "v_proj"
  ]
}
```

The exact files and metadata depend on the PEFT/version and serialization setup.

---

# 11. Important: An adapter is not a complete model

A common misunderstanding is:

> "I fine-tuned a model using LoRA. I have the adapter, so I can run it alone."

Usually, **no**.

You need:

```text
Base Model
+
Compatible LoRA Adapter
```

The adapter only contains the learned modifications.

Mathematically:

[
W_{adapter}
\neq
W_{model}
]

The adapter provides:

[
\Delta W
]

while the base model contains:

[
W_0
]

Together:

[
W_{final}
=========

W_0 + \Delta W
]

So:

```text
Base model alone
=
General model


LoRA adapter alone
=
Not a complete LLM


Base model + adapter
=
Specialized LLM
```

---

# 12. Can the adapter be merged?

Yes, often.

Before merging:

```text
Base Model
+
Adapter
```

At inference:

[
W_{effective}
=============

W_0
+
\frac{\alpha}{r}BA
]

You can merge the adapter:

```python
merged_model = model.merge_and_unload()
```

Conceptually:

```text
Before:

W₀
+
ΔW

After:

Wfinal
```

Now:

```text
Wfinal
```

contains the adaptation directly.

This can simplify some deployment setups, though it removes some of the flexibility of swapping adapters.

---

# 13. Real-world example

Imagine a company builds an enterprise AI platform.

They use:

```text
Base Model
```

for all customers.

Customer A:

```text
Insurance Company
```

Customer B:

```text
Financial Company
```

Customer C:

```text
Retail Company
```

Architecture:

```text
                    API Request
                         │
                         ▼
                 Identify Customer
                         │
          ┌──────────────┼──────────────┐
          │              │              │
          ▼              ▼              ▼

      Customer A     Customer B     Customer C
      Insurance       Finance         Retail
      Adapter         Adapter         Adapter
          │              │              │
          └──────────────┼──────────────┘
                         │
                         ▼
                     Base Model
```

Instead of maintaining:

```text
3 separate 70GB models
```

you can maintain:

```text
1 × base model
+
3 × small adapters
```

This is especially useful for:

* multi-tenant AI systems
* domain-specific assistants
* customer-specific behavior
* different output formats
* multiple languages or styles

---

# Best interview answer

> **The base model is the original pre-trained LLM containing billions of general-purpose parameters and learned knowledge. During LoRA fine-tuning, these base weights are typically frozen. A LoRA adapter is a small set of trainable low-rank matrices attached to selected layers, which learns task-specific weight updates. At inference, the effective model combines the frozen base weights with the LoRA update, commonly represented as (W' = W_0 + (\alpha/r)BA). The main advantage is that one large base model can be reused with multiple small adapters for different tasks or customers.**

## Simple memory trick

```text
Base Model
=
The original brain


LoRA Adapter
=
A small specialization layer


Base + Adapter
=
Specialized model
```
