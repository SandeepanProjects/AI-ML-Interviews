# Does QLoRA modify the quantized base model?

**No — not during normal QLoRA training.**

The key idea is:

> **QLoRA keeps the quantized base model frozen and trains only the LoRA adapter weights.**

The base model is used in the forward/backward computation, but its parameters are **not updated by the optimizer**.

---

# 1. What happens in QLoRA?

Suppose the original layer has weights:

[
W_0
]

QLoRA stores the base weights in 4-bit form:

[
Q(W_0)
]

and adds a LoRA update:

[
\Delta W = BA
]

The effective weight used during computation is conceptually:

[
W_{\text{effective}}
====================

W_0 + \frac{\alpha}{r}BA
]

where:

* (W_0) = frozen pretrained weights
* (Q(W_0)) = 4-bit representation of those weights
* (A,B) = trainable LoRA matrices
* (r) = LoRA rank
* (\alpha) = LoRA scaling factor

So:

```text
                  QLoRA

              4-bit Base Model
                    │
                    │ frozen ❄️
                    │
                    ▼
                  W₀
                    │
                    │
Input ──────────────┼──────────────►
                    │
                    │
              LoRA Adapter
                 A → B
                    │
                    │ trainable ✓
                    ▼
                 ΔW = BA
                    │
                    ▼
             Combined Output
```

---

# 2. What does "frozen" actually mean?

Suppose we inspect a base-model parameter:

```python
for name, param in model.named_parameters():
    print(
        name,
        param.requires_grad
    )
```

For the base model, you will generally see:

```text
model.layers.0.self_attn.q_proj.weight    False
model.layers.0.self_attn.k_proj.weight    False
model.layers.0.self_attn.v_proj.weight    False
...
```

For LoRA parameters:

```text
...q_proj.lora_A.default.weight    True
...q_proj.lora_B.default.weight    True
```

So:

```text
Base model
requires_grad = False
        ❄️

LoRA
requires_grad = True
        ✓
```

---

# 3. Let's see it with actual code

Install:

```bash
pip install transformers peft bitsandbytes accelerate
```

Then:

```python
import torch

from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

from peft import (
    LoraConfig,
    prepare_model_for_kbit_training,
    get_peft_model,
    TaskType,
)
```

---

# 4. Load the model in 4-bit

```python
MODEL_NAME = "your-model"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
)
```

At this point:

```text
Base model
    ↓
4-bit quantized
```

But we haven't added LoRA yet.

---

# 5. Prepare for k-bit training

```python
model = prepare_model_for_kbit_training(
    model
)
```

This prepares the quantized model for parameter-efficient training.

Importantly, this does **not** mean:

```text
"convert the entire model into trainable 4-bit weights"
```

Instead, the base model remains frozen while trainable adapters can be attached.

---

# 6. Add LoRA

```python
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,

    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ],

    bias="none",

    task_type=TaskType.CAUSAL_LM,
)
```

Apply:

```python
model = get_peft_model(
    model,
    lora_config
)
```

Now the model looks like:

```text
                 Model

        ┌─────────────────────┐
        │ 4-bit Base Model    │
        │                     │
        │ Frozen ❄️           │
        └──────────┬──────────┘
                   │
                   │
             LoRA Adapter
                   │
             ┌─────┴─────┐
             │           │
             A           B
             │           │
             └─────┬─────┘
                   │
              Trainable ✓
```

---

# 7. Check which parameters are trainable

Run:

```python
for name, param in model.named_parameters():

    if param.requires_grad:
        print("TRAINABLE:", name)
```

You'll get something similar to:

```text
TRAINABLE:
base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight

TRAINABLE:
base_model.model.model.layers.0.self_attn.q_proj.lora_B.default.weight

TRAINABLE:
base_model.model.model.layers.0.self_attn.v_proj.lora_A.default.weight

TRAINABLE:
base_model.model.model.layers.0.self_attn.v_proj.lora_B.default.weight
```

The enormous base model parameters are not listed.

---

# 8. Verify the base model is frozen

You can explicitly check:

```python
trainable = 0
frozen = 0

for name, param in model.named_parameters():

    if param.requires_grad:
        trainable += param.numel()
    else:
        frozen += param.numel()

print("Trainable:", trainable)
print("Frozen:", frozen)
```

For a large model you might conceptually get:

```text
Trainable:       10,000,000
Frozen:     70,000,000,000
```

The exact number depends on:

* model architecture
* LoRA rank
* target modules

---

# 9. What happens during backpropagation?

This is the important part.

Suppose:

```python
loss.backward()
```

The computation graph is approximately:

```text
Input
  │
  ▼
4-bit Base Model
  │
  │
  ├───────────────┐
  │               │
  ▼               ▼
Base path       LoRA path
Frozen          Trainable
  │               │
  └───────┬───────┘
          ▼
        Output
          │
          ▼
         Loss
          │
          ▼
    Backpropagation
          │
          ├───────────────► LoRA gradients ✓
          │
          └───────────────► Base weights NOT updated ❌
```

The gradient can still **flow through the computation involving the base model** to determine how the LoRA parameters should change.

But the base model's parameters are not optimized.

This distinction is extremely important.

---

# 10. A simple PyTorch example

Consider:

```python
import torch
import torch.nn as nn

base = nn.Linear(
    100,
    100,
    bias=False
)

adapter = nn.Linear(
    100,
    100,
    bias=False
)
```

Freeze base:

```python
for p in base.parameters():
    p.requires_grad = False
```

Train adapter:

```python
for p in adapter.parameters():
    p.requires_grad = True
```

Forward:

```python
x = torch.randn(4, 100)

base_output = base(x)

adapter_output = adapter(x)

output = base_output + adapter_output
```

Calculate loss:

```python
target = torch.randn(4, 100)

loss = ((output - target) ** 2).mean()

loss.backward()
```

Now:

```python
print(base.weight.grad)
```

Because the base parameter is frozen:

```text
None
```

while:

```python
print(adapter.weight.grad is None)
```

gives:

```text
False
```

So:

```text
Base:
gradient update ❌

Adapter:
gradient update ✓
```

---

# 11. What does the optimizer contain?

This is another important point.

Suppose:

```python
optimizer = torch.optim.AdamW(
    filter(
        lambda p: p.requires_grad,
        model.parameters()
    ),
    lr=2e-4
)
```

The optimizer receives only trainable parameters.

Therefore:

```text
Optimizer

LoRA A ✓
LoRA B ✓

Base model ❌
```

During:

```python
optimizer.step()
```

only the LoRA parameters are changed.

---

# 12. Does the 4-bit base model ever get dequantized?

**Yes, during computation, values may be dequantized into a higher-precision representation for matrix multiplication.**

This does **not** mean the base model is being fine-tuned.

Conceptually:

```text
Stored:

4-bit weights
     │
     ▼
Dequantize for computation
     │
     ▼
BF16/FP16 computation
     │
     ▼
Forward pass
```

The important distinction is:

```text
Storage precision
        ≠
Computation precision
        ≠
Training/update precision
```

For example:

```text
Base weights:
stored → 4-bit

Computation:
BF16

LoRA weights:
trainable → typically BF16/FP16

Base weights:
updated → NO
```

---

# 13. Does QLoRA change the quantized weights permanently?

### During training:

**No.**

The base model remains frozen.

```text
Before training:

4-bit Base
   ↓
Frozen


After training:

4-bit Base
   ↓
Still Frozen

LoRA
   ↓
Updated ✓
```

What changes is:

```text
LoRA A
LoRA B
```

---

# 14. What actually gets saved?

Typically:

```python
model.save_pretrained(
    "./my-qlora-adapter"
)
```

You primarily save the adapter:

```text
my-qlora-adapter/
│
├── adapter_config.json
├── adapter_model.safetensors
└── ...
```

You don't need to save a newly modified 70B base model.

At inference:

```text
Original Base Model
       +
QLoRA Adapter
       ↓
Fine-tuned model
```

---

# 15. Important distinction: "modify" can mean two things

This is where interviewers sometimes try to confuse candidates.

### Does QLoRA modify the model's behavior?

**Yes.**

Because:

[
W_{\text{effective}}
====================

W_0 + \Delta W
]

The LoRA adapter changes the effective computation.

### Does QLoRA modify the stored base weights?

**No.**

The base weights remain frozen.

So the best answer is:

> **QLoRA modifies the effective behavior of the model through LoRA adapters, but it does not update the quantized base-model weights during training.**

---

# 16. What happens if you remove the adapter?

Suppose you have:

```text
Base model
+
Financial-domain LoRA
```

Remove LoRA:

```text
Base model only
```

The model goes back to its original behavior.

This proves that the adaptation is stored in the adapter.

```text
Base model
    │
    ├── General-purpose behavior
    │
    +
    │
Financial LoRA
    │
    ▼
Financial-specialized behavior
```

You can even have multiple adapters:

```text
             4-bit Base
                 │
       ┌─────────┼─────────┐
       │         │         │
       ▼         ▼         ▼
   Finance    Medical    Coding
     LoRA       LoRA       LoRA
```

The same base model can be reused.

---

# 17. What about merging?

This is a separate operation.

During normal QLoRA training:

```text
4-bit Base
    ❄️
+
LoRA
    ✓
```

If you later merge the adapter:

[
W_{\text{merged}}
=================

W_0 + \Delta W
]

you create a model whose effective weights include the LoRA update.

But **merging is not what happens during normal QLoRA training**.

---

# 18. Interview answer

If an interviewer asks:

> **"Does QLoRA modify the quantized base model?"**

Answer:

> **No. In standard QLoRA training, the quantized base model remains frozen. The model's base weights are stored in 4-bit form, typically NF4, and are used for forward computation. LoRA adapters are added to selected layers and are the only trainable parameters. During backpropagation, gradients are used to update the LoRA matrices, while the base model's parameters are not updated. So QLoRA changes the model's effective behavior through the adapters, but it doesn't modify the stored quantized base weights.**

### Remember this:

```text
             QLoRA

       4-bit Base Model
             ❄️
          FROZEN
             │
             │
             ▼
       + LoRA Adapter
             ✓
          TRAINED
             │
             ▼
      Fine-tuned behavior
```

**The base model is the foundation; the LoRA adapter is what learns.**
