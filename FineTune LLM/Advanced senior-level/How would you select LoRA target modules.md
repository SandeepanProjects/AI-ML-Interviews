# How would you select LoRA target modules?

This is an important **PEFT/LoRA interview question**.

The short answer is:

> **I select LoRA target modules based on the model architecture, the task, available GPU memory, and the quality/parameter-efficiency trade-off. I first inspect the model's linear layer names, then start with attention projections such as `q_proj` and `v_proj`. If I need more adaptation capacity, I expand to `k_proj` and `o_proj`, and for harder tasks I may target the MLP projections as well. I validate the choice experimentally rather than assuming one configuration works for every model.**

---

# 1. What is a LoRA target module?

Suppose a Transformer contains:

```text
Attention
   │
   ├── q_proj
   ├── k_proj
   ├── v_proj
   └── o_proj

MLP
   │
   ├── gate_proj
   ├── up_proj
   └── down_proj
```

Normally:

```text
W
```

is the original pretrained weight.

LoRA doesn't directly update `W`.

Instead:

```text
W' = W + ΔW
```

where:

```text
ΔW = B × A
```

and:

```text
W = frozen
A, B = trainable
```

So when you write:

```python
target_modules=["q_proj", "v_proj"]
```

you're saying:

> "Insert LoRA adapters into the query and value projection layers."

---

# 2. Why does target-module selection matter?

Because it controls the trade-off between:

```text
Adaptation capacity
        vs
Trainable parameters
        vs
GPU memory
        vs
Training time
```

For example:

```text
q_proj + v_proj
```

might be:

```text
Small
Fast
Memory efficient
```

while:

```text
q_proj
k_proj
v_proj
o_proj
gate_proj
up_proj
down_proj
```

gives:

```text
Much more adaptation capacity
More trainable parameters
More optimizer memory
Potentially better performance
```

---

# 3. Typical Transformer modules

A Llama-style Transformer block roughly looks like:

```text
              Transformer Block
                     │
        ┌────────────┴────────────┐
        │                         │
    Attention                    MLP
        │                         │
 ┌──────┼──────┐           ┌──────┼──────┐
 │      │      │           │      │      │
 Q      K      V          Gate    Up     Down
 │      │      │           │      │      │
q_proj k_proj v_proj   gate_proj up_proj down_proj
        │
        ▼
    o_proj
```

Therefore common LoRA targets are:

```text
Attention:
q_proj
k_proj
v_proj
o_proj

MLP:
gate_proj
up_proj
down_proj
```

---

# 4. Start with `q_proj` and `v_proj`

For a resource-constrained fine-tuning job, I'd often start with:

```python
from peft import LoraConfig

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "q_proj",
        "v_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
```

Why?

Because attention projections are highly influential in determining how the model attends to information.

This gives you:

```text
Small adapter
      ↓
Low memory
      ↓
Fast training
      ↓
Good baseline
```

Then evaluate.

---

# 5. Expand to all attention projections

If `q_proj + v_proj` isn't enough:

```python
target_modules = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj"
]
```

Full attention adaptation:

```text
Q ──┐
K ──┤
V ──┼── LoRA
O ──┘
```

This gives the adapter more control over the attention mechanism.

I'd consider this for:

```text
Complex instruction following
Domain adaptation
Reasoning-related adaptation
Significant behavioral changes
```

---

# 6. Add MLP modules for stronger adaptation

For more demanding domain adaptation:

```python
target_modules = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",

    "gate_proj",
    "up_proj",
    "down_proj"
]
```

Now LoRA affects:

```text
Attention
+
Feed-forward network
```

Conceptually:

```text
Transformer

Attention ──────── LoRA
    │
    ▼
MLP ────────────── LoRA
```

This provides substantially more trainable capacity.

---

# 7. What does each module do?

You don't need to memorize the exact mathematical role for interviews, but understanding the intuition helps.

## `q_proj`

Creates:

```text
Query
```

Used to determine:

> "What information am I looking for?"

---

## `k_proj`

Creates:

```text
Key
```

Used to determine:

> "What information does this token represent for matching?"

---

## `v_proj`

Creates:

```text
Value
```

Carries:

> "What information should actually be passed forward?"

---

## `o_proj`

Projects the attention output back into the model's hidden representation.

---

## `gate_proj`

Part of the MLP gating mechanism in architectures such as Llama.

---

## `up_proj`

Projects the hidden representation into the larger intermediate representation.

---

## `down_proj`

Projects the intermediate representation back to hidden size.

---

# 8. My target-module selection strategy

I would use this progression:

```text
              Start
                │
                ▼
         q_proj + v_proj
                │
                ▼
           Evaluate
                │
        ┌───────┴────────┐
        │                │
      Good             Poor
        │                │
        ▼                ▼
      Keep       q+k+v+o_proj
                         │
                         ▼
                      Evaluate
                         │
                    ┌────┴────┐
                    │         │
                  Good       Poor
                    │         │
                    ▼         ▼
                  Keep    Add MLP
                              │
                              ▼
                    gate + up + down
```

This is better than immediately targeting every layer.

---

# 9. How do I know what modules exist?

**Don't guess the module names.**

Inspect the model.

```python
for name, module in model.named_modules():

    if isinstance(module, torch.nn.Linear):

        print(name)
```

You might see:

```text
model.layers.0.self_attn.q_proj
model.layers.0.self_attn.k_proj
model.layers.0.self_attn.v_proj
model.layers.0.self_attn.o_proj

model.layers.0.mlp.gate_proj
model.layers.0.mlp.up_proj
model.layers.0.mlp.down_proj
```

Then you know what to target.

---

# 10. Filter the output

Instead of printing thousands of layers:

```python
for name, module in model.named_modules():

    if any(
        key in name
        for key in [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj"
        ]
    ):
        print(name)
```

Output:

```text
model.layers.0.self_attn.q_proj
model.layers.0.self_attn.k_proj
model.layers.0.self_attn.v_proj
model.layers.0.self_attn.o_proj

model.layers.0.mlp.gate_proj
model.layers.0.mlp.up_proj
model.layers.0.mlp.down_proj
...
```

This is the safest way to choose target modules.

---

# 11. Use `target_modules="all-linear"`

PEFT can also target all linear layers in supported workflows.

For example:

```python
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,

    target_modules="all-linear",

    lora_dropout=0.05,

    bias="none",

    task_type="CAUSAL_LM"
)
```

This is conceptually:

```text
All relevant Linear layers
        ↓
LoRA
```

This is useful when you want broad adaptation without manually specifying architecture-specific names.

But:

> **"All linear" is not automatically the best choice.**

It increases the number of trainable parameters and optimizer state.

For limited GPU resources, I would benchmark it against a smaller target set.

---

# 12. How target modules affect parameter count

For a linear layer:

```text
W shape = d_out × d_in
```

LoRA replaces training of the full matrix with:

```text
A shape = r × d_in

B shape = d_out × r
```

Trainable parameters:

```text
r × d_in + d_out × r
```

or:

```text
r(d_in + d_out)
```

Therefore:

```text
LoRA rank ↑
    ↓
Trainable parameters ↑
    ↓
Memory ↑
```

And:

```text
Target modules ↑
    ↓
Number of LoRA matrices ↑
    ↓
Trainable parameters ↑
```

---

# 13. Example calculation

Suppose a projection has:

```text
d_in = 4096
d_out = 4096
```

and:

```text
r = 16
```

LoRA parameters:

```text
16 × (4096 + 4096)

= 16 × 8192

= 131,072
```

The original layer has:

```text
4096 × 4096

= 16,777,216
```

So instead of training:

```text
16.8 million
```

parameters, you're training:

```text
131K
```

for that projection.

That's the core reason LoRA is memory efficient.

---

# 14. Target modules vs LoRA rank

These are two separate knobs.

### Option A

```python
target_modules=[
    "q_proj",
    "v_proj"
]

r=64
```

Few layers but high rank.

### Option B

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

r=16
```

Many layers but lower rank.

Both can have substantial capacity.

So I don't optimize only:

```text
r
```

I optimize:

```text
Target modules
+
Rank
+
Alpha
+
Dropout
```

together.

---

# 15. Task-specific choices

## A. Simple instruction tuning

Start:

```python
target_modules=[
    "q_proj",
    "v_proj"
]
```

Good when:

```text
Small dataset
Limited GPU
Simple behavioral adaptation
```

---

## B. Domain adaptation

For example:

```text
Legal
Finance
Healthcare
Enterprise terminology
```

I might use:

```python
target_modules=[
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj"
]
```

If quality isn't sufficient:

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

---

## C. Strong behavioral adaptation

For significant changes:

```python
target_modules="all-linear"
```

can be a useful experiment.

But evaluate carefully because more trainable parameters can increase:

```text
Training cost
Overfitting risk
Memory
```

---

# 16. Example: 7B model

For a 7B model with reasonable GPU capacity:

### Conservative

```python
LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=[
        "q_proj",
        "v_proj"
    ]
)
```

### Balanced

```python
LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj"
    ]
)
```

### High capacity

```python
LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj"
    ]
)
```

---

# 17. Example: 70B model with limited GPUs

For your previous 70B scenario, I would start more conservatively:

```python
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,

    target_modules=[
        "q_proj",
        "v_proj"
    ],

    lora_dropout=0.05,

    bias="none",

    task_type="CAUSAL_LM"
)
```

If validation quality is poor:

```python
target_modules=[
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj"
]
```

Then potentially:

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

I'd compare the configurations empirically.

---

# 18. Don't target LayerNorm blindly

You generally wouldn't do:

```python
target_modules=[
    "q_proj",
    "v_proj",
    "layernorm"
]
```

LoRA is normally applied to suitable linear/weight modules.

If you want to train normalization parameters, that's a different decision and should be deliberate.

---

# 19. Don't assume module names across architectures

This is a common interview trap.

Llama-style:

```text
q_proj
k_proj
v_proj
o_proj
```

But another model might use:

```text
query
key
value
dense
```

or fused projections such as:

```text
qkv_proj
```

Therefore:

```python
for name, module in model.named_modules():
    print(name)
```

is more reliable than memorizing names.

---

# 20. Automated discovery

You can inspect candidate linear layers:

```python
import torch


candidate_modules = set()

for name, module in model.named_modules():

    if isinstance(module, torch.nn.Linear):

        candidate_modules.add(
            name.split(".")[-1]
        )


print(sorted(candidate_modules))
```

You might get:

```text
[
    "down_proj",
    "gate_proj",
    "k_proj",
    "o_proj",
    "q_proj",
    "up_proj",
    "v_proj"
]
```

Now you can decide what to target.

---

# 21. Verify LoRA actually attached

After:

```python
model = get_peft_model(
    model,
    lora_config
)
```

run:

```python
model.print_trainable_parameters()
```

Also inspect:

```python
for name, param in model.named_parameters():

    if param.requires_grad:

        print(name)
```

You should see names similar to:

```text
base_model.model.model.layers.0.self_attn.q_proj.lora_A
base_model.model.model.layers.0.self_attn.q_proj.lora_B

base_model.model.model.layers.0.self_attn.v_proj.lora_A
base_model.model.model.layers.0.self_attn.v_proj.lora_B
```

This confirms that your target modules were actually adapted.

---

# 22. The correct way to optimize target modules

Don't say:

> "I always use q_proj and v_proj."

A stronger Senior AI Engineer answer is:

```text
Baseline
   ↓
q_proj + v_proj
   ↓
Evaluate
   ↓
q+k+v+o
   ↓
Evaluate
   ↓
All attention + MLP
   ↓
Evaluate
```

Track:

```text
Validation loss
Task accuracy
Instruction following
Hallucination rate
Training time
GPU memory
Adapter size
```

Then select the smallest configuration that reaches the required quality.

---

# 23. Example experiment

Suppose you run:

| Configuration   | Trainable params | Validation score | VRAM |
| --------------- | ---------------: | ---------------: | ---: |
| Q + V           |              10M |              82% | 16GB |
| Q + K + V + O   |              20M |              86% | 17GB |
| Attention + MLP |              60M |              87% | 19GB |
| All linear      |              80M |            87.2% | 20GB |

I would probably choose:

```text
Q + K + V + O
```

because:

```text
86% quality
vs
87% quality
```

may not justify:

```text
Much larger adapter
+
More training
+
More memory
```

This is how I would make the decision in production.

---

# 24. Important distinction: target modules ≠ target layers

This is subtle.

When you write:

```python
target_modules=["q_proj", "v_proj"]
```

you're generally targeting those **module types/names across the Transformer blocks**, not just one specific layer.

So you're effectively doing:

```text
Layer 0: q_proj + v_proj
Layer 1: q_proj + v_proj
Layer 2: q_proj + v_proj
...
Layer N: q_proj + v_proj
```

If you want only particular Transformer layers, that is a different, more specialized configuration.

---

# 25. Interview answer

If the interviewer asks:

**"How would you select LoRA target modules?"**

Answer:

> **I first inspect the model architecture rather than hardcoding assumptions. For a Llama-style Transformer, I would initially target `q_proj` and `v_proj` because they provide a good parameter-efficient baseline. If the task requires more adaptation capacity, I would expand to `k_proj` and `o_proj`, and for stronger domain or behavioral adaptation I would consider the MLP projections—`gate_proj`, `up_proj`, and `down_proj`—or `all-linear`. I would then compare configurations using validation quality, trainable parameter count, GPU memory, training throughput, and overfitting. My goal is to use the smallest target-module set that achieves the required quality.**

### Easy way to remember

```text
Limited GPU
     ↓
q_proj + v_proj
     ↓
Need more capacity?
     ↓
q + k + v + o
     ↓
Still insufficient?
     ↓
Attention + MLP
     ↓
Maximum adaptation
     ↓
all-linear
```

And the **most important interview point**:

> **Don't select LoRA target modules purely from a fixed recipe. Inspect the architecture, start with a lightweight baseline, evaluate, and expand adapter coverage only when the quality improvement justifies the additional memory and training cost.**
