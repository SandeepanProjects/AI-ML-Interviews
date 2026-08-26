# How do you save LoRA adapters and merge them?

These are two very important concepts when working with **LoRA / PEFT**.

```text
Base Model
    +
LoRA Adapter
    ↓
Fine-tuned Model
```

After training, you have two choices:

1. **Save the adapter separately** — common during experimentation and when serving multiple adapters.
2. **Merge the adapter into the base model** — useful for standalone deployment.

---

# 1. How do you save LoRA adapters?

Suppose you created a LoRA model:

```python
from peft import LoraConfig, get_peft_model

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
    task_type="CAUSAL_LM",
)

model = get_peft_model(
    base_model,
    lora_config
)
```

Your architecture is now:

```text
              Base Llama Model
                     │
                     │ Frozen
                     ▼
             Transformer Layers
                     │
                     ├──── LoRA A ✓
                     │
                     └──── LoRA B ✓
```

During training:

```text
Base weights → Frozen

LoRA A → Trainable
LoRA B → Trainable
```

After training:

```python
trainer.train()
```

You can save the adapter.

---

# 2. Simple way: `save_pretrained()`

```python
model.save_pretrained(
    "./my-lora-adapter"
)
```

If `model` is a PEFT model, this saves the adapter-related artifacts rather than a separate copy of the entire original base model.

Typical structure:

```text
my-lora-adapter/
│
├── adapter_config.json
├── adapter_model.safetensors
└── README.md
```

The exact files can vary by PEFT version/configuration.

Conceptually:

```text
Base Model
   ↓
NOT duplicated

LoRA Adapter
   ↓
Saved separately
```

This is one of LoRA's major advantages.

For example:

```text
Base Model:      8 GB
Support Adapter: 100 MB
SQL Adapter:     100 MB
Coding Adapter:  120 MB
```

Instead of storing:

```text
8 GB Support Model
8 GB SQL Model
8 GB Coding Model
```

You can store:

```text
Base Model
   +
Small adapters
```

The official [Hugging Face PEFT documentation](https://huggingface.co/docs/peft/index?utm_source=chatgpt.com) describes adapter-based saving and loading workflows.

---

# 3. Saving after `Trainer`

If you use Hugging Face `Trainer`:

```python
trainer.train()

trainer.save_model(
    "./my-lora-adapter"
)
```

You can also explicitly save the model:

```python
trainer.model.save_pretrained(
    "./my-lora-adapter"
)
```

For a PEFT-wrapped model, both approaches are commonly used.

I usually prefer being explicit:

```python
trainer.model.save_pretrained(
    "./my-lora-adapter"
)

tokenizer.save_pretrained(
    "./my-lora-adapter"
)
```

This gives:

```text
my-lora-adapter/
│
├── adapter_config.json
├── adapter_model.safetensors
├── tokenizer.json
├── tokenizer_config.json
└── special_tokens_map.json
```

---

# 4. Complete example: train and save a LoRA adapter

```python
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)

from peft import (
    LoraConfig,
    get_peft_model,
)


MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"

ADAPTER_PATH = "./customer-support-lora"


# =========================================
# Load tokenizer
# =========================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# =========================================
# Load base model
# =========================================

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
)


# =========================================
# LoRA configuration
# =========================================

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
    task_type="CAUSAL_LM",
)


# =========================================
# Add LoRA
# =========================================

model = get_peft_model(
    base_model,
    lora_config
)


# Check trainable parameters
model.print_trainable_parameters()


# =========================================
# Training
# =========================================

# trainer = Trainer(
#     model=model,
#     ...
# )
#
# trainer.train()


# =========================================
# Save adapter
# =========================================

model.save_pretrained(
    ADAPTER_PATH
)


# =========================================
# Save tokenizer
# =========================================

tokenizer.save_pretrained(
    ADAPTER_PATH
)
```

---

# 5. How do you load a saved LoRA adapter?

You need:

```text
1. Original Base Model
2. LoRA Adapter
```

Architecture:

```text
Original Base Model
        +
Saved LoRA Adapter
        ↓
Fine-tuned Model
```

Code:

```python
from transformers import AutoModelForCausalLM
from peft import PeftModel


MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"

ADAPTER_PATH = "./customer-support-lora"


# Load original base model
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME
)


# Load LoRA adapter
model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_PATH
)
```

Now:

```text
Base Model
   │
   ├── q_proj
   │       +
   │     LoRA
   │
   ├── k_proj
   │       +
   │     LoRA
   │
   ├── v_proj
   │       +
   │     LoRA
   │
   └── o_proj
           +
         LoRA
```

For inference:

```python
model.eval()
```

---

# 6. What does "merging LoRA" mean?

Recall the LoRA equation:

$$
W_{final}
=
W_{base}
+
\Delta W
$$

where:

$$
\Delta W
=
\frac{\alpha}{r}BA
$$

So:

$$
W_{final}
=
W_{base}
+
\frac{\alpha}{r}BA
$$

Before merging:

```text
               Input
                 │
                 ▼
          ┌─────────────┐
          │ Base Weight │
          │      W      │
          └──────┬──────┘
                 │
                 │
          ┌──────▼──────┐
          │ LoRA Update │
          │     BA      │
          └──────┬──────┘
                 │
                 ▼
                Output
```

At inference time, the model conceptually computes:

```text
W(x) + ΔW(x)
```

After merging:

```text
Wmerged = W + ΔW
```

So the model becomes:

```text
               Input
                 │
                 ▼
          ┌─────────────┐
          │Merged Weight│
          │ W + ΔW      │
          └──────┬──────┘
                 │
                 ▼
               Output
```

The separate adapter update is folded into the base weights.

---

# 7. How do you merge a LoRA adapter?

The standard PEFT workflow is:

```python
merged_model = model.merge_and_unload()
```

Example:

```python
from transformers import AutoModelForCausalLM
from peft import PeftModel


MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"

ADAPTER_PATH = "./customer-support-lora"


# =========================================
# Load base model
# =========================================

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
)


# =========================================
# Load adapter
# =========================================

model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_PATH
)


# =========================================
# Merge LoRA into base model
# =========================================

merged_model = model.merge_and_unload()
```

After:

```text
Before merge:

Base Model
    +
LoRA Adapter
    =
PeftModel


After merge:

Merged Model

W = Wbase + ΔW
```

The resulting `merged_model` is a regular model without needing the adapter separately for inference.

PEFT documents `merge_and_unload()` as the standard approach for folding supported adapter weights into the base model. [PEFT model merging guide](https://huggingface.co/docs/peft/developer_guides/model_merging?utm_source=chatgpt.com)

---

# 8. Save the merged model

After merging:

```python
merged_model.save_pretrained(
    "./customer-support-merged-model"
)
```

Also save the tokenizer:

```python
tokenizer.save_pretrained(
    "./customer-support-merged-model"
)
```

Your folder now looks like a normal Hugging Face model:

```text
customer-support-merged-model/
│
├── config.json
├── model.safetensors
├── generation_config.json
├── tokenizer.json
├── tokenizer_config.json
└── special_tokens_map.json
```

Exact filenames can vary.

Now you can load it without PEFT:

```python
from transformers import AutoModelForCausalLM


model = AutoModelForCausalLM.from_pretrained(
    "./customer-support-merged-model"
)
```

No adapter is needed.

---

# 9. Complete production-style merge code

```python
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

from peft import PeftModel


# =========================================
# Paths
# =========================================

BASE_MODEL = (
    "meta-llama/Llama-3.2-3B-Instruct"
)

ADAPTER_PATH = (
    "./customer-support-lora"
)

MERGED_MODEL_PATH = (
    "./customer-support-merged"
)


# =========================================
# Load tokenizer
# =========================================

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL
)


# =========================================
# Load base model
# =========================================

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
)


# =========================================
# Load LoRA adapter
# =========================================

peft_model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_PATH
)


# =========================================
# Merge LoRA
# =========================================

merged_model = (
    peft_model.merge_and_unload()
)


# =========================================
# Save merged model
# =========================================

merged_model.save_pretrained(
    MERGED_MODEL_PATH,
    safe_serialization=True
)


# =========================================
# Save tokenizer
# =========================================

tokenizer.save_pretrained(
    MERGED_MODEL_PATH
)


print(
    "LoRA adapter merged successfully!"
)
```

---

# 10. Important: Merge dtype matters

Suppose your base model was trained in:

```text
BF16
```

For merging, you should generally load the base model in an appropriate floating-point dtype.

Example:

```python
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
)
```

Then:

```python
peft_model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_PATH
)

merged_model = peft_model.merge_and_unload()
```

Why?

Because merging mathematically involves combining:

$$
W_{base}
+
\Delta W
$$

The operation needs an appropriate representation for the resulting weights.

A common production workflow is:

```text
Train with QLoRA
       ↓
Save LoRA adapter
       ↓
Load original model in BF16/FP16
       ↓
Load adapter
       ↓
Merge adapter
       ↓
Save merged model
       ↓
Optional deployment quantization
```

---

# 11. Important: Can you merge directly into a 4-bit QLoRA base model?

This requires care.

During QLoRA:

```text
Base Model
    ↓
4-bit quantized
    +
LoRA adapters
```

The LoRA adapters are separate trainable weights.

For a deployment workflow, it is often safer to:

```text
1. Load original base model
   in BF16 / FP16

2. Load trained LoRA adapter

3. Merge LoRA

4. Save merged model

5. Optionally quantize the merged model
```

Example:

```python
import torch

from transformers import AutoModelForCausalLM
from peft import PeftModel


# Load full precision / BF16 base model
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
)


# Load QLoRA-trained adapter
model = PeftModel.from_pretrained(
    base_model,
    "./qlora-adapter"
)


# Merge adapter
merged_model = model.merge_and_unload()


# Save
merged_model.save_pretrained(
    "./merged-model"
)
```

Then, for inference, you can load the saved model using an appropriate quantization workflow.

---

# 12. Multiple adapters

One major benefit of keeping adapters separate:

```text
                 Base Llama
                     │
        ┌────────────┼────────────┐
        ▼            ▼            ▼
    Support       SQL          Coding
    Adapter      Adapter       Adapter
```

You can have:

```text
Llama
 +
support_adapter
```

or:

```text
Llama
 +
sql_adapter
```

or:

```text
Llama
 +
coding_adapter
```

Example:

```python
model.load_adapter(
    "./sql-adapter",
    adapter_name="sql"
)

model.load_adapter(
    "./support-adapter",
    adapter_name="support"
)
```

Then select one:

```python
model.set_adapter(
    "sql"
)
```

or:

```python
model.set_adapter(
    "support"
)
```

This is much harder if you merge everything permanently.

---

# 13. Can you merge multiple LoRA adapters?

Yes, PEFT supports adapter merging strategies for combining adapters, subject to compatibility and the selected merge method.

Conceptually:

```text
Base Model
    │
    ├── Support LoRA
    │
    ├── SQL LoRA
    │
    └── Coding LoRA
             │
             ▼
       Adapter Merge
             │
             ▼
       Combined Model
```

A typical workflow can look like:

```python
model.add_weighted_adapter(
    adapters=[
        "support",
        "sql"
    ],
    weights=[
        0.7,
        0.3
    ],
    adapter_name="combined",
    combination_type="linear"
)

model.set_adapter(
    "combined"
)
```

The exact supported merge methods depend on the PEFT version and adapter types. [PEFT adapter merging documentation](https://huggingface.co/docs/peft/developer_guides/model_merging?utm_source=chatgpt.com)

---

# 14. Adapter saving vs merged model

| Feature                | Separate LoRA Adapter | Merged Model        |
| ---------------------- | --------------------- | ------------------- |
| Storage                | Small                 | Large               |
| Needs base model       | Yes                   | No separate adapter |
| Adapter switching      | Easy                  | No                  |
| Multi-task support     | Excellent             | Difficult           |
| Deployment simplicity  | Slightly more setup   | Simple              |
| Training artifact      | Ideal                 | Usually not primary |
| Inference architecture | Base + adapter        | Single model        |

---

# 15. When should you keep the adapter separate?

Keep adapters separate when:

```text
✓ You have multiple tasks
✓ You want adapter switching
✓ Storage is important
✓ You are experimenting
✓ You want to continue fine-tuning
✓ Multiple customers have different adapters
```

Example enterprise architecture:

```text
                Base LLM
                    │
        ┌───────────┼───────────┐
        │           │           │
        ▼           ▼           ▼
   Customer A   Customer B   Customer C
     Adapter      Adapter      Adapter
```

This is particularly useful for multi-tenant AI systems.

---

# 16. When should you merge?

Merge when:

```text
✓ You want one standalone model
✓ Deployment expects standard model weights
✓ You do not need adapter switching
✓ You want a simpler inference artifact
```

Example:

```text
Training:

Base Model
   +
LoRA
   ↓
Fine-tuned Adapter


Deployment:

Base + LoRA
   ↓
Merge
   ↓
Single Model
   ↓
Inference Server
```

---

# 17. Real-world QLoRA deployment workflow

A strong interview answer is:

```text
                TRAINING

Original Model
      │
      ▼
4-bit Quantization
      │
      ▼
Add LoRA
      │
      ▼
Train Adapters
      │
      ▼
Save Adapter
      │
      │
      ▼
                DEPLOYMENT

Original Model
      │
      ▼
Load in BF16/FP16
      │
      ▼
Load Adapter
      │
      ▼
merge_and_unload()
      │
      ▼
Merged Model
      │
      ▼
Optional Quantization
      │
      ▼
vLLM / TGI / Other Inference Server
```

---

# 18. Interview-ready answer

> **After LoRA fine-tuning, I usually save the PEFT adapter separately using `model.save_pretrained()` or `trainer.save_model()`. The adapter contains the learned LoRA parameters and configuration, while the original base model is stored separately.**
>
> **To load the adapter, I first load the same compatible base model and then use `PeftModel.from_pretrained(base_model, adapter_path)`.**
>
> **To merge the adapter, I call `merge_and_unload()`. Conceptually, LoRA adds a low-rank update to the original weight matrix: \(W' = W + \frac{\alpha}{r}BA\). Merging folds this update into the base model weights and produces a standalone model that no longer needs the separate adapter.**
>
> **For QLoRA, I usually keep the trained adapter separately and, when needed, load the original base model in an appropriate floating-point dtype, load the adapter, merge it, save the merged model, and optionally quantize it again for inference.**

## The 3 most important code patterns

### Save adapter

```python
model.save_pretrained(
    "./lora-adapter"
)

tokenizer.save_pretrained(
    "./lora-adapter"
)
```

### Load adapter

```python
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME
)

model = PeftModel.from_pretrained(
    base_model,
    "./lora-adapter"
)
```

### Merge adapter

```python
merged_model = model.merge_and_unload()

merged_model.save_pretrained(
    "./merged-model"
)
```

The key thing to remember is:

```text
SAVE SEPARATELY
Base Model + Adapter

MERGE
Base Model + LoRA Update
        ↓
Standalone Model
```
