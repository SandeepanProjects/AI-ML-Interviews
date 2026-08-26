# How do you load LoRA adapters during inference?

During LoRA inference, you usually **do not load a complete fine-tuned model**.

Instead, you load:

```text
1. Base model
       +
2. LoRA adapter
       ↓
3. Inference
```

The architecture looks like:

```text
        Base LLM
      (frozen weights)
             │
             │
       Load adapter
             │
             ▼
       LoRA updates
             │
             ▼
       Fine-tuned output
```

---

# 1. Basic workflow

Assume you saved your adapter after training:

```python
model.save_pretrained("./customer-support-lora")
tokenizer.save_pretrained("./customer-support-lora")
```

Now during inference:

```text
Load Base Model
       ↓
Load LoRA Adapter
       ↓
Set evaluation mode
       ↓
Generate response
```

---

# 2. Load the base model + adapter

The standard PEFT approach is:

```python
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

from peft import PeftModel


BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"

ADAPTER_PATH = "./customer-support-lora"


# ==========================================
# Load tokenizer
# ==========================================

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL
)


# ==========================================
# Load base model
# ==========================================

base_model = AutoModelForCausalLM.from_pretrained(

    BASE_MODEL,

    torch_dtype=torch.bfloat16,

    device_map="auto"
)


# ==========================================
# Load LoRA adapter
# ==========================================

model = PeftModel.from_pretrained(

    base_model,

    ADAPTER_PATH
)


# ==========================================
# Evaluation mode
# ==========================================

model.eval()
```

Now:

```text
Before adapter:

Base Model
    │
    ▼
Generic LLM


After adapter:

Base Model
    +
Customer Support LoRA
    │
    ▼
Customer Support LLM
```

---

# 3. Run inference

```python
prompt = """
Customer:
My payment was deducted twice.
What should I do?
"""


inputs = tokenizer(

    prompt,

    return_tensors="pt"
)


inputs = inputs.to(
    model.device
)


with torch.no_grad():

    outputs = model.generate(

        **inputs,

        max_new_tokens=200,

        temperature=0.7,

        do_sample=True
    )


response = tokenizer.decode(

    outputs[0],

    skip_special_tokens=True
)


print(response)
```

---

# 4. Complete inference example

Here is a realistic complete example.

```python
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

from peft import PeftModel


# =====================================================
# Configuration
# =====================================================

BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"

ADAPTER_PATH = "./customer-support-lora"


# =====================================================
# Load Tokenizer
# =====================================================

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL
)


if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# =====================================================
# Load Base Model
# =====================================================

base_model = AutoModelForCausalLM.from_pretrained(

    BASE_MODEL,

    torch_dtype=torch.bfloat16,

    device_map="auto"
)


# =====================================================
# Load LoRA Adapter
# =====================================================

model = PeftModel.from_pretrained(

    base_model,

    ADAPTER_PATH
)


# =====================================================
# Evaluation Mode
# =====================================================

model.eval()


# =====================================================
# Create Prompt
# =====================================================

messages = [

    {
        "role": "system",
        "content": (
            "You are a helpful customer support assistant."
        )
    },

    {
        "role": "user",
        "content": (
            "My credit card was charged twice. "
            "How can I get a refund?"
        )
    }
]


# =====================================================
# Apply Chat Template
# =====================================================

inputs = tokenizer.apply_chat_template(

    messages,

    tokenize=True,

    add_generation_prompt=True,

    return_tensors="pt",

    return_dict=True
)


inputs = inputs.to(
    model.device
)


# =====================================================
# Generate
# =====================================================

with torch.inference_mode():

    outputs = model.generate(

        **inputs,

        max_new_tokens=300,

        temperature=0.7,

        do_sample=True,

        pad_token_id=tokenizer.eos_token_id
    )


# =====================================================
# Decode only new tokens
# =====================================================

input_length = inputs["input_ids"].shape[1]

generated_tokens = outputs[
    0,
    input_length:
]


response = tokenizer.decode(

    generated_tokens,

    skip_special_tokens=True
)


print(response)
```

---

# 5. What happens internally?

Suppose a base model layer has:

$$
W
$$

LoRA adds:

$$
\Delta W = \frac{\alpha}{r}BA
$$

During inference:

$$
W_{effective}
=
W + \Delta W
$$

So:

```text
Base weight W
       +
LoRA update ΔW
       ↓
Effective weight
       ↓
Output
```

For a linear layer:

```text
Original:

y = Wx
```

With LoRA:

```text
y = Wx + ΔWx
```

Where:

```text
ΔW = scaling × B × A
```

The adapter changes only targeted layers.

For example:

```text
Transformer Layer

q_proj ─── Base + LoRA
k_proj ─── Base + LoRA
v_proj ─── Base + LoRA
o_proj ─── Base + LoRA
```

Other layers may remain unchanged.

---

# 6. Load the adapter directly with `AutoPeftModelForCausalLM`

PEFT can also load the adapter model more conveniently:

```python
from peft import AutoPeftModelForCausalLM


model = AutoPeftModelForCausalLM.from_pretrained(

    "./customer-support-lora",

    torch_dtype=torch.bfloat16,

    device_map="auto"
)

model.eval()
```

Why is this useful?

Because the adapter's configuration contains information about the base model.

Conceptually:

```text
Adapter Config
      │
      ├── Base model name
      ├── LoRA rank
      ├── Target modules
      └── Other adapter settings
```

PEFT can use this information to reconstruct the model + adapter.

---

# 7. Loading adapters with a quantized base model

This is very common in production.

For example:

```text
Base Llama Model
       │
       ▼
4-bit quantized
       │
       +
LoRA Adapter
       │
       ▼
Inference
```

Code:

```python
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)

from peft import PeftModel


BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"

ADAPTER_PATH = "./customer-support-lora"


# ==========================================
# Quantization config
# ==========================================

quantization_config = BitsAndBytesConfig(

    load_in_4bit=True,

    bnb_4bit_quant_type="nf4",

    bnb_4bit_compute_dtype=torch.bfloat16,

    bnb_4bit_use_double_quant=True
)


# ==========================================
# Load tokenizer
# ==========================================

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL
)


# ==========================================
# Load quantized base model
# ==========================================

base_model = AutoModelForCausalLM.from_pretrained(

    BASE_MODEL,

    quantization_config=quantization_config,

    device_map="auto"
)


# ==========================================
# Attach adapter
# ==========================================

model = PeftModel.from_pretrained(

    base_model,

    ADAPTER_PATH
)


model.eval()
```

This gives:

```text
                GPU

┌─────────────────────────────┐
│                             │
│ Base Model                  │
│                             │
│ 4-bit Quantized             │
│                             │
│ Mostly Frozen               │
│                             │
├─────────────────────────────┤
│                             │
│ LoRA Adapter                │
│                             │
│ Higher precision            │
│                             │
└─────────────────────────────┘
```

---

# 8. Important: QLoRA training vs inference

During training:

```text
4-bit Base Model
       +
Trainable LoRA
       ↓
QLoRA Training
```

During inference:

You have multiple choices.

### Option 1: Keep the adapter separate

```text
4-bit Base Model
       +
LoRA Adapter
       ↓
Inference
```

Advantages:

```text
✓ Low storage
✓ Can switch adapters
✓ Easy multi-tenant setup
```

---

### Option 2: Merge the adapter

```text
BF16 Base Model
       +
LoRA Adapter
       ↓
Merge
       ↓
Standalone Model
```

Then optionally quantize:

```text
Merged Model
       ↓
4-bit / 8-bit Quantization
       ↓
Inference
```

---

# 9. Loading multiple adapters

Suppose you have:

```text
Base Llama Model
       │
       ├── SQL Adapter
       │
       ├── Customer Support Adapter
       │
       └── Coding Adapter
```

First load one adapter:

```python
from peft import PeftModel

model = PeftModel.from_pretrained(

    base_model,

    "./customer-support-adapter",

    adapter_name="support"
)
```

Then load additional adapters:

```python
model.load_adapter(

    "./sql-adapter",

    adapter_name="sql"
)


model.load_adapter(

    "./coding-adapter",

    adapter_name="coding"
)
```

Now switch adapters.

### Use the support adapter

```python
model.set_adapter(
    "support"
)
```

### Use the SQL adapter

```python
model.set_adapter(
    "sql"
)
```

### Use the coding adapter

```python
model.set_adapter(
    "coding"
)
```

Architecture:

```text
                 Base Model
                     │
        ┌────────────┼────────────┐
        │            │            │
        ▼            ▼            ▼
     Support        SQL         Coding
     Adapter       Adapter      Adapter
```

At runtime:

```python
def select_adapter(task: str):

    if task == "support":

        model.set_adapter("support")

    elif task == "sql":

        model.set_adapter("sql")

    elif task == "coding":

        model.set_adapter("coding")
```

---

# 10. Production example: adapter routing

Imagine an enterprise system.

```text
                  User Request
                       │
                       ▼
                  Task Router
                       │
          ┌────────────┼────────────┐
          │            │            │
          ▼            ▼            ▼
       Support         SQL        Coding
          │            │            │
          ▼            ▼            ▼
      Support        SQL         Coding
      Adapter        Adapter      Adapter
          │            │            │
          └────────────┼────────────┘
                       ▼
                    Response
```

Simple implementation:

```python
from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI()


class QueryRequest(BaseModel):

    task: str

    question: str


@app.post("/generate")
async def generate(
    request: QueryRequest
):

    adapter_map = {

        "support": "support",

        "sql": "sql",

        "coding": "coding"
    }


    adapter_name = adapter_map.get(
        request.task
    )


    if adapter_name is None:

        return {
            "error": "Unknown task"
        }


    # Select adapter
    model.set_adapter(
        adapter_name
    )


    # Tokenize
    inputs = tokenizer(
        request.question,
        return_tensors="pt"
    ).to(model.device)


    # Generate
    with torch.inference_mode():

        outputs = model.generate(

            **inputs,

            max_new_tokens=200
        )


    response = tokenizer.decode(

        outputs[0],

        skip_special_tokens=True
    )


    return {
        "adapter": adapter_name,
        "response": response
    }
```

### Production warning

The above example is conceptually correct, but **do not blindly call `set_adapter()` on one globally shared model concurrently**. If multiple requests arrive simultaneously and each changes the active adapter, requests can interfere with each other.

For production, use an adapter-serving design that supports request isolation, such as:

```text
Request
   ↓
Adapter router
   ↓
Worker/model instance with selected adapter
   ↓
Inference
```

or a serving framework with safe dynamic adapter handling.

---

# 11. Loading an adapter for a merged model is different

If you already merged:

```python
merged_model = model.merge_and_unload()
```

and saved:

```python
merged_model.save_pretrained(
    "./merged-model"
)
```

Then inference is just:

```python
model = AutoModelForCausalLM.from_pretrained(
    "./merged-model",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
```

No `PeftModel` is needed.

```text
Separate Adapter:

Base Model + Adapter
       ↓
PeftModel


Merged:

Merged Model
       ↓
AutoModelForCausalLM
```

---

# 12. Check which adapters are loaded

You can inspect the PEFT model:

```python
print(
    model.peft_config.keys()
)
```

Example:

```text
dict_keys([
    'support',
    'sql',
    'coding'
])
```

You can also inspect:

```python
print(model.active_adapter)
```

Depending on your PEFT version/configuration, APIs and return types may differ slightly.

---

# 13. Common inference errors

## Error 1: Wrong base model

Suppose your adapter was trained on:

```text
Llama Model Version A
```

But you load:

```text
Llama Model Version B
```

This may fail or produce incorrect results.

Always use the same compatible base model architecture/checkpoint used for training.

```text
Training:

Base Model A
     +
Adapter A
     ↓
Save


Inference:

Base Model A  ✓
     +
Adapter A
     ↓
Correct
```

---

## Error 2: Tokenizer mismatch

Use the tokenizer compatible with the training/base model.

```python
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL
)
```

Or, if you saved a modified tokenizer with the adapter:

```python
tokenizer = AutoTokenizer.from_pretrained(
    ADAPTER_PATH
)
```

The second approach can be important if you added special tokens during fine-tuning.

---

## Error 3: Forgetting evaluation mode

Always do:

```python
model.eval()
```

This disables training-specific behavior such as dropout.

---

## Error 4: Gradients enabled during inference

Prefer:

```python
with torch.inference_mode():

    outputs = model.generate(
        **inputs
    )
```

rather than normal training mode.

---

# 14. Recommended inference patterns

## Pattern A — Normal LoRA inference

```python
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_PATH
)

model.eval()
```

---

## Pattern B — Memory-efficient 4-bit inference

```python
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=quantization_config,
    device_map="auto"
)

model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_PATH
)

model.eval()
```

---

## Pattern C — Merged model inference

```python
model = AutoModelForCausalLM.from_pretrained(
    "./merged-model",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

model.eval()
```

---

# Interview-ready answer

> **During LoRA inference, I first load the same compatible base model used during training and then attach the saved adapter using `PeftModel.from_pretrained()`. The base model provides the original weights, while the LoRA adapter provides the learned low-rank updates.**
>
> **For memory-efficient inference, I can load the base model in 4-bit or 8-bit and attach the adapter. If I need a standalone model, I can load the base model in BF16/FP16, load the adapter, call `merge_and_unload()`, and deploy the merged model.**
>
> **For multi-task systems, I can load multiple adapters and select the appropriate adapter based on the request. In a concurrent production system, however, adapter selection must be request-safe and isolated rather than mutating one shared model's active adapter.**

## The most important code to remember

```python
from transformers import AutoModelForCausalLM
from peft import PeftModel


# 1. Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL
)


# 2. Attach LoRA adapter
model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_PATH
)


# 3. Inference mode
model.eval()
```

That is the core answer:

```text
Load Base Model
       +
Load Adapter
       ↓
PeftModel
       ↓
Inference
```
