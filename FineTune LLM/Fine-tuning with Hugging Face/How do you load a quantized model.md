# How do you load a quantized model?

Quantization means storing model weights using fewer bits.

For example:

```text
FP32 → 32 bits per value
FP16 → 16 bits per value
INT8 → 8 bits per value
4-bit → 4 bits per value
```

This reduces GPU memory.

```text
Normal model
     ↓
FP16 / BF16 weights
     ↓
Large GPU memory

Quantized model
     ↓
INT8 or 4-bit weights
     ↓
Much lower GPU memory
```

For Hugging Face models, a common approach is **Transformers + bitsandbytes**. The main API is `BitsAndBytesConfig`, which you pass to `from_pretrained()`. ([Hugging Face][1])

---

# 1. Install required libraries

```bash
pip install torch transformers accelerate bitsandbytes peft
```

`bitsandbytes` provides the 8-bit and 4-bit quantization integration used by Transformers. ([Hugging Face][1])

---

# 2. Load a normal model first

Without quantization:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME
)
```

Conceptually:

```text
Model Weights
    ↓
FP32 / BF16 / FP16
    ↓
High memory usage
```

---

# 3. Load an 8-bit quantized model

The simplest configuration is:

```python
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)

MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"


# Create 8-bit configuration
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True
)


# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


# Load model in 8-bit
model = AutoModelForCausalLM.from_pretrained(

    MODEL_NAME,

    quantization_config=quantization_config,

    device_map="auto"
)
```

Architecture:

```text
Hugging Face Model
       │
       ▼
from_pretrained()
       │
       ▼
BitsAndBytesConfig
       │
       ▼
load_in_8bit=True
       │
       ▼
8-bit Quantized Model
```

8-bit quantization roughly halves model-weight memory compared with 16-bit representations, though actual total memory usage depends on non-quantized modules and runtime overhead. `device_map="auto"` can distribute a large model across available devices. ([Hugging Face][1])

---

# 4. What does `device_map="auto"` do?

```python
device_map="auto"
```

Automatically places model layers on available hardware.

For example:

```text
GPU 0
├── Transformer layers 0–15
└── Transformer layers 16–25

GPU 1
├── Transformer layers 26–40
└── Output layer
```

Or, depending on available hardware:

```text
GPU
 ↓
Some layers

CPU
 ↓
Remaining layers
```

For large models, this is often useful for inference/loading. ([Hugging Face][1])

---

# 5. Load a 4-bit quantized model

4-bit quantization is particularly important for **QLoRA**.

```python
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)


MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"


# Configure 4-bit quantization
quantization_config = BitsAndBytesConfig(

    load_in_4bit=True,

    bnb_4bit_quant_type="nf4",

    bnb_4bit_compute_dtype=torch.bfloat16,

    bnb_4bit_use_double_quant=True
)


# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


# Load model
model = AutoModelForCausalLM.from_pretrained(

    MODEL_NAME,

    quantization_config=quantization_config,

    device_map="auto"
)
```

This is a common QLoRA-style base-model loading configuration: 4-bit storage, NF4 quantization, optional double quantization, and BF16 computation. ([Hugging Face][2])

---

# 6. Understand each parameter

## `load_in_4bit=True`

```python
load_in_4bit=True
```

This tells Transformers:

```text
Load model
   ↓
Quantize supported linear weights
   ↓
Store in 4-bit representation
```

Conceptually:

```text
Original:

Weight = 0.12345678
        ↑
    16/32-bit


Quantized:

Weight ≈ 0.12
        ↑
       4-bit
```

The value is approximated, so quantization trades some numerical precision for memory savings.

---

# 7. What is `bnb_4bit_quant_type="nf4"`?

```python
bnb_4bit_quant_type="nf4"
```

NF4 means:

> **NormalFloat 4**

It is a 4-bit quantization data type designed for values that are approximately normally distributed and is commonly recommended for training 4-bit base models in QLoRA workflows. ([Hugging Face][2])

Code:

```python
quantization_config = BitsAndBytesConfig(

    load_in_4bit=True,

    bnb_4bit_quant_type="nf4"
)
```

You may also see:

```python
bnb_4bit_quant_type="fp4"
```

But for a typical QLoRA fine-tuning setup, NF4 is a common choice.

---

# 8. What is `bnb_4bit_compute_dtype`?

Important distinction:

```text
Storage precision ≠ Compute precision
```

You can store weights in:

```text
4-bit
```

but perform calculations in:

```text
BF16
```

Code:

```python
bnb_4bit_compute_dtype=torch.bfloat16
```

Conceptually:

```text
4-bit stored weights
        │
        ▼
Dequantization during operation
        │
        ▼
BF16 computation
        │
        ▼
Output
```

Example:

```python
import torch

quantization_config = BitsAndBytesConfig(

    load_in_4bit=True,

    bnb_4bit_compute_dtype=torch.bfloat16
)
```

Common choices:

```python
torch.float16
```

or:

```python
torch.bfloat16
```

BF16 is commonly preferred on hardware that supports it well.

---

# 9. What is double quantization?

```python
bnb_4bit_use_double_quant=True
```

Normally:

```text
Original Weights
      │
      ▼
4-bit Quantization
      │
      ▼
Quantized Weights
```

With double/nested quantization:

```text
Original Weights
      │
      ▼
4-bit Quantization
      │
      ▼
Quantization Constants
      │
      ▼
Quantize those constants too
```

Code:

```python
quantization_config = BitsAndBytesConfig(

    load_in_4bit=True,

    bnb_4bit_use_double_quant=True
)
```

Hugging Face documents nested/double quantization as a technique for additional memory savings. ([Hugging Face][2])

---

# 10. Complete 4-bit inference example

Here is a complete example.

```python
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)


# ==========================================
# Configuration
# ==========================================

MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"


# ==========================================
# 4-bit Quantization Config
# ==========================================

quantization_config = BitsAndBytesConfig(

    # Load weights in 4-bit
    load_in_4bit=True,

    # Use NF4 quantization
    bnb_4bit_quant_type="nf4",

    # Use BF16 for computations
    bnb_4bit_compute_dtype=torch.bfloat16,

    # Additional memory optimization
    bnb_4bit_use_double_quant=True
)


# ==========================================
# Load Tokenizer
# ==========================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


# ==========================================
# Load Quantized Model
# ==========================================

model = AutoModelForCausalLM.from_pretrained(

    MODEL_NAME,

    quantization_config=quantization_config,

    device_map="auto"
)


# ==========================================
# Prompt
# ==========================================

prompt = "Explain what QLoRA is."


# ==========================================
# Tokenize
# ==========================================

inputs = tokenizer(

    prompt,

    return_tensors="pt"
)


# Move inputs to model device
inputs = inputs.to(
    model.device
)


# ==========================================
# Generate
# ==========================================

with torch.no_grad():

    output = model.generate(

        **inputs,

        max_new_tokens=200,

        temperature=0.7,

        do_sample=True
    )


# ==========================================
# Decode
# ==========================================

response = tokenizer.decode(

    output[0],

    skip_special_tokens=True
)


print(response)
```

---

# 11. Better inference code for an instruction model

For a chat/instruction model, use the model's chat template when available:

```python
import torch

messages = [

    {
        "role": "system",
        "content": "You are a helpful AI assistant."
    },

    {
        "role": "user",
        "content": "Explain QLoRA in simple terms."
    }
]


inputs = tokenizer.apply_chat_template(

    messages,

    add_generation_prompt=True,

    tokenize=True,

    return_tensors="pt",

    return_dict=True
)


inputs = inputs.to(
    model.device
)


with torch.no_grad():

    outputs = model.generate(

        **inputs,

        max_new_tokens=300,

        temperature=0.7,

        do_sample=True
    )


# Decode only generated tokens
generated_tokens = outputs[
    0,
    inputs["input_ids"].shape[1]:
]


response = tokenizer.decode(

    generated_tokens,

    skip_special_tokens=True
)


print(response)
```

This is generally safer than manually constructing model-specific chat markers.

---

# 12. How do you check memory usage?

Hugging Face models provide:

```python
print(
    model.get_memory_footprint()
)
```

Example:

```python
memory_bytes = model.get_memory_footprint()

print(
    f"Memory: "
    f"{memory_bytes / 1024**3:.2f} GB"
)
```

The Transformers documentation specifically recommends `get_memory_footprint()` for inspecting the memory footprint of a quantized model. ([Hugging Face][1])

---

# 13. Compare FP16 vs 8-bit vs 4-bit

Approximate model-weight memory for a 7B parameter model:

| Precision | Approximate raw weight memory |
| --------- | ----------------------------: |
| FP32      |                        ~28 GB |
| FP16/BF16 |                        ~14 GB |
| INT8      |                         ~7 GB |
| 4-bit     |                       ~3.5 GB |

These are simplified estimates:

$$
Memory \approx Parameters \times Bytes\ per\ parameter
$$

For FP16:

```python
7_000_000_000 * 2
```

≈ 14 GB.

For 4-bit:

```python
7_000_000_000 * 0.5
```

≈ 3.5 GB.

Actual runtime memory is higher because of metadata, activations, temporary buffers, KV cache, and other components.

---

# 14. Loading an 8-bit model

Complete code:

```python
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)


MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"


# 8-bit configuration
quantization_config = BitsAndBytesConfig(

    load_in_8bit=True
)


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


model = AutoModelForCausalLM.from_pretrained(

    MODEL_NAME,

    quantization_config=quantization_config,

    device_map="auto"
)
```

Conceptually:

```text
Model
 │
 ▼
FP16/BF16 weights
 │
 ▼
8-bit quantization
 │
 ▼
Smaller GPU memory footprint
```

8-bit and 4-bit loading are configured through `BitsAndBytesConfig`; the configuration options are mutually exclusive for a given load. ([Hugging Face][3])

---

# 15. QLoRA: loading a quantized model for training

This is extremely important.

For QLoRA:

```text
Step 1
Load base model in 4-bit
        ↓
Step 2
Prepare model for k-bit training
        ↓
Step 3
Add LoRA adapters
        ↓
Step 4
Train LoRA adapters
```

Code:

```python
import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)

from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model
)
```

---

## Step 1: Quantization configuration

```python
quantization_config = BitsAndBytesConfig(

    load_in_4bit=True,

    bnb_4bit_quant_type="nf4",

    bnb_4bit_use_double_quant=True,

    bnb_4bit_compute_dtype=torch.bfloat16
)
```

---

## Step 2: Load the base model

```python
model = AutoModelForCausalLM.from_pretrained(

    MODEL_NAME,

    quantization_config=quantization_config,

    device_map="auto"
)
```

Now:

```text
Base Model
    │
    ▼
4-bit Quantized
```

The base model is quantized.

---

## Step 3: Prepare for k-bit training

```python
model = prepare_model_for_kbit_training(
    model
)
```

This performs preprocessing needed for efficient training on a quantized model.

The PEFT documentation explicitly recommends preparing the quantized model with `prepare_model_for_kbit_training()` before adding/trainable adapters. ([Hugging Face][2])

---

## Step 4: Configure LoRA

```python
lora_config = LoraConfig(

    r=16,

    lora_alpha=32,

    lora_dropout=0.05,

    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj"
    ],

    bias="none",

    task_type="CAUSAL_LM"
)
```

---

## Step 5: Add LoRA adapters

```python
model = get_peft_model(

    model,

    lora_config
)
```

Now the architecture is:

```text
                 QLoRA

       Quantized Base Model
               │
               │
         Frozen 4-bit
               │
               ▼
        LoRA Adapters
               │
          FP16/BF16
               │
            Trainable
```

---

# 16. Complete QLoRA loading code

```python
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)


# ========================================
# Model
# ========================================

MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"


# ========================================
# Tokenizer
# ========================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


if tokenizer.pad_token is None:

    tokenizer.pad_token = tokenizer.eos_token


# ========================================
# 4-bit Configuration
# ========================================

quantization_config = BitsAndBytesConfig(

    load_in_4bit=True,

    bnb_4bit_quant_type="nf4",

    bnb_4bit_compute_dtype=torch.bfloat16,

    bnb_4bit_use_double_quant=True
)


# ========================================
# Load Quantized Base Model
# ========================================

model = AutoModelForCausalLM.from_pretrained(

    MODEL_NAME,

    quantization_config=quantization_config,

    device_map="auto"
)


# ========================================
# Prepare for QLoRA Training
# ========================================

model = prepare_model_for_kbit_training(
    model
)


# ========================================
# LoRA Configuration
# ========================================

lora_config = LoraConfig(

    r=16,

    lora_alpha=32,

    lora_dropout=0.05,

    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj"
    ],

    bias="none",

    task_type="CAUSAL_LM"
)


# ========================================
# Add LoRA Adapters
# ========================================

model = get_peft_model(

    model,

    lora_config
)


# ========================================
# Check Trainable Parameters
# ========================================

model.print_trainable_parameters()
```

Expected architecture:

```text
┌───────────────────────────────────┐
│         Llama Base Model          │
│                                   │
│        4-bit Quantized            │
│                                   │
│           Frozen ❄️               │
└───────────────────┬───────────────┘
                    │
                    ▼
          ┌─────────────────┐
          │  LoRA Adapter   │
          │                 │
          │  A Matrix       │
          │  B Matrix       │
          │                 │
          │   Trainable ✓   │
          └────────┬────────┘
                   │
                   ▼
                Output
```

---

# 17. Important difference: quantized inference vs QLoRA training

## Quantized inference

```python
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=quantization_config
)
```

Purpose:

```text
Reduce inference memory
```

---

## QLoRA training

```python
# 1. Load quantized model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=quantization_config
)

# 2. Prepare model
model = prepare_model_for_kbit_training(
    model
)

# 3. Add LoRA
model = get_peft_model(
    model,
    lora_config
)
```

Purpose:

```text
Quantized Base Model
        +
Trainable LoRA Adapters
        =
QLoRA Fine-Tuning
```

A key point is that 4-bit/8-bit base weights are not ordinarily fully trained directly in this workflow; training is performed on added parameters such as LoRA adapters. ([Hugging Face][1])

---

# 18. Common errors

## Error 1: `bitsandbytes` is missing

```text
ImportError:
bitsandbytes is required
```

Install:

```bash
pip install bitsandbytes
```

---

## Error 2: CUDA/GPU compatibility issue

Check:

```python
import torch

print(torch.cuda.is_available())

if torch.cuda.is_available():

    print(
        torch.cuda.get_device_name(0)
    )
```

Output:

```text
True
NVIDIA GPU ...
```

---

## Error 3: Out of memory

Try:

```python
quantization_config = BitsAndBytesConfig(

    load_in_4bit=True,

    bnb_4bit_quant_type="nf4",

    bnb_4bit_use_double_quant=True,

    bnb_4bit_compute_dtype=torch.bfloat16
)
```

Also reduce:

```text
Batch size
Sequence length
Max generated tokens
KV-cache usage when appropriate
```

---

# Interview-ready answer

> **To load a quantized Hugging Face model, I typically use Transformers with bitsandbytes. I create a `BitsAndBytesConfig` and pass it to `AutoModelForCausalLM.from_pretrained()`. For 8-bit inference, I use `load_in_8bit=True`. For 4-bit loading, especially for QLoRA, I commonly use `load_in_4bit=True`, NF4 quantization, optional double quantization, and BF16 as the compute dtype.**
>
> **For QLoRA training, I load the base model in 4-bit, prepare it with `prepare_model_for_kbit_training()`, attach LoRA adapters using PEFT, and train only the adapter parameters while the quantized base model remains frozen.**

### Most important code to remember

```python
quantization_config = BitsAndBytesConfig(

    load_in_4bit=True,

    bnb_4bit_quant_type="nf4",

    bnb_4bit_use_double_quant=True,

    bnb_4bit_compute_dtype=torch.bfloat16
)


model = AutoModelForCausalLM.from_pretrained(

    MODEL_NAME,

    quantization_config=quantization_config,

    device_map="auto"
)
```

Official references: [Transformers bitsandbytes quantization guide](https://huggingface.co/docs/transformers/quantization/bitsandbytes?utm_source=chatgpt.com) and [PEFT quantization guide](https://huggingface.co/docs/peft/developer_guides/quantization?utm_source=chatgpt.com).

[1]: https://huggingface.co/docs/transformers/quantization/bitsandbytes?utm_source=chatgpt.com "Bitsandbytes · Hugging Face"
[2]: https://huggingface.co/docs/peft/developer_guides/quantization?utm_source=chatgpt.com "Quantization · Hugging Face"
[3]: https://huggingface.co/docs/transformers/main_classes/quantization?utm_source=chatgpt.com "Quantization · Hugging Face"
