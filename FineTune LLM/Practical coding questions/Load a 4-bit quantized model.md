# Load a 4-bit Quantized Model for QLoRA

In QLoRA, the workflow is:

```text
Base Model (FP16/BF16)
        ↓
Load in 4-bit NF4
        ↓
Base model stays frozen
        ↓
Attach LoRA adapters
        ↓
Train only LoRA adapters
```

The base model is stored in **4-bit quantized form**, while LoRA adapters are trainable at higher precision.

---

## 1. Install required packages

```bash
pip install -U torch transformers accelerate bitsandbytes peft trl datasets
```

`bitsandbytes` provides 4-bit quantization support.

---

# 2. Import required libraries

```python
import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
```

---

# 3. Configure 4-bit quantization

The key object is:

```python
BitsAndBytesConfig
```

Example:

```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)
```

Let's understand each option.

---

# 4. `load_in_4bit=True`

```python
load_in_4bit=True
```

This tells Transformers:

```text
Load model weights using 4-bit quantization
```

Conceptually:

```text
Original model:

FP16

16 bits per weight

        ↓

4-bit quantized

4 bits per weight
```

Approximate memory comparison:

```text
FP32 = 4 bytes/parameter
FP16 = 2 bytes/parameter
INT8  = 1 byte/parameter
4-bit = 0.5 bytes/parameter
```

Ignoring quantization metadata and runtime overhead:

```text
7B model

FP16 ≈ 14 GB

4-bit ≈ 3.5 GB
```

In practice, actual memory usage is higher because of quantization metadata, activations, CUDA/runtime memory, and other components.

---

# 5. `bnb_4bit_quant_type="nf4"`

```python
bnb_4bit_quant_type="nf4"
```

NF4 means:

```text
NormalFloat 4-bit
```

It is designed for weights that approximately follow a normal distribution.

For QLoRA, a common configuration is:

```python
bnb_4bit_quant_type="nf4"
```

---

# 6. `bnb_4bit_compute_dtype`

The weights may be stored in 4-bit, but computations need a higher precision.

```python
bnb_4bit_compute_dtype=torch.bfloat16
```

Conceptually:

```text
Stored weights:

4-bit
   ↓
Dequantized during computation
   ↓
BF16 computation
```

So:

```text
Storage precision ≠ Compute precision
```

This is very important.

### Typical configuration

```python
torch.bfloat16
```

if your GPU supports BF16.

Otherwise:

```python
torch.float16
```

A safe detection pattern:

```python
import torch


if torch.cuda.is_available() and torch.cuda.is_bf16_supported():

    compute_dtype = torch.bfloat16

else:

    compute_dtype = torch.float16
```

Then:

```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=True
)
```

---

# 7. `bnb_4bit_use_double_quant=True`

```python
bnb_4bit_use_double_quant=True
```

This enables **double quantization**.

Normal quantization:

```text
Original weights
      ↓
Quantized weights
```

Double quantization:

```text
Original weights
      ↓
Quantized weights
      +
Quantization constants
      ↓
Quantize some quantization metadata
```

This further reduces memory usage.

For QLoRA, a common setting is:

```python
bnb_4bit_use_double_quant=True
```

---

# 8. Load the tokenizer

Always load the tokenizer associated with the base model.

Example:

```python
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=True
)
```

For many causal language models, you may need a padding token:

```python
if tokenizer.pad_token is None:

    tokenizer.pad_token = tokenizer.eos_token
```

---

# 9. Load the model in 4-bit

```python
model = AutoModelForCausalLM.from_pretrained(

    MODEL_NAME,

    quantization_config=bnb_config,

    device_map="auto",

    torch_dtype=compute_dtype
)
```

Complete example:

```python
import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)


# =====================================================
# MODEL
# =====================================================

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


# =====================================================
# COMPUTE PRECISION
# =====================================================

if torch.cuda.is_available() and torch.cuda.is_bf16_supported():

    compute_dtype = torch.bfloat16

else:

    compute_dtype = torch.float16


# =====================================================
# 4-BIT CONFIGURATION
# =====================================================

bnb_config = BitsAndBytesConfig(

    load_in_4bit=True,

    bnb_4bit_quant_type="nf4",

    bnb_4bit_compute_dtype=compute_dtype,

    bnb_4bit_use_double_quant=True
)


# =====================================================
# TOKENIZER
# =====================================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=True
)


if tokenizer.pad_token is None:

    tokenizer.pad_token = tokenizer.eos_token


# =====================================================
# LOAD 4-BIT MODEL
# =====================================================

model = AutoModelForCausalLM.from_pretrained(

    MODEL_NAME,

    quantization_config=bnb_config,

    device_map="auto"
)


print("Model loaded successfully")
```

---

# 10. What does `device_map="auto"` do?

```python
device_map="auto"
```

Transformers automatically decides where to place model components.

For example:

```text
GPU available:

Transformer Layer 0 → GPU
Transformer Layer 1 → GPU
Transformer Layer 2 → GPU
```

If memory is insufficient, depending on the environment and offloading configuration, parts may be distributed across available devices.

For explicit single-GPU training, you may instead control placement more carefully, especially in distributed training setups.

---

# 11. Verify the model is quantized

Inspect modules:

```python
for name, module in model.named_modules():

    if "Linear4bit" in str(type(module)):

        print(name)
```

You may see:

```text
model.layers.0.self_attn.q_proj
model.layers.0.self_attn.k_proj
model.layers.0.self_attn.v_proj
model.layers.0.self_attn.o_proj
```

Depending on the architecture and library version.

You can also check:

```python
print(model)
```

Some linear layers may appear conceptually as:

```text
Linear4bit(...)
```

---

# 12. Prepare the model for QLoRA training

Loading in 4-bit is not the entire QLoRA setup.

Next:

```python
from peft import prepare_model_for_kbit_training

model = prepare_model_for_kbit_training(
    model
)
```

This prepares the model for k-bit training.

Conceptually:

```text
4-bit Base Model
       │
       ▼
prepare_model_for_kbit_training()
       │
       ▼
Prepare model for adapter training
       │
       ▼
Attach LoRA adapters
```

Then:

```python
from peft import (
    LoraConfig,
    get_peft_model
)


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


model = get_peft_model(
    model,
    lora_config
)
```

---

# 13. Complete QLoRA loading pipeline

```python
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)

from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model
)


MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


# -----------------------------------------------------
# Select compute dtype
# -----------------------------------------------------

compute_dtype = (
    torch.bfloat16
    if torch.cuda.is_available()
    and torch.cuda.is_bf16_supported()
    else torch.float16
)


# -----------------------------------------------------
# Configure 4-bit quantization
# -----------------------------------------------------

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=True
)


# -----------------------------------------------------
# Load tokenizer
# -----------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# -----------------------------------------------------
# Load quantized base model
# -----------------------------------------------------

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto"
)


# -----------------------------------------------------
# Prepare model for k-bit training
# -----------------------------------------------------

model = prepare_model_for_kbit_training(
    model
)


# -----------------------------------------------------
# Configure LoRA
# -----------------------------------------------------

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


# -----------------------------------------------------
# Attach LoRA adapters
# -----------------------------------------------------

model = get_peft_model(
    model,
    lora_config
)


# -----------------------------------------------------
# Verify trainable parameters
# -----------------------------------------------------

model.print_trainable_parameters()
```

---

# 14. What is frozen and what is trainable?

After QLoRA setup:

```text
                    QLoRA Model

┌─────────────────────────────────────┐
│ Base Model                          │
│                                     │
│ 4-bit NF4 quantized                 │
│                                     │
│ q_proj  ───── FROZEN               │
│ k_proj  ───── FROZEN               │
│ v_proj  ───── FROZEN               │
│ o_proj  ───── FROZEN               │
└───────────────────┬─────────────────┘
                    │
                    ▼
┌─────────────────────────────────────┐
│ LoRA Adapters                       │
│                                     │
│ A matrix ─── TRAINABLE             │
│ B matrix ─── TRAINABLE             │
│                                     │
│ FP16/BF16 training                 │
└─────────────────────────────────────┘
```

The important idea is:

```text
Base model:
4-bit
Frozen

LoRA adapters:
Higher precision
Trainable
```

The 4-bit base weights themselves are **not directly updated** during QLoRA training.

---

# 15. Common errors

## Error: bitsandbytes not installed

```text
PackageNotFoundError
```

Install:

```bash
pip install -U bitsandbytes
```

---

## Error: CUDA unavailable

Check:

```python
import torch

print(torch.cuda.is_available())

if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
```

4-bit `bitsandbytes` workflows are generally intended for supported hardware backends; exact support depends on your installed versions and platform.

---

## Error: CUDA out of memory

Reduce:

```python
per_device_train_batch_size=1
```

Use:

```python
gradient_accumulation_steps=16
```

Enable:

```python
gradient_checkpointing=True
```

Reduce sequence length:

```python
max_length=1024
```

And keep:

```python
load_in_4bit=True
```

---

# Interview answer

> **To load a model for QLoRA, I use Hugging Face `BitsAndBytesConfig` with `load_in_4bit=True`. I typically use NF4 quantization because it is designed for normally distributed neural network weights, enable double quantization to reduce memory further, and use BF16 or FP16 as the compute dtype. The base model is loaded in 4-bit and kept frozen. I then call `prepare_model_for_kbit_training()` and attach trainable LoRA adapters using PEFT. During training, only the LoRA adapter parameters are updated, while the quantized base model remains unchanged.**

The essential configuration is:

```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto"
)
```

This gives you the **4-bit frozen base model** required for the next stage: attaching LoRA adapters and starting QLoRA fine-tuning.
