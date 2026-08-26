# Complete QLoRA Training Script

Below is a practical **QLoRA fine-tuning script** using:

* Hugging Face Transformers
* PEFT
* TRL
* BitsAndBytes
* 4-bit NF4 quantization
* Double quantization
* LoRA adapters
* Gradient checkpointing
* BF16
* Training/validation split

The key difference from normal LoRA is:

```text
Normal LoRA:
Base Model → BF16/FP16
             +
             LoRA adapters

QLoRA:
Base Model → 4-bit NF4
             +
             LoRA adapters → Trainable
```

---

# 1. Install dependencies

```bash
pip install -U torch transformers datasets accelerate peft trl bitsandbytes tensorboard
```

Check your environment:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

For NVIDIA GPU training, this should ideally print:

```text
True
```

---

# 2. Dataset format

Create `train.jsonl`:

```json
{"messages":[{"role":"system","content":"You are a helpful customer support assistant."},{"role":"user","content":"My payment was deducted twice."},{"role":"assistant","content":"I’m sorry about that. Please share the transaction ID so we can investigate the duplicate charge."}]}
{"messages":[{"role":"system","content":"You are a helpful customer support assistant."},{"role":"user","content":"I forgot my password."},{"role":"assistant","content":"You can reset your password using the Forgot Password option on the login screen."}]}
{"messages":[{"role":"system","content":"You are a helpful customer support assistant."},{"role":"user","content":"Where is my order?"},{"role":"assistant","content":"Please share your order ID and I can help check the current order status."}]}
```

Real projects should contain many more examples.

---

# 3. Complete QLoRA script

Save this as:

```text
train_qlora.py
```

```python
import os
import torch

from datasets import load_dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

from peft import (
    LoraConfig,
    prepare_model_for_kbit_training,
    get_peft_model,
)

from trl import (
    SFTTrainer,
    SFTConfig,
)


# =========================================================
# CONFIGURATION
# =========================================================

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

DATASET_PATH = "train.jsonl"

OUTPUT_DIR = "./qlora-output"

ADAPTER_PATH = "./customer-support-qlora"


# =========================================================
# GPU CHECK
# =========================================================

if not torch.cuda.is_available():
    raise RuntimeError(
        "QLoRA training requires a CUDA-capable GPU."
    )

print("GPU:", torch.cuda.get_device_name(0))


# =========================================================
# SELECT COMPUTE DTYPE
# =========================================================

# BF16 is preferred on supported GPUs.
if torch.cuda.is_bf16_supported():
    compute_dtype = torch.bfloat16
    use_bf16 = True
    use_fp16 = False
else:
    compute_dtype = torch.float16
    use_bf16 = False
    use_fp16 = True


print("Compute dtype:", compute_dtype)


# =========================================================
# LOAD DATASET
# =========================================================

dataset = load_dataset(
    "json",
    data_files=DATASET_PATH,
    split="train",
)


# =========================================================
# TRAIN / VALIDATION SPLIT
# =========================================================

dataset = dataset.train_test_split(
    test_size=0.1,
    seed=42,
)

train_dataset = dataset["train"]
eval_dataset = dataset["test"]


# =========================================================
# LOAD TOKENIZER
# =========================================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=False,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "right"


# =========================================================
# FORMAT CONVERSATIONAL DATA
# =========================================================

def format_example(example):

    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )

    return {
        "text": text
    }


train_dataset = train_dataset.map(
    format_example,
    remove_columns=train_dataset.column_names,
)

eval_dataset = eval_dataset.map(
    format_example,
    remove_columns=eval_dataset.column_names,
)


# =========================================================
# 4-BIT QUANTIZATION CONFIGURATION
# =========================================================

bnb_config = BitsAndBytesConfig(

    # Load base model in 4-bit
    load_in_4bit=True,

    # NF4 quantization
    bnb_4bit_quant_type="nf4",

    # Compute in BF16/FP16
    bnb_4bit_compute_dtype=compute_dtype,

    # Double quantization
    bnb_4bit_use_double_quant=True,
)


# =========================================================
# LOAD QUANTIZED BASE MODEL
# =========================================================

model = AutoModelForCausalLM.from_pretrained(

    MODEL_NAME,

    quantization_config=bnb_config,

    torch_dtype=compute_dtype,

    device_map="auto",
)


# =========================================================
# IMPORTANT FOR TRAINING
# =========================================================

# Disable KV cache because it conflicts with
# gradient checkpointing.
model.config.use_cache = False


# =========================================================
# PREPARE MODEL FOR K-BIT TRAINING
# =========================================================

model = prepare_model_for_kbit_training(
    model
)


# =========================================================
# ENABLE GRADIENT CHECKPOINTING
# =========================================================

model.gradient_checkpointing_enable()


# =========================================================
# CONFIGURE LoRA
# =========================================================

lora_config = LoraConfig(

    # Low-rank dimension
    r=16,

    # LoRA scaling
    lora_alpha=32,

    # Dropout
    lora_dropout=0.05,

    # Common transformer targets
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ],

    # Don't train biases
    bias="none",

    # Decoder-only LLM
    task_type="CAUSAL_LM",
)


# =========================================================
# ADD LoRA ADAPTERS
# =========================================================

model = get_peft_model(
    model,
    lora_config,
)


# =========================================================
# VERIFY TRAINABLE PARAMETERS
# =========================================================

model.print_trainable_parameters()


# =========================================================
# TRAINING CONFIGURATION
# =========================================================

training_args = SFTConfig(

    output_dir=OUTPUT_DIR,

    # -----------------------------------------------------
    # Training
    # -----------------------------------------------------

    num_train_epochs=3,

    per_device_train_batch_size=2,

    per_device_eval_batch_size=2,

    gradient_accumulation_steps=8,

    # -----------------------------------------------------
    # Optimizer
    # -----------------------------------------------------

    learning_rate=2e-4,

    optim="paged_adamw_8bit",

    weight_decay=0.01,

    # -----------------------------------------------------
    # Learning-rate scheduler
    # -----------------------------------------------------

    lr_scheduler_type="cosine",

    warmup_ratio=0.03,

    # -----------------------------------------------------
    # Precision
    # -----------------------------------------------------

    bf16=use_bf16,

    fp16=use_fp16,

    # -----------------------------------------------------
    # Memory
    # -----------------------------------------------------

    gradient_checkpointing=True,

    # -----------------------------------------------------
    # Sequence length
    # -----------------------------------------------------

    max_length=2048,

    # -----------------------------------------------------
    # Logging
    # -----------------------------------------------------

    logging_strategy="steps",

    logging_steps=10,

    # -----------------------------------------------------
    # Evaluation
    # -----------------------------------------------------

    eval_strategy="steps",

    eval_steps=100,

    # -----------------------------------------------------
    # Checkpoints
    # -----------------------------------------------------

    save_strategy="steps",

    save_steps=100,

    save_total_limit=2,

    # -----------------------------------------------------
    # Best model
    # -----------------------------------------------------

    load_best_model_at_end=True,

    metric_for_best_model="eval_loss",

    greater_is_better=False,

    # -----------------------------------------------------
    # Logging tool
    # -----------------------------------------------------

    report_to="tensorboard",

    # -----------------------------------------------------
    # Reproducibility
    # -----------------------------------------------------

    seed=42,
)


# =========================================================
# CREATE SFT TRAINER
# =========================================================

trainer = SFTTrainer(

    model=model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=eval_dataset,

    processing_class=tokenizer,
)


# =========================================================
# TRAIN
# =========================================================

print("\nStarting QLoRA training...\n")

trainer.train()


# =========================================================
# SAVE QLoRA ADAPTER
# =========================================================

trainer.model.save_pretrained(
    ADAPTER_PATH
)


# =========================================================
# SAVE TOKENIZER
# =========================================================

tokenizer.save_pretrained(
    ADAPTER_PATH
)


print(
    f"\nQLoRA training completed!\n"
    f"Adapter saved to: {ADAPTER_PATH}"
)
```

---

# 4. What happens at every stage?

## Stage 1: Load the base model in 4-bit

```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
```

Architecture:

```text
Original Model

FP16 / BF16 weights
        ↓
4-bit NF4 quantization
        ↓
Quantized Base Model
```

The base model weights are not directly updated.

---

# 5. Prepare the model for k-bit training

```python
model = prepare_model_for_kbit_training(
    model
)
```

Conceptually:

```text
4-bit Model
    │
    ▼
Freeze Base Weights
    │
    ▼
Prepare for Stable Training
```

This preparation handles important training details required when combining low-bit quantized models with PEFT training.

---

# 6. Add LoRA

```python
model = get_peft_model(
    model,
    lora_config
)
```

Now:

```text
                    Transformer

                 Quantized Layer
                       │
             ┌─────────┴─────────┐
             │                   │
             ▼                   ▼
      4-bit Base Weight        LoRA
         Frozen               Trainable
```

Mathematically:

$$
W_{effective}
=
W_{quantized}
+
\frac{\alpha}{r}BA
$$

Only:

$$
A, B
$$

are trained.

---

# 7. Why `paged_adamw_8bit`?

```python
optim="paged_adamw_8bit"
```

A normal optimizer can consume significant GPU memory because it stores optimizer state.

Conceptually:

```text
Normal AdamW

Parameters
   +
Gradient
   +
Momentum
   +
Variance
```

For very large models, optimizer memory can become a problem.

Paged optimizers help manage memory pressure and are commonly used in QLoRA-style setups.

---

# 8. Why gradient checkpointing?

```python
gradient_checkpointing=True
```

Normally:

```text
Forward Pass

Layer 1 → Save activations
Layer 2 → Save activations
Layer 3 → Save activations
Layer 4 → Save activations

High memory usage
```

With checkpointing:

```text
Forward Pass

Layer 1 → Don't save everything
Layer 2 → Recompute later
Layer 3 → Recompute later

Lower GPU memory
More computation
```

Tradeoff:

```text
GPU Memory ↓
Training Time ↑
```

---

# 9. Why `use_cache=False`?

```python
model.config.use_cache = False
```

The KV cache is useful for generation:

```text
Inference
    ↓
Reuse previous key/value tensors
    ↓
Faster generation
```

But during training with gradient checkpointing:

```text
KV Cache
+
Gradient Checkpointing
```

can cause incompatibilities or unnecessary memory usage.

Therefore:

```text
Training → use_cache=False
Inference → use_cache=True
```

---

# 10. Why `r=16` and `alpha=32`?

```python
r=16
lora_alpha=32
```

The scaling factor is:

$$
\frac{\alpha}{r}
=
\frac{32}{16}
=
2
$$

So:

$$
\Delta W
=
2BA
$$

Architecture:

```text
Input Dimension = 4096

      A
4096 → 16

      B
16 → 4096
```

Instead of training:

$$
4096 \times 4096
=
16,777,216
$$

parameters, LoRA approximately trains:

$$
4096 \times 16
+
16 \times 4096
=
131,072
$$

parameters for that projection.

---

# 11. Why the base model is frozen

During QLoRA:

```text
Base Model

4-bit weights
     ↓
Frozen ❄️
```

Only:

```text
LoRA A
LoRA B
```

are updated:

```text
LoRA Adapter

A → Trainable ✓
B → Trainable ✓
```

Therefore:

```text
Gradient
   ↓
LoRA parameters
   ↓
Optimizer
   ↓
Update A and B
```

The quantized base model remains unchanged.

---

# 12. Expected output

You should see something similar to:

```text
GPU: NVIDIA ...

Compute dtype: torch.bfloat16

trainable params:
8,388,608

all params:
1,500,000,000

trainable%:
0.55%

Starting QLoRA training...

{'loss': 2.34}
{'loss': 1.85}
{'loss': 1.42}

QLoRA training completed!
```

Exact values depend on the model and dataset.

---

# 13. Monitor TensorBoard

Start:

```bash
tensorboard --logdir ./qlora-output
```

You can monitor:

```text
Training Loss
Validation Loss
Learning Rate
Gradient Norm
```

Healthy training:

```text
Train Loss ↓
Eval Loss  ↓
```

Overfitting:

```text
Train Loss ↓
Eval Loss  ↑
```

---

# 14. How to resume training

If training crashes, you can resume from a checkpoint.

```python
trainer.train(
    resume_from_checkpoint=True
)
```

Or:

```python
trainer.train(
    resume_from_checkpoint="./qlora-output/checkpoint-500"
)
```

This is important for long-running training jobs.

---

# 15. How to load the QLoRA adapter for inference

After training, you have:

```text
Base Model
     +
QLoRA Adapter
```

Load it like this:

```python
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

from peft import PeftModel


MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

ADAPTER_PATH = "./customer-support-qlora"


# -----------------------------------------
# Quantization configuration
# -----------------------------------------

bnb_config = BitsAndBytesConfig(

    load_in_4bit=True,

    bnb_4bit_quant_type="nf4",

    bnb_4bit_compute_dtype=torch.bfloat16,

    bnb_4bit_use_double_quant=True,
)


# -----------------------------------------
# Load tokenizer
# -----------------------------------------

tokenizer = AutoTokenizer.from_pretrained(
    ADAPTER_PATH
)


# -----------------------------------------
# Load quantized base model
# -----------------------------------------

base_model = AutoModelForCausalLM.from_pretrained(

    MODEL_NAME,

    quantization_config=bnb_config,

    device_map="auto",
)


# -----------------------------------------
# Attach adapter
# -----------------------------------------

model = PeftModel.from_pretrained(

    base_model,

    ADAPTER_PATH
)


model.eval()


# -----------------------------------------
# Prompt
# -----------------------------------------

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
            "My payment was deducted twice."
        )
    }
]


# -----------------------------------------
# Format prompt
# -----------------------------------------

inputs = tokenizer.apply_chat_template(

    messages,

    tokenize=True,

    add_generation_prompt=True,

    return_tensors="pt",

    return_dict=True,
)


inputs = inputs.to(
    model.device
)


# -----------------------------------------
# Generate
# -----------------------------------------

with torch.inference_mode():

    outputs = model.generate(

        **inputs,

        max_new_tokens=200,

        temperature=0.7,

        do_sample=True,

        pad_token_id=tokenizer.eos_token_id,
    )


# -----------------------------------------
# Decode generated tokens only
# -----------------------------------------

input_length = inputs[
    "input_ids"
].shape[1]


generated_tokens = outputs[
    0,
    input_length:
]


response = tokenizer.decode(

    generated_tokens,

    skip_special_tokens=True,
)


print(response)
```

---

# 16. Production improvements

For a serious project, I would add:

```text
qlora_project/
│
├── configs/
│   └── train.yaml
│
├── data/
│   ├── train.jsonl
│   └── validation.jsonl
│
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── model.py
│   ├── trainer.py
│   └── evaluation.py
│
├── train.py
├── inference.py
├── requirements.txt
└── README.md
```

Also add:

```text
✓ MLflow / W&B experiment tracking
✓ Dataset versioning
✓ Checkpoint recovery
✓ Validation metrics
✓ Early stopping
✓ Gradient clipping
✓ Logging
✓ Reproducibility
✓ Automated evaluation
```

---

# Interview-ready answer

> **To train with QLoRA, I load the pretrained base model using 4-bit quantization with `BitsAndBytesConfig`, typically using NF4 and optional double quantization. The base model remains frozen. I then call `prepare_model_for_kbit_training()` and attach LoRA adapters to selected transformer layers such as `q_proj`, `k_proj`, `v_proj`, and `o_proj`.**
>
> **The LoRA adapters are trained in higher precision while the base model stays quantized, which dramatically reduces GPU memory. I typically use gradient checkpointing and a memory-efficient optimizer such as `paged_adamw_8bit`. After training, I save only the adapter weights and load them together with the original compatible base model during inference.**

The complete flow is:

```text
Dataset
   ↓
4-bit NF4 Base Model
   ↓
prepare_model_for_kbit_training()
   ↓
Add LoRA Adapters
   ↓
SFTTrainer
   ↓
Train Only Adapters
   ↓
Save QLoRA Adapter
```
