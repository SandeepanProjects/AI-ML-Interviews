# Save and Reload LoRA Adapters

After LoRA or QLoRA training, you usually **do not save the entire base model**.

You save only the trained adapter weights.

The architecture is:

```text
Base Model
   +
LoRA Adapter
   ↓
Fine-tuning
   ↓
Save Adapter Only
```

This is one of the biggest advantages of LoRA.

---

# 1. What gets saved?

Suppose the base model is:

```text
Qwen / Llama Base Model
≈ several GB
```

Your LoRA adapter may contain only a small number of trainable parameters:

```text
Base Model
    ↓ Frozen

LoRA A + B matrices
    ↓ Trainable

Save only:
✓ adapter weights
✓ adapter configuration
```

Typically:

```text
adapter/
├── adapter_config.json
├── adapter_model.safetensors
└── tokenizer files (optional)
```

---

# 2. Save LoRA adapters

After training:

```python
trainer.train()
```

Save the PEFT model:

```python
OUTPUT_DIR = "./my_lora_adapter"

trainer.model.save_pretrained(
    OUTPUT_DIR
)

tokenizer.save_pretrained(
    OUTPUT_DIR
)
```

Or:

```python
model.save_pretrained(
    "./my_lora_adapter"
)
```

If `model` is a PEFT model, this saves the adapter rather than a full copy of the base model.

---

# 3. Complete training + save example

```python
# Train
trainer.train()


# Save LoRA adapter
ADAPTER_DIR = "./outputs/customer-support-lora"

trainer.model.save_pretrained(
    ADAPTER_DIR
)


# Save tokenizer
tokenizer.save_pretrained(
    ADAPTER_DIR
)

print("LoRA adapter saved successfully")
```

Result:

```text
customer-support-lora/
├── adapter_config.json
├── adapter_model.safetensors
├── tokenizer.json
├── tokenizer_config.json
└── special_tokens_map.json
```

The exact files can vary depending on library versions and serialization settings.

---

# 4. Why save the tokenizer?

The tokenizer may include:

```text
Special tokens
Chat template
Padding configuration
```

For example:

```text
<system>
<user>
<assistant>
```

Saving the tokenizer ensures inference uses compatible preprocessing.

---

# 5. Reload LoRA adapters for inference

To use the trained adapter, you need:

```text
1. Base model
2. LoRA adapter
```

The flow is:

```text
Base Model
    ↓
Load
    +
LoRA Adapter
    ↓
Attach
    ↓
Inference
```

---

# 6. Load the base model

For normal LoRA:

```python
import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)


BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

ADAPTER_DIR = "./outputs/customer-support-lora"


tokenizer = AutoTokenizer.from_pretrained(
    ADAPTER_DIR
)


base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
```

---

# 7. Load the LoRA adapter

```python
from peft import PeftModel


model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_DIR
)
```

Now:

```text
Base Model
+
Trained LoRA Adapter
=
Fine-tuned Model
```

Put the model into evaluation mode:

```python
model.eval()
```

---

# 8. Run inference

For a simple prompt:

```python
prompt = "My payment was deducted twice. What should I do?"

inputs = tokenizer(
    prompt,
    return_tensors="pt"
)

inputs = {
    key: value.to(model.device)
    for key, value in inputs.items()
}
```

Generate:

```python
with torch.inference_mode():

    output = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7,
        do_sample=True
    )
```

Decode:

```python
response = tokenizer.decode(
    output[0],
    skip_special_tokens=True
)

print(response)
```

---

# 9. Correct chat inference

For an instruction/chat model, use the model's chat template.

```python
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
            "My payment was deducted twice. "
            "What should I do?"
        )
    }
]
```

Apply the chat template:

```python
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True
)

inputs = {
    key: value.to(model.device)
    for key, value in inputs.items()
}
```

Generate:

```python
with torch.inference_mode():

    output = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7,
        do_sample=True
    )
```

Important: `generate()` returns:

```text
Prompt tokens
+
Generated tokens
```

To decode only the new response:

```python
input_length = inputs["input_ids"].shape[1]

generated_tokens = output[
    0,
    input_length:
]

response = tokenizer.decode(
    generated_tokens,
    skip_special_tokens=True
)

print(response)
```

This avoids printing the entire prompt again.

---

# 10. Full reload and inference example

```python
import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)

from peft import PeftModel


# ==============================================
# CONFIGURATION
# ==============================================

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

ADAPTER_DIR = "./outputs/customer-support-lora"


# ==============================================
# LOAD TOKENIZER
# ==============================================

tokenizer = AutoTokenizer.from_pretrained(
    ADAPTER_DIR
)


if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# ==============================================
# LOAD BASE MODEL
# ==============================================

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)


# ==============================================
# LOAD LORA ADAPTER
# ==============================================

model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_DIR
)


# ==============================================
# INFERENCE MODE
# ==============================================

model.eval()


# ==============================================
# CHAT INPUT
# ==============================================

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
            "My payment was deducted twice. "
            "What should I do?"
        )
    }
]


# ==============================================
# APPLY CHAT TEMPLATE
# ==============================================

inputs = tokenizer.apply_chat_template(

    messages,

    tokenize=True,

    add_generation_prompt=True,

    return_tensors="pt",

    return_dict=True
)


inputs = {

    key: value.to(model.device)

    for key, value in inputs.items()
}


# ==============================================
# GENERATE RESPONSE
# ==============================================

with torch.inference_mode():

    outputs = model.generate(

        **inputs,

        max_new_tokens=200,

        temperature=0.7,

        do_sample=True,

        pad_token_id=tokenizer.eos_token_id
    )


# ==============================================
# EXTRACT GENERATED TOKENS
# ==============================================

input_length = inputs[
    "input_ids"
].shape[1]


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

# 11. Reload a QLoRA adapter

For QLoRA, the adapter is saved similarly:

```python
model.save_pretrained(
    "./qlora_adapter"
)
```

But when reloading, the base model should generally be loaded with the appropriate quantization configuration if you want to preserve the memory-efficient QLoRA-style setup.

```python
import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)

from peft import PeftModel


BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

ADAPTER_DIR = "./qlora_adapter"


# ==============================================
# 4-BIT CONFIGURATION
# ==============================================

compute_dtype = (
    torch.bfloat16
    if torch.cuda.is_available()
    and torch.cuda.is_bf16_supported()
    else torch.float16
)


bnb_config = BitsAndBytesConfig(

    load_in_4bit=True,

    bnb_4bit_quant_type="nf4",

    bnb_4bit_compute_dtype=compute_dtype,

    bnb_4bit_use_double_quant=True
)


# ==============================================
# LOAD TOKENIZER
# ==============================================

tokenizer = AutoTokenizer.from_pretrained(
    ADAPTER_DIR
)


if tokenizer.pad_token is None:

    tokenizer.pad_token = tokenizer.eos_token


# ==============================================
# LOAD 4-BIT BASE MODEL
# ==============================================

base_model = AutoModelForCausalLM.from_pretrained(

    BASE_MODEL,

    quantization_config=bnb_config,

    device_map="auto"
)


# ==============================================
# ATTACH QLORA ADAPTER
# ==============================================

model = PeftModel.from_pretrained(

    base_model,

    ADAPTER_DIR
)


model.eval()
```

The architecture is:

```text
                Inference

        Base Model
              │
              │ 4-bit NF4
              ▼
      Frozen Quantized Weights
              +
              │
              ▼
         LoRA Adapter
              │
              │ trained weights
              ▼
          Final Output
```

---

# 12. Load directly using `AutoPeftModelForCausalLM`

PEFT can also load an adapter model directly when the adapter configuration contains the base model reference.

```python
from peft import AutoPeftModelForCausalLM

model = AutoPeftModelForCausalLM.from_pretrained(
    ADAPTER_DIR,
    device_map="auto"
)
```

Then:

```python
model.eval()
```

However, for QLoRA or specific memory requirements, explicitly loading the base model with your desired `BitsAndBytesConfig` gives you more control.

---

# 13. Multiple adapters

One major LoRA advantage is that one base model can support multiple adapters.

```text
                Base Model
                    │
        ┌───────────┼───────────┐
        │           │           │
        ▼           ▼           ▼

 Customer       Finance       Legal
 Support        Adapter       Adapter
 Adapter
```

Example:

```python
from peft import PeftModel


model = PeftModel.from_pretrained(
    base_model,
    "./customer_support_adapter",
    adapter_name="customer_support"
)
```

Load another:

```python
model.load_adapter(
    "./finance_adapter",
    adapter_name="finance"
)
```

Select an adapter:

```python
model.set_adapter(
    "customer_support"
)
```

Switch:

```python
model.set_adapter(
    "finance"
)
```

This is useful in enterprise systems.

---

# 14. How to inspect adapter information

```python
print(model.peft_config)
```

You can inspect:

```text
LoRA rank
LoRA alpha
Target modules
Dropout
Base model
Task type
```

You can also inspect trainable parameters:

```python
model.print_trainable_parameters()
```

After loading for inference, you may see only the adapter parameters as the PEFT-specific trainable components depending on inference/training configuration.

---

# 15. Important: adapter path vs base model path

A common mistake is:

```python
AutoModelForCausalLM.from_pretrained(
    "./my_lora_adapter"
)
```

This may not be the correct loading path because an adapter directory generally contains:

```text
adapter_config.json
adapter_model.safetensors
```

but not necessarily the full base model weights.

Correct:

```python
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL
)

model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_DIR
)
```

---

# 16. Resume LoRA training from a saved adapter

If you want to continue training:

```python
from peft import PeftModel


model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_DIR,
    is_trainable=True
)
```

The important part is:

```python
is_trainable=True
```

Then verify:

```python
model.print_trainable_parameters()
```

Without explicitly loading it as trainable, the adapter may be configured for inference rather than continued training.

---

# 17. Checkpoint saving vs final adapter saving

During training:

```text
output/
├── checkpoint-100/
├── checkpoint-200/
└── checkpoint-300/
```

The trainer checkpoints can contain information needed to resume training, such as:

```text
Adapter weights
Optimizer state
Scheduler state
Trainer state
```

The final model save:

```python
trainer.save_model(
    "./final_adapter"
)
```

is primarily what you deploy for inference.

A good workflow:

```python
trainer.train()

trainer.save_model(
    "./final_adapter"
)

tokenizer.save_pretrained(
    "./final_adapter"
)
```

---

# Interview answer

> **With LoRA or QLoRA, I usually save only the adapter weights rather than the entire base model. After training, I call `model.save_pretrained()` or `trainer.save_model()`, which stores the PEFT adapter weights and adapter configuration. I also save the tokenizer. During inference, I load the original base model and attach the adapter using `PeftModel.from_pretrained()`.**
>
> **For QLoRA, I normally reload the base model with the same 4-bit quantization configuration when memory efficiency is important, then attach the saved adapter. This allows one large base model to be reused with many small task-specific adapters.**

## Minimal version

### Save

```python
model.save_pretrained("./adapter")
tokenizer.save_pretrained("./adapter")
```

### Reload

```python
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME
)

model = PeftModel.from_pretrained(
    base_model,
    "./adapter"
)

model.eval()
```

### QLoRA reload

```python
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto"
)

model = PeftModel.from_pretrained(
    base_model,
    "./adapter"
)
```

This is the standard **save → deploy → reload** lifecycle for LoRA and QLoRA adapters.
