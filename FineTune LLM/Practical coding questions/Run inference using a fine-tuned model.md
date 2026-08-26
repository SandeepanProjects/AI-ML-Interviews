# Run Inference Using a Fine-Tuned LoRA / QLoRA Model

After fine-tuning, your model consists of:

```text
Base Model
    +
Fine-tuned LoRA Adapter
    =
Fine-tuned Model
```

For a QLoRA model:

```text
4-bit Quantized Base Model
        +
Trained LoRA Adapter
        =
Fine-tuned Model
```

The inference process is:

```text
User Input
    ↓
Tokenizer / Chat Template
    ↓
Base Model + LoRA Adapter
    ↓
Token Generation
    ↓
Decode Tokens
    ↓
Response
```

---

# 1. Complete inference with a LoRA fine-tuned model

Assume:

```text
Base model:
Qwen/Qwen2.5-1.5B-Instruct

Fine-tuned adapter:
./outputs/customer-support-lora
```

## Code

```python
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

from peft import PeftModel


# =========================================================
# CONFIGURATION
# =========================================================

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

ADAPTER_PATH = "./outputs/customer-support-lora"


# =========================================================
# LOAD TOKENIZER
# =========================================================

tokenizer = AutoTokenizer.from_pretrained(
    ADAPTER_PATH
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# =========================================================
# LOAD BASE MODEL
# =========================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

dtype = (
    torch.bfloat16
    if torch.cuda.is_available()
    and torch.cuda.is_bf16_supported()
    else torch.float16
    if torch.cuda.is_available()
    else torch.float32
)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=dtype,
    device_map="auto" if device == "cuda" else None
)

if device == "cpu":
    base_model = base_model.to(device)


# =========================================================
# LOAD LORA ADAPTER
# =========================================================

model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_PATH
)

model.eval()

print("Fine-tuned model loaded successfully")
```

---

# 2. Run a simple prompt

```python
prompt = """
My payment was deducted twice.

What should I do?
"""
```

Tokenize:

```python
inputs = tokenizer(
    prompt,
    return_tensors="pt"
)

# Move input tensors to the model input device
input_device = model.get_input_embeddings().weight.device

inputs = {
    key: value.to(input_device)
    for key, value in inputs.items()
}
```

Generate:

```python
with torch.inference_mode():

    output = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
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

# 3. Why use `torch.inference_mode()`?

```python
with torch.inference_mode():
```

During training, PyTorch stores information for:

```text
Forward Pass
    ↓
Save activations
    ↓
Backward Pass
    ↓
Update weights
```

During inference, there is no backward pass.

So:

```python
with torch.inference_mode():
    output = model.generate(...)
```

results in:

```text
No gradients
↓
Less memory
↓
Faster inference
```

---

# 4. Use the chat template — recommended for instruction models

If you fine-tuned an instruct/chat model, do not manually invent the prompt format unless your training data used that exact format.

Use:

```python
tokenizer.apply_chat_template()
```

Example:

```python
messages = [
    {
        "role": "system",
        "content": (
            "You are a helpful customer support assistant. "
            "Be polite, concise, and helpful."
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

Convert to tokens:

```python
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True
)
```

Move to the input device:

```python
input_device = model.get_input_embeddings().weight.device

inputs = {
    key: value.to(input_device)
    for key, value in inputs.items()
}
```

Generate:

```python
with torch.inference_mode():

    output = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id
    )
```

---

# 5. Decode only the generated response

This is important.

`model.generate()` returns:

```text
Input tokens
+
Generated tokens
```

Suppose:

```text
Input:
Hello, how can I help?

Generated:
Hello, how can I help?
Sure! What do you need?
```

Usually, we want only:

```text
Sure! What do you need?
```

Code:

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

---

# 6. Complete chat inference function

```python
import torch


def generate_response(
    model,
    tokenizer,
    messages,
    max_new_tokens=256,
    temperature=0.7,
    top_p=0.9
):

    # -------------------------------------------------
    # Apply model-specific chat template
    # -------------------------------------------------

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True
    )

    # -------------------------------------------------
    # Move input to model device
    # -------------------------------------------------

    input_device = model.get_input_embeddings().weight.device

    inputs = {
        key: value.to(input_device)
        for key, value in inputs.items()
    }

    # -------------------------------------------------
    # Generate
    # -------------------------------------------------

    with torch.inference_mode():

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id
        )

    # -------------------------------------------------
    # Extract only generated tokens
    # -------------------------------------------------

    input_length = inputs["input_ids"].shape[1]

    generated_tokens = outputs[
        0,
        input_length:
    ]

    # -------------------------------------------------
    # Decode
    # -------------------------------------------------

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    )

    return response
```

Use it:

```python
messages = [
    {
        "role": "system",
        "content": (
            "You are a helpful customer-support assistant."
        )
    },
    {
        "role": "user",
        "content": (
            "My payment was deducted twice. "
            "How can I get my money back?"
        )
    }
]


response = generate_response(
    model=model,
    tokenizer=tokenizer,
    messages=messages
)

print(response)
```

---

# 7. QLoRA inference

For QLoRA, load the base model in 4-bit.

```python
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)

from peft import PeftModel
```

## Configure quantization

```python
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
```

## Load model and adapter

```python
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

ADAPTER_PATH = "./outputs/customer-support-qlora"


tokenizer = AutoTokenizer.from_pretrained(
    ADAPTER_PATH
)


base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto"
)


model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_PATH
)

model.eval()
```

Now use the same `generate_response()` function.

---

# 8. Generation parameters explained

## `max_new_tokens`

```python
max_new_tokens=200
```

Maximum number of tokens generated.

```text
Prompt tokens = 100
max_new_tokens = 200

Maximum output ≈ 200 tokens
```

Prefer this over relying on `max_length`, because `max_length` can include the input length.

---

## `temperature`

```python
temperature=0.7
```

Controls randomness.

```text
0.0 → Most deterministic
0.3 → More focused
0.7 → Balanced
1.0 → More creative
```

For customer support:

```python
temperature=0.2
```

or:

```python
do_sample=False
```

is often preferable for consistent answers.

---

## `top_p`

```python
top_p=0.9
```

Nucleus sampling.

The model considers tokens whose cumulative probability is within approximately the top `p` probability mass.

Typical:

```python
top_p=0.9
```

---

## `do_sample`

```python
do_sample=True
```

Enables probabilistic sampling.

For deterministic inference:

```python
do_sample=False
```

A good pattern:

```python
do_sample=temperature > 0
```

---

## `repetition_penalty`

```python
repetition_penalty=1.1
```

Can discourage repetitive output.

Be careful: too high a penalty can reduce response quality.

---

# 9. Deterministic inference

For structured tasks like:

```text
JSON generation
SQL generation
Classification
Customer support workflows
```

you often want predictable results.

```python
with torch.inference_mode():

    output = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )
```

You may omit sampling-related parameters when `do_sample=False`.

---

# 10. Streaming inference

For a chatbot, users should see tokens while they are generated.

```python
from transformers import TextStreamer
```

Create a streamer:

```python
streamer = TextStreamer(
    tokenizer,
    skip_prompt=True,
    skip_special_tokens=True
)
```

Generate:

```python
model.generate(
    **inputs,
    max_new_tokens=200,
    do_sample=True,
    temperature=0.7,
    streamer=streamer
)
```

The response will appear progressively.

For production APIs, a background thread plus a streamer is commonly used so the request handler can stream tokens as they are produced.

---

# 11. Compare base model vs fine-tuned model

A good way to evaluate fine-tuning is:

```python
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="auto"
)

fine_tuned_model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_PATH
)
```

Use the same prompt:

```python
prompt = "Explain the company's refund policy."
```

Compare:

```text
Base Model
    ↓
Generic answer

Fine-Tuned Model
    ↓
Company-specific terminology
Company-specific behavior
Correct response style
```

In a real evaluation pipeline, don't rely only on a few manual examples. Use a held-out evaluation dataset and task-specific metrics.

---

# 12. Production inference architecture

A production system might look like:

```text
                 Client
                    │
                    ▼
                FastAPI
                    │
                    ▼
            Input Validation
                    │
                    ▼
             Prompt / Messages
                    │
                    ▼
        Fine-Tuned LoRA Model
                    │
                    ▼
             Generation
                    │
                    ▼
          Output Validation
                    │
                    ▼
                 Response
```

Example FastAPI endpoint:

```python
from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI()


class ChatRequest(BaseModel):
    message: str


@app.post("/chat")
def chat(request: ChatRequest):

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful customer support assistant."
            )
        },
        {
            "role": "user",
            "content": request.message
        }
    ]

    response = generate_response(
        model=model,
        tokenizer=tokenizer,
        messages=messages,
        temperature=0.2
    )

    return {
        "response": response
    }
```

---

# 13. Recommended inference settings

For your fine-tuned **customer-support model**:

```python
output = model.generate(
    **inputs,
    max_new_tokens=256,
    do_sample=False,
    repetition_penalty=1.05,
    pad_token_id=tokenizer.eos_token_id
)
```

For a creative assistant:

```python
output = model.generate(
    **inputs,
    max_new_tokens=512,
    do_sample=True,
    temperature=0.8,
    top_p=0.9,
    pad_token_id=tokenizer.eos_token_id
)
```

For structured JSON:

```python
output = model.generate(
    **inputs,
    max_new_tokens=256,
    do_sample=False,
    pad_token_id=tokenizer.eos_token_id
)
```

---

# Interview-ready answer

> **To run inference with a LoRA or QLoRA fine-tuned model, I first load the original base model and then attach the trained PEFT adapter using `PeftModel.from_pretrained()`. For QLoRA, I typically reload the base model with the same 4-bit quantization configuration to reduce memory usage. I then put the model in evaluation mode, format the input using the tokenizer's chat template, tokenize it, and call `model.generate()` inside `torch.inference_mode()`. Finally, I decode only the newly generated tokens.**
>
> **In production, I also control generation using parameters such as `max_new_tokens`, `temperature`, `top_p`, and `do_sample`, and I choose deterministic decoding for structured or business-critical tasks.**

## Minimal inference code

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_PATH = "./my_adapter"


tokenizer = AutoTokenizer.from_pretrained(
    ADAPTER_PATH
)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="auto"
)

model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_PATH
)

model.eval()


messages = [
    {
        "role": "user",
        "content": "My payment was deducted twice. What should I do?"
    }
]


inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True
)

input_device = model.get_input_embeddings().weight.device

inputs = {
    key: value.to(input_device)
    for key, value in inputs.items()
}


with torch.inference_mode():

    output = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )


input_length = inputs["input_ids"].shape[1]

response = tokenizer.decode(
    output[0, input_length:],
    skip_special_tokens=True
)

print(response)
```

This is the core **fine-tuned model inference pipeline** used for LoRA/QLoRA deployments.
