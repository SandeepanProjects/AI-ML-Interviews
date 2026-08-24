# What format should an instruction-tuning dataset have?

An **instruction-tuning dataset** should contain examples of:

> **Instruction/Input → Desired Assistant Response**

The exact format depends on the model and training framework, but the two most common formats are:

1. **Instruction–Input–Output format**
2. **Chat/messages format** ← usually preferred for modern chat LLMs

---

# 1. Basic instruction format

A classic instruction-tuning example looks like this:

```json
{
  "instruction": "Explain LoRA in simple terms.",
  "input": "",
  "output": "LoRA is a parameter-efficient fine-tuning technique that trains small adapter matrices instead of updating the entire model."
}
```

If additional context is needed:

```json
{
  "instruction": "Summarize the following text.",
  "input": "LoRA reduces the number of trainable parameters by keeping the pretrained model frozen and training low-rank matrices.",
  "output": "LoRA fine-tunes a model efficiently by training small low-rank adapters while keeping the original model frozen."
}
```

Conceptually:

```text
Instruction
    +
Optional Input/Context
    ↓
Desired Output
```

This is commonly called an **Alpaca-style format**.

---

# 2. The most common modern format: Chat/messages

For modern instruction/chat models, I recommend storing the dataset as conversations.

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful AI assistant."
    },
    {
      "role": "user",
      "content": "Explain LoRA in simple terms."
    },
    {
      "role": "assistant",
      "content": "LoRA is a fine-tuning technique that keeps the original model frozen and trains small additional matrices."
    }
  ]
}
```

This is conceptually:

```text
System
  │
  ▼
Defines behavior

User
  │
  ▼
Instruction / Question

Assistant
  │
  ▼
Desired answer
```

For most modern chat LLM fine-tuning, this is the most flexible format.

---

# 3. JSON vs JSONL

You may store the data as a JSON array:

```json
[
  {
    "instruction": "Explain LoRA.",
    "input": "",
    "output": "LoRA is..."
  },
  {
    "instruction": "Explain QLoRA.",
    "input": "",
    "output": "QLoRA is..."
  }
]
```

But large fine-tuning datasets are commonly stored as **JSONL**.

Example:

```text
{"instruction":"Explain LoRA.","input":"","output":"LoRA is a parameter-efficient fine-tuning technique."}
{"instruction":"Explain QLoRA.","input":"","output":"QLoRA combines quantization with LoRA."}
{"instruction":"Explain RAG.","input":"","output":"RAG retrieves relevant information before generating an answer."}
```

Each line is one independent example.

```text
Line 1 → Example 1
Line 2 → Example 2
Line 3 → Example 3
```

---

# 4. Multi-turn instruction tuning

Instruction tuning does not have to be single-turn.

Example:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a Python tutor."
    },
    {
      "role": "user",
      "content": "What is a list in Python?"
    },
    {
      "role": "assistant",
      "content": "A list is an ordered and mutable collection of values."
    },
    {
      "role": "user",
      "content": "Can you show an example?"
    },
    {
      "role": "assistant",
      "content": "Sure. Example: numbers = [1, 2, 3]"
    }
  ]
}
```

This teaches the model:

* context handling
* follow-up questions
* conversational flow

---

# 5. Code: create an instruction-tuning dataset

Let's create a dataset programmatically.

```python
import json

examples = [
    {
        "instruction": "Explain LoRA in simple terms.",
        "input": "",
        "output": (
            "LoRA is a parameter-efficient fine-tuning technique "
            "that keeps the original model frozen and trains small "
            "low-rank adapter matrices."
        )
    },
    {
        "instruction": "Explain QLoRA.",
        "input": "",
        "output": (
            "QLoRA combines 4-bit quantization with LoRA. "
            "The base model remains quantized and frozen while "
            "LoRA adapters are trained."
        )
    },
    {
        "instruction": "Summarize the following text.",
        "input": (
            "Retrieval-Augmented Generation retrieves relevant "
            "documents from an external knowledge source and "
            "provides them to the LLM as context."
        ),
        "output": (
            "RAG retrieves relevant information and gives it to "
            "the LLM as context before generating an answer."
        )
    }
]

with open(
    "instruction_dataset.jsonl",
    "w",
    encoding="utf-8"
) as f:

    for example in examples:
        f.write(json.dumps(example) + "\n")
```

The generated file contains:

```text
instruction_dataset.jsonl
│
├── Example 1
├── Example 2
└── Example 3
```

---

# 6. Convert instruction format to messages format

Suppose you initially have:

```python
example = {
    "instruction": "Explain LoRA.",
    "input": "",
    "output": "LoRA is..."
}
```

Convert it:

```python
def convert_to_messages(example):

    user_content = example["instruction"]

    if example.get("input"):
        user_content += (
            "\n\nContext:\n"
            + example["input"]
        )

    return {
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful AI assistant."
            },
            {
                "role": "user",
                "content": user_content
            },
            {
                "role": "assistant",
                "content": example["output"]
            }
        ]
    }
```

Usage:

```python
chat_example = convert_to_messages(example)

print(
    json.dumps(
        chat_example,
        indent=2
    )
)
```

Output:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful AI assistant."
    },
    {
      "role": "user",
      "content": "Explain LoRA."
    },
    {
      "role": "assistant",
      "content": "LoRA is..."
    }
  ]
}
```

---

# 7. The model-specific chat template is important

This is one of the most important concepts.

Different models expect different special tokens.

For example, one model may expect:

```text
<|user|>
Explain LoRA.
<|assistant|>
LoRA is...
```

Another may expect:

```text
<s>[INST]
Explain LoRA.
[/INST]

LoRA is...
```

Therefore, don't manually hardcode formats unless necessary.

Use:

```python
tokenizer.apply_chat_template()
```

Example:

```python
from transformers import AutoTokenizer

MODEL_NAME = "your-model"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)
```

Then:

```python
messages = [
    {
        "role": "system",
        "content": "You are a helpful AI assistant."
    },
    {
        "role": "user",
        "content": "Explain LoRA."
    },
    {
        "role": "assistant",
        "content": "LoRA is a parameter-efficient fine-tuning technique."
    }
]

formatted_text = tokenizer.apply_chat_template(
    messages,
    tokenize=False
)

print(formatted_text)
```

The tokenizer converts your generic `messages` format into the exact prompt format expected by the base model.

---

# 8. Dataset format for SFT

For **Supervised Fine-Tuning (SFT)**, the core idea is:

```text
Prompt
   +
Expected Answer
   ↓
Training Example
```

For a chat model:

```text
┌─────────────────────────────┐
│ System                      │
│ You are a coding assistant  │
└──────────────┬──────────────┘
               │
┌──────────────▼──────────────┐
│ User                        │
│ Write a Python function     │
└──────────────┬──────────────┘
               │
┌──────────────▼──────────────┐
│ Assistant                   │
│ def factorial(...):         │
└─────────────────────────────┘
```

During training, the model learns to generate the assistant's response.

---

# 9. Should the loss be calculated on everything?

Usually, for instruction tuning, we primarily want the model to learn the **assistant response**.

Conceptually:

```text
System Prompt
██████████████
Loss ignored

User Instruction
██████████████
Loss ignored

Assistant Response
██████████████
Loss calculated ✓
```

For example:

```text
Tokens:

<System> You are helpful
<User> Explain LoRA
<Assistant> LoRA is...
```

Labels can conceptually look like:

```text
System tokens:
-100 -100 -100

User tokens:
-100 -100 -100

Assistant tokens:
 123  456  789
```

`-100` means:

> Ignore these positions when calculating the loss.

This is called **completion-only training** or **assistant-only loss masking**.

---

# 10. Code: Load an instruction dataset with Hugging Face

Install:

```bash
pip install datasets transformers
```

Load:

```python
from datasets import load_dataset

dataset = load_dataset(
    "json",
    data_files="instruction_dataset.jsonl",
    split="train"
)

print(dataset[0])
```

Example output:

```python
{
    "instruction": "Explain LoRA in simple terms.",
    "input": "",
    "output": "LoRA is a parameter-efficient..."
}
```

Convert to messages:

```python
def to_messages(example):

    user_message = example["instruction"]

    if example["input"].strip():
        user_message += (
            "\n\n"
            + example["input"]
        )

    return {
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful AI assistant."
            },
            {
                "role": "user",
                "content": user_message
            },
            {
                "role": "assistant",
                "content": example["output"]
            }
        ]
    }


dataset = dataset.map(
    to_messages
)
```

---

# 11. Apply the chat template

```python
def apply_template(example):

    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False
    )

    return {
        "text": text
    }


dataset = dataset.map(
    apply_template
)
```

Now inspect:

```python
print(dataset[0]["text"])
```

The output will use the exact template required by your model.

---

# 12. Tokenize

```python
def tokenize(example):

    return tokenizer(
        example["text"],
        truncation=True,
        max_length=2048
    )


tokenized_dataset = dataset.map(
    tokenize,
    batched=True
)
```

The complete pipeline is:

```text
Raw Dataset

{
 instruction,
 input,
 output
}

        │
        ▼

Convert to Messages

{
 messages: [
   system,
   user,
   assistant
 ]
}

        │
        ▼

Apply Chat Template

Model-specific tokens

        │
        ▼

Tokenization

input_ids

        │
        ▼

Create Labels

assistant tokens

        │
        ▼

SFT / Fine-tuning
```

---

# 13. Recommended formats for different use cases

| Use case                   | Recommended format                    |
| -------------------------- | ------------------------------------- |
| Simple instruction tuning  | `instruction`, `input`, `output`      |
| Modern chat LLM            | `messages`                            |
| Multi-turn conversation    | `messages`                            |
| Text completion            | `prompt`, `completion`                |
| Classification             | `instruction`, `input`, `output`      |
| Structured JSON generation | `messages` with JSON assistant output |
| Tool/function calling      | `messages` + tool-call schema         |

---

# 14. Structured output example

Suppose you want to fine-tune an LLM to return JSON.

```json
{
  "messages": [
    {
      "role": "system",
      "content": "Extract customer information and return valid JSON."
    },
    {
      "role": "user",
      "content": "John is 30 years old and lives in Bangalore."
    },
    {
      "role": "assistant",
      "content": "{\"name\": \"John\", \"age\": 30, \"city\": \"Bangalore\"}"
    }
  ]
}
```

Include many examples:

```text
Normal input
Missing fields
Multiple people
Ambiguous information
Invalid input
Different wording
```

This teaches a reliable input → structured output behavior.

---

# 15. What makes a good instruction-tuning example?

A high-quality example should have:

### 1. Clear instruction

```text
Explain LoRA in simple language.
```

### 2. Relevant input

```text
Optional context or document.
```

### 3. High-quality expected answer

```text
LoRA freezes the pretrained model and trains
small low-rank adapter matrices.
```

### 4. Consistent style

If you want concise answers:

```text
Keep examples concise.
```

If you want detailed answers:

```text
Include detailed examples consistently.
```

The dataset teaches behavior.

---

# 16. Bad vs good example

### ❌ Bad

```json
{
  "instruction": "Explain LoRA",
  "output": "LoRA is good"
}
```

Problems:

* vague
* incomplete
* poor-quality target

### ✅ Better

```json
{
  "instruction": "Explain LoRA in simple terms.",
  "input": "",
  "output": "LoRA is a parameter-efficient fine-tuning method that keeps the original model frozen and trains small low-rank matrices to adapt the model to a new task."
}
```

The model learns from the **quality of the answer**.

---

# Interview answer

If an interviewer asks:

> **What format should an instruction-tuning dataset have?**

A strong answer is:

> **An instruction-tuning dataset should contain examples of an instruction or user request and the ideal assistant response. A classic format is instruction, optional input, and output. For modern chat models, I prefer a messages format containing system, user, and assistant roles. The dataset is then converted using the target model's chat template so the special tokens match the pretrained model. During supervised fine-tuning, we usually train the model to predict the assistant response, often masking the system and user tokens from the loss.**

### Remember this:

```text
INSTRUCTION TUNING DATA

User Instruction
        │
        ▼
Expected Ideal Response
        │
        ▼
Model learns desired behavior
```

For a modern production LLM fine-tuning pipeline, I would generally recommend:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "..."
    },
    {
      "role": "user",
      "content": "..."
    },
    {
      "role": "assistant",
      "content": "..."
    }
  ]
}
```

because it naturally supports **chat, multi-turn conversations, system instructions, and modern model chat templates**.
