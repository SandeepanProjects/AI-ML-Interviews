# How to Prepare an Instruction Dataset for LLM Fine-Tuning

An **instruction dataset** teaches an LLM:

> Given an instruction, optionally some context, produce the desired response.

The basic pattern is:

```text
Instruction + Input/Context → Ideal Output
```

For example:

```text
Instruction:
Explain Python decorators.

Input:
None

Output:
A decorator is a function that modifies or extends another function's behavior...
```

---

# 1. Common instruction dataset format

A simple JSONL format:

```json
{"instruction":"Explain Python decorators.","input":"","output":"A decorator is a function that modifies or extends another function's behavior."}
{"instruction":"Summarize the text.","input":"Python is a popular programming language.","output":"Python is a popular programming language."}
{"instruction":"Translate English to French.","input":"Hello, how are you?","output":"Bonjour, comment allez-vous ?"}
```

Each line is one training example.

File:

```text
instruction_dataset.jsonl
```

---

# 2. Why JSONL?

JSONL means:

```text
JSON Lines
```

Each line is independent:

```text
Example 1
Example 2
Example 3
...
```

Advantages:

```text
✓ Easy to stream
✓ Good for large datasets
✓ Easy to process
✓ Easy to append
✓ Supported by Hugging Face datasets
```

---

# 3. Recommended dataset schema

For a production dataset, I recommend:

```json
{
  "id": "support_0001",
  "instruction": "Help the customer solve their problem.",
  "input": "My payment was deducted twice.",
  "output": "I’m sorry about the duplicate charge. Please share the transaction ID so we can investigate and resolve the issue.",
  "category": "payment",
  "source": "synthetic",
  "quality_score": 0.95
}
```

Not all fields must be sent to the LLM during training.

For example:

```text
id              → metadata
category        → metadata
source          → metadata
quality_score   → filtering

instruction
input
output
      ↓
Training data
```

---

# 4. Customer-support instruction dataset

Example `support_dataset.jsonl`:

```json
{"id":"001","instruction":"You are a helpful customer support agent. Respond professionally and clearly.","input":"My payment was deducted twice.","output":"I’m sorry about the duplicate charge. Please provide the transaction ID so we can investigate the issue and help resolve it.","category":"payment"}
{"id":"002","instruction":"You are a helpful customer support agent. Respond professionally and clearly.","input":"I forgot my password.","output":"You can reset your password by selecting the Forgot Password option on the login page and following the instructions sent to your registered email address.","category":"account"}
{"id":"003","instruction":"You are a helpful customer support agent. Respond professionally and clearly.","input":"Where is my order?","output":"Please provide your order ID, and I can help you check the current delivery status.","category":"order"}
```

---

# 5. Prepare the dataset with Python

Suppose your raw data is:

```python
raw_data = [
    {
        "question": "My payment was deducted twice.",
        "answer": "Please provide your transaction ID."
    },
    {
        "question": "I forgot my password.",
        "answer": "Use the Forgot Password option."
    }
]
```

Convert it:

```python
import json


INSTRUCTION = """
You are a helpful customer support assistant.
Provide accurate, professional, and concise responses.
"""


training_examples = []


for index, item in enumerate(raw_data):

    example = {

        "id": f"support_{index}",

        "instruction": INSTRUCTION.strip(),

        "input": item["question"].strip(),

        "output": item["answer"].strip(),

        "category": "customer_support"
    }

    training_examples.append(example)


with open(
    "instruction_dataset.jsonl",
    "w",
    encoding="utf-8"
) as file:

    for example in training_examples:

        file.write(
            json.dumps(
                example,
                ensure_ascii=False
            )
            + "\n"
        )
```

Output:

```text
instruction_dataset.jsonl
```

---

# 6. Load the dataset with Hugging Face

```python
from datasets import load_dataset


dataset = load_dataset(
    "json",
    data_files="instruction_dataset.jsonl",
    split="train"
)


print(dataset[0])
```

Output:

```python
{
    "id": "support_0",

    "instruction":
        "You are a helpful customer support assistant...",

    "input":
        "My payment was deducted twice.",

    "output":
        "Please provide your transaction ID.",

    "category":
        "customer_support"
}
```

---

# 7. Format instruction data into a training prompt

The model cannot directly train on arbitrary columns such as:

```text
instruction
input
output
```

We must convert them into a sequence.

For example:

```text
### Instruction:
You are a helpful customer support assistant.

### Input:
My payment was deducted twice.

### Response:
Please provide your transaction ID.
```

Code:

```python
def format_instruction(example):

    instruction = example["instruction"]
    user_input = example["input"]
    output = example["output"]

    prompt = f"""### Instruction:
{instruction}

### Input:
{user_input}

### Response:
{output}"""

    return {
        "text": prompt
    }


dataset = dataset.map(
    format_instruction
)
```

Now:

```python
print(dataset[0]["text"])
```

Output:

```text
### Instruction:
You are a helpful customer support assistant.

### Input:
My payment was deducted twice.

### Response:
Please provide your transaction ID.
```

---

# 8. A better format: conversational messages

Modern instruction-tuned models are usually trained with chat messages.

For example:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful customer support assistant."
    },
    {
      "role": "user",
      "content": "My payment was deducted twice."
    },
    {
      "role": "assistant",
      "content": "I’m sorry about the duplicate charge. Please provide your transaction ID."
    }
  ]
}
```

JSONL:

```json
{"messages":[{"role":"system","content":"You are a helpful customer support assistant."},{"role":"user","content":"My payment was deducted twice."},{"role":"assistant","content":"I’m sorry about the duplicate charge. Please provide your transaction ID."}]}
{"messages":[{"role":"system","content":"You are a helpful customer support assistant."},{"role":"user","content":"I forgot my password."},{"role":"assistant","content":"You can reset your password using the Forgot Password option."}]}
```

This is usually preferable for modern chat models.

---

# 9. Convert instruction format to conversational format

Suppose you start with:

```python
{
    "instruction": "You are a helpful customer support assistant.",
    "input": "My payment was deducted twice.",
    "output": "Please provide your transaction ID."
}
```

Convert it:

```python
def convert_to_messages(example):

    return {

        "messages": [

            {
                "role": "system",
                "content": example["instruction"]
            },

            {
                "role": "user",
                "content": example["input"]
            },

            {
                "role": "assistant",
                "content": example["output"]
            }
        ]
    }
```

Apply:

```python
dataset = dataset.map(
    convert_to_messages
)
```

Result:

```python
print(dataset[0]["messages"])
```

---

# 10. Use the model's chat template

Do **not manually invent special tokens** for a chat model unless you have a strong reason.

Different models have different templates.

For example:

```text
Qwen
Llama
Mistral
Gemma
```

may use different formatting.

Use:

```python
text = tokenizer.apply_chat_template(

    example["messages"],

    tokenize=False,

    add_generation_prompt=False
)
```

Full code:

```python
from transformers import AutoTokenizer


MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


def format_messages(example):

    text = tokenizer.apply_chat_template(

        example["messages"],

        tokenize=False,

        add_generation_prompt=False
    )

    return {
        "text": text
    }


dataset = dataset.map(
    format_messages
)
```

Now each training example becomes correctly formatted for the selected model.

---

# 11. Clean the dataset

Before training, clean the data.

A production pipeline:

```text
Raw Data
    │
    ▼
Remove Empty Records
    │
    ▼
Remove Duplicates
    │
    ▼
Validate Schema
    │
    ▼
Remove Bad Examples
    │
    ▼
Length Filtering
    │
    ▼
PII/Sensitive Data Review
    │
    ▼
Quality Checks
    │
    ▼
Train / Validation Split
```

---

# 12. Remove empty examples

```python
def is_valid(example):

    required_fields = [
        "instruction",
        "input",
        "output"
    ]

    for field in required_fields:

        if not example.get(field):

            return False

        if len(
            example[field].strip()
        ) == 0:

            return False

    return True


dataset = dataset.filter(
    is_valid
)
```

---

# 13. Remove duplicates

```python
import hashlib


def generate_hash(example):

    text = (
        example["instruction"]
        +
        example["input"]
        +
        example["output"]
    )

    return hashlib.sha256(
        text.encode()
    ).hexdigest()
```

For a simple in-memory dataset:

```python
seen = set()


def is_unique(example):

    example_hash = generate_hash(example)

    if example_hash in seen:

        return False

    seen.add(example_hash)

    return True


dataset = dataset.filter(
    is_unique
)
```

For millions of examples, you would generally use scalable approaches such as distributed processing or deduplication systems rather than keeping everything in a Python set.

---

# 14. Remove excessively long examples

Suppose:

```text
Maximum sequence length = 2048 tokens
```

Check length using the tokenizer:

```python
MAX_LENGTH = 2048


def is_short_enough(example):

    text = (
        example["instruction"]
        +
        example["input"]
        +
        example["output"]
    )

    tokens = tokenizer(
        text,
        truncation=False
    )

    return len(
        tokens["input_ids"]
    ) <= MAX_LENGTH
```

Apply:

```python
dataset = dataset.filter(
    is_short_enough
)
```

---

# 15. Detect suspicious examples

Example bad record:

```text
Input:
What is the capital of France?

Output:
Banana
```

A simple automated validation:

```python
def basic_quality_check(example):

    output = example["output"]

    # Too short
    if len(output) < 5:
        return False

    # Too long
    if len(output) > 10000:
        return False

    return True
```

Apply:

```python
dataset = dataset.filter(
    basic_quality_check
)
```

In production, you may additionally use:

```text
Rule-based validation
LLM-as-a-judge
Human review
Schema validation
Domain-specific validators
```

---

# 16. Split train and validation correctly

```python
dataset = dataset.train_test_split(

    test_size=0.1,

    seed=42
)


train_dataset = dataset["train"]

validation_dataset = dataset["test"]
```

Typical:

```text
Dataset

90% → Training
10% → Validation
```

For large datasets:

```text
98% → Training
1%  → Validation
1%  → Test
```

The important part is preventing data leakage.

---

# 17. Prevent data leakage

Imagine:

```text
Training:

Question:
My payment failed.

Answer:
Try another payment method.
```

Validation:

```text
Question:
My payment failed.

Answer:
Try another payment method.
```

This is leakage.

A better approach is deduplication before splitting.

```python
# Step 1
dataset = remove_duplicates(dataset)

# Step 2
dataset = dataset.train_test_split(
    test_size=0.1,
    seed=42
)
```

For customer conversations, you may also split by:

```text
Customer ID
Conversation ID
Ticket ID
Time period
```

Example concept:

```text
Customer A → only training

Customer B → only validation
```

Do not allow the same conversation or near-duplicate conversation into both sets.

---

# 18. Complete dataset preparation script

```python
import json
import hashlib

from datasets import load_dataset
from transformers import AutoTokenizer


# =========================================================
# CONFIGURATION
# =========================================================

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

INPUT_FILE = "raw_data.jsonl"

MAX_LENGTH = 2048


# =========================================================
# LOAD TOKENIZER
# =========================================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


# =========================================================
# LOAD DATASET
# =========================================================

dataset = load_dataset(
    "json",
    data_files=INPUT_FILE,
    split="train"
)


# =========================================================
# REMOVE EMPTY EXAMPLES
# =========================================================

def is_valid(example):

    required_fields = [
        "instruction",
        "input",
        "output"
    ]

    for field in required_fields:

        value = example.get(field, "")

        if not value.strip():
            return False

    return True


dataset = dataset.filter(
    is_valid
)


# =========================================================
# REMOVE DUPLICATES
# =========================================================

seen = set()


def is_unique(example):

    text = "||".join([
        example["instruction"].strip(),
        example["input"].strip(),
        example["output"].strip()
    ])

    example_hash = hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()

    if example_hash in seen:

        return False

    seen.add(example_hash)

    return True


dataset = dataset.filter(
    is_unique
)


# =========================================================
# BASIC QUALITY CHECK
# =========================================================

def quality_check(example):

    output = example["output"].strip()

    # Reject very short outputs
    if len(output) < 5:
        return False

    # Reject extremely long outputs
    if len(output) > 10000:
        return False

    return True


dataset = dataset.filter(
    quality_check
)


# =========================================================
# TOKEN LENGTH CHECK
# =========================================================

def is_within_length(example):

    text = (
        example["instruction"]
        + "\n"
        + example["input"]
        + "\n"
        + example["output"]
    )

    tokens = tokenizer(
        text,
        truncation=False
    )

    return (
        len(tokens["input_ids"])
        <= MAX_LENGTH
    )


dataset = dataset.filter(
    is_within_length
)


# =========================================================
# CONVERT TO CHAT FORMAT
# =========================================================

def to_messages(example):

    return {

        "messages": [

            {
                "role": "system",
                "content": example["instruction"].strip()
            },

            {
                "role": "user",
                "content": example["input"].strip()
            },

            {
                "role": "assistant",
                "content": example["output"].strip()
            }
        ]
    }


dataset = dataset.map(
    to_messages
)


# =========================================================
# TRAIN / VALIDATION SPLIT
# =========================================================

dataset = dataset.train_test_split(

    test_size=0.1,

    seed=42
)


train_dataset = dataset["train"]

validation_dataset = dataset["test"]


# =========================================================
# FORMAT USING CHAT TEMPLATE
# =========================================================

def format_for_model(example):

    text = tokenizer.apply_chat_template(

        example["messages"],

        tokenize=False,

        add_generation_prompt=False
    )

    return {
        "text": text
    }


train_dataset = train_dataset.map(
    format_for_model
)

validation_dataset = validation_dataset.map(
    format_for_model
)


# =========================================================
# SAVE DATASETS
# =========================================================

train_dataset.to_json(
    "train.jsonl"
)

validation_dataset.to_json(
    "validation.jsonl"
)


print("Dataset preparation completed!")

print(
    "Training examples:",
    len(train_dataset)
)

print(
    "Validation examples:",
    len(validation_dataset)
)
```

---

# 19. What should you inspect before training?

Always inspect random examples.

```python
import random


for _ in range(5):

    index = random.randint(
        0,
        len(train_dataset) - 1
    )

    print("=" * 80)

    print(
        train_dataset[index]["text"]
    )
```

Check:

```text
✓ Correct system instruction
✓ Correct user input
✓ Correct assistant response
✓ No broken formatting
✓ No accidental duplicate
✓ No PII/secrets
✓ No irrelevant output
✓ Correct chat template
✓ Sequence length is reasonable
```

---

# 20. Recommended production dataset pipeline

For a real customer-support or enterprise fine-tuning project:

```text
Raw Conversations
        │
        ▼
Remove PII / Secrets
        │
        ▼
Normalize Text
        │
        ▼
Remove Duplicates
        │
        ▼
Remove Low-Quality Examples
        │
        ▼
Validate Conversation Structure
        │
        ▼
Create Instruction/Response Pairs
        │
        ▼
Apply Model Chat Template
        │
        ▼
Token-Length Analysis
        │
        ▼
Train / Validation / Test Split
        │
        ▼
Human Quality Review
        │
        ▼
Version Dataset
        │
        ▼
Fine-Tuning
```

---

# Interview-ready answer

> **To prepare an instruction dataset, I first define the desired model behavior and create high-quality instruction-input-output examples. For modern chat models, I usually convert these into a structured `messages` format containing system, user, and assistant roles.**
>
> **I clean the data by removing empty records, duplicates, malformed examples, excessively long sequences, and low-quality responses. I check token lengths using the target model's tokenizer and use the model's native chat template rather than manually inventing special tokens.**
>
> **Before splitting, I deduplicate the dataset to prevent leakage. Then I create training and validation sets, often using grouped splits for conversations so that the same customer or conversation does not appear in both sets. Finally, I manually inspect samples, version the dataset, and train the model.**

The most important principle is:

```text
Model Quality
      ↑
Dataset Quality
      ↑
High-quality examples
      +
Correct formatting
      +
No leakage
      +
Strong validation
```
