# How would you create a fine-tuning dataset?

Creating a good fine-tuning dataset is often **more important than choosing LoRA vs QLoRA**.

A fine-tuning dataset teaches the model:

* how to respond
* what style to use
* how to follow instructions
* what output format to produce
* domain-specific behavior

It is usually **not just a collection of documents**.

---

# 1. First: What problem are you solving?

Before creating data, define the objective.

For example, suppose we want to build:

> **An enterprise financial assistant**

We might want the model to:

* answer financial questions in a professional style
* summarize reports
* extract structured information
* classify financial requests
* refuse unsupported actions
* return JSON when requested

So the training data must represent these desired behaviors.

```text
Business Requirement
        ↓
Define Tasks
        ↓
Create Examples
        ↓
Validate Quality
        ↓
Split Dataset
        ↓
Tokenize
        ↓
Fine-tune
        ↓
Evaluate
```

---

# 2. Choose the correct dataset format

For instruction fine-tuning, a common conceptual format is:

```json
{
  "instruction": "Explain compound interest",
  "input": "Principal: 10000, Rate: 10%, Years: 2",
  "output": "Compound interest is..."
}
```

However, modern chat models are often better trained using a **messages format**.

Example:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful financial assistant."
    },
    {
      "role": "user",
      "content": "Explain compound interest."
    },
    {
      "role": "assistant",
      "content": "Compound interest is interest calculated on the principal and accumulated interest."
    }
  ]
}
```

This is useful because many modern LLMs are already trained with chat templates.

---

# 3. Create a dataset manually

Suppose we create a file:

```text
data/
├── raw/
│   └── financial_examples.jsonl
│
├── processed/
│   ├── train.jsonl
│   └── validation.jsonl
│
└── test.jsonl
```

Let's create the raw data.

```python
import json

examples = [
    {
        "instruction": "Explain compound interest.",
        "input": "",
        "output": (
            "Compound interest is interest calculated on both "
            "the original principal and previously accumulated interest."
        )
    },
    {
        "instruction": "Summarize the following financial result.",
        "input": (
            "Revenue increased from $10 million to $12 million, "
            "while operating expenses increased from $7 million to $8 million."
        ),
        "output": (
            "Revenue increased by 20%, while operating expenses "
            "increased by approximately 14.3%."
        )
    },
    {
        "instruction": "Classify the request.",
        "input": "What is my current account balance?",
        "output": "ACCOUNT_BALANCE_QUERY"
    }
]

with open(
    "financial_examples.jsonl",
    "w"
) as f:

    for example in examples:

        f.write(
            json.dumps(example) + "\n"
        )
```

This produces JSONL:

```text
{"instruction": "...", "input": "...", "output": "..."}
{"instruction": "...", "input": "...", "output": "..."}
{"instruction": "...", "input": "...", "output": "..."}
```

---

# 4. Why JSONL?

JSONL means:

> **JSON Lines**

Each line is one training example.

Example:

```text
Line 1 → Training example 1
Line 2 → Training example 2
Line 3 → Training example 3
```

This is convenient for:

* large datasets
* streaming
* Hugging Face datasets
* distributed training
* preprocessing pipelines

---

# 5. Convert instruction data to chat format

Let's say our base model is a chat model.

We can convert:

```text
Instruction
    +
Input
    ↓
User message

Output
    ↓
Assistant message
```

Code:

```python
def convert_to_chat(example):

    user_content = example["instruction"]

    if example["input"].strip():

        user_content += (
            "\n\n"
            + example["input"]
        )

    return {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a helpful and accurate "
                    "financial assistant."
                )
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

Example:

```python
example = {
    "instruction": "Explain compound interest.",
    "input": "",
    "output": "Compound interest is interest..."
}

chat_example = convert_to_chat(example)

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
      "content": "You are a helpful and accurate financial assistant."
    },
    {
      "role": "user",
      "content": "Explain compound interest."
    },
    {
      "role": "assistant",
      "content": "Compound interest is interest..."
    }
  ]
}
```

---

# 6. Use Hugging Face `datasets`

Install:

```bash
pip install datasets
```

Load:

```python
from datasets import load_dataset

dataset = load_dataset(
    "json",
    data_files="financial_examples.jsonl"
)

print(dataset)
```

You might get:

```text
DatasetDict({
    train: Dataset({
        features: [
            'instruction',
            'input',
            'output'
        ],
        num_rows: 3
    })
})
```

---

# 7. Create a larger dataset programmatically

In a real project, your data may come from:

```text
┌───────────────────────┐
│ PDFs                  │
├───────────────────────┤
│ Knowledge Bases       │
├───────────────────────┤
│ Support Tickets       │
├───────────────────────┤
│ Human-written examples│
├───────────────────────┤
│ Existing Conversations│
└───────────┬───────────┘
            ↓
     Data Processing
            ↓
    Fine-tuning examples
```

Suppose we have support tickets:

```python
tickets = [
    {
        "question": "How do I reset my password?",
        "answer": (
            "Go to the login page, select "
            "'Forgot Password', and follow the instructions."
        )
    },
    {
        "question": "How can I update my email address?",
        "answer": (
            "Open account settings, select profile, "
            "and update your registered email address."
        )
    }
]
```

Convert them:

```python
training_examples = []

for ticket in tickets:

    training_examples.append(
        {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful customer support assistant."
                    )
                },
                {
                    "role": "user",
                    "content": ticket["question"]
                },
                {
                    "role": "assistant",
                    "content": ticket["answer"]
                }
            ]
        }
    )
```

---

# 8. But real dataset creation is more complicated

You should not simply do:

```text
Company PDFs
    ↓
Copy everything
    ↓
Fine-tune
```

This is often a bad idea.

Why?

Because fine-tuning is not primarily designed to store frequently changing knowledge.

For documents like:

```text
Product documentation
Company policies
Current prices
Daily financial reports
```

usually use:

> **RAG**

For fine-tuning, use examples of desired behavior:

```text
User question
        ↓
Desired model response
```

---

# 9. Create examples from documents

Suppose you have this source:

```text
Policy:

Employees can work remotely for up to
three days per week with manager approval.
```

You can create instruction examples:

```python
examples = [
    {
        "instruction":
            "What is the remote work policy?",

        "input": "",

        "output":
            "Employees may work remotely for up to three days "
            "per week with manager approval."
    },

    {
        "instruction":
            "How many days can an employee work remotely?",

        "input": "",

        "output":
            "An employee can work remotely for up to three days per week."
    }
]
```

This teaches multiple phrasings:

```text
Question A ─┐
Question B ─┼──► Same concept
Question C ─┘
```

But for changing policies, RAG is usually safer.

---

# 10. Synthetic data generation

A common approach is:

```text
Small human-written dataset
          │
          ▼
    Strong LLM
          │
          ▼
Synthetic examples
          │
          ▼
Quality validation
          │
          ▼
Fine-tuning dataset
```

For example:

```python
seed_examples = [
    {
        "question": "How do I reset my password?",
        "answer": "Use the Forgot Password option."
    }
]
```

You can ask a stronger model to generate:

* paraphrases
* difficult cases
* edge cases
* different user styles

Example prompt:

```text
Generate 5 diverse user questions that have the same intent:

Original question:
How do I reset my password?

Requirements:
- Include casual language
- Include formal language
- Include short queries
- Include long queries
- Do not change the intent
```

Possible generated dataset:

```python
examples = [
    {
        "input":
            "I forgot my password. How can I log in?",

        "output":
            "Select 'Forgot Password' on the login screen..."
    },

    {
        "input":
            "Please provide the procedure for resetting my password.",

        "output":
            "Go to the login page and select 'Forgot Password'..."
    }
]
```

### Important

Synthetic data should be reviewed.

Bad pipeline:

```text
LLM generates data
        ↓
Immediately train
```

Better pipeline:

```text
LLM generates data
        ↓
Automatic validation
        ↓
Deduplication
        ↓
Human review/sample audit
        ↓
Fine-tuning
```

---

# 11. Data cleaning

Real data contains:

```text
Duplicates
Broken examples
Wrong answers
PII
Inconsistent formats
HTML
Empty responses
```

Let's clean a dataset.

```python
def is_valid(example):

    # Check required fields
    required_fields = [
        "instruction",
        "output"
    ]

    for field in required_fields:

        if not example.get(field):
            return False

    # Remove very short outputs
    if len(example["output"].strip()) < 10:
        return False

    return True
```

Apply:

```python
clean_examples = [
    example
    for example in examples
    if is_valid(example)
]
```

---

# 12. Deduplication

Duplicates can cause:

* overfitting
* wasted compute
* biased training

Example:

```python
import hashlib


def example_hash(example):

    content = (
        example["instruction"]
        +
        example.get("input", "")
        +
        example["output"]
    )

    return hashlib.sha256(
        content.encode()
    ).hexdigest()
```

Deduplicate:

```python
seen = set()

deduplicated = []

for example in clean_examples:

    h = example_hash(example)

    if h not in seen:

        seen.add(h)

        deduplicated.append(example)
```

---

# 13. More realistic normalization

For text:

```python
import re


def normalize_text(text):

    # Remove extra spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()
```

Apply:

```python
def clean_example(example):

    example["instruction"] = normalize_text(
        example["instruction"]
    )

    example["input"] = normalize_text(
        example.get("input", "")
    )

    example["output"] = normalize_text(
        example["output"]
    )

    return example
```

---

# 14. Remove PII

This is extremely important for enterprise data.

Suppose:

```text
My email is john@example.com
My phone is 9876543210
```

You may need to redact this before training.

Simple demonstration:

```python
import re


def remove_pii(text):

    # Email
    text = re.sub(
        r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
        "[EMAIL_REDACTED]",
        text,
    )

    # Simple phone pattern
    text = re.sub(
        r"\b\d{10}\b",
        "[PHONE_REDACTED]",
        text,
    )

    return text
```

Use:

```python
text = """
Contact me at john@example.com.
My phone number is 9876543210.
"""

print(
    remove_pii(text)
)
```

Output:

```text
Contact me at [EMAIL_REDACTED].
My phone number is [PHONE_REDACTED].
```

For production systems, use stronger PII detection and organization-specific governance rather than relying only on simple regex.

---

# 15. Balance your dataset

Suppose your dataset contains:

```text
90% password reset
5% billing
5% account deletion
```

The model may become disproportionately good at password resets.

Instead:

```text
Dataset

Password reset      30%
Billing             25%
Account management  25%
Technical support   20%
```

Check distribution:

```python
from collections import Counter


categories = [
    example["category"]
    for example in dataset
]

counts = Counter(categories)

print(counts)
```

---

# 16. Dataset diversity

You want variation.

Bad dataset:

```text
How do I reset my password?
How do I reset my password?
How do I reset my password?
```

Better:

```text
I forgot my password.

How can I recover access to my account?

I cannot remember my login credentials.

What's the process for changing my password?

My account is locked because I forgot my password.
```

Same intent, different language.

---

# 17. Split the dataset

Never evaluate only on training data.

Typical split:

```text
Dataset
   │
   ├── 80% Train
   │
   ├── 10% Validation
   │
   └── 10% Test
```

Code:

```python
from datasets import Dataset

dataset = Dataset.from_list(
    clean_examples
)

train_test = dataset.train_test_split(
    test_size=0.2,
    seed=42
)

train_dataset = train_test["train"]

temp_dataset = train_test["test"]
```

Split validation/test:

```python
validation_test = temp_dataset.train_test_split(
    test_size=0.5,
    seed=42
)

validation_dataset = validation_test["train"]

test_dataset = validation_test["test"]
```

Now approximately:

```text
Train        80%
Validation   10%
Test         10%
```

---

# 18. Avoid data leakage

This is critical.

Suppose:

### Training:

```text
Question:
How do I reset my password?

Answer:
Click Forgot Password.
```

### Test:

```text
Question:
How do I reset my password?

Answer:
Click Forgot Password.
```

This is leakage.

The model may appear highly accurate simply because it memorized the answer.

Better:

```text
Training:
Password reset examples A, B, C

Test:
New examples and paraphrases not present in training
```

For document-based datasets, a good practice is to split by:

* customer
* document
* conversation
* time period
* topic

rather than randomly splitting individual rows.

---

# 19. Apply the model's chat template

This is very important.

Different models expect different prompt formats.

For example:

```text
Model A:

<user>
Hello
</user>

<assistant>
Hi
</assistant>
```

Another model:

```text
<|user|>
Hello
<|assistant|>
Hi
```

Instead of manually guessing the format, use the tokenizer's chat template:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)
```

Example:

```python
messages = [
    {
        "role": "system",
        "content": "You are helpful."
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
```

Apply:

```python
formatted_text = tokenizer.apply_chat_template(
    messages,
    tokenize=False
)

print(formatted_text)
```

This converts your dataset into the exact format expected by the model.

---

# 20. Preprocess the dataset

Using Hugging Face:

```python
def format_example(example):

    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False
    )

    return {
        "text": text
    }
```

Apply:

```python
formatted_dataset = dataset.map(
    format_example
)
```

Now:

```python
print(
    formatted_dataset[0]["text"]
)
```

This gives the final training text.

---

# 21. Tokenize

```python
def tokenize(example):

    return tokenizer(
        example["text"],
        truncation=True,
        max_length=2048,
    )
```

Apply:

```python
tokenized_dataset = formatted_dataset.map(
    tokenize,
    batched=True,
)
```

The model doesn't understand:

```text
"Explain LoRA"
```

directly.

It sees token IDs:

```text
[128000, 849, 527, ...]
```

So the pipeline is:

```text
Raw JSON
   ↓
Clean
   ↓
Validate
   ↓
Chat Messages
   ↓
Apply Chat Template
   ↓
Tokenize
   ↓
Token IDs
   ↓
Fine-tuning
```

---

# 22. Labels for supervised fine-tuning

For causal language models, we usually train the model to predict the next token.

Example:

```text
User:
Explain LoRA

Assistant:
LoRA is...
```

A simplified token sequence:

```text
[USER_TOKENS] [ASSISTANT_TOKENS]
```

The labels might initially be:

```python
labels = input_ids.copy()
```

But often for instruction tuning, we want to calculate loss primarily on the **assistant response**, not the user prompt.

Conceptually:

```text
System Prompt
████████████  ignored

User Prompt
████████████  ignored

Assistant Response
████████████  calculate loss
```

The ignored tokens are assigned:

```python
-100
```

because PyTorch's cross-entropy loss ignores them.

Example:

```text
Input IDs:

[101, 102, 103, 104, 105, 106]

Labels:

[-100, -100, -100, 104, 105, 106]
```

This means:

```text
System/User tokens
      ↓
No loss

Assistant tokens
      ↓
Loss ✓
```

This is common in supervised instruction fine-tuning.

---

# 23. Production-style dataset pipeline

Here is a simplified architecture:

```text
Raw Data Sources
      │
      ├── Documents
      ├── Support Tickets
      ├── FAQs
      ├── Human Examples
      └── Approved Conversations
               │
               ▼
         Ingestion
               │
               ▼
      ┌─────────────────┐
      │ Data Cleaning   │
      │                 │
      │ Remove HTML     │
      │ Normalize       │
      │ Remove PII      │
      │ Remove Duplicates│
      └────────┬────────┘
               │
               ▼
       Quality Validation
               │
               ▼
      Human / Expert Review
               │
               ▼
         Dataset Versioning
               │
               ▼
       Train/Val/Test Split
               │
               ▼
       Chat Formatting
               │
               ▼
         Tokenization
               │
               ▼
          Fine-Tuning
               │
               ▼
          Evaluation
```

---

# 24. A complete small dataset builder

Here is a more realistic example.

```python
import json
import re
import hashlib


# -----------------------------------------
# 1. Normalize text
# -----------------------------------------

def normalize_text(text: str) -> str:

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# -----------------------------------------
# 2. Remove basic PII
# -----------------------------------------

def remove_pii(text: str) -> str:

    # Email
    text = re.sub(
        r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
        "[EMAIL_REDACTED]",
        text,
    )

    return text


# -----------------------------------------
# 3. Clean example
# -----------------------------------------

def clean_example(example):

    question = normalize_text(
        example["question"]
    )

    answer = normalize_text(
        example["answer"]
    )

    question = remove_pii(question)

    answer = remove_pii(answer)

    return {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a helpful enterprise assistant. "
                    "Provide accurate and concise answers."
                ),
            },
            {
                "role": "user",
                "content": question,
            },
            {
                "role": "assistant",
                "content": answer,
            },
        ]
    }


# -----------------------------------------
# 4. Validation
# -----------------------------------------

def is_valid(example):

    messages = example.get(
        "messages",
        []
    )

    if len(messages) < 3:
        return False

    user_message = messages[-2]["content"]

    assistant_message = messages[-1]["content"]

    if not user_message:
        return False

    if len(assistant_message) < 10:
        return False

    return True


# -----------------------------------------
# 5. Hash example
# -----------------------------------------

def get_hash(example):

    content = json.dumps(
        example,
        sort_keys=True
    )

    return hashlib.sha256(
        content.encode()
    ).hexdigest()


# -----------------------------------------
# 6. Build dataset
# -----------------------------------------

raw_examples = [
    {
        "question":
            "How do I reset my password?",

        "answer":
            "Go to the login page and select "
            "'Forgot Password'. Follow the instructions "
            "to create a new password.",
    },
    {
        "question":
            "How can I update my email?",

        "answer":
            "Open account settings, select your profile, "
            "and update your registered email address.",
    },
]


cleaned_examples = []

seen = set()


for raw in raw_examples:

    example = clean_example(raw)

    if not is_valid(example):
        continue

    example_hash = get_hash(example)

    if example_hash in seen:
        continue

    seen.add(example_hash)

    cleaned_examples.append(example)


# -----------------------------------------
# 7. Save JSONL
# -----------------------------------------

output_file = "finetuning_dataset.jsonl"


with open(
    output_file,
    "w",
    encoding="utf-8"
) as f:

    for example in cleaned_examples:

        f.write(
            json.dumps(example)
            + "\n"
        )


print(
    f"Saved {len(cleaned_examples)} examples"
)
```

---

# 25. Example output

Your final file looks like:

```json
{"messages":[
  {
    "role":"system",
    "content":"You are a helpful enterprise assistant. Provide accurate and concise answers."
  },
  {
    "role":"user",
    "content":"How do I reset my password?"
  },
  {
    "role":"assistant",
    "content":"Go to the login page and select 'Forgot Password'. Follow the instructions to create a new password."
  }
]}
```

One line = one training example.

---

# 26. How much data should you create?

There is no universal number.

A rough guideline:

| Task                         |       Typical starting point |
| ---------------------------- | ---------------------------: |
| Simple style adaptation      |                     Hundreds |
| Structured output task       |        Hundreds to thousands |
| Narrow domain behavior       |                    Thousands |
| General instruction tuning   |           Tens of thousands+ |
| Foundation model pretraining | Billions/trillions of tokens |

But:

> **1,000 high-quality examples can be more valuable than 100,000 poor synthetic examples.**

For a real project, I would start with a small, carefully curated dataset and establish a baseline before scaling.

---

# 27. Example: Fine-tuning for JSON output

Suppose your application must always extract structured data.

Training example:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "Extract customer information and return valid JSON."
    },
    {
      "role": "user",
      "content": "My name is John and I live in Bangalore."
    },
    {
      "role": "assistant",
      "content": "{\"name\":\"John\",\"city\":\"Bangalore\"}"
    }
  ]
}
```

You would include many variations:

```text
Short input
Long input
Missing values
Ambiguous values
Invalid data
Multiple entities
Edge cases
```

This is where fine-tuning can be very useful: **teaching consistent behavior or output format**.

---

# 28. Real-world recommendation

If you were building a production AI application, I would use:

```text
1. Define the exact behavior to improve
              ↓
2. Collect high-quality examples
              ↓
3. Remove PII and sensitive data
              ↓
4. Clean and normalize
              ↓
5. Remove duplicates
              ↓
6. Validate answers
              ↓
7. Add difficult edge cases
              ↓
8. Split train/validation/test
              ↓
9. Apply the model's chat template
              ↓
10. Tokenize
              ↓
11. Fine-tune with LoRA/QLoRA
              ↓
12. Evaluate against a held-out test set
```

---

# Interview answer

If asked:

> **How would you create a fine-tuning dataset?**

A strong answer is:

> **First, I define the exact capability I want to improve, such as instruction following, domain-specific behavior, classification, or structured output. Then I collect high-quality input-output examples from approved sources, clean and normalize the data, remove duplicates and sensitive information, and validate the responses. I format the data according to the target model's chat template, typically as system, user, and assistant messages. Then I create train, validation, and test splits while avoiding data leakage. Finally, I tokenize the data and use supervised fine-tuning where the loss is usually calculated on the assistant response. I would prioritize data quality and diversity over simply increasing dataset size.**

## The most important rule

```text
Documents ≠ Fine-tuning dataset
```

Instead:

```text
Desired user input
        +
Desired model behavior/output
        =
Fine-tuning example
```

For frequently changing enterprise knowledge, use **RAG**. For changing the model's **behavior, style, format, or specialized task performance**, use **fine-tuning**.
