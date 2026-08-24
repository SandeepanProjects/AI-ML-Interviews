# Suppose you have 1 million customer-support conversations. How would you prepare them for fine-tuning?

This is an excellent **Senior AI/LLM Engineer interview question**.

The important answer is: **I would not directly dump 1 million conversations into an LLM fine-tuning job.**

I would build a data pipeline:

```text
1M Raw Conversations
        │
        ▼
Data ingestion
        │
        ▼
PII / secrets removal
        │
        ▼
Schema validation
        │
        ▼
Conversation reconstruction
        │
        ▼
Quality filtering
        │
        ▼
Deduplication
        │
        ▼
Near-duplicate detection
        │
        ▼
Remove bad / noisy examples
        │
        ▼
Language / domain filtering
        │
        ▼
Length analysis
        │
        ▼
Convert to chat format
        │
        ▼
Train / Validation / Test split
        │
        ▼
Tokenization
        │
        ▼
Fine-tuning dataset
```

Let's build it step by step.

---

# 1. First define the goal

Before processing 1 million conversations, I would ask:

> **What exactly am I trying to teach the model?**

For customer support, possible goals are:

```text
Goal 1 → Answer customer questions
Goal 2 → Generate support-agent replies
Goal 3 → Classify support tickets
Goal 4 → Summarize conversations
Goal 5 → Suggest next actions
Goal 6 → Extract structured information
```

Let's assume the goal is:

> **Fine-tune an LLM to generate high-quality customer-support responses.**

The model should learn:

```text
Customer message
        ↓
Understand problem
        ↓
Generate helpful support response
```

---

# 2. Raw data format

Suppose the raw data comes from:

```text
Zendesk
Salesforce
Intercom
Email
Chat support
```

A raw conversation may look like:

```python
raw_conversation = {
    "conversation_id": "conv_123",
    "customer_id": "user_456",
    "created_at": "2026-01-15T10:30:00",
    "messages": [
        {
            "sender": "customer",
            "text": "Hi, I was charged twice for my subscription."
        },
        {
            "sender": "agent",
            "text": "Sorry about that. Could you share your order number?"
        },
        {
            "sender": "customer",
            "text": "My order number is 123456."
        },
        {
            "sender": "agent",
            "text": "Thanks. I can see the duplicate charge and have initiated a refund."
        }
    ]
}
```

We should **not keep customer IDs or raw personal information in the training data unless there is a lawful, approved reason and proper controls**.

---

# 3. Step 1 — Create a standard schema

Different systems produce different formats.

Normalize them.

```python
from pydantic import BaseModel
from typing import Literal
from datetime import datetime


class Message(BaseModel):
    role: Literal[
        "customer",
        "agent",
        "system"
    ]

    content: str


class Conversation(BaseModel):
    conversation_id: str
    created_at: datetime
    messages: list[Message]
```

Normalize:

```python
def normalize_conversation(raw: dict) -> Conversation:

    normalized_messages = []

    for message in raw["messages"]:

        sender = message["sender"]

        if sender == "customer":
            role = "customer"

        elif sender == "agent":
            role = "agent"

        else:
            role = "system"

        normalized_messages.append(
            Message(
                role=role,
                content=message["text"].strip()
            )
        )

    return Conversation(
        conversation_id=raw["conversation_id"],
        created_at=raw["created_at"],
        messages=normalized_messages
    )
```

Now every source becomes:

```text
conversation_id
created_at
messages
```

---

# 4. Step 2 — Remove PII and secrets

This is one of the most important steps.

Customer support conversations may contain:

```text
Names
Email addresses
Phone numbers
Addresses
Credit card numbers
Account IDs
Passwords
API keys
Tokens
```

Example:

```text
My name is John Smith.
My email is john@gmail.com.
My card is 4111-1111-1111-1111.
```

We should redact them before training.

## Basic example

```python
import re


def redact_pii(text: str) -> str:

    # Email
    text = re.sub(
        r"\b[\w\.-]+@[\w\.-]+\.\w+\b",
        "[EMAIL]",
        text
    )

    # Phone numbers (basic example)
    text = re.sub(
        r"\b\d{10}\b",
        "[PHONE]",
        text
    )

    # Credit card-like numbers
    text = re.sub(
        r"\b(?:\d[ -]*?){13,16}\b",
        "[CARD_NUMBER]",
        text
    )

    return text
```

Example:

```python
text = """
My email is john@example.com.
My card number is 4111 1111 1111 1111.
"""

print(
    redact_pii(text)
)
```

Output:

```text
My email is [EMAIL].
My card number is [CARD_NUMBER].
```

In production, I would **not rely only on regex**. I would combine:

```text
Regex
+
NER / PII detection
+
Secrets detection
+
Organization-specific patterns
+
Manual audits
```

Also, depending on the domain and policy, I may remove entire conversations containing certain sensitive information rather than trying to redact them.

---

# 5. Apply PII cleaning to conversations

```python
def clean_pii(
    conversation: Conversation
) -> Conversation:

    cleaned_messages = []

    for message in conversation.messages:

        cleaned_text = redact_pii(
            message.content
        )

        cleaned_messages.append(
            Message(
                role=message.role,
                content=cleaned_text
            )
        )

    return Conversation(
        conversation_id=conversation.conversation_id,
        created_at=conversation.created_at,
        messages=cleaned_messages
    )
```

---

# 6. Step 3 — Remove empty and corrupted messages

Bad:

```text
""
"   "
"asdfasdf"
"??????"
```

Basic cleaning:

```python
def clean_text(text: str) -> str:

    text = text.strip()

    # Remove repeated whitespace
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text
```

Validate:

```python
def is_valid_message(
    text: str
) -> bool:

    if not text:
        return False

    if len(text) < 2:
        return False

    return True
```

Apply:

```python
def remove_invalid_messages(
    conversation: Conversation
):

    messages = []

    for message in conversation.messages:

        text = clean_text(
            message.content
        )

        if is_valid_message(text):

            messages.append(
                Message(
                    role=message.role,
                    content=text
                )
            )

    conversation.messages = messages

    return conversation
```

---

# 7. Step 4 — Validate conversation structure

We need meaningful conversations.

Bad:

```text
Agent
Agent
Agent
```

No customer request.

Bad:

```text
Customer
Customer
Customer
```

No useful answer.

Basic validation:

```python
def is_valid_conversation(
    conversation: Conversation
) -> bool:

    roles = [
        message.role
        for message in conversation.messages
    ]

    # Minimum messages
    if len(roles) < 2:
        return False

    # Must contain customer
    if "customer" not in roles:
        return False

    # Must contain agent
    if "agent" not in roles:
        return False

    return True
```

You may also validate:

```text
Maximum conversation length
Expected role transitions
Timestamp order
No corrupted encoding
```

---

# 8. Step 5 — Filter low-quality agent responses

Remember:

> During SFT, the model learns from the agent responses.

If the agent says:

```text
idk
```

the model may learn:

```text
idk
```

We need to filter poor responses.

Example:

```python
BAD_RESPONSES = {
    "idk",
    "i don't know",
    "ok",
    "okay",
    "thanks",
    "done"
}


def is_good_agent_response(
    text: str
) -> bool:

    normalized = text.lower().strip()

    if normalized in BAD_RESPONSES:
        return False

    if len(normalized) < 20:
        return False

    return True
```

Example:

```text
Customer:
My payment failed.

Agent:
ok
```

Remove.

But:

```text
Customer:
My payment failed.

Agent:
I'm sorry you're experiencing this issue. Please verify that your card has sufficient funds and try again.
```

Keep.

---

# 9. Better quality filtering

For 1 million conversations, I would use multiple signals.

```text
Quality Score
     │
     ├── Response length
     ├── Completeness
     ├── Toxicity
     ├── PII
     ├── Agent professionalism
     ├── Relevance
     └── Resolution quality
```

Example:

```python
def calculate_quality_score(
    customer_text: str,
    agent_text: str
):

    score = 0

    # Reasonable response length
    if 20 <= len(agent_text) <= 3000:
        score += 1

    # Not empty
    if len(agent_text.strip()) > 0:
        score += 1

    # Basic apology/helpfulness signals
    helpful_words = [
        "sorry",
        "help",
        "please",
        "check",
        "confirm"
    ]

    if any(
        word in agent_text.lower()
        for word in helpful_words
    ):
        score += 1

    return score
```

Keep:

```python
if score >= 2:
    keep_example()
```

In a real system, I would use a more robust quality pipeline and periodically review samples manually.

---

# 10. Step 6 — Exact deduplication

With 1 million conversations, duplicates are common.

Example:

```text
Customer: I forgot my password.
Agent: Please click Forgot Password.
```

Repeated 100,000 times.

We don't want the model over-trained on duplicates.

Create a hash.

```python
import hashlib


def create_hash(text: str) -> str:

    normalized = (
        text
        .lower()
        .strip()
    )

    return hashlib.sha256(
        normalized.encode()
    ).hexdigest()
```

Create a conversation representation:

```python
def conversation_to_text(
    conversation: Conversation
):

    return "\n".join(
        f"{message.role}: "
        f"{message.content}"
        for message in conversation.messages
    )
```

Deduplicate:

```python
def remove_exact_duplicates(
    conversations
):

    seen = set()

    unique = []

    for conversation in conversations:

        text = conversation_to_text(
            conversation
        )

        text_hash = create_hash(
            text
        )

        if text_hash not in seen:

            seen.add(text_hash)

            unique.append(
                conversation
            )

    return unique
```

For 1 million records, don't keep everything in Python memory. In production, use distributed processing or a database/object-store-based deduplication pipeline.

---

# 11. Step 7 — Near-duplicate detection

Exact hashing doesn't catch:

```text
I cannot log in.
```

and:

```text
I am unable to log into my account.
```

For semantic deduplication, use embeddings.

```text
Conversation
      ↓
Embedding Model
      ↓
Vector
      ↓
Similarity Search
      ↓
Remove highly similar duplicates
```

Example:

```python
from sentence_transformers import SentenceTransformer


embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)
```

Create embeddings:

```python
texts = [
    conversation_to_text(conv)
    for conv in conversations
]

embeddings = embedding_model.encode(
    texts,
    normalize_embeddings=True,
    batch_size=256
)
```

At 1 million examples, **do not create a full NxN similarity matrix**:

```text
1,000,000 × 1,000,000
```

That is impractical.

Instead use approximate nearest-neighbor search, for example:

```text
Embeddings
    ↓
Vector Index
    ↓
Find nearest neighbors
    ↓
Compare similarity
```

Pseudo-production flow:

```python
# Conceptual code
index.add(embeddings)

distances, neighbors = index.search(
    embeddings,
    k=5
)
```

Then:

```python
SIMILARITY_THRESHOLD = 0.98

duplicates = []

for i, neighbor_ids in enumerate(neighbors):

    for j in neighbor_ids:

        if i == j:
            continue

        similarity = cosine_similarity(
            embeddings[i],
            embeddings[j]
        )

        if similarity > SIMILARITY_THRESHOLD:

            duplicates.append(
                (i, j)
            )
```

In practice, thresholds must be tuned on manually labeled duplicate/non-duplicate examples.

---

# 12. Step 8 — Convert conversations into training examples

This is extremely important.

A conversation:

```text
Customer:
My payment failed.

Agent:
Can you share the error?

Customer:
It says insufficient funds.

Agent:
Please contact your bank or try another payment method.
```

Can become multiple SFT examples.

## Example 1

```text
Input:
Customer: My payment failed.

Target:
Can you share the error?
```

## Example 2

```text
Input:
Customer: My payment failed.
Agent: Can you share the error?
Customer: It says insufficient funds.

Target:
Please contact your bank or try another payment method.
```

This teaches the model:

```text
Conversation history
        ↓
Next agent response
```

---

# 13. Code to create training turns

```python
def conversation_to_examples(
    conversation: Conversation
):

    examples = []

    history = []

    for message in conversation.messages:

        if message.role == "customer":

            history.append({
                "role": "user",
                "content": message.content
            })

        elif message.role == "agent":

            if not history:
                continue

            examples.append({
                "conversation_id":
                    conversation.conversation_id,

                "messages":
                    history.copy(),

                "response":
                    message.content
            })

            history.append({
                "role": "assistant",
                "content": message.content
            })

    return examples
```

Example output:

```python
examples = conversation_to_examples(
    conversation
)

for example in examples:
    print(example)
```

Output conceptually:

```python
{
    "conversation_id": "conv_123",
    "messages": [
        {
            "role": "user",
            "content": "My payment failed."
        }
    ],
    "response": "Can you share the error?"
}
```

Second example:

```python
{
    "conversation_id": "conv_123",
    "messages": [
        {
            "role": "user",
            "content": "My payment failed."
        },
        {
            "role": "assistant",
            "content": "Can you share the error?"
        },
        {
            "role": "user",
            "content": "It says insufficient funds."
        }
    ],
    "response": "Please contact your bank or try another payment method."
}
```

---

# 14. Prevent low-quality context from entering training

Suppose an earlier agent message was bad:

```text
Agent: idk
```

Should that be included in the context for later training examples?

Probably not.

A safer approach:

```python
def conversation_to_examples(
    conversation: Conversation
):

    examples = []

    history = []

    for message in conversation.messages:

        if message.role == "customer":

            history.append({
                "role": "user",
                "content": message.content
            })

        elif message.role == "agent":

            if not is_good_agent_response(
                message.content
            ):
                continue

            if history:

                examples.append({
                    "conversation_id":
                        conversation.conversation_id,

                    "messages":
                        history.copy(),

                    "response":
                        message.content
                })

                history.append({
                    "role": "assistant",
                    "content":
                        message.content
                })

    return examples
```

---

# 15. Step 9 — Convert to chat template

Modern instruction-tuned models expect a specific chat format.

For example:

```python
example = {
    "messages": [
        {
            "role": "system",
            "content": (
                "You are a helpful customer "
                "support assistant."
            )
        },
        {
            "role": "user",
            "content":
                "I was charged twice."
        },
        {
            "role": "assistant",
            "content":
                "I'm sorry about that. "
                "I'll help you investigate the duplicate charge."
        }
    ]
}
```

Use the tokenizer's chat template:

```python
def format_chat_example(
    example,
    tokenizer
):

    return tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False
    )
```

This is better than manually guessing special tokens because different models use different templates.

---

# 16. Step 10 — Split train/validation/test correctly

This is one of the most important interview points.

## Wrong

Split individual messages randomly.

```text
Conversation 1:

Customer → TRAIN
Agent    → VALIDATION
Customer → TRAIN
Agent    → TEST
```

This is catastrophic leakage.

The model can effectively see the same conversation during training and validation.

---

## Correct

Split by conversation ID.

```text
Conversation 1 → TRAIN
Conversation 2 → TRAIN
Conversation 3 → VALIDATION
Conversation 4 → TEST
```

Code:

```python
from sklearn.model_selection import (
    GroupShuffleSplit
)
```

Assume:

```python
examples = [
    ...
]
```

Each example has:

```python
example["conversation_id"]
```

Create groups:

```python
groups = [
    example["conversation_id"]
    for example in examples
]
```

First split:

```python
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_indices, temp_indices = next(
    splitter.split(
        examples,
        groups=groups
    )
)
```

Create datasets:

```python
train_examples = [
    examples[i]
    for i in train_indices
]

temp_examples = [
    examples[i]
    for i in temp_indices
]
```

Now:

```text
Train = 80%
Temporary = 20%
```

Split temporary:

```python
temp_groups = [
    example["conversation_id"]
    for example in temp_examples
]

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.50,
    random_state=42
)

val_indices, test_indices = next(
    splitter.split(
        temp_examples,
        groups=temp_groups
    )
)
```

Create:

```python
val_examples = [
    temp_examples[i]
    for i in val_indices
]

test_examples = [
    temp_examples[i]
    for i in test_indices
]
```

Final:

```text
1,000,000 conversations

        │
        ├── Train: 800,000
        │
        ├── Validation: 100,000
        │
        └── Test: 100,000
```

---

# 17. Verify no leakage

```python
train_ids = {
    example["conversation_id"]
    for example in train_examples
}

val_ids = {
    example["conversation_id"]
    for example in val_examples
}

test_ids = {
    example["conversation_id"]
    for example in test_examples
}
```

Assertions:

```python
assert (
    train_ids.isdisjoint(val_ids)
)

assert (
    train_ids.isdisjoint(test_ids)
)

assert (
    val_ids.isdisjoint(test_ids)
)
```

If all pass:

```text
No conversation-level leakage
```

---

# 18. Customer-level leakage

There is another subtle issue.

Suppose the same customer has:

```text
Conversation 1 → TRAIN
Conversation 2 → VALIDATION
Conversation 3 → TEST
```

Potentially:

```text
Same writing style
Same account context
Repeated problem
```

For strict evaluation, you might split by:

```text
customer_id
```

instead of:

```text
conversation_id
```

```text
Customer A
    ├── Conversation 1 → TRAIN
    ├── Conversation 2 → TRAIN
    └── Conversation 3 → TRAIN
```

Code:

```python
groups = [
    example["customer_group_id"]
    for example in examples
]
```

However, for customer support, you need to decide whether your production scenario expects returning users. The split strategy should match the evaluation question.

---

# 19. Time-based split

For production systems, I often also use a time-based test.

```text
January → Training
February → Training
March → Training
April → Validation
May → Test
```

Why?

Because production changes:

```text
New policies
New products
New issues
New language
```

Code:

```python
from datetime import datetime


sorted_examples = sorted(
    examples,
    key=lambda x: x["created_at"]
)
```

Then:

```python
n = len(sorted_examples)

train_end = int(n * 0.8)

val_end = int(n * 0.9)


train_data = sorted_examples[
    :train_end
]

val_data = sorted_examples[
    train_end:val_end
]

test_data = sorted_examples[
    val_end:
]
```

A time-based test gives a more realistic future generalization evaluation.

---

# 20. Step 11 — Analyze token lengths

Now tokenize.

```python
from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained(
    "your-model"
)
```

Count tokens:

```python
def count_tokens(
    example,
    tokenizer
):

    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False
    )

    tokens = tokenizer(
        text,
        truncation=False
    )

    return len(
        tokens["input_ids"]
    )
```

Analyze:

```python
lengths = [
    count_tokens(
        example,
        tokenizer
    )
    for example in train_examples
]
```

Calculate:

```python
import numpy as np


print(
    "P50:",
    np.percentile(lengths, 50)
)

print(
    "P90:",
    np.percentile(lengths, 90)
)

print(
    "P95:",
    np.percentile(lengths, 95)
)

print(
    "P99:",
    np.percentile(lengths, 99)
)
```

Suppose:

```text
P50 = 350
P90 = 1200
P95 = 2000
P99 = 6000
```

A reasonable starting point:

```python
MAX_SEQ_LENGTH = 4096
```

Then measure truncation.

---

# 21. Handle long conversations

Suppose:

```text
Conversation = 50 messages
Total = 10,000 tokens
```

But:

```python
MAX_SEQ_LENGTH = 4096
```

Do not blindly keep:

```text
First 4096 tokens
```

You may lose the latest customer problem.

For conversational support, a reasonable policy might be:

```text
System prompt
        +
First user issue summary
        +
Recent conversation turns
        +
Current customer message
        +
Target agent response
```

Example:

```python
def build_context(
    messages,
    max_turns=8
):

    # Keep the latest messages
    recent_messages = messages[
        -max_turns:
    ]

    return recent_messages
```

But production implementations should usually be token-budget based, not just turn-count based.

---

# 22. Token-budget-based truncation

```python
def trim_messages_to_token_budget(
    messages,
    tokenizer,
    max_tokens
):

    selected = []

    total_tokens = 0

    # Start from latest messages
    for message in reversed(messages):

        token_count = len(
            tokenizer(
                message["content"],
                add_special_tokens=False
            )["input_ids"]
        )

        if (
            total_tokens
            + token_count
            > max_tokens
        ):
            break

        selected.append(
            message
        )

        total_tokens += token_count

    return list(
        reversed(selected)
    )
```

Usage:

```python
context = trim_messages_to_token_budget(
    messages=conversation_history,
    tokenizer=tokenizer,
    max_tokens=3000
)
```

Reserve space:

```text
4096 total tokens

3000 → Conversation context
 100 → System prompt
 996 → Agent response
```

The exact budgeting should use the model's actual chat template, because role tokens and special tokens also consume context.

---

# 23. Step 12 — Label masking for SFT

For causal language modeling, we typically do not want to train the model to predict the user's message.

Example:

```text
User:
My payment failed.

Assistant:
Please check whether your card has sufficient funds.
```

We want loss mainly on:

```text
Assistant response
```

Conceptually:

```text
Input IDs:

[User tokens] [Assistant tokens]

Labels:

[-100]        [Actual token IDs]
```

`-100` means:

```text
Ignore this token in loss calculation
```

Example:

```python
input_ids = [
    10, 11, 12, 13, 14,
    20, 21, 22, 23
]

labels = [
    -100,
    -100,
    -100,
    -100,
    -100,
    20,
    21,
    22,
    23
]
```

Loss is calculated only for:

```text
20, 21, 22, 23
```

For multi-turn chat, use a trainer/collator compatible with the model's chat template and verify that only the intended assistant spans are supervised.

---

# 24. Step 13 — Save in JSONL

A common format:

```json
{"messages":[{"role":"system","content":"You are a helpful customer support assistant."},{"role":"user","content":"My payment failed."},{"role":"assistant","content":"I'm sorry you're experiencing this. Please check your card details and try again."}]}
```

Code:

```python
import json


def save_jsonl(
    examples,
    file_path
):

    with open(
        file_path,
        "w",
        encoding="utf-8"
    ) as file:

        for example in examples:

            json.dump(
                {
                    "messages":
                        example["messages"]
                },
                file
            )

            file.write("\n")
```

Save:

```python
save_jsonl(
    train_examples,
    "train.jsonl"
)

save_jsonl(
    val_examples,
    "validation.jsonl"
)

save_jsonl(
    test_examples,
    "test.jsonl"
)
```

---

# 25. For 1 million records: don't use a simple Python list pipeline

This is important.

This:

```python
conversations = []

for record in one_million_records:
    conversations.append(record)
```

may be inefficient or fail depending on record size and machine memory.

Instead use:

```text
Object Storage
     ↓
Parquet / JSONL
     ↓
Distributed processing
     ↓
Cleaned Parquet
     ↓
Quality filtered data
     ↓
Deduplicated dataset
     ↓
Train/Val/Test files
```

Possible technologies:

```text
Small/medium scale:
Python + Hugging Face Datasets

Large scale:
Spark
Ray
Polars
DuckDB
Dataflow systems
```

A practical scalable pattern is:

```text
Raw data
   ↓
Partition by date/source
   ↓
Distributed cleaning
   ↓
PII redaction
   ↓
Quality filtering
   ↓
Deduplication
   ↓
Versioned dataset
   ↓
Fine-tuning
```

---

# 26. Example using Hugging Face Datasets

```python
from datasets import load_dataset


dataset = load_dataset(
    "json",
    data_files={
        "train":
            "raw/*.jsonl"
    }
)
```

Map:

```python
def process_record(record):

    # normalize
    conversation = normalize_conversation(
        record
    )

    # PII cleaning
    conversation = clean_pii(
        conversation
    )

    # remove invalid messages
    conversation = remove_invalid_messages(
        conversation
    )

    return {
        "conversation_id":
            conversation.conversation_id,

        "messages": [
            {
                "role":
                    message.role,

                "content":
                    message.content
            }
            for message in conversation.messages
        ]
    }
```

Apply:

```python
processed_dataset = dataset.map(
    process_record,
    batched=False,
    num_proc=8
)
```

Filter:

```python
processed_dataset = processed_dataset.filter(
    lambda x:
        len(x["messages"]) >= 2,
    num_proc=8
)
```

For a million conversations, tune parallelism based on CPU, memory, I/O, and storage.

---

# 27. Data quality audit

Before training, randomly sample examples.

```python
import random


samples = random.sample(
    train_examples,
    10
)

for example in samples:

    print("=" * 80)

    for message in example["messages"]:

        print(
            message["role"],
            ":",
            message["content"]
        )
```

Look for:

```text
PII leaks
Bad agent answers
Wrong roles
Broken conversations
Duplicate data
Toxic content
Incorrect answers
Internal system messages
Secrets
Prompt injections
```

For large datasets, random sampling should be combined with targeted sampling of rare categories and high-risk filters.

---

# 28. Dataset statistics dashboard

Before fine-tuning, I would generate:

```text
Total conversations
Total messages
Total training examples

Languages
Support categories
Average conversation length
P50 / P90 / P95 token length

PII detection rate
Duplicate rate
Near duplicate rate
Removed records
Quality score distribution
Train/Validation/Test sizes
```

Example:

```text
Raw conversations:          1,000,000

After schema validation:      980,000
After PII/security filtering: 940,000
After quality filtering:      820,000
After deduplication:          700,000

Train:                        560,000
Validation:                    70,000
Test:                          70,000
```

These numbers are just illustrative; the actual retention rate depends heavily on the source data and filtering policy.

---

# 29. The complete pipeline architecture

```text
                    1 MILLION RAW CONVERSATIONS
                              │
                              ▼
                     ┌──────────────────┐
                     │ DATA INGESTION   │
                     └────────┬─────────┘
                              │
                              ▼
                     ┌──────────────────┐
                     │ NORMALIZATION    │
                     │ Schema / roles   │
                     └────────┬─────────┘
                              │
                              ▼
                     ┌──────────────────┐
                     │ PRIVACY & SAFETY │
                     │ PII / secrets    │
                     └────────┬─────────┘
                              │
                              ▼
                     ┌──────────────────┐
                     │ QUALITY FILTER   │
                     │ Bad conversations│
                     └────────┬─────────┘
                              │
                              ▼
                     ┌──────────────────┐
                     │ DEDUPLICATION    │
                     │ Exact + semantic │
                     └────────┬─────────┘
                              │
                              ▼
                     ┌──────────────────┐
                     │ CREATE SFT TURNS │
                     └────────┬─────────┘
                              │
                              ▼
                     ┌──────────────────┐
                     │ GROUPED SPLIT    │
                     │ Train/Val/Test   │
                     └────────┬─────────┘
                              │
                              ▼
                     ┌──────────────────┐
                     │ TOKEN ANALYSIS   │
                     │ Max sequence     │
                     └────────┬─────────┘
                              │
                              ▼
                     ┌──────────────────┐
                     │ DATASET VERSION  │
                     └────────┬─────────┘
                              │
                              ▼
                         FINE-TUNING
```

# Strong interview answer

> If I had one million customer-support conversations, I would first define the fine-tuning objective rather than training on all raw data blindly. I would normalize conversations into a common schema, remove or appropriately handle PII and secrets, validate conversation structure, and filter low-quality or unsafe agent responses. Then I would perform exact and scalable semantic deduplication, convert multi-turn conversations into supervised chat examples, and use the model's chat template. I would split the data at the conversation or customer level to prevent leakage, and potentially maintain a time-based test set for future generalization. Next, I would analyze token-length distributions, handle long conversations using token budgets and context selection, and verify supervision labels so the loss is applied to the intended assistant responses. Finally, I would version the dataset, run automated and human quality audits, and only then start fine-tuning.

The key principle is:

```text
1 million conversations
        ≠
1 million good training examples

Quality
+
Privacy
+
Correct supervision
+
No leakage
>
Raw data volume
```
