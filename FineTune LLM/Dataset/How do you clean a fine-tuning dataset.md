# How do you clean a fine-tuning dataset?

Cleaning a fine-tuning dataset is a critical step because:

> **The model learns patterns, quality, mistakes, biases, and formatting from your training data.**

A typical pipeline looks like:

```text
Raw Data
   │
   ▼
Schema Validation
   │
   ▼
Text Cleaning
   │
   ▼
PII / Sensitive Data Removal
   │
   ▼
Duplicate Detection
   │
   ▼
Quality Checks
   │
   ▼
Bad Example Detection
   │
   ▼
Human Review
   │
   ▼
Train / Validation / Test Split
```

We'll build this step by step.

---

# 1. Example of a raw dataset

Suppose we have conversational fine-tuning data:

```python
raw_examples = [
    {
        "messages": [
            {
                "role": "user",
                "content": "What is LoRA?"
            },
            {
                "role": "assistant",
                "content": "LoRA is a parameter efficient fine tuning method."
            }
        ]
    },

    # Duplicate
    {
        "messages": [
            {
                "role": "user",
                "content": "What is LoRA?"
            },
            {
                "role": "assistant",
                "content": "LoRA is a parameter efficient fine tuning method."
            }
        ]
    },

    # Bad example - empty answer
    {
        "messages": [
            {
                "role": "user",
                "content": "Explain QLoRA"
            },
            {
                "role": "assistant",
                "content": ""
            }
        ]
    },

    # Bad example - meaningless answer
    {
        "messages": [
            {
                "role": "user",
                "content": "What is RAG?"
            },
            {
                "role": "assistant",
                "content": "yes yes yes yes yes"
            }
        ]
    }
]
```

Our job is to transform:

```text
Messy Data
    ↓
Validated
    ↓
Clean
    ↓
Deduplicated
    ↓
Quality Checked
    ↓
Fine-Tuning Dataset
```

---

# 2. Step 1: Validate the schema

Before checking quality, make sure every example has the correct structure.

For a conversational dataset:

```text
messages
   │
   ├── role
   └── content
```

We can use Pydantic.

```python
from pydantic import BaseModel
from typing import Literal


class Message(BaseModel):
    role: Literal[
        "system",
        "user",
        "assistant"
    ]

    content: str


class Conversation(BaseModel):
    messages: list[Message]
```

Now validate:

```python
def validate_schema(example):

    try:
        Conversation.model_validate(example)
        return True

    except Exception as e:

        print(
            "Invalid schema:",
            e
        )

        return False
```

Usage:

```python
valid_examples = []

for example in raw_examples:

    if validate_schema(example):
        valid_examples.append(example)
```

This removes examples with:

```text
❌ Missing messages
❌ Missing role
❌ Missing content
❌ Invalid role
❌ Wrong data type
```

---

# 3. Step 2: Validate conversation structure

Schema validation is not enough.

This is valid JSON:

```json
{
  "messages": [
    {
      "role": "assistant",
      "content": "Hello"
    },
    {
      "role": "user",
      "content": "What is LoRA?"
    }
  ]
}
```

But it's usually not a valid training conversation.

We want:

```text
system (optional)
       ↓
user
       ↓
assistant
       ↓
user
       ↓
assistant
```

Code:

```python
def validate_conversation_structure(example):

    messages = example["messages"]

    if len(messages) < 2:
        return False

    # System can only appear at beginning
    start_index = 0

    if messages[0]["role"] == "system":
        start_index = 1

    # Remaining conversation must exist
    conversation = messages[start_index:]

    if len(conversation) < 2:
        return False

    # Must start with user
    if conversation[0]["role"] != "user":
        return False

    # Must alternate user -> assistant
    for i, message in enumerate(conversation):

        expected_role = (
            "user"
            if i % 2 == 0
            else "assistant"
        )

        if message["role"] != expected_role:
            return False

    # Usually end with assistant
    if conversation[-1]["role"] != "assistant":
        return False

    return True
```

Usage:

```python
clean_structure_examples = [
    example
    for example in valid_examples
    if validate_conversation_structure(example)
]
```

---

# 4. Step 3: Clean text

Raw enterprise data may contain:

```text
Extra spaces
HTML
Tabs
Newlines
Broken Unicode
Control characters
```

Example:

```text
"  What    is   LoRA?   \n\n"
```

We normalize it.

```python
import re
import html


def clean_text(text: str) -> str:

    # Decode HTML entities
    text = html.unescape(text)

    # Remove HTML tags
    text = re.sub(
        r"<[^>]+>",
        "",
        text
    )

    # Remove control characters
    text = re.sub(
        r"[\x00-\x08\x0B\x0C\x0E-\x1F]",
        "",
        text
    )

    # Normalize whitespace
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()
```

Example:

```python
text = "  <p>What   is LoRA?</p>\n\n"

print(clean_text(text))
```

Output:

```text
What is LoRA?
```

Apply to the entire dataset:

```python
def clean_conversation(example):

    cleaned_messages = []

    for message in example["messages"]:

        cleaned_messages.append(
            {
                "role": message["role"],
                "content": clean_text(
                    message["content"]
                )
            }
        )

    return {
        "messages": cleaned_messages
    }
```

---

# 5. Step 4: Remove empty messages

A very common problem:

```json
{
    "role": "assistant",
    "content": ""
}
```

or:

```text
"     "
```

Validation:

```python
def has_empty_messages(example):

    for message in example["messages"]:

        if not message["content"].strip():
            return True

    return False
```

Filter:

```python
non_empty_examples = [
    example
    for example in clean_structure_examples
    if not has_empty_messages(example)
]
```

---

# 6. How do you remove exact duplicate examples?

The simplest technique is hashing.

Example:

```text
Example
   ↓
Canonical JSON
   ↓
SHA256 Hash
   ↓
Seen before?
   │
 ┌─┴──┐
Yes   No
↓      ↓
Drop   Keep
```

Code:

```python
import json
import hashlib


def example_hash(example):

    canonical_json = json.dumps(
        example,
        sort_keys=True,
        ensure_ascii=False
    )

    return hashlib.sha256(
        canonical_json.encode("utf-8")
    ).hexdigest()
```

Now deduplicate:

```python
def remove_exact_duplicates(examples):

    seen = set()

    unique_examples = []

    for example in examples:

        hash_value = example_hash(example)

        if hash_value in seen:
            continue

        seen.add(hash_value)

        unique_examples.append(example)

    return unique_examples
```

Usage:

```python
unique_examples = remove_exact_duplicates(
    non_empty_examples
)

print(
    "Before:",
    len(non_empty_examples)
)

print(
    "After:",
    len(unique_examples)
)
```

---

# 7. Why exact deduplication is not enough

Consider:

```text
Example 1:
What is LoRA?

LoRA is a parameter-efficient fine-tuning technique.
```

Example 2:

```text
What is LoRA?

LoRA is a parameter efficient fine tuning technique.
```

These are almost identical, but the hashes are different.

This is called:

> **Near-duplicate detection**

---

# 8. Normalize before hashing

We can improve exact deduplication.

```python
def normalize_for_hash(text: str):

    text = text.lower()

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    text = re.sub(
        r"[^\w\s]",
        "",
        text
    )

    return text.strip()
```

Create a normalized representation:

```python
def normalized_example_hash(example):

    normalized_messages = []

    for message in example["messages"]:

        normalized_messages.append(
            {
                "role": message["role"],
                "content": normalize_for_hash(
                    message["content"]
                )
            }
        )

    content = json.dumps(
        normalized_messages,
        sort_keys=True,
        ensure_ascii=False
    )

    return hashlib.sha256(
        content.encode()
    ).hexdigest()
```

Then:

```python
def remove_normalized_duplicates(examples):

    seen = set()

    unique_examples = []

    for example in examples:

        h = normalized_example_hash(example)

        if h not in seen:

            seen.add(h)

            unique_examples.append(example)

    return unique_examples
```

---

# 9. Detect semantic duplicates

Sometimes examples are different textually:

```text
How can I reset my password?
```

and:

```text
I forgot my password. How do I regain access?
```

These may represent the same intent.

For semantic duplicate detection, use embeddings.

Conceptually:

```text
Example 1
    ↓
Embedding
    [0.12, 0.91, ...]


Example 2
    ↓
Embedding
    [0.11, 0.89, ...]


Cosine similarity
    ↓
0.96

Near duplicate
```

Code using sentence-transformers:

```bash
pip install sentence-transformers
```

Then:

```python
from sentence_transformers import SentenceTransformer
```

Load model:

```python
embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)
```

Create a representation:

```python
def conversation_to_text(example):

    parts = []

    for message in example["messages"]:

        parts.append(
            f"{message['role']}: "
            f"{message['content']}"
        )

    return "\n".join(parts)
```

Generate embeddings:

```python
texts = [
    conversation_to_text(example)
    for example in unique_examples
]

embeddings = embedding_model.encode(
    texts,
    normalize_embeddings=True
)
```

Cosine similarity:

```python
from sklearn.metrics.pairwise import cosine_similarity


similarity_matrix = cosine_similarity(
    embeddings
)
```

Find duplicates:

```python
threshold = 0.95

duplicate_pairs = []

for i in range(len(texts)):

    for j in range(i + 1, len(texts)):

        similarity = similarity_matrix[i][j]

        if similarity >= threshold:

            duplicate_pairs.append(
                {
                    "example_1": i,
                    "example_2": j,
                    "similarity": float(similarity)
                }
            )
```

Print:

```python
for pair in duplicate_pairs:

    print(pair)
```

Example:

```text
{
    'example_1': 5,
    'example_2': 17,
    'similarity': 0.97
}
```

---

# 10. A more scalable semantic deduplication approach

The previous method creates an:

[
N \times N
]

similarity matrix.

For 100,000 examples:

```text
100,000 × 100,000
```

This is expensive.

In production, use a vector index such as:

* FAISS
* Qdrant
* approximate nearest neighbor search

Example with FAISS:

```python
import faiss
import numpy as np
```

Assuming normalized embeddings:

```python
embeddings_np = np.array(
    embeddings,
    dtype="float32"
)

dimension = embeddings_np.shape[1]

index = faiss.IndexFlatIP(
    dimension
)

index.add(
    embeddings_np
)
```

Search nearest neighbors:

```python
k = 5

scores, indices = index.search(
    embeddings_np,
    k
)
```

Find near duplicates:

```python
threshold = 0.95

near_duplicates = []

for i in range(len(texts)):

    for score, j in zip(
        scores[i],
        indices[i]
    ):

        if i == j:
            continue

        if score >= threshold:

            near_duplicates.append(
                {
                    "source": i,
                    "duplicate": int(j),
                    "similarity": float(score)
                }
            )
```

For very large datasets, use an approximate index rather than brute-force similarity.

---

# 11. How do you detect bad training examples?

There is no single rule.

You should use multiple quality checks.

```text
Training Example
       │
       ├── Schema quality
       │
       ├── Structural quality
       │
       ├── Length quality
       │
       ├── Repetition
       │
       ├── Language quality
       │
       ├── Relevance
       │
       ├── Factual correctness
       │
       └── Safety / policy checks
```

---

# 12. Detect very short answers

Bad:

```text
yes
```

```text
okay
```

```text
LoRA
```

Code:

```python
def is_too_short(text, min_words=3):

    words = text.split()

    return len(words) < min_words
```

Usage:

```python
answer = "yes"

print(
    is_too_short(answer)
)
```

Output:

```text
True
```

But be careful: some tasks legitimately have short answers.

For example:

```text
User:
What is 2 + 2?

Assistant:
4
```

So quality rules should be **task-specific**, not blindly applied.

---

# 13. Detect extremely long examples

Very long examples can cause:

* truncation
* wasted GPU memory
* irrelevant content

Simple character check:

```python
def is_too_long(
    text,
    max_chars=20000
):

    return len(text) > max_chars
```

Better: check tokens.

```python
def count_tokens(text, tokenizer):

    return len(
        tokenizer.encode(
            text,
            add_special_tokens=False
        )
    )
```

Then:

```python
def is_too_long_for_model(
    text,
    tokenizer,
    max_tokens=2048
):

    token_count = count_tokens(
        text,
        tokenizer
    )

    return token_count > max_tokens
```

---

# 14. Detect repeated text

Bad answer:

```text
LoRA is good. LoRA is good. LoRA is good.
LoRA is good. LoRA is good.
```

One simple method is checking repeated n-grams.

```python
from collections import Counter


def repetition_ratio(
    text,
    n=3
):

    words = text.lower().split()

    if len(words) < n:
        return 0

    ngrams = [
        tuple(words[i:i + n])
        for i in range(
            len(words) - n + 1
        )
    ]

    counts = Counter(ngrams)

    repeated = sum(
        count
        for count in counts.values()
        if count > 1
    )

    return repeated / len(ngrams)
```

Example:

```python
text = (
    "LoRA is useful LoRA is useful "
    "LoRA is useful LoRA is useful"
)

score = repetition_ratio(text)

print(score)
```

If:

```text
repetition_ratio > 0.5
```

you might flag it for review.

---

# 15. Detect low diversity

Another problem:

```text
aaaaaaaaaaaaaaa
```

or:

```text
!!!!!!!!!!!!!!!
```

Check unique word ratio:

```python
def lexical_diversity(text):

    words = text.lower().split()

    if not words:
        return 0

    unique_words = set(words)

    return (
        len(unique_words)
        / len(words)
    )
```

Example:

```python
text = "yes yes yes yes yes"

print(
    lexical_diversity(text)
)
```

Output:

```text
0.2
```

A very low score can indicate repetitive or low-quality output.

Again, use it as a signal—not an automatic truth.

---

# 16. Detect prompt-answer mismatch

Example:

```text
User:
What is LoRA?

Assistant:
Python is a programming language.
```

The schema is valid.

The text is clean.

But the answer is wrong for the question.

We need semantic relevance.

One approach:

```text
User Question
      ↓
Embedding

Assistant Answer
      ↓
Embedding

Cosine Similarity
      ↓
Low similarity → flag
```

Code:

```python
from sentence_transformers import util


def relevance_score(
    question,
    answer,
    model
):

    question_embedding = model.encode(
        question,
        convert_to_tensor=True
    )

    answer_embedding = model.encode(
        answer,
        convert_to_tensor=True
    )

    return float(
        util.cos_sim(
            question_embedding,
            answer_embedding
        )[0][0]
    )
```

Example:

```python
score = relevance_score(
    "What is LoRA?",
    "Python is a programming language.",
    embedding_model
)

print(score)
```

Low relevance:

```text
0.10 → suspicious
```

High relevance:

```text
0.80 → likely relevant
```

### Important

Embedding similarity alone does **not** prove correctness.

An answer can be relevant but factually wrong.

Example:

```text
Question:
What is LoRA?

Answer:
LoRA is a full fine-tuning technique where every model parameter is updated.
```

Semantically relevant.

Factually incorrect.

So you need additional validation.

---

# 17. Detect factual errors

For high-quality fine-tuning data, factual correctness is one of the hardest problems.

Possible strategies:

```text
Source Documents
      ↓
Claim Extraction
      ↓
Evidence Retrieval
      ↓
LLM / Rule-based Verification
      ↓
Confidence Score
      ↓
Human Review
```

For example, if your dataset comes from a trusted FAQ:

```python
source_of_truth = {
    "What is LoRA?": (
        "LoRA is a parameter-efficient fine-tuning "
        "method that trains low-rank adapters."
    )
}
```

You can compare generated answers against approved references.

For production datasets, a common approach is:

```text
Candidate Example
       ↓
Retriever
       ↓
Trusted Source
       ↓
LLM Judge
       ↓
Score
```

Example evaluation prompt:

```text
You are a dataset quality evaluator.

Question:
{question}

Reference Answer:
{reference}

Candidate Answer:
{candidate}

Evaluate:

1. Factual correctness
2. Relevance
3. Completeness

Return JSON:

{
    "correctness": 0-10,
    "relevance": 0-10,
    "completeness": 0-10,
    "reason": "..."
}
```

Then filter:

```python
def should_keep(result):

    return (
        result["correctness"] >= 8
        and result["relevance"] >= 8
    )
```

In a real production pipeline, I would sample and manually audit the automatically accepted examples.

---

# 18. Detect conflicting examples

Suppose your dataset contains:

### Example 1

```text
Question:
What is LoRA?

Answer:
LoRA trains low-rank adapters.
```

### Example 2

```text
Question:
What is LoRA?

Answer:
LoRA updates every parameter.
```

The model receives conflicting signals.

You should detect this.

Conceptually:

```text
Similar Questions
       │
       ▼
Different Answers
       │
       ▼
Semantic Comparison
       │
       ▼
Contradiction?
       │
      Yes
       │
       ▼
Human Review
```

This is especially important for:

* enterprise policy data
* customer support data
* medical/legal datasets
* financial datasets

---

# 19. Detect low-quality synthetic data

Synthetic data can contain:

```text
Generic responses
Repeated patterns
Hallucinations
Unrealistic conversations
Overly formal language
Model-generated artifacts
```

For example:

```text
Certainly! I'd be happy to help you with that!
```

repeated thousands of times.

We can inspect common phrases:

```python
from collections import Counter


def find_common_phrases(
    examples,
    phrase_length=3
):

    counter = Counter()

    for example in examples:

        for message in example["messages"]:

            if message["role"] != "assistant":
                continue

            words = message["content"].lower().split()

            for i in range(
                len(words) - phrase_length + 1
            ):

                phrase = tuple(
                    words[
                        i:i + phrase_length
                    ]
                )

                counter[phrase] += 1

    return counter.most_common(20)
```

If you find:

```text
"certainly i'd be" → 20,000 times
```

that may indicate poor diversity.

---

# 20. PII detection

Before fine-tuning, remove sensitive information.

Examples:

```text
Email addresses
Phone numbers
Credit cards
Account numbers
Employee IDs
```

Basic demonstration:

```python
import re


def redact_pii(text):

    # Email
    text = re.sub(
        r"\b[\w.%+-]+@[\w.-]+\.[A-Za-z]{2,}\b",
        "[EMAIL_REDACTED]",
        text
    )

    # Phone number - simple pattern
    text = re.sub(
        r"\b\d{10}\b",
        "[PHONE_REDACTED]",
        text
    )

    # Credit card - simplified
    text = re.sub(
        r"\b(?:\d[ -]*?){13,16}\b",
        "[CARD_REDACTED]",
        text
    )

    return text
```

Apply:

```python
def redact_example(example):

    messages = []

    for message in example["messages"]:

        messages.append(
            {
                "role": message["role"],
                "content": redact_pii(
                    message["content"]
                )
            }
        )

    return {
        "messages": messages
    }
```

For production, use dedicated PII detection and data-governance tooling, plus human review for high-risk datasets.

---

# 21. Build a quality score

Instead of immediately deleting examples, assign a quality score.

Example:

```text
Schema Valid          20 points
Conversation Valid    20 points
Not Empty             10 points
Good Length           10 points
Low Repetition        10 points
Relevant Answer       20 points
Verified Correct      10 points
────────────────────────────
Total                100
```

Code:

```python
def calculate_quality_score(
    example,
    relevance=None
):

    score = 0

    # Conversation structure
    if validate_conversation_structure(example):
        score += 20

    # Empty messages
    if not has_empty_messages(example):
        score += 10

    # Assistant answer
    answer = example["messages"][-1]["content"]

    # Length
    if 10 <= len(answer) <= 5000:
        score += 10

    # Repetition
    if repetition_ratio(answer) < 0.3:
        score += 10

    # Relevance
    if relevance is not None:

        if relevance >= 0.5:
            score += 20

    return score
```

Then:

```python
for example in unique_examples:

    score = calculate_quality_score(
        example
    )

    if score >= 40:

        print(
            "KEEP:",
            score
        )

    else:

        print(
            "REVIEW/DROP:",
            score
        )
```

The thresholds should be calibrated against your actual task and manually reviewed samples.

---

# 22. Complete dataset cleaning pipeline

Now let's combine everything.

```python
import json
import re
import html
import hashlib

from typing import Any
from pydantic import BaseModel
from typing import Literal


# ==========================================
# 1. Schema
# ==========================================

class Message(BaseModel):

    role: Literal[
        "system",
        "user",
        "assistant"
    ]

    content: str


class Conversation(BaseModel):

    messages: list[Message]


# ==========================================
# 2. Schema Validation
# ==========================================

def validate_schema(
    example: dict[str, Any]
) -> bool:

    try:

        Conversation.model_validate(
            example
        )

        return True

    except Exception:

        return False


# ==========================================
# 3. Clean Text
# ==========================================

def clean_text(text: str) -> str:

    text = html.unescape(text)

    text = re.sub(
        r"<[^>]+>",
        "",
        text
    )

    text = re.sub(
        r"[\x00-\x08\x0B\x0C\x0E-\x1F]",
        "",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ==========================================
# 4. Clean Conversation
# ==========================================

def clean_conversation(
    example: dict
) -> dict:

    messages = []

    for message in example["messages"]:

        messages.append(
            {
                "role": message["role"],
                "content": clean_text(
                    message["content"]
                )
            }
        )

    return {
        "messages": messages
    }


# ==========================================
# 5. Conversation Structure
# ==========================================

def validate_structure(
    example: dict
) -> bool:

    messages = example["messages"]

    if len(messages) < 2:
        return False

    start = 0

    if messages[0]["role"] == "system":
        start = 1

    conversation = messages[start:]

    if len(conversation) < 2:
        return False

    if conversation[0]["role"] != "user":
        return False

    for i, message in enumerate(conversation):

        expected = (
            "user"
            if i % 2 == 0
            else "assistant"
        )

        if message["role"] != expected:
            return False

        if not message["content"].strip():
            return False

    return (
        conversation[-1]["role"]
        == "assistant"
    )


# ==========================================
# 6. PII Redaction
# ==========================================

def redact_pii(text: str) -> str:

    text = re.sub(
        r"\b[\w.%+-]+@[\w.-]+\.[A-Za-z]{2,}\b",
        "[EMAIL_REDACTED]",
        text
    )

    text = re.sub(
        r"\b\d{10}\b",
        "[PHONE_REDACTED]",
        text
    )

    return text


# ==========================================
# 7. Normalize for Hashing
# ==========================================

def normalize_for_hash(
    text: str
) -> str:

    text = text.lower()

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    text = re.sub(
        r"[^\w\s]",
        "",
        text
    )

    return text.strip()


# ==========================================
# 8. Hash
# ==========================================

def get_hash(
    example: dict
) -> str:

    normalized = []

    for message in example["messages"]:

        normalized.append(
            {
                "role": message["role"],
                "content": normalize_for_hash(
                    message["content"]
                )
            }
        )

    data = json.dumps(
        normalized,
        sort_keys=True,
        ensure_ascii=False
    )

    return hashlib.sha256(
        data.encode()
    ).hexdigest()


# ==========================================
# 9. Deduplication
# ==========================================

def remove_duplicates(
    examples: list[dict]
) -> list[dict]:

    seen = set()

    unique = []

    for example in examples:

        example_hash = get_hash(
            example
        )

        if example_hash in seen:
            continue

        seen.add(example_hash)

        unique.append(example)

    return unique


# ==========================================
# 10. Main Pipeline
# ==========================================

def clean_dataset(
    raw_examples: list[dict]
) -> list[dict]:

    cleaned = []

    # Schema + text cleaning
    for example in raw_examples:

        if not validate_schema(example):
            continue

        example = clean_conversation(
            example
        )

        if not validate_structure(example):
            continue

        cleaned.append(example)

    # Deduplicate
    cleaned = remove_duplicates(
        cleaned
    )

    return cleaned
```

Usage:

```python
final_dataset = clean_dataset(
    raw_examples
)

print(
    f"Final examples: {len(final_dataset)}"
)
```

---

# 23. Production-grade pipeline architecture

For a serious fine-tuning project, I would structure it like this:

```text
                    DATA SOURCES
                         │
          ┌──────────────┼──────────────┐
          │              │              │
     Conversations    Documents      Synthetic
          │              │              │
          └──────────────┼──────────────┘
                         │
                         ▼
                 INGESTION LAYER
                         │
                         ▼
              SCHEMA VALIDATION
                         │
                         ▼
               TEXT NORMALIZATION
                         │
                         ▼
                 PII REDACTION
                         │
                         ▼
              EXACT DEDUPLICATION
                         │
                         ▼
             SEMANTIC DEDUPLICATION
                         │
                         ▼
              QUALITY SCORING
                         │
              ┌──────────┴──────────┐
              │                     │
           High Score            Low Score
              │                     │
              ▼                     ▼
          Auto Accept           Human Review
              │                     │
              └──────────┬──────────┘
                         │
                         ▼
                  DATA VERSIONING
                         │
                         ▼
                TRAIN / VAL / TEST
                         │
                         ▼
                    FINE-TUNING
```

---

# Interview answer

If an interviewer asks:

### **How do you clean a fine-tuning dataset?**

> I start with schema validation and enforce the expected structure, such as system, user, and assistant messages. Then I normalize text, remove empty or malformed examples, redact PII, and validate conversation role ordering. I perform exact deduplication using normalized hashes and near-duplicate detection using embeddings or approximate nearest-neighbor search. After that, I run quality checks for length, repetition, relevance, factual correctness, contradictions, and synthetic-data artifacts. I typically assign quality scores and send borderline examples for human review rather than relying completely on automatic filtering.

### **How do you remove duplicates?**

> For exact duplicates, I normalize the text and hash a canonical representation using SHA-256, keeping only the first occurrence. For near duplicates, I generate embeddings and compare semantic similarity, using an ANN index such as FAISS for large datasets. Similarity thresholds should be tuned using manually reviewed examples.

### **How do you detect bad training examples?**

> I use multiple layers: schema and structural validation, empty-content checks, token-length limits, repetition detection, prompt-response relevance scoring, factual verification against trusted sources, contradiction detection, and safety/PII checks. High-confidence examples are automatically accepted, suspicious examples are sent to human review, and the thresholds are calibrated on the specific task.

## Key principle

```text
Bad data
   ↓
Bad model behavior

High-quality + diverse + correct data
   ↓
Better fine-tuning results
```

For fine-tuning, **data quality is usually more valuable than simply increasing dataset size**.
