# Implement Tokenization for LLM Fine-Tuning

After preparing the instruction dataset, the next step is to convert text into **token IDs** that the model can process.

The pipeline is:

```text
Instruction Dataset
       ↓
Chat Formatting
       ↓
Tokenizer
       ↓
input_ids
attention_mask
labels
       ↓
LLM Training
```

---

# 1. What does tokenization do?

Suppose we have:

```text
Hello, how are you?
```

The tokenizer might convert it conceptually into:

```text
["Hello", ",", " how", " are", " you", "?"]
```

Then:

```text
[15496, 11, 703, 389, 345, 30]
```

The exact token IDs depend on the model tokenizer.

For LLM fine-tuning:

```python
text
↓
tokenizer
↓
{
    "input_ids": [...],
    "attention_mask": [...]
}
```

---

# 2. Load the tokenizer

Use the **same tokenizer as the base model**.

```python
from transformers import AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
```

Why?

```text
Base Model
    ↓
Was trained using
    ↓
Specific tokenizer vocabulary
    ↓
Use the same tokenizer
```

You should generally not tokenize a Qwen model using a Llama tokenizer, for example.

---

# 3. Basic tokenization

```python
text = "My payment was deducted twice."

encoded = tokenizer(
    text
)

print(encoded)
```

Typical structure:

```python
{
    "input_ids": [
        123,
        456,
        789
    ],
    "attention_mask": [
        1,
        1,
        1
    ]
}
```

You can decode:

```python
decoded = tokenizer.decode(
    encoded["input_ids"]
)

print(decoded)
```

---

# 4. Tokenize your instruction dataset

Suppose your dataset contains:

```json
{
    "instruction": "You are a helpful customer support assistant.",
    "input": "My payment was deducted twice.",
    "output": "Please provide your transaction ID."
}
```

For a chat model, first convert it to:

```python
messages = [
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
        "content": "Please provide your transaction ID."
    }
]
```

---

# 5. Apply the model's chat template

```python
def format_messages(example):

    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False
    )

    return {
        "text": text
    }
```

Apply:

```python
dataset = dataset.map(
    format_messages
)
```

Now inspect:

```python
print(dataset[0]["text"])
```

The model's tokenizer will format the conversation using the correct special tokens.

---

# 6. Implement tokenization

Now convert the text to tokens.

```python
MAX_LENGTH = 2048


def tokenize_function(example):

    return tokenizer(

        example["text"],

        truncation=True,

        max_length=MAX_LENGTH,

        padding="max_length"
    )
```

Apply:

```python
tokenized_dataset = dataset.map(

    tokenize_function,

    batched=True,

    remove_columns=dataset.column_names
)
```

Now:

```python
print(tokenized_dataset[0])
```

You should get:

```python
{
    "input_ids": [...],
    "attention_mask": [...]
}
```

---

# 7. Why `input_ids`?

Example:

```text
"My payment failed"
```

Tokenizer:

```text
My        → 1234
payment   → 4567
failed    → 8910
```

The model receives:

```python
input_ids = [
    1234,
    4567,
    8910
]
```

The embedding layer does:

```text
Token ID
   ↓
Embedding Lookup
   ↓
Vector
```

Example:

```text
1234
 ↓
[0.12, -0.83, 0.45, ...]
```

---

# 8. Why `attention_mask`?

Suppose two examples have different lengths.

```text
Example 1:

[10, 20, 30]

Example 2:

[40, 50]
```

For batching, padding is added:

```text
Example 1:

[10, 20, 30]

Example 2:

[40, 50, PAD]
```

Attention masks:

```text
Example 1:

[1, 1, 1]

Example 2:

[1, 1, 0]
```

Meaning:

```text
1 → Real token
0 → Padding token
```

The model should not treat padding as meaningful input.

---

# 9. Dynamic padding vs fixed padding

### Fixed padding

```python
tokenizer(
    text,
    truncation=True,
    max_length=2048,
    padding="max_length"
)
```

Every example becomes:

```text
2048 tokens
```

Even:

```text
Example A = 100 tokens
Example B = 200 tokens
```

Both become:

```text
2048 tokens
```

This can waste memory.

---

### Dynamic padding

Usually better:

```python
tokenizer(
    text,
    truncation=True,
    max_length=2048,
    padding=False
)
```

Then use a data collator:

```python
from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)
```

This pads only to the longest sequence in each batch.

Example:

```text
Batch:

Example A = 100 tokens
Example B = 150 tokens
Example C = 120 tokens

Longest = 150
```

All examples are padded to:

```text
150 tokens
```

instead of:

```text
2048 tokens
```

This can significantly reduce memory usage.

---

# 10. Create labels for causal language modeling

For causal language modeling:

```text
Input:

The capital of France is

Target:

Paris
```

The model predicts the next token.

Usually labels are based on the input IDs.

```python
def tokenize_with_labels(example):

    tokenized = tokenizer(

        example["text"],

        truncation=True,

        max_length=MAX_LENGTH,

        padding=False
    )

    tokenized["labels"] = (
        tokenized["input_ids"].copy()
    )

    return tokenized
```

Apply:

```python
tokenized_dataset = dataset.map(

    tokenize_with_labels,

    batched=False,

    remove_columns=dataset.column_names
)
```

Example:

```python
{
    "input_ids": [
        10,
        20,
        30,
        40
    ],

    "labels": [
        10,
        20,
        30,
        40
    ]
}
```

The causal language model internally shifts labels.

Conceptually:

```text
Input:

The   capital   of   France

Target:

capital   of   France   is
```

---

# 11. Important: ignore padding in labels

Padding should not contribute to the loss.

The ignore value is usually:

```python
-100
```

Example:

```text
input_ids:

[10, 20, 30, PAD, PAD]

labels:

[10, 20, 30, -100, -100]
```

Implement:

```python
def tokenize_with_labels(example):

    result = tokenizer(

        example["text"],

        truncation=True,

        max_length=MAX_LENGTH,

        padding="max_length"
    )

    labels = result[
        "input_ids"
    ].copy()

    labels = [

        token if token != tokenizer.pad_token_id
        else -100

        for token in labels
    ]

    result["labels"] = labels

    return result
```

---

# 12. Assistant-only loss

This is especially important for instruction tuning.

Suppose:

```text
System:
You are a helpful assistant.

User:
What is Python?

Assistant:
Python is a programming language.
```

We often want to train the model primarily on:

```text
Python is a programming language.
```

not on reproducing:

```text
System prompt
User prompt
```

Conceptually:

```text
Tokens:

<System>
You are helpful
<User>
What is Python?
<Assistant>
Python is a programming language.
```

Labels:

```text
-100
-100
-100
-100
-100
-100
-100
-100
Python
is
a
programming
language
```

Only assistant tokens contribute to the loss.

---

# 13. Manual implementation of assistant-only labels

For illustration, we can tokenize the prompt and the full conversation separately.

```python
def tokenize_chat(example):

    messages = example["messages"]

    # Prompt without assistant response
    prompt_messages = messages[:-1]

    prompt_text = tokenizer.apply_chat_template(

        prompt_messages,

        tokenize=False,

        add_generation_prompt=True
    )

    # Full conversation
    full_text = tokenizer.apply_chat_template(

        messages,

        tokenize=False,

        add_generation_prompt=False
    )

    # Tokenize prompt
    prompt_tokens = tokenizer(

        prompt_text,

        add_special_tokens=False
    )

    # Tokenize complete conversation
    full_tokens = tokenizer(

        full_text,

        truncation=True,

        max_length=2048,

        add_special_tokens=False
    )

    input_ids = full_tokens[
        "input_ids"
    ]

    attention_mask = full_tokens[
        "attention_mask"
    ]

    # Initially ignore everything
    labels = [-100] * len(input_ids)

    # Train only after prompt
    prompt_length = len(
        prompt_tokens["input_ids"]
    )

    labels[prompt_length:] = input_ids[
        prompt_length:
    ]

    return {

        "input_ids": input_ids,

        "attention_mask": attention_mask,

        "labels": labels
    }
```

Apply:

```python
tokenized_dataset = dataset.map(
    tokenize_chat,
    remove_columns=dataset.column_names
)
```

---

# 14. Important caveat about assistant-only masking

The manual approach above assumes:

```text
prompt token sequence
=
exact prefix of
full conversation token sequence
```

That can be incorrect for some chat templates because:

* special tokens may differ
* generation prompts may add tokens
* tool/function messages can change formatting
* templates may have special assistant boundaries

Therefore, in production, prefer trainer/template support that understands the tokenizer's chat template when available.

Always verify the token boundaries.

A useful debugging technique:

```python
example = tokenized_dataset[0]

for token_id, label in zip(
    example["input_ids"],
    example["labels"]
):

    token = tokenizer.decode(
        [token_id]
    )

    print(
        repr(token),
        "LOSS" if label != -100 else "IGNORE"
    )
```

You want to see:

```text
<System>      IGNORE
You           IGNORE
are           IGNORE
...
<User>        IGNORE
What          IGNORE
...
<Assistant>   IGNORE
Python        LOSS
is            LOSS
...
```

---

# 15. Best practical approach with TRL

If you use `SFTTrainer`, you often do not need to manually create token IDs before passing the dataset.

For a simple text dataset:

```python
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer
)
```

The trainer can tokenize internally.

Your dataset might simply contain:

```text
text
```

However, you should understand the tokenization process because debugging fine-tuning issues often requires inspecting:

```text
input_ids
labels
attention_mask
```

---

# 16. Production-quality tokenization script

```python
from datasets import load_dataset
from transformers import AutoTokenizer


# =====================================================
# CONFIGURATION
# =====================================================

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

DATASET_PATH = "instruction_dataset.jsonl"

MAX_LENGTH = 2048


# =====================================================
# LOAD TOKENIZER
# =====================================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=True
)


if tokenizer.pad_token is None:

    tokenizer.pad_token = tokenizer.eos_token


# =====================================================
# LOAD DATASET
# =====================================================

dataset = load_dataset(

    "json",

    data_files=DATASET_PATH,

    split="train"
)


# =====================================================
# CONVERT TO CHAT FORMAT
# =====================================================

def create_messages(example):

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


dataset = dataset.map(
    create_messages
)


# =====================================================
# TOKENIZE WITH ASSISTANT-ONLY LABELS
# =====================================================

def tokenize_example(example):

    messages = example["messages"]

    # Separate prompt and assistant response
    prompt_messages = messages[:-1]

    # Create prompt
    prompt_text = tokenizer.apply_chat_template(

        prompt_messages,

        tokenize=False,

        add_generation_prompt=True
    )

    # Create complete conversation
    full_text = tokenizer.apply_chat_template(

        messages,

        tokenize=False,

        add_generation_prompt=False
    )

    # Tokenize prompt
    prompt_ids = tokenizer(

        prompt_text,

        add_special_tokens=False

    )["input_ids"]


    # Tokenize full conversation
    encoded = tokenizer(

        full_text,

        truncation=True,

        max_length=MAX_LENGTH,

        add_special_tokens=False
    )


    input_ids = encoded[
        "input_ids"
    ]


    attention_mask = encoded[
        "attention_mask"
    ]


    # Ignore all tokens initially
    labels = [
        -100
    ] * len(input_ids)


    # Calculate assistant token start
    prompt_length = min(
        len(prompt_ids),
        len(input_ids)
    )


    # Only assistant response contributes to loss
    labels[
        prompt_length:
    ] = input_ids[
        prompt_length:
    ]


    return {

        "input_ids": input_ids,

        "attention_mask": attention_mask,

        "labels": labels
    }


# =====================================================
# TOKENIZE
# =====================================================

tokenized_dataset = dataset.map(

    tokenize_example,

    remove_columns=dataset.column_names,

    desc="Tokenizing dataset"
)


# =====================================================
# SAVE TOKENIZED DATASET
# =====================================================

tokenized_dataset.save_to_disk(
    "./tokenized_dataset"
)


print(
    "Tokenization completed!"
)


print(
    "Number of examples:",
    len(tokenized_dataset)
)


print(
    tokenized_dataset[0]
)
```

---

# 17. Use dynamic padding during training

Since examples have different lengths, use a collator.

For standard causal LM training:

```python
from transformers import DataCollatorForLanguageModeling


data_collator = DataCollatorForLanguageModeling(

    tokenizer=tokenizer,

    mlm=False
)
```

But if you already manually created `labels` with assistant-only masking, be careful: a generic language-modeling collator may recreate or overwrite labels depending on the setup/version.

For precomputed labels, use a collator that preserves them.

A simple custom collator:

```python
import torch


class CausalLMCollator:

    def __init__(
        self,
        tokenizer
    ):

        self.tokenizer = tokenizer


    def __call__(
        self,
        features
    ):

        input_features = []

        for feature in features:

            input_features.append({

                "input_ids": feature["input_ids"],

                "attention_mask": feature[
                    "attention_mask"
                ]
            })


        batch = self.tokenizer.pad(

            input_features,

            padding=True,

            return_tensors="pt"
        )


        max_length = batch[
            "input_ids"
        ].shape[1]


        padded_labels = []


        for feature in features:

            labels = feature["labels"]

            padding_length = (
                max_length
                - len(labels)
            )

            # Right padding assumed
            labels = (
                labels
                + [-100] * padding_length
            )

            padded_labels.append(
                labels
            )


        batch["labels"] = torch.tensor(
            padded_labels,
            dtype=torch.long
        )


        return batch
```

Use:

```python
data_collator = CausalLMCollator(
    tokenizer
)
```

---

# 18. Tokenization pipeline for your QLoRA script

Your full pipeline now becomes:

```text
Raw JSONL
   │
   ▼
Clean Dataset
   │
   ▼
Instruction / Input / Output
   │
   ▼
Convert to Messages
   │
   ▼
Apply Model Chat Template
   │
   ▼
Tokenize
   │
   ├── input_ids
   ├── attention_mask
   └── labels
           │
           ▼
      Data Collator
           │
           ▼
       QLoRA Training
```

---

# Interview answer

> **I implement tokenization using the tokenizer that belongs to the exact base model being fine-tuned. For instruction or chat fine-tuning, I first convert examples into the model's expected conversational format and apply its native chat template. I then tokenize into `input_ids` and `attention_mask`, applying truncation based on the chosen maximum sequence length.**
>
> **For causal language modeling, labels are derived from token IDs, but padding tokens are masked with `-100` so they don't contribute to the loss. For instruction tuning, I often use assistant-only loss, masking system and user tokens with `-100` and computing loss only on assistant responses. I prefer dynamic padding at batch time to reduce wasted GPU memory.**

The core implementation is:

```python
encoded = tokenizer(
    text,
    truncation=True,
    max_length=2048,
    padding=False
)

input_ids = encoded["input_ids"]
attention_mask = encoded["attention_mask"]
labels = input_ids.copy()
```

For assistant-only training:

```python
labels = [-100] * len(input_ids)

labels[assistant_start:] = input_ids[assistant_start:]
```

This is the tokenized data that is ultimately passed into the LLM during LoRA or QLoRA fine-tuning.
