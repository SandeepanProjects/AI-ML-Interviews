# What is Conversational Fine-Tuning?

**Conversational fine-tuning** is a type of supervised fine-tuning where you train an LLM using **multi-turn conversations** instead of only isolated instruction → answer examples.

The goal is to teach the model how to:

* maintain context across multiple turns
* answer follow-up questions
* behave consistently during a conversation
* remember information within the context window
* ask clarifying questions
* follow a specific conversational style
* handle role-based interactions
* respond appropriately to corrections

---

# 1. Instruction tuning vs conversational fine-tuning

## Instruction tuning

Usually a single interaction:

```text
User:
Explain LoRA.

Assistant:
LoRA is a parameter-efficient fine-tuning technique.
```

Dataset:

```json
{
  "instruction": "Explain LoRA.",
  "input": "",
  "output": "LoRA is a parameter-efficient fine-tuning technique."
}
```

---

## Conversational fine-tuning

Multiple turns:

```text
User:
What is LoRA?

Assistant:
LoRA is a parameter-efficient fine-tuning technique.

User:
Why is it memory efficient?

Assistant:
Because it freezes the original model and trains only small adapter matrices.

User:
What are those matrices?

Assistant:
They are low-rank trainable matrices that approximate the weight update.
```

The model learns:

```text
Conversation history
        +
Current user message
        ↓
Generate context-aware response
```

---

# 2. Why conversational fine-tuning is needed

Imagine this conversation:

```text
User:
What is LoRA?

Assistant:
LoRA is a PEFT technique.

User:
How does it reduce memory?
```

The second question:

> "How does it reduce memory?"

depends on the previous conversation.

Without history:

```text
How does it reduce memory?
```

is ambiguous.

With history:

```text
What is LoRA?
        ↓
LoRA is a PEFT technique
        ↓
How does it reduce memory?
        ↓
Correct context-aware answer
```

Conversational fine-tuning teaches the model to work with this structure.

---

# 3. Dataset format

The most common format is a `messages` list.

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful AI tutor."
    },
    {
      "role": "user",
      "content": "What is LoRA?"
    },
    {
      "role": "assistant",
      "content": "LoRA is a parameter-efficient fine-tuning technique."
    },
    {
      "role": "user",
      "content": "Why does it use less memory?"
    },
    {
      "role": "assistant",
      "content": "Because the original model remains frozen and only small adapter matrices are trained."
    }
  ]
}
```

The model sees the whole conversation:

```text
System
   │
   ▼
User question
   │
   ▼
Assistant answer
   │
   ▼
User follow-up
   │
   ▼
Assistant answer
```

---

# 4. A real conversational dataset

Create:

```text
data/
├── train.jsonl
├── validation.jsonl
└── test.jsonl
```

Each line contains one conversation.

### `train.jsonl`

```json
{"messages":[{"role":"system","content":"You are a helpful AI tutor."},{"role":"user","content":"What is LoRA?"},{"role":"assistant","content":"LoRA is a parameter-efficient fine-tuning technique that freezes the original model and trains small low-rank adapter matrices."},{"role":"user","content":"Why does that reduce memory?"},{"role":"assistant","content":"It reduces memory because gradients and optimizer states are stored only for the small LoRA adapters instead of the entire base model."}]}
```

Another example:

```json
{"messages":[{"role":"system","content":"You are a helpful AI tutor."},{"role":"user","content":"What is QLoRA?"},{"role":"assistant","content":"QLoRA combines LoRA with low-bit quantization of the frozen base model."},{"role":"user","content":"Is the base model updated during training?"},{"role":"assistant","content":"No. In standard QLoRA training, the quantized base model remains frozen and only the LoRA adapter parameters are updated."}]}
```

---

# 5. Create the dataset with Python

```python
import json


conversations = [
    {
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful AI tutor."
            },
            {
                "role": "user",
                "content": "What is LoRA?"
            },
            {
                "role": "assistant",
                "content": (
                    "LoRA is a parameter-efficient fine-tuning "
                    "technique that freezes the pretrained model "
                    "and trains small low-rank adapter matrices."
                )
            },
            {
                "role": "user",
                "content": "Why does it reduce memory usage?"
            },
            {
                "role": "assistant",
                "content": (
                    "It reduces memory because only the small LoRA "
                    "adapter parameters require gradients and optimizer "
                    "states instead of the full model."
                )
            }
        ]
    },
    {
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful AI tutor."
            },
            {
                "role": "user",
                "content": "What is QLoRA?"
            },
            {
                "role": "assistant",
                "content": (
                    "QLoRA combines a quantized frozen base model "
                    "with trainable LoRA adapters."
                )
            },
            {
                "role": "user",
                "content": "What precision is the base model stored in?"
            },
            {
                "role": "assistant",
                "content": (
                    "QLoRA commonly stores the base model in 4-bit "
                    "quantized form, often using NF4."
                )
            }
        ]
    }
]


with open(
    "conversations.jsonl",
    "w",
    encoding="utf-8"
) as f:

    for conversation in conversations:
        f.write(
            json.dumps(conversation) + "\n"
        )
```

---

# 6. Load the dataset

Install:

```bash
pip install datasets transformers trl peft
```

Load:

```python
from datasets import load_dataset


dataset = load_dataset(
    "json",
    data_files="conversations.jsonl",
    split="train"
)

print(dataset[0])
```

You will get something conceptually like:

```python
{
    "messages": [
        {
            "role": "system",
            "content": "You are a helpful AI tutor."
        },
        {
            "role": "user",
            "content": "What is LoRA?"
        },
        ...
    ]
}
```

---

# 7. Apply the model's chat template

This is very important.

Do **not** manually assume every model uses the same special tokens.

Load the tokenizer:

```python
from transformers import AutoTokenizer


MODEL_NAME = "your-base-model"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)
```

Format each conversation:

```python
def format_conversation(example):

    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False
    )

    return {
        "text": text
    }


dataset = dataset.map(
    format_conversation
)
```

Now:

```python
print(dataset[0]["text"])
```

The tokenizer converts:

```text
system
user
assistant
user
assistant
```

into the exact format expected by the model.

Conceptually:

```text
<system>
You are a helpful AI tutor.

<user>
What is LoRA?

<assistant>
LoRA is...

<user>
Why does it reduce memory?

<assistant>
It reduces memory because...
```

---

# 8. Tokenization

The model ultimately sees token IDs.

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

Pipeline:

```text
JSONL conversation
       │
       ▼
messages
       │
       ▼
Chat template
       │
       ▼
Text with model tokens
       │
       ▼
Tokenization
       │
       ▼
input_ids
       │
       ▼
Fine-tuning
```

---

# 9. What does the model learn?

Suppose the conversation is:

```text
User:
What is LoRA?

Assistant:
LoRA is a PEFT technique.

User:
Why does it reduce memory?

Assistant:
Because only adapters are trained.
```

During training, the model learns the probability of the next assistant tokens.

Conceptually:

[
P(\text{Assistant Response} \mid \text{Conversation History})
]

For the second answer:

[
P(A_2 \mid S, U_1, A_1, U_2)
]

Where:

* (S) = system prompt
* (U_1) = first user message
* (A_1) = first assistant response
* (U_2) = second user message
* (A_2) = second assistant response

This is the key difference from a simple one-turn example.

---

# 10. How is loss calculated?

Usually, we want the model to learn the assistant responses.

Conceptually:

```text
System message
──────────────
Loss ignored

User message
──────────────
Loss ignored

Assistant response 1
──────────────
Loss calculated ✓

User message
──────────────
Loss ignored

Assistant response 2
──────────────
Loss calculated ✓
```

So:

```text
Conversation:

System       → context only
User         → context only
Assistant    → train ✓
User         → context only
Assistant    → train ✓
```

---

# 11. Manual assistant-only loss masking

Let's understand how this works conceptually.

Suppose tokens are:

```text
[SYS] You are helpful
[USER] What is LoRA?
[ASSISTANT] LoRA is PEFT
[USER] Why memory efficient?
[ASSISTANT] Only adapters train
```

Labels could conceptually be:

```text
Input IDs:

[10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
```

Labels:

```text
[-100, -100, -100,
 -100, -100,
 15, 16,
 -100,
 18, 19]
```

`-100` means:

> Ignore this token when calculating cross-entropy loss.

The loss is calculated only on assistant response tokens.

---

# 12. Using TRL `SFTTrainer`

A common way to fine-tune conversational datasets is using TRL.

```python
from trl import SFTTrainer
from transformers import TrainingArguments
```

Example configuration:

```python
training_args = TrainingArguments(
    output_dir="./conversation-model",

    num_train_epochs=3,

    per_device_train_batch_size=2,

    gradient_accumulation_steps=8,

    learning_rate=2e-5,

    logging_steps=10,

    save_strategy="epoch",

    bf16=True,

    report_to="none",
)
```

Then:

```python
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
)
```

Depending on the TRL version, conversational formatting and completion-only loss can be handled through its dataset preparation/configuration APIs. The exact arguments differ across versions, so check the installed TRL version rather than copying old examples blindly.

Train:

```python
trainer.train()
```

---

# 13. Conversational fine-tuning with LoRA

Usually, you don't full fine-tune a large model.

Instead:

```text
Base LLM
   │
   │ frozen
   ▼
LoRA adapters
   │
   │ trained on conversations
   ▼
Conversational model
```

Example:

```python
from peft import LoraConfig, TaskType


lora_config = LoraConfig(
    r=16,

    lora_alpha=32,

    lora_dropout=0.05,

    target_modules=[
        "q_proj",
        "v_proj"
    ],

    bias="none",

    task_type=TaskType.CAUSAL_LM,
)
```

Then attach LoRA to the base model:

```python
from peft import get_peft_model


model = get_peft_model(
    model,
    lora_config
)

model.print_trainable_parameters()
```

Conceptually:

```text
Conversation Dataset
        │
        ▼
Base Model
Frozen ❄️
        +
LoRA
Trainable ✓
        │
        ▼
Fine-Tuned Conversational Model
```

---

# 14. Full example architecture

```text
Raw Conversations
       │
       ▼
┌──────────────────┐
│ Data Cleaning    │
│                  │
│ • Remove PII     │
│ • Remove noise   │
│ • Deduplicate    │
│ • Validate roles │
└────────┬─────────┘
         │
         ▼
Conversation Dataset
         │
         ▼
Chat Template
         │
         ▼
Tokenization
         │
         ▼
Assistant-only Labels
         │
         ▼
SFT
         │
         ▼
LoRA / QLoRA Training
         │
         ▼
Evaluation
```

---

# 15. How to validate conversational data

A valid conversation should generally follow a role sequence.

For example:

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
def validate_conversation(example):

    messages = example.get("messages", [])

    if len(messages) < 2:
        return False

    # First message may be system or user
    allowed_first_roles = {
        "system",
        "user"
    }

    if messages[0]["role"] not in allowed_first_roles:
        return False

    # Check message content
    for message in messages:

        if not message.get("content"):
            return False

    # Extract non-system roles
    conversation_roles = [
        message["role"]
        for message in messages
        if message["role"] != "system"
    ]

    # Should start with user
    if not conversation_roles:
        return False

    if conversation_roles[0] != "user":
        return False

    # Alternate user -> assistant
    for i, role in enumerate(conversation_roles):

        expected = (
            "user"
            if i % 2 == 0
            else "assistant"
        )

        if role != expected:
            return False

    # Usually conversation should end with assistant
    return conversation_roles[-1] == "assistant"
```

Test:

```python
valid = validate_conversation(
    conversations[0]
)

print(valid)
```

Output:

```text
True
```

---

# 16. Should you train on entire conversations?

Usually yes, but there are trade-offs.

### Short conversations

```text
Pros:
✓ Less GPU memory
✓ More examples per batch

Cons:
✗ Less context learning
```

### Long conversations

```text
Pros:
✓ Better follow-up/context behavior
✓ More realistic interactions

Cons:
✗ Higher memory usage
✗ More padding/truncation
✗ Fewer conversations per batch
```

A practical dataset often contains a mixture:

```text
Single-turn       30%
2–4 turns         40%
5–10 turns        20%
Long conversations 10%
```

The exact distribution should match your production traffic.

---

# 17. A realistic enterprise example

Imagine an internal HR assistant.

Conversation:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are an internal HR assistant. Answer only using approved HR policies."
    },
    {
      "role": "user",
      "content": "How many vacation days do I get?"
    },
    {
      "role": "assistant",
      "content": "Vacation entitlement depends on employee level and applicable company policy."
    },
    {
      "role": "user",
      "content": "I am a senior engineer."
    },
    {
      "role": "assistant",
      "content": "For senior engineers, I would need your applicable location and policy version to provide the correct entitlement."
    }
  ]
}
```

Notice what the model learns:

1. It does not blindly assume missing information.
2. It asks for clarification.
3. It uses previous conversation context.
4. It follows organizational behavior.

This is a good use case for conversational fine-tuning.

However, for the **actual current HR policy content**, RAG may still be needed because policies change.

---

# 18. Conversational fine-tuning vs RAG

This distinction is important.

### Conversational fine-tuning teaches:

```text
HOW to behave
HOW to respond
HOW to ask follow-ups
HOW to maintain style
```

### RAG provides:

```text
WHAT current information to use
WHAT policies say
WHAT the latest documents contain
```

Together:

```text
          User
           │
           ▼
    Conversation History
           │
           ▼
        Retrieve
           │
           ▼
        LLM
     ┌─────┴─────┐
     │           │
 Fine-tuned      RAG
 Behavior       Knowledge
     │           │
     └─────┬─────┘
           ▼
        Response
```

This is a common enterprise architecture.

---

# 19. How would I create a production conversational dataset?

I would follow:

```text
1. Define desired behavior
        ↓
2. Collect approved conversations
        ↓
3. Remove PII/sensitive data
        ↓
4. Clean formatting
        ↓
5. Validate role ordering
        ↓
6. Remove duplicates
        ↓
7. Create diverse conversations
        ↓
8. Add edge cases
        ↓
9. Split by conversation/session
        ↓
10. Apply model chat template
        ↓
11. Train
        ↓
12. Evaluate on unseen conversations
```

### Important:

Do not split individual turns randomly.

Bad:

```text
Conversation A turn 1 → Train
Conversation A turn 2 → Test
```

This causes leakage.

Good:

```text
Conversation A → Train
Conversation B → Test
```

---

# 20. Interview answer

If asked:

> **What is conversational fine-tuning?**

A strong answer is:

> **Conversational fine-tuning is supervised fine-tuning using multi-turn dialogue data. Instead of training only on isolated instruction-response pairs, the model is trained on sequences containing conversation history, user messages, and assistant responses. This helps the model learn context-aware follow-ups, conversational flow, clarification behavior, and consistent responses across multiple turns. The data is usually stored in a messages format and converted using the target model's chat template. During training, loss is typically calculated on assistant messages while system and user messages provide context. In production, I would combine conversational fine-tuning for behavior with RAG for dynamic or frequently changing knowledge.**

## The simplest way to remember it

```text
Instruction Fine-Tuning:

User
 ↓
Assistant


Conversational Fine-Tuning:

User
 ↓
Assistant
 ↓
User
 ↓
Assistant
 ↓
User
 ↓
Assistant
```

### Core idea

[
\boxed{
\text{Learn response} \mid \text{entire conversation history}
}
]

That is the fundamental idea behind **conversational fine-tuning**.
