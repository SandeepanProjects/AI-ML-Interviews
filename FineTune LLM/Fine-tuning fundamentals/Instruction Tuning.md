# What is Instruction Tuning?

**Instruction tuning** is a type of **supervised fine-tuning (SFT)** where we train an LLM on many examples of:

```text
Instruction
     +
Optional Input
     ↓
Expected Response
```

The goal is to teach the model:

> **How to follow human instructions reliably.**

---

# 1. Simple example

Suppose we give a normal pre-trained model:

```text
Translate this sentence to French:
Hello, how are you?
```

A base model might continue text unpredictably because it was mainly trained to predict the next token.

With instruction tuning, we train examples like:

```text
Instruction:
Translate the following sentence into French.

Input:
Hello, how are you?

Response:
Bonjour, comment allez-vous ?
```

Another example:

```text
Instruction:
Summarize the following text in one sentence.

Input:
Artificial intelligence is changing...

Response:
AI is transforming many industries.
```

Another:

```text
Instruction:
Classify the sentiment.

Input:
This product is amazing!

Response:
positive
```

The model sees **many different instructions and tasks**.

Eventually, it learns the general pattern:

```text
Human gives instruction
          ↓
Understand task
          ↓
Perform requested task
          ↓
Return appropriate answer
```

---

# 2. Instruction tuning vs normal SFT

This distinction is important.

## Normal SFT

SFT is a broad concept.

```text
Input
  ↓
Desired Output
```

Example:

```text
"I was charged twice"
        ↓
"billing"
```

This could be a very narrow task.

---

## Instruction tuning

Instruction tuning is usually an SFT setup focused on teaching the model to follow explicit instructions.

```text
Instruction
     +
Input
     ↓
Response
```

Example:

```text
Instruction:
Explain the concept simply.

Input:
What is recursion?

Response:
Recursion is when a function calls itself...
```

So:

```text
SFT
│
├── Classification tuning
├── Domain tuning
├── Style tuning
├── Code tuning
└── Instruction tuning
```

> **Instruction tuning is generally a specialized form of supervised fine-tuning.**

---

# 3. Why is instruction tuning needed?

A pre-trained model learns primarily from large-scale prediction objectives.

Conceptually:

```text
Input:
The capital of France is

Target:
Paris
```

The model becomes good at predicting:

```text
Next token
```

But a user interacts differently:

```text
User:
Explain quantum computing like I am five.
```

The model must understand:

* What does "explain" mean?
* What information is being requested?
* What does "like I am five" imply?
* How simple should the answer be?
* What format should the answer have?

Instruction tuning improves this behavior.

```text
Pre-trained Model
       ↓
Instruction Dataset
       ↓
Instruction Tuning
       ↓
Instruction-following Model
```

---

# 4. What does an instruction-tuning dataset look like?

A common structure:

```json
{
  "instruction": "Summarize the following text in one sentence.",
  "input": "Machine learning is a field of artificial intelligence...",
  "output": "Machine learning enables computers to learn patterns from data."
}
```

Another:

```json
{
  "instruction": "Classify the sentiment as positive, negative, or neutral.",
  "input": "The product is excellent.",
  "output": "positive"
}
```

Another:

```json
{
  "instruction": "Extract the name and age as JSON.",
  "input": "John is 32 years old.",
  "output": "{\"name\": \"John\", \"age\": 32}"
}
```

The key is **task diversity**:

```text
Summarization
Translation
Classification
Question Answering
Extraction
Reasoning
Code Generation
Formatting
```

This teaches a general ability to follow different instructions.

---

# 5. Dataset example in JSONL

Create:

```text
data/instructions.jsonl
```

```json
{"instruction":"Classify the sentiment as positive, negative, or neutral.","input":"This product is amazing.","output":"positive"}
{"instruction":"Classify the sentiment as positive, negative, or neutral.","input":"This product is terrible.","output":"negative"}
{"instruction":"Summarize the text in one sentence.","input":"Artificial intelligence is a field of computer science that enables machines to perform tasks requiring human intelligence.","output":"AI enables machines to perform tasks that normally require human intelligence."}
{"instruction":"Translate the text into French.","input":"Hello, how are you?","output":"Bonjour, comment allez-vous ?"}
{"instruction":"Extract the person's name and age as JSON.","input":"Sarah is 28 years old.","output":"{\"name\":\"Sarah\",\"age\":28}"}
```

Each line is a supervised training example.

---

# 6. Step 1: Load the dataset

```python
from datasets import load_dataset

dataset = load_dataset(
    "json",
    data_files="data/instructions.jsonl",
    split="train"
)

print(dataset[0])
```

Output:

```python
{
    "instruction": "Classify the sentiment...",
    "input": "This product is amazing.",
    "output": "positive"
}
```

---

# 7. Step 2: Convert examples into prompts

We need to format the training data.

```python
def format_example(example):
    return f"""### Instruction:
{example["instruction"]}

### Input:
{example["input"]}

### Response:
{example["output"]}"""
```

Example:

```python
print(format_example(dataset[0]))
```

Output:

```text
### Instruction:
Classify the sentiment as positive, negative, or neutral.

### Input:
This product is amazing.

### Response:
positive
```

---

# 8. What does the model actually learn?

During training, the model sees:

```text
### Instruction:
Classify the sentiment.

### Input:
This product is amazing.

### Response:
positive
```

But we normally want the loss to focus primarily on:

```text
positive
```

Not necessarily on reproducing:

```text
### Instruction:
Classify the sentiment.
```

Conceptually:

```text
Prompt tokens:
Instruction + Input
        ↓
Ignored for loss

Response tokens:
positive
        ↓
Used for loss
```

---

# 9. Code: tokenize and mask the prompt

Here is the important implementation.

```python
from transformers import AutoTokenizer

MODEL_NAME = "gpt2"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

tokenizer.pad_token = tokenizer.eos_token
```

Now tokenize:

```python
def tokenize_example(example):

    prompt = f"""### Instruction:
{example["instruction"]}

### Input:
{example["input"]}

### Response:
"""

    response = example["output"] + tokenizer.eos_token

    full_text = prompt + response

    # Tokenize full input
    full_tokens = tokenizer(
        full_text,
        truncation=True,
        max_length=256
    )

    # Tokenize prompt separately
    prompt_tokens = tokenizer(
        prompt,
        truncation=True,
        max_length=256
    )

    input_ids = full_tokens["input_ids"]

    # Initially labels are all tokens
    labels = input_ids.copy()

    # Number of prompt tokens
    prompt_length = len(prompt_tokens["input_ids"])

    # Ignore prompt tokens during loss calculation
    labels[:prompt_length] = [-100] * prompt_length

    return {
        "input_ids": input_ids,
        "attention_mask": full_tokens["attention_mask"],
        "labels": labels
    }
```

The important part is:

```python
labels[:prompt_length] = [-100] * prompt_length
```

`-100` is commonly used as an ignore index for cross-entropy loss in many PyTorch/Transformers training setups.

Conceptually:

```text
Tokens:

[###] [Instruction] [Classify] [Input] [Amazing] [Response] [positive]
   │         │           │        │         │          │         │
   ▼         ▼           ▼        ▼         ▼          ▼         ▼
 Ignore    Ignore      Ignore   Ignore    Ignore     Ignore     Train
 -100       -100        -100    -100      -100       -100      positive
```

---

# 10. Apply tokenization to the dataset

```python
tokenized_dataset = dataset.map(
    tokenize_example,
    remove_columns=dataset.column_names
)
```

Now:

```python
print(tokenized_dataset[0])
```

Conceptually:

```python
{
    "input_ids": [...],
    "attention_mask": [...],
    "labels": [
        -100,
        -100,
        -100,
        ...,
        1234,  # token for "positive"
        50256  # EOS
    ]
}
```

---

# 11. Step 3: Load the pre-trained model

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME
)

model.config.pad_token_id = tokenizer.pad_token_id
```

Architecture:

```text
Instruction + Input
        ↓
     Tokenizer
        ↓
     Token IDs
        ↓
   Pre-trained LLM
        ↓
     Predictions
```

---

# 12. Step 4: Train the model

Let's use Hugging Face's Trainer.

```python
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
```

Training configuration:

```python
training_args = TrainingArguments(
    output_dir="./instruction_model",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    learning_rate=2e-5,
    logging_steps=1,
    save_strategy="epoch",
    report_to="none"
)
```

Data collator:

```python
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True
)
```

Trainer:

```python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator
)
```

Train:

```python
trainer.train()
```

Save:

```python
trainer.save_model("./instruction_model")
tokenizer.save_pretrained("./instruction_model")
```

---

# 13. What happens during `trainer.train()`?

Internally, conceptually:

```text
Example
   ↓
Tokenization

Instruction:
Classify sentiment

Input:
This product is amazing

Response:
positive

        ↓

Model Forward Pass

        ↓

Predicted token probabilities

positive → 0.40
negative → 0.30
neutral  → 0.20

        ↓

Correct label = positive

        ↓

Calculate Cross-Entropy Loss

        ↓

loss.backward()

        ↓

Calculate gradients

        ↓

optimizer.step()

        ↓

Update trainable parameters
```

Repeated thousands of times:

```text
Batch 1
 ↓
Update

Batch 2
 ↓
Update

Batch 3
 ↓
Update
```

Eventually, the model becomes better at recognizing:

```text
Instruction
     ↓
Task
     ↓
Expected response
```

---

# 14. Complete training example

Here is everything together.

```python
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)

MODEL_NAME = "gpt2"


# ----------------------------------
# Load dataset
# ----------------------------------

dataset = load_dataset(
    "json",
    data_files="data/instructions.jsonl",
    split="train"
)


# ----------------------------------
# Load tokenizer
# ----------------------------------

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

tokenizer.pad_token = tokenizer.eos_token


# ----------------------------------
# Tokenize instruction data
# ----------------------------------

def tokenize_example(example):

    prompt = f"""### Instruction:
{example["instruction"]}

### Input:
{example["input"]}

### Response:
"""

    response = example["output"] + tokenizer.eos_token

    full_text = prompt + response

    full_tokens = tokenizer(
        full_text,
        truncation=True,
        max_length=256
    )

    prompt_tokens = tokenizer(
        prompt,
        truncation=True,
        max_length=256
    )

    input_ids = full_tokens["input_ids"]

    labels = input_ids.copy()

    prompt_length = len(prompt_tokens["input_ids"])

    labels[:prompt_length] = [-100] * prompt_length

    return {
        "input_ids": input_ids,
        "attention_mask": full_tokens["attention_mask"],
        "labels": labels
    }


tokenized_dataset = dataset.map(
    tokenize_example,
    remove_columns=dataset.column_names
)


# ----------------------------------
# Load model
# ----------------------------------

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME
)

model.config.pad_token_id = tokenizer.pad_token_id


# ----------------------------------
# Training arguments
# ----------------------------------

training_args = TrainingArguments(
    output_dir="./instruction_model",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    learning_rate=2e-5,
    logging_steps=1,
    save_strategy="epoch",
    report_to="none"
)


# ----------------------------------
# Data collator
# ----------------------------------

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True
)


# ----------------------------------
# Trainer
# ----------------------------------

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator
)


# ----------------------------------
# Train
# ----------------------------------

trainer.train()


# ----------------------------------
# Save
# ----------------------------------

trainer.save_model("./instruction_model")

tokenizer.save_pretrained(
    "./instruction_model"
)
```

---

# 15. Inference after instruction tuning

```python
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)
import torch


MODEL_PATH = "./instruction_model"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH
)

model.eval()


prompt = """### Instruction:
Classify the sentiment as positive, negative, or neutral.

### Input:
The product is absolutely fantastic.

### Response:
"""


inputs = tokenizer(
    prompt,
    return_tensors="pt"
)


with torch.no_grad():

    output = model.generate(
        **inputs,
        max_new_tokens=10,
        do_sample=False
    )


generated_text = tokenizer.decode(
    output[0],
    skip_special_tokens=True
)

print(generated_text)
```

Expected conceptually:

```text
### Instruction:
Classify the sentiment as positive, negative, or neutral.

### Input:
The product is absolutely fantastic.

### Response:
positive
```

---

# 16. Instruction tuning with LoRA

In practice, you often don't want to update every model parameter.

```text
Traditional Instruction Tuning:

Base Model
    ↓
Update all parameters
    ↓
Expensive


Instruction Tuning + LoRA:

Base Model (Frozen)
       +
Small Trainable Adapters
       ↓
Cheaper Training
```

Example:

```python
from peft import (
    LoraConfig,
    get_peft_model
)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)


model = get_peft_model(
    model,
    lora_config
)

model.print_trainable_parameters()
```

Then use the same `Trainer`.

So:

```text
Instruction Dataset
       ↓
SFT Objective
       ↓
LoRA updates adapters
       ↓
Instruction-following model
```

---

# 17. Instruction tuning vs prompt engineering

This is a very important distinction.

## Prompt engineering

You tell the model at runtime:

```text
"Classify sentiment.
Return only positive, negative, or neutral."
```

The model parameters do not change.

```text
Prompt
  ↓
Temporary behavior
```

---

## Instruction tuning

You train the model using thousands of examples:

```text
Instruction A → Correct Response
Instruction B → Correct Response
Instruction C → Correct Response
...
```

The model parameters/adapters are updated.

```text
Training
   ↓
Learned behavior
```

---

# 18. Instruction tuning vs domain fine-tuning

Suppose you have 10,000 examples:

```text
Financial Question
      ↓
Financial Answer
```

That is primarily:

> **Domain fine-tuning**

Suppose you have:

```text
Summarize
Translate
Classify
Extract
Generate JSON
Write Code
Answer Questions
```

That is primarily:

> **Instruction tuning**

The difference is the goal.

```text
Domain Fine-Tuning
→ Learn a specific domain/task

Instruction Tuning
→ Learn to follow many instructions/tasks
```

---

# 19. Real-world production example

Imagine your enterprise AI system.

Users ask:

```text
"Summarize this document"
"Extract risks"
"Return JSON"
"Explain for executives"
"Compare these contracts"
```

You want the model to consistently follow instructions.

Training data:

```json
{
    "instruction": "Extract risks from the contract and return JSON.",
    "input": "Contract text...",
    "output": "{\"risks\": [...]}"
}
```

```json
{
    "instruction": "Summarize this contract for an executive.",
    "input": "Contract text...",
    "output": "Executive summary..."
}
```

Then:

```text
Enterprise Dataset
        ↓
Instruction-response pairs
        ↓
Instruction Tuning
        ↓
Model follows company tasks
        ↓
RAG adds current documents
        ↓
Final Enterprise AI System
```

This is where **instruction tuning + RAG** can work together:

```text
                 USER REQUEST
                      │
                      ▼
               Instruction
                      │
                      ▼
                    RAG
              Retrieve Documents
                      │
                      ▼
         Instruction-Tuned LLM
                      │
                      ▼
           Structured Response
```

---

# 20. Interview answer

If asked:

> **What is instruction tuning?**

A strong answer is:

> "Instruction tuning is a form of supervised fine-tuning where an LLM is trained on diverse instruction-input-output examples to improve its ability to follow human instructions. During training, the instruction and input form the context, while the desired response acts as the supervised target. The model uses token-level cross-entropy loss and backpropagation to update its weights or adapter parameters. In instruction-tuning datasets, we typically include diverse tasks such as summarization, classification, extraction, translation, and question answering so the model learns generalized instruction-following behavior. In practice, I would often use parameter-efficient methods such as LoRA and compute loss primarily on the assistant response tokens."

## Remember this hierarchy

```text
Pre-training
      ↓
General language capability
      ↓
Supervised Fine-Tuning
      ↓
├── Instruction tuning
├── Domain tuning
├── Classification tuning
└── Style tuning
      ↓
Deployment / Inference
```

### One-line summary

> **Instruction tuning teaches an LLM: "When a human gives you an instruction, understand what they want and produce the appropriate response."**
