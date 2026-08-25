# Instruction Tuning — Complete Explanation with Code

## 1. What is instruction tuning?

**Instruction tuning** is a type of **supervised fine-tuning (SFT)** where we train an LLM to understand and follow human instructions.

The model is trained on examples like:

```text
Instruction:
"Explain recursion in simple terms."

Expected response:
"Recursion is a technique where a function calls itself..."
```

The goal is not just to teach the model facts. The goal is to teach a general pattern:

```text
Human instruction
        ↓
Understand task
        ↓
Follow instruction
        ↓
Generate useful response
```

### Formal representation

An instruction-tuning example can be represented as:

[
(x, y)
]

where:

* (x) = instruction + optional context/input
* (y) = desired response

For example:

```text
x:
Instruction: Translate English to French
Input: Hello, how are you?

y:
Bonjour, comment allez-vous ?
```

The model is trained to maximize:

[
P(y \mid x)
]

Meaning:

> Given an instruction (x), generate the correct response (y).

---

# 2. Why do we need instruction tuning?

A pretrained LLM mainly learns from large-scale text prediction.

For example, during pre-training:

```text
The capital of France is ___
```

The model learns:

```text
Paris
```

But a raw pretrained model may not consistently behave like an assistant.

For example:

```text
User:
Explain Python in 3 bullet points.
```

A base model may simply continue text:

```text
Python is a programming language...
Python is widely used...
Python was created...
```

It may not reliably follow:

* "3 bullet points"
* output format
* tone
* constraints
* role

Instruction tuning teaches the model:

```text
"Explain in 3 bullet points"
            ↓
Understand requested task
            ↓
Generate exactly 3 useful bullets
```

---

# 3. Pre-training vs instruction tuning

## Pre-training

The model learns from massive amounts of text.

Typical objective:

```text
Predict next token
```

Example:

```text
Input:
The capital of India is

Target:
New Delhi
```

The model learns general:

* language
* grammar
* reasoning patterns
* programming patterns
* factual associations

### Pre-training data

```text
Books
Wikipedia
Web pages
Research papers
Code
Documentation
```

---

## Instruction tuning

The model learns:

```text
How should I respond to a user request?
```

Example:

```text
Instruction:
Summarize this text in two sentences.

Input:
<document>

Response:
<two-sentence summary>
```

The focus is:

```text
Instruction following
+
Correct response
+
Desired format
```

---

# 4. How is instruction tuning different from traditional supervised learning?

This is an important interview question.

## Traditional supervised learning

Suppose we build a spam classifier.

Input:

```text
"Congratulations! You won $1 million"
```

Output:

```text
Spam = 1
```

The model learns:

[
X \rightarrow Y
]

Example:

| Input   | Label    |
| ------- | -------- |
| Email A | Spam     |
| Email B | Not Spam |
| Image A | Cat      |
| Image B | Dog      |

The output is usually from a **fixed label space**.

```text
[0, 1]
```

or:

```text
cat
dog
bird
```

---

## Instruction tuning

The model receives a natural-language task.

Example:

```text
Instruction:
Classify the sentiment.

Input:
I love this product!

Response:
Positive
```

But another example may be:

```text
Instruction:
Summarize this article.

Input:
<article>

Response:
<generated summary>
```

Another:

```text
Instruction:
Write Python code to sort a list.

Input:
[5, 3, 1]

Response:
sorted([5, 3, 1])
```

So one model learns many tasks:

```text
Instruction A → Answer A
Instruction B → Answer B
Instruction C → Answer C
```

The output space is essentially open-ended.

---

# 5. Main difference

| Traditional Supervised Learning                          | Instruction Tuning                            |
| -------------------------------------------------------- | --------------------------------------------- |
| Usually fixed task                                       | Many tasks                                    |
| Often fixed labels                                       | Natural-language output                       |
| Input → Label                                            | Instruction + Input → Response                |
| Example: spam classification                             | Example: explain, summarize, translate        |
| Task often implicit in model architecture/training setup | Task explicitly described in natural language |
| Usually task-specific model                              | General-purpose instruction-following model   |

### Important clarification

Instruction tuning **is still supervised learning**.

The difference is not:

```text
Traditional supervised learning ❌
vs
Instruction tuning ❌
```

The correct relationship is:

```text
Supervised Fine-Tuning
        │
        ├── Classification fine-tuning
        │
        ├── Task-specific fine-tuning
        │
        └── Instruction tuning
```

Instruction tuning is a **specific style of supervised training data** for teaching LLMs to follow natural-language instructions.

---

# 6. What does an instruction dataset look like?

There are several common formats.

---

## Format 1: Instruction + Input + Output

```json
{
  "instruction": "Translate the following English text to French.",
  "input": "Good morning, how are you?",
  "output": "Bonjour, comment allez-vous ?"
}
```

Another:

```json
{
  "instruction": "Summarize the following text in one sentence.",
  "input": "Artificial intelligence is a field of computer science focused on creating systems capable of performing tasks that normally require human intelligence.",
  "output": "Artificial intelligence develops computer systems capable of performing tasks associated with human intelligence."
}
```

---

# 7. Dataset example as JSONL

A common training format is **JSON Lines (`.jsonl`)**.

Each line is one training example.

```json
{"instruction": "Explain recursion simply.", "input": "", "output": "Recursion is when a function calls itself to solve a smaller version of the same problem."}
{"instruction": "Translate English to French.", "input": "Hello world", "output": "Bonjour le monde"}
{"instruction": "Classify the sentiment.", "input": "I love this product.", "output": "Positive"}
```

Save as:

```text
instruction_dataset.jsonl
```

---

# 8. Example: create an instruction dataset with Python

```python
import json
```

Create examples:

```python
data = [

    {
        "instruction": "Explain recursion in simple terms.",
        "input": "",
        "output": (
            "Recursion is a programming technique "
            "where a function calls itself to solve "
            "a smaller version of the same problem."
        )
    },

    {
        "instruction": "Translate English to French.",
        "input": "Good morning",
        "output": "Bonjour"
    },

    {
        "instruction": "Classify the sentiment as Positive or Negative.",
        "input": "I really enjoyed this movie.",
        "output": "Positive"
    },

    {
        "instruction": "Write a Python function that adds two numbers.",
        "input": "",
        "output": (
            "def add(a, b):\n"
            "    return a + b"
        )
    }
]
```

Save as JSONL:

```python
with open(
    "instruction_dataset.jsonl",
    "w"
) as file:

    for example in data:

        file.write(
            json.dumps(example)
            +
            "\n"
        )
```

The output file contains:

```text
{"instruction":"Explain recursion in simple terms.","input":"","output":"Recursion is when..."}
{"instruction":"Translate English to French.","input":"Good morning","output":"Bonjour"}
...
```

---

# 9. Convert instruction data into an LLM prompt

LLMs need token sequences.

Suppose our dataset contains:

```python
example = {

    "instruction":
        "Explain recursion simply.",

    "input": "",

    "output":
        "Recursion is when a function calls itself."
}
```

We need to format it into text.

For example:

```text
### Instruction:
Explain recursion simply.

### Response:
Recursion is when a function calls itself.
```

Create a formatting function:

```python
def format_instruction(
    example
):

    instruction = example[
        "instruction"
    ]

    user_input = example[
        "input"
    ]

    output = example[
        "output"
    ]

    prompt = (
        f"### Instruction:\n"
        f"{instruction}\n\n"
    )

    if user_input:

        prompt += (
            f"### Input:\n"
            f"{user_input}\n\n"
        )

    prompt += (
        f"### Response:\n"
        f"{output}"
    )

    return prompt
```

Test:

```python
print(
    format_instruction(
        data[0]
    )
)
```

Output:

```text
### Instruction:
Explain recursion in simple terms.

### Response:
Recursion is when a function calls itself to solve a smaller version of the same problem.
```

---

# 10. Load the dataset using Hugging Face Datasets

```python
from datasets import load_dataset
```

Load:

```python
dataset = load_dataset(

    "json",

    data_files=
        "instruction_dataset.jsonl"
)
```

Check:

```python
print(
    dataset["train"][0]
)
```

You get:

```python
{
    "instruction":
        "Explain recursion in simple terms.",

    "input": "",

    "output":
        "Recursion is when a function calls itself..."
}
```

---

# 11. Format the entire dataset

```python
def formatting_function(
    examples
):

    texts = []

    for instruction, user_input, output in zip(

        examples["instruction"],

        examples["input"],

        examples["output"]
    ):

        text = (
            f"### Instruction:\n"
            f"{instruction}\n\n"
        )

        if user_input:

            text += (
                f"### Input:\n"
                f"{user_input}\n\n"
            )

        text += (
            f"### Response:\n"
            f"{output}"
        )

        texts.append(
            text
        )

    return {

        "text": texts
    }
```

Apply it:

```python
dataset = dataset.map(

    formatting_function,

    batched=True
)
```

Now:

```python
print(
    dataset["train"][0]["text"]
)
```

---

# 12. Tokenization

Load a tokenizer:

```python
from transformers import (
    AutoTokenizer
)
```

```python
model_name = "your-model"

tokenizer = AutoTokenizer.from_pretrained(
    model_name
)
```

Tokenize:

```python
def tokenize_function(
    examples
):

    return tokenizer(

        examples["text"],

        truncation=True,

        max_length=512
    )
```

Apply:

```python
tokenized_dataset = dataset.map(

    tokenize_function,

    batched=True
)
```

Now each text becomes:

```text
Text
 ↓
Tokenizer
 ↓
Token IDs
```

Example:

```python
{
    "input_ids": [
        101,
        2023,
        2003,
        1037,
        ...
    ],

    "attention_mask": [
        1,
        1,
        1,
        ...
    ]
}
```

---

# 13. How does the model know what the correct answer is?

For causal language models, the training objective predicts the next token.

Example:

```text
### Instruction:
Explain recursion.

### Response:
Recursion is when...
```

Token sequence:

```text
T1 → T2 → T3 → T4 → T5
```

The model learns:

[
P(T_{t+1} \mid T_1,T_2,...,T_t)
]

A basic implementation can use the same sequence as labels:

```python
def tokenize_function(
    examples
):

    tokens = tokenizer(

        examples["text"],

        truncation=True,

        max_length=512
    )

    tokens["labels"] = (
        tokens["input_ids"].copy()
    )

    return tokens
```

Conceptually:

```text
Input IDs:
[10, 20, 30, 40, 50]

Labels:
[10, 20, 30, 40, 50]
```

The model learns to predict the sequence token by token.

---

# 14. Important: response-only loss masking

For instruction tuning, we usually want the model to learn most strongly from the **assistant response**, not necessarily from reproducing the user instruction.

Example:

```text
### Instruction:
Explain recursion.

### Response:
Recursion is a technique...
```

Ideally:

```text
Instruction tokens → context
Response tokens → prediction target
```

So:

```text
Input:
[Instruction tokens][Response tokens]

Labels:
[-100][-100][-100][Response tokens]
```

In PyTorch and Hugging Face:

```text
-100 = ignore this token in loss calculation
```

---

# 15. Code: response-only loss masking

First, format:

```python
def create_prompt(
    instruction,
    user_input,
    output
):

    prompt = (
        "### Instruction:\n"
        f"{instruction}\n\n"
    )

    if user_input:

        prompt += (
            "### Input:\n"
            f"{user_input}\n\n"
        )

    response_prefix = (
        "### Response:\n"
    )

    return prompt, response_prefix + output
```

Create:

```python
prompt, response = create_prompt(

    instruction=
        "Explain recursion.",

    user_input="",

    output=
        "Recursion is when a function calls itself."
)
```

Full training text:

```python
full_text = (
    prompt
    +
    response
)
```

Tokenize separately:

```python
prompt_tokens = tokenizer(
    prompt,
    add_special_tokens=False
)

full_tokens = tokenizer(
    full_text,
    truncation=True,
    max_length=512
)
```

Create labels:

```python
labels = full_tokens[
    "input_ids"
].copy()
```

Ignore the prompt portion:

```python
prompt_length = len(
    prompt_tokens[
        "input_ids"
    ]
)
```

```python
labels[:prompt_length] = (
    [-100] * prompt_length
)
```

Result conceptually:

```text
Input IDs:

[Instruction][Instruction][Instruction][Response][Response]


Labels:

[-100][-100][-100][Correct][Correct]
```

Now calculate:

```python
tokenized_example = {

    "input_ids":
        full_tokens["input_ids"],

    "attention_mask":
        full_tokens["attention_mask"],

    "labels":
        labels
}
```

This means:

> The instruction is context, and the response is what the model is explicitly optimized to generate.

---

# 16. Modern chat-format instruction tuning

Modern LLMs often use a conversational structure instead.

Example:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful AI assistant."
    },
    {
      "role": "user",
      "content": "Explain recursion simply."
    },
    {
      "role": "assistant",
      "content": "Recursion is when a function solves a problem by calling itself on a smaller version of the problem."
    }
  ]
}
```

Another example:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Translate to French: Good morning"
    },
    {
      "role": "assistant",
      "content": "Bonjour"
    }
  ]
}
```

This format is often preferred for instruction-tuning chat models.

---

# 17. Apply a model's chat template

Modern tokenizers often provide:

```python
tokenizer.apply_chat_template()
```

Example:

```python
messages = [

    {
        "role": "system",
        "content":
            "You are a helpful programming assistant."
    },

    {
        "role": "user",
        "content":
            "Explain recursion simply."
    },

    {
        "role": "assistant",
        "content":
            "Recursion is when a function calls itself to solve a smaller version of a problem."
    }
]
```

Apply the template:

```python
text = tokenizer.apply_chat_template(

    messages,

    tokenize=False,

    add_generation_prompt=False
)
```

The actual output depends on the model's template.

It might look like:

```text
<system>
You are a helpful programming assistant.

<user>
Explain recursion simply.

<assistant>
Recursion is when a function calls itself...
```

Then tokenize:

```python
tokens = tokenizer(

    text,

    truncation=True,

    max_length=1024
)
```

### Important production rule

Use the **correct chat template for the base model**.

Do not invent:

```text
### Instruction:
```

if the model was originally trained with a specific format such as:

```text
<|user|>
<|assistant|>
```

Different models may expect different templates.

---

# 18. Full example: instruction tuning with TRL + LoRA

Here is a realistic setup.

Install dependencies:

```bash
pip install transformers datasets peft trl accelerate
```

Imports:

```python
from datasets import load_dataset

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)

from peft import (
    LoraConfig,
    get_peft_model
)

from trl import SFTConfig, SFTTrainer
```

Load dataset:

```python
dataset = load_dataset(

    "json",

    data_files={
        "train":
            "train.jsonl",

        "validation":
            "validation.jsonl"
    }
)
```

Example dataset:

```json
{
  "instruction": "Explain dependency injection.",
  "input": "",
  "output": "Dependency injection is a design pattern where an object receives its dependencies from outside instead of creating them itself."
}
```

Format dataset:

```python
def format_example(
    example
):

    text = (
        "### Instruction:\n"
        f"{example['instruction']}\n\n"
    )

    if example["input"]:

        text += (
            "### Input:\n"
            f"{example['input']}\n\n"
        )

    text += (
        "### Response:\n"
        f"{example['output']}"
    )

    return {
        "text": text
    }
```

Apply:

```python
dataset = dataset.map(
    format_example
)
```

Load model:

```python
model_name = (
    "your-base-model"
)
```

```python
tokenizer = AutoTokenizer.from_pretrained(
    model_name
)

model = AutoModelForCausalLM.from_pretrained(
    model_name
)
```

Add padding token if needed:

```python
if tokenizer.pad_token is None:

    tokenizer.pad_token = (
        tokenizer.eos_token
    )
```

Configure LoRA:

```python
lora_config = LoraConfig(

    r=16,

    lora_alpha=32,

    lora_dropout=0.05,

    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj"
    ],

    task_type="CAUSAL_LM"
)
```

Apply LoRA:

```python
model = get_peft_model(
    model,
    lora_config
)
```

Check trainable parameters:

```python
model.print_trainable_parameters()
```

You may see something conceptually like:

```text
Trainable params: 10M
All params: 7B
Trainable: 0.14%
```

This means the base model is mostly frozen.

---

# 19. Configure training

```python
training_args = SFTConfig(

    output_dir="./instruction_model",

    num_train_epochs=3,

    learning_rate=2e-4,

    per_device_train_batch_size=2,

    gradient_accumulation_steps=8,

    max_length=1024,

    eval_strategy="steps",

    eval_steps=100,

    save_steps=100,

    logging_steps=10,

    warmup_ratio=0.05,

    bf16=True,

    report_to="none"
)
```

Create trainer:

```python
trainer = SFTTrainer(

    model=model,

    args=training_args,

    train_dataset=dataset["train"],

    eval_dataset=dataset["validation"],

    processing_class=tokenizer,

    dataset_text_field="text"
)
```

Train:

```python
trainer.train()
```

Save:

```python
trainer.save_model(
    "./instruction_model"
)

tokenizer.save_pretrained(
    "./instruction_model"
)
```

---

# 20. What is the model learning internally?

Consider this training example:

```text
Instruction:
Explain FastAPI in 3 points.

Response:
1. FastAPI is a Python framework for building APIs.
2. It supports type validation using Pydantic.
3. It provides automatic API documentation.
```

During training:

```text
Instruction
     ↓
Tokenizer
     ↓
Token IDs
     ↓
Transformer
     ↓
Predicted tokens
     ↓
Compare with expected response
     ↓
Calculate loss
     ↓
Backpropagation
     ↓
Update trainable parameters
```

The model gradually learns:

```text
"Explain X in 3 points"
        ↓
Generate
1.
2.
3.
```

But a good instruction-tuned model should generalize:

```text
Explain Kubernetes in 3 points
```

even if that exact example was not present in the dataset.

That is why dataset diversity matters.

---

# 21. Multi-task instruction tuning

A strong instruction dataset contains different tasks.

```python
dataset = [

    {
        "instruction":
            "Summarize the text.",

        "input":
            "Long article...",

        "output":
            "Short summary..."
    },

    {
        "instruction":
            "Translate English to French.",

        "input":
            "Hello",

        "output":
            "Bonjour"
    },

    {
        "instruction":
            "Explain this Python error.",

        "input":
            "TypeError: ...",

        "output":
            "This error occurs because..."
    },

    {
        "instruction":
            "Write unit tests.",

        "input":
            "def add(a, b): ...",

        "output":
            "import pytest..."
    }
]
```

This teaches:

```text
Many instructions
       ↓
One model
       ↓
General instruction-following ability
```

---

# 22. Instruction tuning vs traditional supervised learning with code

## Traditional classification

```python
from sklearn.linear_model import (
    LogisticRegression
)
```

```python
X = [

    [0.1, 0.2],

    [0.9, 0.8],

    [0.2, 0.1],

    [0.8, 0.9]
]

y = [

    0,
    1,
    0,
    1
]
```

Train:

```python
model = LogisticRegression()

model.fit(
    X,
    y
)
```

Output:

```text
Input → fixed class

0 or 1
```

---

## Instruction tuning

```python
{
    "instruction":
        "Classify the sentiment.",

    "input":
        "I love this product.",

    "output":
        "Positive"
}
```

Another task:

```python
{
    "instruction":
        "Summarize the text.",

    "input":
        "Long document...",

    "output":
        "Summary..."
}
```

The model is not limited to:

```text
Class 0
Class 1
```

It generates a sequence:

[
y_1, y_2, y_3, ..., y_n
]

The same model can perform many tasks because the **instruction defines the task**.

---

# 23. Real-world example: customer-support instruction tuning

Suppose a company wants a support assistant.

A bad dataset:

```json
{
  "question": "Where is my order?",
  "answer": "Your order is on the way."
}
```

A better instruction dataset:

```json
{
  "instruction": "Answer the customer professionally and concisely.",
  "input": "Where is my order?",
  "output": "You can check your order status in the Orders section of your account. If you share your order number, I can help you further."
}
```

Another:

```json
{
  "instruction": "Respond empathetically and explain the refund process.",
  "input": "I want a refund for my subscription.",
  "output": "I'm sorry to hear that. You can request a refund through the Billing section of your account. Refund eligibility depends on your purchase date and applicable policy."
}
```

Now the model learns not just:

```text
Question → Answer
```

but:

```text
Instruction
    +
Desired tone
    +
Input
    ↓
Appropriate response
```

---

# 24. Dataset quality matters

Bad example:

```json
{
  "instruction": "Explain Python",
  "output": "Python good language lol"
}
```

Good example:

```json
{
  "instruction": "Explain Python to a beginner in three concise points.",
  "output": "1. Python is a readable general-purpose programming language. 2. It is commonly used for web development, automation, data science, and AI. 3. Its simple syntax makes it beginner-friendly."
}
```

The model learns from:

```text
Instruction quality
Response quality
Consistency
Correctness
Format
Tone
```

So:

> **Fine-tuning cannot magically fix a low-quality dataset. The model will learn the behavior present in the training examples.**

---

# 25. Best practices for an instruction dataset

## 1. Use diverse instructions

Avoid:

```text
Explain Python
Explain Java
Explain FastAPI
Explain Docker
```

only.

Also include:

```text
Compare X and Y
Summarize X
Debug this code
Write tests
Explain for beginners
Explain for experts
Return JSON
Return 3 bullet points
```

---

## 2. Include realistic user language

Real users write:

```text
why my api is not working
```

not always:

```text
Please provide a detailed technical diagnosis of my API connectivity issue.
```

Include both.

---

## 3. Keep output high quality

The output becomes the target behavior.

Bad output:

```text
idk try again
```

The model can learn that style.

---

## 4. Avoid duplicate examples

Bad:

```text
Explain FastAPI
Explain FastAPI
Explain FastAPI
Explain FastAPI
```

This increases the risk of overfitting.

---

## 5. Use train/validation/test splits

For example:

```text
Train       80%
Validation  10%
Test        10%
```

Example:

```python
dataset = dataset["train"].train_test_split(

    test_size=0.2,

    seed=42
)
```

Then split again:

```python
temp = dataset["test"].train_test_split(

    test_size=0.5,

    seed=42
)
```

Final:

```python
train_dataset = dataset["train"]

validation_dataset = temp["train"]

test_dataset = temp["test"]
```

---

# 26. Important distinction: instruction tuning does not add unlimited knowledge

Suppose you want an LLM to answer:

```text
What was our company's revenue yesterday?
```

You generally should not instruction-tune the model every day with new revenue data.

Use:

```text
RAG / Database / API
```

Instruction tuning is better for:

```text
How should the model behave?
How should it respond?
What format should it use?
What tasks should it follow?
```

RAG is better for:

```text
What does the latest company document say?
What is today's policy?
What is the current inventory?
What happened in a new document?
```

---

# 27. Instruction tuning vs RAG

```text
INSTRUCTION TUNING

Teach behavior

User instruction
      ↓
Fine-tuned model
      ↓
Desired behavior
```

```text
RAG

Provide external knowledge

User question
      ↓
Retriever
      ↓
Relevant documents
      ↓
LLM
      ↓
Answer
```

Example:

### Instruction tuning

Teach:

```text
Always answer support questions politely.
Return troubleshooting steps as numbered lists.
```

### RAG

Retrieve:

```text
Latest company refund policy
```

A production system often combines both:

```text
Instruction-tuned LLM
        +
        RAG
        +
        Tools/APIs
```

---

# Interview-ready answer

> **Instruction tuning is a form of supervised fine-tuning where an LLM is trained on instruction-response examples so it learns to follow natural-language requests. A typical example contains an instruction, optional input or context, and an expected output. Unlike traditional supervised learning, which often trains a model for one fixed task with a limited label space, instruction tuning can train one generative model across many tasks, with the task specified directly in natural language. During training, the instruction is provided as context and the model learns to generate the target response token by token. In practice, modern instruction tuning often uses chat-formatted datasets and applies the base model's official chat template.**

# Final mental model

```text
PRETRAINING

Massive text
    ↓
Learn language and knowledge


INSTRUCTION TUNING

Instruction + Input
        ↓
Learn desired behavior
        ↓
Generate expected response


RAG

Question
   ↓
Retrieve current knowledge
   ↓
LLM generates grounded answer
```

The most important relationship to remember is:

> **Pre-training teaches an LLM language and broad patterns; instruction tuning teaches it how to behave and follow requests; RAG gives it external, current, or private knowledge at inference time.**
