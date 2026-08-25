# How would you fine-tune an LLM for coding?

Fine-tuning a coding model means training an existing LLM to generate better code for a **specific programming language, framework, coding style, domain, or internal codebase**.

For example:

```text
User requirement
      ↓
Fine-tuned Code LLM
      ↓
Generated code
      ↓
Tests + Linting + Security checks
      ↓
Valid code
```

A model may already know Python, but you might fine-tune it to:

* follow your company's coding standards
* generate FastAPI services
* follow Clean Architecture
* use your internal SDK
* write unit tests
* generate Swift/SwiftUI code
* generate secure code
* fix bugs in a specific codebase
* generate infrastructure code
* convert legacy code

---

# 1. When should you fine-tune a coding model?

Suppose you ask a general model:

```text
Create a FastAPI endpoint for creating a user.
```

It might generate:

```python
@app.post("/users")
def create_user(user: User):
    return user
```

But your company architecture is:

```text
Router
   ↓
Service
   ↓
Repository
   ↓
Database
```

And you require:

```text
FastAPI
+
Pydantic
+
Dependency Injection
+
Async SQLAlchemy
+
Repository Pattern
+
Structured Logging
+
Exception Handling
```

Fine-tuning can teach the model your preferred patterns.

---

# 2. Fine-tuning vs RAG for coding

This is a very important distinction.

## Fine-tuning

Use it for stable behavior:

```text
Coding style
Architecture patterns
Preferred frameworks
Output format
Common implementation patterns
Testing style
```

## RAG

Use it for changing knowledge:

```text
Current repository
Current APIs
Current functions
Internal documentation
Latest library versions
```

A production coding assistant often looks like:

```text
                    Developer Request
                           │
                           ▼
                     Code Retrieval
                           │
                ┌──────────┴──────────┐
                ▼                     ▼
             Relevant              API Docs
             Source Code
                │                     │
                └──────────┬──────────┘
                           ▼
                    Fine-tuned LLM
                           │
                           ▼
                         Code
                           │
             ┌─────────────┼──────────────┐
             ▼             ▼              ▼
           Tests          Linter       Security Scan
             │             │              │
             └─────────────┴──────────────┘
                           │
                           ▼
                         Result
```

> You should usually **not fine-tune a model on the entire codebase every time the repository changes**. Use retrieval for current code.

---

# 3. What does a coding fine-tuning dataset look like?

A basic example:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are an expert Python backend engineer. Generate production-quality code following Clean Architecture."
    },
    {
      "role": "user",
      "content": "Create a FastAPI endpoint that returns a user by ID."
    },
    {
      "role": "assistant",
      "content": "@router.get('/users/{user_id}')\nasync def get_user(user_id: int, service: UserService = Depends(get_user_service)):\n    user = await service.get_user(user_id)\n    if user is None:\n        raise HTTPException(status_code=404, detail='User not found')\n    return user"
    }
  ]
}
```

The model learns:

```text
Requirement
     ↓
Correct implementation
```

---

# 4. Better dataset: include context

For coding, context is often essential.

Example:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "Follow the project's async Python and Clean Architecture conventions."
    },
    {
      "role": "user",
      "content": "Existing repository interface:\n\nclass UserRepository(Protocol):\n    async def get_by_id(self, user_id: int) -> User | None: ...\n\nImplement UserService.get_user."
    },
    {
      "role": "assistant",
      "content": "class UserService:\n    def __init__(self, repository: UserRepository):\n        self.repository = repository\n\n    async def get_user(self, user_id: int) -> User:\n        user = await self.repository.get_by_id(user_id)\n        if user is None:\n            raise UserNotFoundError(user_id)\n        return user"
    }
  ]
}
```

This is better because the model learns:

```text
Context
+
Requirement
↓
Implementation
```

---

# 5. Types of coding examples to include

A good coding dataset should contain multiple task types.

## A. Code generation

```text
Requirement → Code
```

Example:

```text
Create a retry decorator.
```

↓

```python
async def retry(...):
    ...
```

---

## B. Code completion

```text
Existing code → Missing implementation
```

Example:

```python
class UserRepository:

    async def get_by_id(
        self,
        user_id: int
    ):
        # TODO
```

Target:

```python
return await self.session.get(
    User,
    user_id
)
```

---

## C. Bug fixing

Input:

```python
async def get_user():
    user = db.query(User).first()
```

Problem:

```text
Blocking database operation inside async code.
```

Target:

```python
async def get_user(
    session: AsyncSession
):
    result = await session.execute(
        select(User)
    )

    return result.scalar_one_or_none()
```

---

## D. Code explanation

```text
Code → Explanation
```

---

## E. Test generation

```text
Code → pytest tests
```

Example:

```text
Function:
calculate_discount(...)
```

↓

```python
def test_discount_for_premium_user():
    ...
```

---

## F. Refactoring

```text
Bad code → Improved code
```

---

# 6. Create a coding fine-tuning dataset

Let's build one using Python.

```python
import json
```

```python
examples = [
    {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a senior Python backend engineer. "
                    "Write production-quality async code. "
                    "Use type hints and proper error handling."
                )
            },
            {
                "role": "user",
                "content": (
                    "Implement an async function that fetches "
                    "a user from a repository by ID."
                )
            },
            {
                "role": "assistant",
                "content": """
async def get_user(
    user_id: int,
    repository: UserRepository
) -> User:

    user = await repository.get_by_id(
        user_id
    )

    if user is None:
        raise UserNotFoundError(
            user_id
        )

    return user
"""
            }
        ]
    },

    {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a senior Python backend engineer. "
                    "Write secure production-quality code."
                )
            },
            {
                "role": "user",
                "content": (
                    "Create a FastAPI endpoint that "
                    "retrieves a user by ID."
                )
            },
            {
                "role": "assistant",
                "content": """
@router.get(
    "/users/{user_id}",
    response_model=UserResponse
)
async def get_user_endpoint(
    user_id: int,
    service: UserService = Depends(
        get_user_service
    )
) -> UserResponse:

    user = await service.get_user(
        user_id
    )

    return UserResponse.model_validate(
        user
    )
"""
            }
        ]
    }
]
```

Save as JSONL:

```python
with open(
    "coding_dataset.jsonl",
    "w",
    encoding="utf-8"
) as file:

    for example in examples:

        file.write(
            json.dumps(example)
            + "\n"
        )
```

---

# 7. Dataset quality is extremely important

For coding fine-tuning:

```text
Bad code
   ↓
Model learns bad code
```

You should validate examples.

Check:

* Does the code compile?
* Does it pass tests?
* Does it follow linting?
* Is it secure?
* Is it duplicated?
* Does it use deprecated APIs?
* Is the instruction clear?
* Is the answer actually correct?

A production pipeline:

```text
Raw Code
    ↓
License Filtering
    ↓
Secret Detection
    ↓
Deduplication
    ↓
Parsing
    ↓
Compilation
    ↓
Unit Tests
    ↓
Linting
    ↓
Security Scan
    ↓
Training Dataset
```

---

# 8. Validate Python code before training

You can compile Python code.

```python
def validate_python_code(
    code: str
) -> bool:

    try:

        compile(
            code,
            "<generated>",
            "exec"
        )

        return True

    except SyntaxError:

        return False
```

Example:

```python
code = """
def add(a, b):
    return a + b
"""
```

```python
print(
    validate_python_code(
        code
    )
)
```

Output:

```text
True
```

---

# 9. Detect duplicate training examples

A simple approach:

```python
import hashlib
```

```python
def hash_example(
    example: dict
) -> str:

    content = json.dumps(
        example,
        sort_keys=True
    )

    return hashlib.sha256(
        content.encode()
    ).hexdigest()
```

Remove duplicates:

```python
def remove_duplicates(
    examples: list[dict]
) -> list[dict]:

    seen = set()

    unique_examples = []

    for example in examples:

        example_hash = hash_example(
            example
        )

        if example_hash not in seen:

            seen.add(
                example_hash
            )

            unique_examples.append(
                example
            )

    return unique_examples
```

---

# 10. Split train and validation data correctly

```python
from datasets import Dataset
```

```python
dataset = Dataset.from_list(
    examples
)
```

```python
splits = dataset.train_test_split(
    test_size=0.1,
    seed=42
)

train_dataset = splits["train"]

validation_dataset = splits["test"]
```

But random splitting alone may cause leakage.

Example:

```text
Train:
Implement UserService.get_user()

Validation:
Implement UserService.get_user_by_id()
```

These may be nearly identical.

Better:

```text
Split by:
Repository
Project
Function family
Problem category
```

This measures real generalization.

---

# 11. Format examples using the model's chat template

Install:

```bash
pip install transformers datasets peft trl accelerate
```

Load the model:

```python
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)
```

```python
MODEL_NAME = "your-instruct-model"
```

```python
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME
)
```

Format:

```python
def format_example(example):

    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False
    )

    return {
        "text": text
    }
```

Load the JSONL dataset:

```python
from datasets import load_dataset
```

```python
dataset = load_dataset(
    "json",
    data_files={
        "train": "coding_train.jsonl",
        "validation": "coding_validation.jsonl"
    }
)
```

Apply:

```python
dataset = dataset.map(
    format_example
)
```

---

# 12. Apply LoRA

For coding fine-tuning, LoRA is often a practical choice.

```python
from peft import (
    LoraConfig,
    get_peft_model
)
```

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

Apply:

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

Conceptually:

```text
Base Model Parameters

Frozen:
████████████████████

LoRA Parameters

Trainable:
██
```

---

# 13. Fine-tune using supervised fine-tuning

```python
from trl import (
    SFTTrainer,
    SFTConfig
)
```

Configuration:

```python
training_args = SFTConfig(

    output_dir="./coding_model",

    num_train_epochs=2,

    learning_rate=2e-4,

    per_device_train_batch_size=2,

    gradient_accumulation_steps=8,

    max_length=2048,

    warmup_ratio=0.05,

    lr_scheduler_type="cosine",

    logging_steps=10,

    eval_strategy="steps",

    eval_steps=100,

    save_steps=100,

    bf16=True,

    report_to="none"
)
```

Trainer:

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
    "./coding_lora_adapter"
)
```

---

# 14. Assistant-only training loss

For code generation, this is often important.

Suppose:

```text
SYSTEM:
You are a Python expert.

USER:
Create a retry decorator.

ASSISTANT:

def retry(...):
    ...
```

We generally want to optimize:

```text
System instruction → no loss

User question → no loss

Assistant code → training loss
```

Conceptually:

```text
SYSTEM TOKENS
labels = -100

USER TOKENS
labels = -100

ASSISTANT CODE TOKENS
labels = token IDs
```

Why?

Because we want the model to learn:

```text
Prompt
   ↓
Correct code
```

rather than spending optimization capacity learning to reproduce the prompt.

The exact implementation depends on the model and trainer. Many modern chat templates and SFT training tools provide an assistant-only loss option.

---

# 15. Code-specific training example with context

A stronger example:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "Follow async Python, dependency injection, repository pattern, and type hints."
    },
    {
      "role": "user",
      "content": "Existing interface:\n\nclass UserRepository(Protocol):\n    async def get_by_email(self, email: str) -> User | None: ...\n\nImplement UserService.find_by_email. Raise UserNotFoundError when the user does not exist."
    },
    {
      "role": "assistant",
      "content": "class UserService:\n    def __init__(self, repository: UserRepository) -> None:\n        self.repository = repository\n\n    async def find_by_email(self, email: str) -> User:\n        user = await self.repository.get_by_email(email)\n        if user is None:\n            raise UserNotFoundError(email)\n        return user"
    }
  ]
}
```

This teaches:

```text
Existing API
+
Requirement
+
Architecture rules
↓
Compatible code
```

---

# 16. Inference

After training:

```python
messages = [
    {
        "role": "system",
        "content": (
            "You are a senior Python engineer. "
            "Generate production-quality async code."
        )
    },
    {
        "role": "user",
        "content": (
            "Implement a repository method that "
            "returns a user by email."
        )
    }
]
```

Tokenize:

```python
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt"
).to(model.device)
```

Generate:

```python
outputs = model.generate(
    inputs,
    max_new_tokens=512,
    do_sample=False
)
```

Decode only new tokens:

```python
generated_tokens = outputs[
    0,
    inputs.shape[1]:
]

code = tokenizer.decode(
    generated_tokens,
    skip_special_tokens=True
)

print(code)
```

---

# 17. Validate generated code

Generated code should not directly be trusted.

A production pipeline:

```text
Generated Code
      │
      ▼
Syntax Check
      │
      ▼
Type Checking
      │
      ▼
Unit Tests
      │
      ▼
Linting
      │
      ▼
Security Scan
      │
      ▼
Sandbox Execution
      │
      ▼
Return Result
```

---

## Syntax validation

```python
def check_syntax(
    code: str
):

    try:

        compile(
            code,
            "<generated_code>",
            "exec"
        )

        return {
            "valid": True,
            "error": None
        }

    except SyntaxError as error:

        return {
            "valid": False,
            "error": str(error)
        }
```

---

# 18. Run tests in a sandbox

Conceptually:

```python
def evaluate_generated_code(
    generated_file: str
):

    # 1. Create isolated environment
    # 2. Write generated code
    # 3. Run tests
    # 4. Capture output
    # 5. Destroy environment

    pass
```

You should avoid blindly executing untrusted LLM-generated code on your production host.

A safer design:

```text
LLM
 ↓
Generated Code
 ↓
Isolated Container / Sandbox
 ↓
Resource Limits
 ↓
Tests
 ↓
Result
```

Use:

* CPU limits
* memory limits
* network isolation when possible
* execution timeouts
* restricted filesystem access
* non-root users

---

# 19. Test-driven fine-tuning examples

One powerful dataset pattern is:

```text
Task
+
Existing Code
+
Tests
↓
Implementation
```

Example:

```text
Task:
Implement calculate_discount.

Tests:

assert calculate_discount(100, "premium") == 80
assert calculate_discount(100, "regular") == 95
```

Target:

```python
def calculate_discount(
    amount: float,
    customer_type: str
) -> float:

    discounts = {
        "premium": 0.20,
        "regular": 0.05
    }

    discount = discounts.get(
        customer_type,
        0
    )

    return amount * (
        1 - discount
    )
```

This helps the model learn **functional correctness**, not just style.

---

# 20. How would you evaluate a coding model?

Do not evaluate only with:

```text
Does the code look good?
```

Use multiple metrics.

## 1. Compilation rate

```text
Generated programs
        ↓
How many compile?
```

```text
Compile Rate =
Compiled Programs / Total Programs
```

---

## 2. Test pass rate

```text
Generated Code
       ↓
Unit Tests
       ↓
Pass / Fail
```

```text
Pass@1
=
Tasks solved by first generation
/
Total tasks
```

---

## 3. Pass@k

If you generate multiple solutions:

```text
Attempt 1 → Fail
Attempt 2 → Fail
Attempt 3 → Pass
```

Then the task contributes to:

```text
Pass@3
```

---

## 4. Static analysis

Use tools appropriate to the language:

```text
Python
→ Ruff / type checker

JavaScript
→ ESLint

Java
→ compiler + static analysis

Swift
→ Swift compiler + SwiftLint
```

---

## 5. Security evaluation

Check for:

```text
SQL injection
Command injection
Hardcoded secrets
Path traversal
Unsafe deserialization
Authentication bypass
```

---

# 21. Fine-tuning a model for a specific codebase

Suppose your company has:

```text
project/
│
├── api/
├── services/
├── repositories/
├── models/
├── schemas/
└── tests/
```

You should create examples like:

```text
Existing Repository Interface
        +
Relevant Models
        +
Task
        ↓
Expected Implementation
```

But do not dump the entire repository into every training example.

Better:

```text
Codebase
    ↓
Code Parser
    ↓
Functions / Classes
    ↓
Dependency Graph
    ↓
Related Code
    ↓
Training Example
```

Example:

```text
Context:
UserRepository.get_by_id()

Task:
Implement UserService.get_user()

Target:
Service implementation
```

---

# 22. Fine-tuning vs RAG for internal code

Use this rule:

| Requirement                  | Best approach           |
| ---------------------------- | ----------------------- |
| Coding style                 | Fine-tuning             |
| Stable architecture patterns | Fine-tuning             |
| Internal SDK usage           | Fine-tuning + RAG       |
| Current repository           | RAG                     |
| Recently changed APIs        | RAG                     |
| Current function signatures  | RAG                     |
| Bug fixing with current code | RAG + model             |
| New framework knowledge      | Base model/update + RAG |

The strongest production architecture is:

```text
                   Developer
                      │
                      ▼
                 User Task
                      │
                      ▼
               Repository Search
                      │
             ┌────────┴────────┐
             ▼                 ▼
       Relevant Files      API Docs
             │                 │
             └────────┬────────┘
                      ▼
              Fine-tuned Model
                      │
                      ▼
                    Code
                      │
                      ▼
                 Run Tests
                      │
               ┌──────┴──────┐
               ▼             ▼
              Pass          Fail
               │             │
               ▼             ▼
             Return     Error Feedback
                               │
                               ▼
                            Retry
```

---

# 23. A complete simplified training script

```python
from datasets import load_dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

from peft import (
    LoraConfig,
    get_peft_model
)

from trl import (
    SFTTrainer,
    SFTConfig
)


MODEL_NAME = "your-instruct-model"


# ---------------------------------
# Load tokenizer
# ---------------------------------

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


# ---------------------------------
# Load model
# ---------------------------------

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME
)


# ---------------------------------
# Load dataset
# ---------------------------------

dataset = load_dataset(
    "json",
    data_files={
        "train": "coding_train.jsonl",
        "validation": "coding_validation.jsonl"
    }
)


# ---------------------------------
# Convert chat → model format
# ---------------------------------

def format_example(example):

    text = tokenizer.apply_chat_template(

        example["messages"],

        tokenize=False,

        add_generation_prompt=False
    )

    return {
        "text": text
    }


dataset = dataset.map(
    format_example
)


# ---------------------------------
# LoRA
# ---------------------------------

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


model = get_peft_model(
    model,
    lora_config
)


# ---------------------------------
# Training configuration
# ---------------------------------

training_args = SFTConfig(

    output_dir="./coding_model",

    num_train_epochs=2,

    learning_rate=2e-4,

    per_device_train_batch_size=2,

    gradient_accumulation_steps=8,

    max_length=2048,

    warmup_ratio=0.05,

    lr_scheduler_type="cosine",

    eval_strategy="steps",

    eval_steps=100,

    save_steps=100,

    bf16=True,

    logging_steps=10,

    report_to="none"
)


# ---------------------------------
# Trainer
# ---------------------------------

trainer = SFTTrainer(

    model=model,

    args=training_args,

    train_dataset=dataset["train"],

    eval_dataset=dataset["validation"],

    processing_class=tokenizer,

    dataset_text_field="text"
)


# ---------------------------------
# Train
# ---------------------------------

trainer.train()


# ---------------------------------
# Save adapter
# ---------------------------------

trainer.save_model(
    "./coding_lora_adapter"
)
```

---

# Interview-ready answer

> **To fine-tune a model for coding, I would start with an existing instruction-tuned code-capable LLM and build a high-quality supervised dataset containing tasks, relevant code context, and correct implementations. The dataset should include code generation, completion, bug fixing, refactoring, test generation, and realistic repository-level examples.**
>
> **Before training, I would clean the data, remove duplicates, detect secrets and license issues, validate syntax, run tests where possible, and filter deprecated or insecure code. I would usually use LoRA or QLoRA for efficient fine-tuning and calculate training loss primarily on the assistant's code output.**
>
> **In production, I would combine the fine-tuned model with repository-level RAG because source code and APIs change frequently. Generated code would then go through compilation, linting, type checking, unit tests, security scanning, and sandboxed execution.**
>
> **I would evaluate the model using compilation rate, unit-test pass rate, Pass@1/Pass@k, static analysis, security checks, and repository-level compatibility.**

## The most important mental model

```text
Fine-tuning
    ↓
Teaches HOW to write code

RAG
    ↓
Provides CURRENT code and context

Testing
    ↓
Checks whether code works

Security scanning
    ↓
Checks whether code is safe
```

For a real-world coding assistant:

> **Fine-tuned coding model + Code RAG + Tool execution in a sandbox + Automated tests** is much stronger than fine-tuning alone.
