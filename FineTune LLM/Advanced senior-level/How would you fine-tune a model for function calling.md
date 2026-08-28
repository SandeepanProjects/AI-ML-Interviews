# How would you fine-tune a model for function calling?

**Function calling** is very similar to tool calling. The main idea is:

> Train the model to convert a natural-language request into a **structured function name + validated arguments**, then let your application execute that function.

Example:

```text
User: Transfer ₹5,000 to Alice
        ↓
Model
        ↓
{
  "name": "transfer_money",
  "arguments": {
    "recipient": "Alice",
    "amount": 5000,
    "currency": "INR"
  }
}
        ↓
Application validates
        ↓
Function executes
```

## Tool calling vs function calling

In practice, the terms are often used interchangeably:

```text
Tool calling     = broader concept
Function calling = calling a function/API with structured arguments
```

A production system should look like:

```text
User
 │
 ▼
LLM
 │
 ▼
Function call?
 │
 ├── No ──► Direct response
 │
 ▼ Yes
Function name + arguments
 │
 ▼
Schema validation
 │
 ▼
Authorization
 │
 ▼
Function execution
 │
 ▼
Function result
 │
 ▼
LLM
 │
 ▼
Final response
```

---

# 1. Define the functions

Let's build a banking example.

```python
from decimal import Decimal


def get_account_balance(
    account_id: str
) -> dict:
    return {
        "account_id": account_id,
        "balance": 25000,
        "currency": "INR"
    }


def transfer_money(
    recipient: str,
    amount: Decimal,
    currency: str
) -> dict:
    return {
        "status": "success",
        "recipient": recipient,
        "amount": str(amount),
        "currency": currency
    }
```

The model should **not directly execute these functions**.

It should only produce:

```json
{
  "name": "get_account_balance",
  "arguments": {
    "account_id": "ACC-123"
  }
}
```

Your backend decides whether execution is allowed.

---

# 2. Define function schemas

The model needs to understand what functions exist.

```python
FUNCTIONS = [
    {
        "name": "get_account_balance",
        "description": "Get the balance for a bank account",
        "parameters": {
            "type": "object",
            "properties": {
                "account_id": {
                    "type": "string",
                    "description": "The account ID"
                }
            },
            "required": ["account_id"]
        }
    },
    {
        "name": "transfer_money",
        "description": "Transfer money to a recipient",
        "parameters": {
            "type": "object",
            "properties": {
                "recipient": {
                    "type": "string"
                },
                "amount": {
                    "type": "number",
                    "minimum": 0.01
                },
                "currency": {
                    "type": "string",
                    "enum": ["INR", "USD"]
                }
            },
            "required": [
                "recipient",
                "amount",
                "currency"
            ]
        }
    }
]
```

The model learns:

```text
Natural language
       ↓
Function selection
       ↓
Argument extraction
       ↓
Structured output
```

---

# 3. Create the fine-tuning dataset

A basic function-calling training example:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "What is the balance of account ACC-123?"
    },
    {
      "role": "assistant",
      "function_call": {
        "name": "get_account_balance",
        "arguments": {
          "account_id": "ACC-123"
        }
      }
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
      "content": "Send ₹5000 to Alice."
    },
    {
      "role": "assistant",
      "function_call": {
        "name": "transfer_money",
        "arguments": {
          "recipient": "Alice",
          "amount": 5000,
          "currency": "INR"
        }
      }
    }
  ]
}
```

A no-function example:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "What is the capital of France?"
    },
    {
      "role": "assistant",
      "content": "Paris."
    }
  ]
}
```

This is important because you want the model to learn:

```text
Some requests → Function call
Some requests → Normal answer
```

---

# 4. Include the complete function-calling loop

For production training, include examples like:

```text
User
 ↓
Assistant function call
 ↓
Function result
 ↓
Assistant final answer
```

Example:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "What is my account balance?"
    },
    {
      "role": "assistant",
      "function_call": {
        "id": "call_001",
        "name": "get_account_balance",
        "arguments": {
          "account_id": "ACC-123"
        }
      }
    },
    {
      "role": "function",
      "function_call_id": "call_001",
      "content": "{\"balance\":25000,\"currency\":\"INR\"}"
    },
    {
      "role": "assistant",
      "content": "Your account balance is ₹25,000."
    }
  ]
}
```

This teaches both:

### Step 1: Planning

```text
User request
     ↓
Which function should I call?
```

### Step 2: Result interpretation

```text
Function result
     ↓
Generate final answer
```

---

# 5. Dataset preparation in Python

Suppose your `train.jsonl` contains:

```text
data/
├── train.jsonl
└── validation.jsonl
```

Load it:

```python
from datasets import load_dataset

dataset = load_dataset(
    "json",
    data_files={
        "train": "data/train.jsonl",
        "validation": "data/validation.jsonl"
    }
)
```

Check:

```python
print(dataset["train"][0])
```

---

# 6. Use the base model's function-calling format

This is one of the most important points.

Different models may expect different formats:

```text
Model A
→ <tool_call> ... </tool_call>

Model B
→ special tokens

Model C
→ JSON

Model D
→ proprietary chat template
```

So don't arbitrarily invent the format if the base model already supports function calling.

Load the tokenizer:

```python
from transformers import AutoTokenizer

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)
```

Conceptually:

```python
formatted_text = tokenizer.apply_chat_template(
    messages,
    tools=FUNCTIONS,
    tokenize=False
)
```

The tokenizer's chat template can convert structured messages and function definitions into the format expected by that particular model.

---

# 7. Example custom function-call format

If your base model does not provide a usable template, you can train a consistent format.

For example:

```text
<functions>
[
  {
    "name": "get_account_balance",
    "parameters": {
      "account_id": "string"
    }
  }
]
</functions>

<user>
What is the balance of account ACC-123?
</user>

<assistant>
<function_call>
{
  "name": "get_account_balance",
  "arguments": {
    "account_id": "ACC-123"
  }
}
</function_call>
</assistant>
```

Python formatter:

```python
import json


def format_function_call(
    user_message: str,
    function_name: str,
    arguments: dict
) -> str:

    function_call = {
        "name": function_name,
        "arguments": arguments
    }

    return f"""
<user>
{user_message}
</user>

<assistant>
<function_call>
{json.dumps(function_call)}
</function_call>
</assistant>
"""
```

Consistency is critical.

Don't mix:

```text
FUNCTION_CALL
```

with:

```text
TOOL:
```

and:

```text
CALL_FUNCTION:
```

in different training examples unless the model is intentionally expected to support all formats.

---

# 8. Load a model using QLoRA

For a 7B/8B model on limited GPU memory:

```python
import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
```

Configure 4-bit quantization:

```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)
```

Load:

```python
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto"
)
```

Prepare the model:

```python
from peft import prepare_model_for_kbit_training

model = prepare_model_for_kbit_training(
    model
)
```

---

# 9. Configure LoRA

```python
from peft import LoraConfig

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

    bias="none",
    task_type="CAUSAL_LM"
)
```

For function calling, you are training the model to improve:

```text
Intent understanding
Function selection
Argument extraction
Structured formatting
```

I would experiment with:

```text
Rank 8
Rank 16
Rank 32
```

and select based on held-out function-calling evaluation metrics.

---

# 10. Format the dataset

```python
def format_example(example):

    text = tokenizer.apply_chat_template(
        example["messages"],
        tools=FUNCTIONS,
        tokenize=False
    )

    return {
        "text": text
    }


dataset = dataset.map(
    format_example
)
```

Inspect the output:

```python
print(
    dataset["train"][0]["text"]
)
```

Always inspect a few formatted examples before training.

You should verify:

```text
✓ User message is correct
✓ Function definitions are present
✓ Function-call format is correct
✓ Function arguments are valid
✓ EOS tokens are correct
```

---

# 11. Train with SFTTrainer

```python
from trl import SFTTrainer
from transformers import TrainingArguments
```

Training configuration:

```python
training_args = TrainingArguments(
    output_dir="function-calling-model",

    num_train_epochs=3,

    per_device_train_batch_size=2,

    gradient_accumulation_steps=8,

    learning_rate=2e-4,

    logging_steps=10,

    eval_strategy="steps",

    eval_steps=100,

    save_steps=100,

    save_total_limit=2,

    bf16=True,

    optim="paged_adamw_8bit",

    report_to="none"
)
```

Create the trainer:

```python
trainer = SFTTrainer(
    model=model,

    train_dataset=dataset["train"],

    eval_dataset=dataset["validation"],

    peft_config=lora_config,

    args=training_args,

    dataset_text_field="text",

    tokenizer=tokenizer,

    max_seq_length=2048
)
```

Train:

```python
trainer.train()
```

Save the LoRA adapter:

```python
trainer.save_model(
    "function-calling-lora"
)

tokenizer.save_pretrained(
    "function-calling-lora"
)
```

---

# 12. Train primarily on the assistant output

For instruction/function calling, conceptually:

```text
System prompt       → Context
Function schemas    → Context
User request        → Context

Assistant output    → Prediction target
```

You ideally want:

```text
System tokens         label = -100
Function tokens       label = -100
User tokens           label = -100
Assistant tokens      actual labels
```

Example:

```text
INPUT:

<user>
Send ₹5000 to Alice
</user>

<assistant>
<function_call>
{
  "name": "transfer_money",
  "arguments": {
    "recipient": "Alice",
    "amount": 5000,
    "currency": "INR"
  }
}
</function_call>
```

The model should primarily learn to predict:

```text
<function_call>
{
  ...
}
</function_call>
```

The exact masking implementation depends on the model's chat template and the version of TRL you use.

---

# 13. Run inference

Load the base model and LoRA adapter:

```python
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

model = PeftModel.from_pretrained(
    base_model,
    "function-calling-lora"
)

model.eval()
```

Create the request:

```python
messages = [
    {
        "role": "user",
        "content": "Send ₹5000 to Alice."
    }
]
```

Create the prompt:

```python
prompt = tokenizer.apply_chat_template(
    messages,
    tools=FUNCTIONS,
    tokenize=False,
    add_generation_prompt=True
)
```

Generate:

```python
inputs = tokenizer(
    prompt,
    return_tensors="pt"
).to(model.device)


with torch.inference_mode():

    output = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=False
    )
```

Decode only newly generated tokens:

```python
input_length = inputs["input_ids"].shape[1]

generated_tokens = output[
    0,
    input_length:
]

response = tokenizer.decode(
    generated_tokens,
    skip_special_tokens=False
)

print(response)
```

Expected conceptually:

```json
{
  "name": "transfer_money",
  "arguments": {
    "recipient": "Alice",
    "amount": 5000,
    "currency": "INR"
  }
}
```

---

# 14. Parse the function call

Define a Pydantic model:

```python
from typing import Any
from pydantic import BaseModel


class FunctionCall(BaseModel):
    name: str
    arguments: dict[str, Any]
```

Parse:

```python
import json


def parse_function_call(
    response: str
) -> FunctionCall:

    data = json.loads(response)

    return FunctionCall.model_validate(
        data
    )
```

Usage:

```python
function_call = parse_function_call(
    response
)

print(function_call.name)
print(function_call.arguments)
```

---

# 15. Validate each function's arguments

This is essential.

```python
from pydantic import BaseModel, Field
from decimal import Decimal


class BalanceArguments(BaseModel):
    account_id: str


class TransferArguments(BaseModel):
    recipient: str
    amount: Decimal = Field(gt=0)
    currency: str
```

Function registry:

```python
FUNCTION_REGISTRY = {
    "get_account_balance": {
        "function": get_account_balance,
        "schema": BalanceArguments
    },

    "transfer_money": {
        "function": transfer_money,
        "schema": TransferArguments
    }
}
```

Validation:

```python
def validate_function_call(
    function_call: FunctionCall
):

    config = FUNCTION_REGISTRY.get(
        function_call.name
    )

    if config is None:
        raise ValueError(
            f"Function not allowed: "
            f"{function_call.name}"
        )

    argument_schema = config["schema"]

    validated_arguments = (
        argument_schema.model_validate(
            function_call.arguments
        )
    )

    return config, validated_arguments
```

---

# 16. Safely execute the function

Never use:

```python
# ❌ Do not do this

function = globals()[
    function_call.name
]

function(
    **function_call.arguments
)
```

Instead:

```python
def execute_function(
    function_call: FunctionCall
):

    config, arguments = (
        validate_function_call(
            function_call
        )
    )

    function = config["function"]

    return function(
        **arguments.model_dump()
    )
```

Example:

```python
result = execute_function(
    function_call
)

print(result)
```

---

# 17. Add authorization

For functions that modify data, validation is not enough.

For example:

```text
transfer_money
delete_user
cancel_order
```

require authorization.

```python
READ_ONLY_FUNCTIONS = {
    "get_account_balance"
}

SENSITIVE_FUNCTIONS = {
    "transfer_money"
}
```

Example:

```python
def authorize_function(
    user,
    function_name: str
):

    if function_name in READ_ONLY_FUNCTIONS:
        return True

    if function_name in SENSITIVE_FUNCTIONS:

        return (
            "money.transfer"
            in user.permissions
        )

    return False
```

A production flow:

```text
LLM requests function
         │
         ▼
Is function allowlisted?
         │
         ▼
Are arguments valid?
         │
         ▼
Is user authorized?
         │
         ▼
Is confirmation required?
         │
         ▼
Execute
```

For financial or destructive operations, I would usually add a confirmation/human approval step rather than executing immediately.

---

# 18. Complete production function-calling loop

Here is a simplified implementation:

```python
import json


def run_function_calling_agent(
    user,
    user_message: str
):

    # 1. Ask model
    model_response = generate(
        user_message
    )

    # 2. Normal answer
    if not is_function_call(
        model_response):
        return model_response

    # 3. Parse
    function_call = (
        parse_function_call(
            model_response
        )
    )

    # 4. Allowlist and schema validation
    config, validated_arguments = (
        validate_function_call(
            function_call
        )
    )

    # 5. Authorization
    if not authorize_function(
        user,
        function_call.name
    ):
        raise PermissionError(
            "Not authorized"
        )

    # 6. Execute
    result = config["function"](
        **validated_arguments.model_dump()
    )

    # 7. Send result back to model
    messages = [
        {
            "role": "user",
            "content": user_message
        },
        {
            "role": "assistant",
            "content": model_response
        },
        {
            "role": "function",
            "name": function_call.name,
            "content": json.dumps(result)
        }
    ]

    # 8. Final answer
    return generate_from_messages(
        messages
    )
```

---

# 19. Support multiple function calls

A user might say:

```text
Check my account balance and tell me the status of order ORD-123.
```

The model could produce:

```json
[
  {
    "name": "get_account_balance",
    "arguments": {
      "account_id": "ACC-123"
    }
  },
  {
    "name": "get_order",
    "arguments": {
      "order_id": "ORD-123"
    }
  }
]
```

Execute safely:

```python
def execute_multiple_calls(
    function_calls
):

    results = []

    for function_call in function_calls:

        result = execute_function(
            function_call
        )

        results.append({
            "function":
                function_call.name,

            "result":
                result
        })

    return results
```

For independent read-only calls, you could execute concurrently with `asyncio.gather()`. For dependent or side-effecting calls, execute in a controlled order.

---

# 20. Training dataset should include edge cases

A production dataset should include:

### Correct function

```text
"Get my account balance"
        ↓
get_account_balance
```

### No function

```text
"What is machine learning?"
        ↓
Normal answer
```

### Missing argument

```text
"Transfer money to Alice"
        ↓
Ask for amount
```

Do not hallucinate:

```text
amount = 5000
```

unless it was provided.

### Ambiguous request

```text
"Send money to John"
        ↓
Ask for clarification
```

### Invalid argument

```text
"Transfer -5000 INR"
        ↓
Reject / clarify
```

### Multiple functions

```text
"Check my balance and order"
        ↓
Multiple calls
```

### Function result errors

```text
Function:
{
  "error": "Account not found"
}

        ↓

Assistant:
"I couldn't find that account."
```

---

# 21. How would you evaluate the fine-tuned model?

Important metrics:

```text
1. Function selection accuracy
2. Exact function-call match
3. Argument extraction accuracy
4. Argument schema validity
5. No-call accuracy
6. Multi-function accuracy
7. End-to-end task success
```

## Function selection accuracy

```python
def function_selection_accuracy(
    predictions,
    references
):

    correct = sum(
        prediction["name"] == reference["name"]
        for prediction, reference
        in zip(predictions, references)
    )

    return correct / len(predictions)
```

---

## Argument accuracy

```python
def argument_accuracy(
    predicted,
    expected
):

    correct = 0
    total = len(expected)

    for key, value in expected.items():

        if predicted.get(key) == value:
            correct += 1

    return correct / total
```

Example:

```text
Expected:
recipient = Alice
amount = 5000
currency = INR

Predicted:
recipient = Alice
amount = 6000
currency = INR
```

Result:

```text
2 / 3 = 66.7%
```

---

## Schema validity

```python
def schema_validity_rate(
    calls
):

    valid = 0

    for call in calls:

        try:
            validate_function_call(call)
            valid += 1

        except Exception:
            pass

    return valid / len(calls)
```

---

# 22. Training architecture

```text
                   Training Data
                        │
                        ▼
                Function Definitions
                        │
                        ▼
               User Intent Examples
                        │
              ┌─────────┴──────────┐
              │                    │
              ▼                    ▼
        Function Call         Normal Answer
              │
              ▼
        Correct Arguments
              │
              ▼
       Chat/Function Template
              │
              ▼
          SFT + LoRA
              │
              ▼
       Fine-tuned Adapter
```

---

# 23. Production architecture

```text
                   User
                     │
                     ▼
                  FastAPI
                     │
                     ▼
                    LLM
                     │
              Function call?
              /             \
            No               Yes
            │                 │
            ▼                 ▼
        Response          Parse
                              │
                              ▼
                         Allowlist
                              │
                              ▼
                        Validate args
                              │
                              ▼
                         Authorization
                              │
                              ▼
                         Confirmation
                       (if required)
                              │
                              ▼
                        Execute function
                              │
                              ▼
                        Function result
                              │
                              ▼
                             LLM
                              │
                              ▼
                         Final response
```

---

# 24. When would I fine-tune for function calling?

I would **not automatically fine-tune**.

First I would test:

```text
Base model
   +
Function schemas
   +
Good prompting
   +
Constrained output
```

Fine-tune when:

```text
✓ Domain-specific functions
✓ Many functions
✓ Complex argument extraction
✓ Repeated tool-selection mistakes
✓ Need a smaller/cheaper model
✓ High request volume
✓ Specialized terminology
```

For example:

```text
General model:
Function selection = 85%

Fine-tuned model:
Function selection = 96%
```

These numbers are illustrative; the real decision should come from your evaluation dataset.

---

# Interview-quality answer

> **I would fine-tune a model for function calling by creating supervised examples that map natural-language requests to structured function names and arguments. The dataset would include successful calls, no-call cases, missing arguments, ambiguous requests, multiple calls, tool errors, and complete conversations containing function results and final responses.**
>
> **I would use the base model's native chat/function-calling template whenever possible, rather than inventing a custom format, and fine-tune using SFT with LoRA or QLoRA. I would evaluate function-selection accuracy, argument extraction accuracy, schema validity, no-call accuracy, and end-to-end task success.**
>
> **In production, the LLM would only propose a function call. A deterministic execution layer would enforce an allowlist, Pydantic or JSON Schema validation, authentication, RBAC, tenant isolation, rate limits, timeouts, and confirmation for sensitive actions. The application executes the function and passes the result back to the model to generate the final response.**

## One-line answer to remember

```text
Train the model to decide WHAT function to call and WITH WHICH arguments;
let deterministic, authorized application code decide whether and how to execute it.
```
