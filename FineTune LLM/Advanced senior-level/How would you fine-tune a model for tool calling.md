# How would you fine-tune a model for tool calling?

Fine-tuning for **tool calling** means training an LLM to decide:

1. **Does this request require a tool?**
2. **Which tool should I use?**
3. **What arguments should I pass?**
4. **When should I answer directly instead?**
5. **How do I use the tool result to produce the final answer?**

For example:

```text
User: What's the weather in Bangalore?
```

The model should produce something like:

```json
{
  "name": "get_weather",
  "arguments": {
    "city": "Bangalore"
  }
}
```

Your application then executes:

```python
get_weather(city="Bangalore")
```

and sends the result back to the model.

The full flow is:

```text
User
  │
  ▼
LLM
  │
  ├── Direct answer
  │
  └── Tool call
          │
          ▼
       Tool Executor
          │
          ▼
       Tool Result
          │
          ▼
          LLM
          │
          ▼
     Final Answer
```

---

# 1. Define tools

Let's create three tools:

```text
1. get_weather
2. search_products
3. get_order
```

In Python:

```python
def get_weather(
    city: str
):
    """
    Get the current weather for a city.
    """
    return {
        "city": city,
        "temperature": 28,
        "condition": "Sunny"
    }


def get_order(
    order_id: str
):
    """
    Get order details.
    """
    return {
        "order_id": order_id,
        "status": "shipped"
    }


def search_products(
    query: str
):
    """
    Search products.
    """
    return {
        "products": [
            {
                "name": "Laptop",
                "price": 50000
            }
        ]
    }
```

But the model needs to understand the **tool schemas**.

---

# 2. Define tool schemas

A common structure is JSON Schema.

```python
TOOLS = [
    {
        "name": "get_weather",
        "description":
            "Get current weather for a city",

        "parameters": {
            "type": "object",

            "properties": {

                "city": {
                    "type": "string",
                    "description":
                        "City name"
                }
            },

            "required": [
                "city"
            ]
        }
    },

    {
        "name": "get_order",

        "description":
            "Get details for an order",

        "parameters": {

            "type": "object",

            "properties": {

                "order_id": {
                    "type": "string"
                }
            },

            "required": [
                "order_id"
            ]
        }
    }
]
```

Conceptually:

```text
Tool Name
    │
    ▼
Description
    │
    ▼
Input Schema
    │
    ▼
Required Arguments
```

The model needs to learn:

```text
User intent
    ↓
Correct tool
    ↓
Correct arguments
```

---

# 3. Training data for tool calling

Your fine-tuning dataset should contain examples of conversations where the assistant makes tool calls.

A conceptual example:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant."
    },
    {
      "role": "user",
      "content": "What is the weather in Bangalore?"
    },
    {
      "role": "assistant",
      "tool_calls": [
        {
          "name": "get_weather",
          "arguments": {
            "city": "Bangalore"
          }
        }
      ]
    }
  ]
}
```

Another:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Where is order ORD-123?"
    },
    {
      "role": "assistant",
      "tool_calls": [
        {
          "name": "get_order",
          "arguments": {
            "order_id": "ORD-123"
          }
        }
      ]
    }
  ]
}
```

You should also include **no-tool examples**:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "What is 2 + 2?"
    },
    {
      "role": "assistant",
      "content": "4"
    }
  ]
}
```

This is extremely important.

Otherwise, the model may learn:

```text
Every request
      ↓
Always call a tool
```

which is bad behavior.

---

# 4. Training data should include tool results

A production-quality example should teach the model the complete cycle:

```text
User
 ↓
Tool call
 ↓
Tool result
 ↓
Final answer
```

Example:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "What is the weather in Bangalore?"
    },
    {
      "role": "assistant",
      "tool_calls": [
        {
          "id": "call_1",
          "name": "get_weather",
          "arguments": {
            "city": "Bangalore"
          }
        }
      ]
    },
    {
      "role": "tool",
      "tool_call_id": "call_1",
      "content": "{\"city\":\"Bangalore\",\"temperature\":28,\"condition\":\"Sunny\"}"
    },
    {
      "role": "assistant",
      "content": "The current weather in Bangalore is sunny and 28°C."
    }
  ]
}
```

This teaches two behaviors:

### Phase 1

```text
Question
   ↓
Call tool
```

### Phase 2

```text
Tool result
   ↓
Understand result
   ↓
Generate useful answer
```

---

# 5. Tool calling dataset structure

For example:

```text
data/
│
├── train.jsonl
└── validation.jsonl
```

`train.jsonl`:

```json
{"messages":[{"role":"user","content":"What is the weather in Bangalore?"},{"role":"assistant","tool_calls":[{"id":"call_1","name":"get_weather","arguments":{"city":"Bangalore"}}]}]}
{"messages":[{"role":"user","content":"Where is order ORD-123?"},{"role":"assistant","tool_calls":[{"id":"call_2","name":"get_order","arguments":{"order_id":"ORD-123"}}]}]}
{"messages":[{"role":"user","content":"What is 2 + 2?"},{"role":"assistant","content":"4"}]}
```

However, **the exact dataset format depends on the base model**.

For example, different models may represent tool calls using different special tokens or templates.

So the correct rule is:

> **Use the official tool-calling/chat template expected by the base model whenever possible.**

---

# 6. Use the model's chat template

Load a tokenizer:

```python
from transformers import AutoTokenizer


MODEL_NAME = (
    "meta-llama/Llama-3.1-8B-Instruct"
)


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)
```

Suppose we have:

```python
example = {
    "messages": [
        {
            "role": "user",
            "content":
                "What is the weather in Bangalore?"
        },
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_1",
                    "name": "get_weather",
                    "arguments": {
                        "city": "Bangalore"
                    }
                }
            ]
        }
    ]
}
```

Conceptually:

```python
formatted_text = (
    tokenizer.apply_chat_template(
        example["messages"],
        tools=TOOLS,
        tokenize=False
    )
)
```

The tokenizer converts the conversation into the **exact token format the base model expects**.

This is better than manually inventing:

```text
TOOL_CALL:
{"name": "get_weather"}
```

because models may expect special formatting.

---

# 7. A simpler custom tool-calling format

If you are fine-tuning a model without an existing tool-calling template, you can define a controlled format.

For example:

```text
<tools>
[
  {
    "name": "get_weather",
    "parameters": {
      "city": "string"
    }
  }
]
</tools>

User:
What is the weather in Bangalore?

Assistant:
<tool_call>
{
  "name": "get_weather",
  "arguments": {
    "city": "Bangalore"
  }
}
</tool_call>
```

You can generate training examples:

```python
import json


def format_tool_call_example(
    user_message: str,
    tool_name: str,
    arguments: dict
):

    tool_call = {
        "name": tool_name,
        "arguments": arguments
    }

    return f"""
<user>
{user_message}
</user>

<assistant>
<tool_call>
{json.dumps(tool_call)}
</tool_call>
</assistant>
"""
```

Example:

```python
text = format_tool_call_example(
    user_message=(
        "What is the weather "
        "in Bangalore?"
    ),
    tool_name="get_weather",
    arguments={
        "city": "Bangalore"
    }
)
```

---

# 8. Fine-tune using LoRA

Tool calling is often a good candidate for LoRA because you are teaching:

```text
Base model capabilities
        +
Tool selection behavior
        +
Tool output format
        +
Argument extraction
```

Load the model:

```python
import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
```

Configure 4-bit loading:

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
model = (
    AutoModelForCausalLM.from_pretrained(

        MODEL_NAME,

        quantization_config=bnb_config,

        device_map="auto"
    )
)
```

Prepare:

```python
from peft import (
    prepare_model_for_kbit_training
)


model = (
    prepare_model_for_kbit_training(
        model
    )
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

For tool calling, I would experiment with:

```text
r = 8
r = 16
r = 32
```

because tool calling requires both:

```text
Semantic reasoning
+
Strict output formatting
```

---

# 10. Load the dataset

```python
from datasets import load_dataset


dataset = load_dataset(
    "json",
    data_files={
        "train": "data/train.jsonl",
        "validation":
            "data/validation.jsonl"
    }
)
```

Format it:

```python
def format_example(example):

    text = (
        tokenizer.apply_chat_template(

            example["messages"],

            tools=TOOLS,

            tokenize=False
        )
    )

    return {
        "text": text
    }


dataset = dataset.map(
    format_example
)
```

Check:

```python
print(
    dataset["train"][0]["text"]
)
```

This is an important debugging step.

---

# 11. Train with SFTTrainer

```python
from trl import SFTTrainer

from transformers import (
    TrainingArguments
)
```

Training arguments:

```python
training_args = TrainingArguments(

    output_dir="tool-calling-model",

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

Save:

```python
trainer.save_model(
    "tool-calling-lora"
)
```

---

# 12. The most important training concept: completion loss

You generally want the model to learn:

```text
User request
       ↓
Tool call
```

rather than training equally on every token.

Conceptually:

```text
Prompt:

System
Tools
User Question

        ↓

Assistant:

Tool Call
```

The labels should ideally look like:

```text
System tokens        -100
Tool definitions     -100
User tokens          -100
Assistant tokens     TARGET
```

For a tool-calling example:

```text
INPUT:

User:
What is the weather in Bangalore?

ASSISTANT:
<tool_call>
{"name":"get_weather","arguments":{"city":"Bangalore"}}
</tool_call>
```

Labels:

```text
USER TOKENS
-100
-100
-100

TOOL CALL TOKENS
<tool_call>      TARGET
{                TARGET
"name"           TARGET
...              TARGET
</tool_call>     TARGET
```

This makes training more focused.

---

# 13. Inference

Load the fine-tuned adapter:

```python
from peft import PeftModel


base_model = (
    AutoModelForCausalLM.from_pretrained(

        MODEL_NAME,

        torch_dtype=torch.bfloat16,

        device_map="auto"
    )
)


model = PeftModel.from_pretrained(

    base_model,

    "tool-calling-lora"
)


model.eval()
```

Create a user request:

```python
messages = [

    {
        "role": "user",

        "content":
            "What is the weather "
            "in Bangalore?"
    }
]
```

Format:

```python
prompt = (
    tokenizer.apply_chat_template(

        messages,

        tools=TOOLS,

        tokenize=False,

        add_generation_prompt=True
    )
)
```

Tokenize:

```python
inputs = tokenizer(

    prompt,

    return_tensors="pt"

).to(model.device)
```

Generate:

```python
with torch.inference_mode():

    output = model.generate(

        **inputs,

        max_new_tokens=256,

        do_sample=False
    )
```

Decode:

```python
input_length = (
    inputs["input_ids"]
    .shape[1]
)


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

Expected:

```json
{
    "name": "get_weather",
    "arguments": {
        "city": "Bangalore"
    }
}
```

---

# 14. Parse and validate the tool call

Never directly execute model-generated code.

Use a schema.

```python
from pydantic import (
    BaseModel
)


class ToolCall(BaseModel):

    name: str

    arguments: dict
```

Parse:

```python
import json


def parse_tool_call(
    response: str
):

    data = json.loads(
        response
    )

    return ToolCall.model_validate(
        data
    )
```

---

# 15. Validate against allowed tools

This is critical for security.

```python
ALLOWED_TOOLS = {

    "get_weather": get_weather,

    "get_order": get_order,

    "search_products":
        search_products
}
```

Never do:

```python
# ❌ Dangerous

function = globals()[
    tool_call.name
]

result = function(
    **tool_call.arguments
)
```

Instead:

```python
def execute_tool(
    tool_call: ToolCall
):

    tool = ALLOWED_TOOLS.get(
        tool_call.name
    )

    if tool is None:

        raise ValueError(
            f"Unknown tool: "
            f"{tool_call.name}"
        )

    return tool(
        **tool_call.arguments
    )
```

The model can only call explicitly approved tools.

---

# 16. Validate tool arguments

Use a separate schema for each tool.

```python
class WeatherArguments(
    BaseModel
):

    city: str
```

Another:

```python
class OrderArguments(
    BaseModel
):

    order_id: str
```

Registry:

```python
TOOL_ARGUMENT_MODELS = {

    "get_weather":
        WeatherArguments,

    "get_order":
        OrderArguments
}
```

Validation:

```python
def validate_tool_arguments(
    tool_call: ToolCall
):

    schema = (
        TOOL_ARGUMENT_MODELS.get(
            tool_call.name
        )
    )

    if schema is None:

        raise ValueError(
            "Unknown tool"
        )

    return schema.model_validate(
        tool_call.arguments
    )
```

Example:

```python
tool_call = ToolCall(

    name="get_weather",

    arguments={
        "city": "Bangalore"
    }
)


arguments = (
    validate_tool_arguments(
        tool_call
    )
)
```

---

# 17. Complete tool-calling execution loop

This is the core architecture.

```python
def run_agent(
    user_message: str
):

    # 1. Ask model

    response = generate(
        user_message
    )


    # 2. Determine whether model
    #    wants to call a tool

    if not is_tool_call(
        response
    ):

        return response


    # 3. Parse tool call

    tool_call = parse_tool_call(
        response
    )


    # 4. Validate

    validated_arguments = (
        validate_tool_arguments(
            tool_call
        )
    )


    # 5. Execute

    result = execute_tool(
        ToolCall(
            name=tool_call.name,
            arguments=
                validated_arguments
                .model_dump()
        )
    )


    # 6. Give result to LLM

    final_prompt = f"""
User request:

{user_message}

Tool result:

{json.dumps(result)}

Generate the final answer.
"""


    return generate(
        final_prompt
    )
```

Flow:

```text
User
 │
 ▼
LLM
 │
 ▼
Tool Call?
 │
 ├──── No ────► Return Answer
 │
 Yes
 │
 ▼
Validate Tool
 │
 ▼
Validate Arguments
 │
 ▼
Execute Tool
 │
 ▼
Tool Result
 │
 ▼
LLM
 │
 ▼
Final Answer
```

---

# 18. Training examples you should include

A good dataset needs variety.

## A. Correct tool selection

```text
Weather question
       ↓
get_weather
```

```json
{
    "name": "get_weather",
    "arguments": {
        "city": "Bangalore"
    }
}
```

---

## B. No tool needed

```text
User:
What is 2 + 2?

Assistant:
4
```

---

## C. Multiple tools

```text
User:
Check my order ORD-123
and tell me the weather in Delhi.
```

Potential output:

```json
[
    {
        "name": "get_order",
        "arguments": {
            "order_id": "ORD-123"
        }
    },
    {
        "name": "get_weather",
        "arguments": {
            "city": "Delhi"
        }
    }
]
```

---

## D. Missing information

```text
User:
What's my order status?
```

The model should not guess.

It should respond:

```text
Please provide your order ID.
```

Training example:

```json
{
    "messages": [
        {
            "role": "user",
            "content":
                "What is my order status?"
        },
        {
            "role": "assistant",
            "content":
                "Please provide your order ID."
        }
    ]
}
```

---

## E. Invalid tool arguments

User:

```text
Get weather
```

The model should ask:

```text
Which city would you like the weather for?
```

It should **not** generate:

```json
{
    "name": "get_weather",
    "arguments": {
        "city": "Bangalore"
    }
}
```

unless Bangalore is actually provided by context.

---

# 19. Tool selection evaluation

You should not evaluate only training loss.

Measure:

```text
Tool Selection Accuracy
Argument Accuracy
Exact Schema Validity
No-Tool Accuracy
End-to-End Task Success
```

Example:

```python
def tool_selection_accuracy(
    predictions,
    references
):

    correct = 0

    for prediction, reference in zip(
        predictions,
        references
    ):

        if (
            prediction["name"]
            ==
            reference["name"]
        ):

            correct += 1

    return correct / len(
        predictions
    )
```

---

# 20. Argument accuracy

Example:

Expected:

```json
{
    "name": "get_weather",
    "arguments": {
        "city": "Bangalore"
    }
}
```

Predicted:

```json
{
    "name": "get_weather",
    "arguments": {
        "city": "Delhi"
    }
}
```

Tool selection is correct:

```text
100%
```

But task correctness is wrong.

Evaluate arguments:

```python
def argument_accuracy(
    prediction,
    reference
):

    expected = (
        reference["arguments"]
    )

    predicted = (
        prediction["arguments"]
    )

    correct = 0

    total = len(
        expected
    )

    for key, value in expected.items():

        if predicted.get(key) == value:

            correct += 1

    return correct / total
```

---

# 21. Production evaluation

I would create a test set:

```text
evaluation/
│
├── tool_selection.jsonl
├── argument_extraction.jsonl
├── no_tool_cases.jsonl
├── multi_tool_cases.jsonl
└── adversarial_cases.jsonl
```

Example metrics:

```text
Tool selection accuracy      96%
Argument accuracy            93%
Schema validity              99%
No-tool accuracy             97%
End-to-end task success      91%
```

---

# 22. Production architecture

```text
                    User
                     │
                     ▼
                  FastAPI
                     │
                     ▼
              Authentication
                     │
                     ▼
              Tool Authorization
                     │
                     ▼
                    LLM
                     │
          ┌──────────┴──────────┐
          │                     │
      Text Answer           Tool Call
          │                     │
          ▼                     ▼
       Response          Schema Validation
                                │
                                ▼
                          Authorization
                                │
                                ▼
                          Argument Validation
                                │
                                ▼
                           Tool Execution
                                │
                                ▼
                            Tool Result
                                │
                                ▼
                                LLM
                                │
                                ▼
                           Final Answer
```

---

# 23. Important security rules

The model should **never have unrestricted tool access**.

Bad:

```text
LLM
 ↓
Execute anything
```

Good:

```text
LLM
 ↓
Requested tool
 ↓
Allowed?
 ↓
Arguments valid?
 ↓
User authorized?
 ↓
Execute
```

For sensitive tools, additionally enforce:

```text
RBAC
Tenant isolation
Rate limits
Audit logs
Timeouts
Retries
Idempotency
Human approval
```

For example:

```python
def authorize_tool(
    user,
    tool_name
):

    permissions = user.permissions

    required_permission = {
        "get_weather":
            "weather.read",

        "get_order":
            "orders.read"
    }

    permission = (
        required_permission.get(
            tool_name
        )
    )

    return permission in permissions
```

---

# 24. Fine-tuning vs prompt engineering

You don't always need fine-tuning.

Use prompting when:

```text
Few tools
Stable schemas
Strong model
Low request volume
```

Fine-tune when:

```text
Many domain-specific tools
Complex argument extraction
Consistent tool selection needed
Large request volume
Specialized terminology
Weak/open-source base model
```

Example:

```text
Prompt engineering
      ↓
Tool accuracy = 85%

Fine-tuning
      ↓
Tool accuracy = 96%
```

The exact improvement must be measured on your evaluation set rather than assumed.

---

# Strong interview answer

> **To fine-tune a model for tool calling, I would first define every tool using a clear schema containing the tool name, description, parameters, types, and required fields. Then I would create supervised training data containing user requests and the correct tool calls, including the full lifecycle of user request, assistant tool call, tool result, and final assistant response. I would also include no-tool cases, missing-argument cases, ambiguous requests, invalid requests, and multi-tool workflows.**
>
> **I would format the data using the base model's official chat and tool-use template and fine-tune using SFT with LoRA or QLoRA. During training, I would ideally compute loss primarily on assistant-generated tool-call and response tokens.**
>
> **For evaluation, I would measure tool selection accuracy, argument extraction accuracy, tool-call schema validity, no-tool accuracy, and end-to-end task success.**
>
> **In production, I would never allow the LLM to execute arbitrary functions. The tool call would go through schema validation, an allowlist, argument validation, authorization, tenant isolation, rate limiting, and auditing before execution. The tool result would then be sent back to the LLM for the final answer.**

## One-line answer

```text
Fine-tune the model to choose the correct tool and arguments,
but let deterministic application code validate and execute the tool safely.
```
