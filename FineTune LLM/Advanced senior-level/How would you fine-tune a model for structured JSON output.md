# How would you fine-tune a model for structured JSON output?

The goal is to train an LLM so its responses consistently follow a JSON schema.

For example, given:

```text
Extract information:

John is 32 years old and works as a software engineer.
```

We want:

```json
{
  "name": "John",
  "age": 32,
  "occupation": "software engineer"
}
```

A good production solution has **three layers**:

```text
1. Fine-tuning
       +
2. Schema validation
       +
3. Constrained output / retries
```

> Fine-tuning improves the probability of correct JSON, but **you should not rely on fine-tuning alone to guarantee valid JSON**.

---

# 1. Define the JSON schema

Use Pydantic for the application contract.

```python
from pydantic import BaseModel, Field


class Person(BaseModel):
    name: str
    age: int = Field(
        ge=0,
        le=150
    )
    occupation: str
```

This defines the expected output:

```json
{
  "name": "John",
  "age": 32,
  "occupation": "software engineer"
}
```

Validation:

```python
person = Person.model_validate_json(
    """
    {
        "name": "John",
        "age": 32,
        "occupation": "software engineer"
    }
    """
)

print(person)
```

---

# 2. Prepare the training dataset

For structured output fine-tuning, your examples must consistently demonstrate:

```text
Instruction
        ↓
Input
        ↓
Valid JSON Output
```

Example dataset:

```json
{
  "instruction": "Extract person information from the text. Return only valid JSON.",
  "input": "John is 32 years old and works as a software engineer.",
  "output": {
    "name": "John",
    "age": 32,
    "occupation": "software engineer"
  }
}
```

A JSONL dataset:

```text
data/
└── train.jsonl
```

Example:

```json
{"instruction":"Extract person information. Return only JSON.","input":"John is 32 years old and works as a software engineer.","output":{"name":"John","age":32,"occupation":"software engineer"}}
{"instruction":"Extract person information. Return only JSON.","input":"Sarah is 28 and works as a data scientist.","output":{"name":"Sarah","age":28,"occupation":"data scientist"}}
```

---

# 3. Why output consistency matters

This is bad training data:

```text
Example 1:

{
  "name": "John",
  "age": 32
}
```

Example 2:

```json
{
  "name": "Sarah",
  "age": "28"
}
```

````

Example 3:

```text
The extracted data is:

{
    "name": "Mike"
}
````

The model learns inconsistent behavior.

Instead:

```text
Always:

{
    "name": string,
    "age": integer,
    "occupation": string
}
```

Consistency is extremely important.

---

# 4. Generate training prompts

For a chat model, use the model's chat template.

```python
import json


def format_example(example, tokenizer):

    messages = [

        {
            "role": "system",
            "content": (
                "You extract structured information. "
                "Return only valid JSON."
            )
        },

        {
            "role": "user",
            "content": (
                example["instruction"]
                + "\n\n"
                + example["input"]
            )
        },

        {
            "role": "assistant",
            "content": json.dumps(
                example["output"],
                ensure_ascii=False
            )
        }
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False
    )
```

Example:

```python
example = {
    "instruction":
        "Extract person information.",

    "input":
        "John is 32 years old and works as a software engineer.",

    "output": {
        "name": "John",
        "age": 32,
        "occupation": "software engineer"
    }
}
```

Format:

```python
text = format_example(
    example,
    tokenizer
)
```

Conceptually:

```text
SYSTEM:
You extract structured information.
Return only valid JSON.

USER:
Extract person information.

John is 32 years old and works as a software engineer.

ASSISTANT:
{"name":"John","age":32,"occupation":"software engineer"}
```

---

# 5. Load the dataset

Using Hugging Face Datasets:

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

Convert examples:

```python
def format_dataset(example):

    return {
        "text": format_example(
            example,
            tokenizer
        )
    }


dataset = dataset.map(
    format_dataset
)
```

---

# 6. Load the base model with QLoRA

For a 7B/8B model on limited GPU memory:

```python
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
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

Load model:

```python
MODEL_NAME = (
    "meta-llama/Llama-3.1-8B-Instruct"
)


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


if tokenizer.pad_token is None:

    tokenizer.pad_token = (
        tokenizer.eos_token
    )


model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,

    quantization_config=bnb_config,

    device_map="auto"
)
```

Prepare for training:

```python
from peft import (
    prepare_model_for_kbit_training
)


model = prepare_model_for_kbit_training(
    model
)
```

---

# 7. Configure LoRA

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

For a structured-output task, I might start with:

```text
r = 16
alpha = 32
dropout = 0.05
```

Then compare:

```text
r = 8
r = 16
r = 32
```

using the same validation dataset.

---

# 8. Train using SFTTrainer

```python
from trl import SFTTrainer

from transformers import (
    TrainingArguments
)
```

Training configuration:

```python
training_args = TrainingArguments(

    output_dir="json-output-model",

    num_train_epochs=3,

    per_device_train_batch_size=2,

    gradient_accumulation_steps=8,

    learning_rate=2e-4,

    logging_steps=10,

    evaluation_strategy="steps",

    eval_steps=100,

    save_steps=100,

    save_total_limit=2,

    bf16=True,

    optim="paged_adamw_8bit",

    report_to="none"
)
```

Create trainer:

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
    "json-output-lora"
)

tokenizer.save_pretrained(
    "json-output-lora"
)
```

The result is a LoRA adapter:

```text
json-output-lora/
├── adapter_model.safetensors
├── adapter_config.json
└── tokenizer files
```

---

# 9. Important: train only on the assistant response when possible

For instruction tuning, ideally:

```text
User prompt → input context

Assistant response → calculate loss
```

You generally do not want to optimize the model to reproduce the prompt.

Conceptually:

```text
SYSTEM TOKENS       → label = -100
USER TOKENS         → label = -100
ASSISTANT TOKENS    → actual labels
```

Example:

```text
Input IDs:

[SYSTEM TOKENS]
[USER TOKENS]
[ASSISTANT TOKENS]

Labels:

[-100]
[-100]
[JSON TARGET TOKENS]
```

This teaches the model:

```text
Given this prompt
      ↓
Generate this JSON
```

Depending on the TRL version and dataset format, use an appropriate completion-only or assistant-token loss mechanism rather than assuming every token in the formatted conversation should contribute to the loss.

---

# 10. Run inference

Load the base model and adapter:

```python
from peft import PeftModel


base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,

    torch_dtype=torch.bfloat16,

    device_map="auto"
)


model = PeftModel.from_pretrained(
    base_model,
    "json-output-lora"
)

model.eval()
```

Create a prompt:

```python
messages = [

    {
        "role": "system",
        "content": (
            "You extract structured information. "
            "Return only valid JSON."
        )
    },

    {
        "role": "user",
        "content": (
            "Extract person information from:\n"
            "Alice is 30 years old and works as a designer."
        )
    }
]
```

Format:

```python
prompt = tokenizer.apply_chat_template(
    messages,
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

        do_sample=False,

        temperature=None,

        top_p=None
    )
```

Extract generated tokens:

```python
input_length = inputs[
    "input_ids"
].shape[1]


generated_tokens = output[
    0,
    input_length:
]


response = tokenizer.decode(
    generated_tokens,
    skip_special_tokens=True
)

print(response)
```

Expected:

```json
{
  "name": "Alice",
  "age": 30,
  "occupation": "designer"
}
```

---

# 11. Validate the JSON

Never trust raw model output directly.

```python
import json


def parse_json(response):

    try:

        return json.loads(
            response
        )

    except json.JSONDecodeError:

        return None
```

Usage:

```python
result = parse_json(
    response
)


if result is None:

    print(
        "Invalid JSON"
    )

else:

    print(
        "Valid JSON"
    )
```

---

# 12. Validate against the Pydantic schema

Valid JSON does not mean valid business data.

For example:

```json
{
    "name": "Alice",
    "age": "hello",
    "occupation": 123
}
```

This may parse as JSON but violate your schema.

Use:

```python
from pydantic import ValidationError


def validate_response(
    response: str
):

    try:

        person = (
            Person.model_validate_json(
                response
            )
        )

        return {
            "success": True,
            "data": person.model_dump(),
            "error": None
        }

    except ValidationError as error:

        return {
            "success": False,
            "data": None,
            "error": str(error)
        }
```

Usage:

```python
result = validate_response(
    response
)


print(result)
```

Now you have two levels:

```text
Level 1:
Valid JSON?

Level 2:
Valid schema?
```

---

# 13. Measure JSON validity rate

This is one of the most important evaluation metrics.

$$
JSON\ Validity\ Rate =
\frac{Valid\ JSON\ Outputs}
{Total\ Outputs}
$$

Code:

```python
def calculate_json_validity(
    responses
):

    valid = 0

    for response in responses:

        try:

            json.loads(response)

            valid += 1

        except json.JSONDecodeError:

            pass

    return valid / len(responses)
```

Example:

```python
responses = [

    '{"name": "Alice", "age": 30, "occupation": "Designer"}',

    '{"name": "Bob", "age": 25}',

    'Alice is 30 years old'
]


score = calculate_json_validity(
    responses
)


print(score)
```

Output:

```text
0.66
```

---

# 14. Measure schema validity rate

JSON validity is not enough.

```python
def calculate_schema_validity(
    responses
):

    valid = 0

    for response in responses:

        try:

            Person.model_validate_json(
                response
            )

            valid += 1

        except Exception:

            pass

    return valid / len(
        responses
    )
```

Example metrics:

```text
JSON validity:       98%
Schema validity:     94%
Correct extraction:  90%
```

This tells you where failures occur.

---

# 15. Measure field-level accuracy

Suppose:

```json
Expected:

{
    "name": "Alice",
    "age": 30,
    "occupation": "Designer"
}
```

Model output:

```json
{
    "name": "Alice",
    "age": 31,
    "occupation": "Designer"
}
```

The JSON is valid, but the answer is partially wrong.

Field accuracy:

```python
def field_accuracy(
    prediction,
    reference
):

    correct = 0
    total = 0

    for field, expected in reference.items():

        total += 1

        if prediction.get(field) == expected:

            correct += 1

    return correct / total
```

Usage:

```python
reference = {

    "name": "Alice",

    "age": 30,

    "occupation": "Designer"
}


prediction = {

    "name": "Alice",

    "age": 31,

    "occupation": "Designer"
}


print(
    field_accuracy(
        prediction,
        reference
    )
)
```

Output:

```text
0.66
```

---

# 16. Full structured-output evaluation

A production evaluation pipeline:

```python
def evaluate_structured_output(
    predictions,
    references
):

    json_valid_count = 0
    schema_valid_count = 0
    total_field_accuracy = 0

    for prediction, reference in zip(
        predictions,
        references
    ):

        try:

            data = json.loads(
                prediction
            )

            json_valid_count += 1

        except json.JSONDecodeError:

            continue


        try:

            validated = (
                Person.model_validate(
                    data
                )
            )

            schema_valid_count += 1

        except ValidationError:

            continue


        total_field_accuracy += (
            field_accuracy(
                validated.model_dump(),
                reference
            )
        )


    total = len(
        predictions
    )


    return {

        "json_validity_rate":
            json_valid_count / total,

        "schema_validity_rate":
            schema_valid_count / total,

        "field_accuracy":
            total_field_accuracy / total
    }
```

For a stricter implementation, you may report field accuracy separately over schema-valid examples and over all examples; otherwise invalid outputs implicitly count as zero.

---

# 17. Add constrained decoding in production

Fine-tuning improves behavior, but sometimes the model can still produce:

```text
Here is the JSON:

{
    "name": "Alice"
}
```

Or:

```text
{
    "name": "Alice",
}
```

For high-reliability applications, use constrained generation.

Conceptually:

```text
Model
  │
  ▼
Next Token Candidates
  │
  ▼
JSON Schema Constraint
  │
  ├── Valid token → Allow
  │
  └── Invalid token → Block
```

This provides a stronger guarantee than prompt engineering alone.

The exact implementation depends on the inference server. Common approaches include schema/grammar-constrained decoding or provider-native structured-output features.

---

# 18. Add a repair/retry layer

If validation fails:

```text
Model Response
     │
     ▼
JSON Parser
     │
     ├── Valid
     │      │
     │      ▼
     │    Return
     │
     └── Invalid
            │
            ▼
       Retry / Repair
            │
            ▼
       Validate Again
```

Example:

```python
MAX_RETRIES = 2


def generate_structured_response(
    prompt
):

    for attempt in range(
        MAX_RETRIES
    ):

        response = call_model(
            prompt
        )

        try:

            person = (
                Person.model_validate_json(
                    response
                )
            )

            return person

        except ValidationError:

            continue


    raise ValueError(
        "Model failed to generate valid JSON"
    )
```

---

# 19. Better retry strategy

Do not blindly send the same request repeatedly.

Provide the validation error:

```python
def generate_with_repair(
    prompt
):

    response = call_model(
        prompt
    )

    try:

        return Person.model_validate_json(
            response
        )

    except ValidationError as error:

        repair_prompt = f"""
The following response is invalid.

Response:
{response}

Validation error:
{error}

Return only corrected JSON.
"""

        repaired = call_model(
            repair_prompt
        )

        return Person.model_validate_json(
            repaired
        )
```

In sensitive applications, validate that repair does not silently invent values.

---

# 20. Production FastAPI example

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ValidationError


app = FastAPI()


class ExtractRequest(BaseModel):
    text: str


class Person(BaseModel):
    name: str
    age: int
    occupation: str


@app.post(
    "/extract",
    response_model=Person
)
async def extract_person(
    request: ExtractRequest
):

    prompt = f"""
Extract person information.

Text:
{request.text}

Return only JSON with:

name
age
occupation
"""

    response = call_model(
        prompt
    )

    try:

        person = Person.model_validate_json(
            response
        )

        return person

    except ValidationError:

        raise HTTPException(

            status_code=422,

            detail=(
                "Invalid structured output "
                "generated by model"
            )
        )
```

The client always receives the expected structure:

```json
{
    "name": "Alice",
    "age": 30,
    "occupation": "Designer"
}
```

---

# 21. Production architecture

```text
                     Training Data
                          │
                          ▼
                 Validate JSON Schema
                          │
                          ▼
                    SFT / QLoRA
                          │
                          ▼
                    LoRA Adapter
                          │
                          ▼
                      Inference
                          │
                          ▼
                  Generate Response
                          │
                          ▼
                  Constrained Decoding
                          │
                          ▼
                     JSON Parse
                          │
                  ┌───────┴────────┐
                  │                │
                Valid            Invalid
                  │                │
                  ▼                ▼
            Schema Validation    Retry/Repair
                  │
                  ▼
               API Response
```

---

# 22. Best practices for training data

For structured JSON output:

### Keep schema consistent

```text
Always:
name
age
occupation
```

Avoid:

```text
Example 1 → age
Example 2 → user_age
Example 3 → years
```

### Use correct data types

```json
{
    "age": 30
}
```

Not:

```json
{
    "age": "30"
}
```

unless the schema expects a string.

### Include difficult examples

Train on:

```text
Missing values
Null values
Multiple entities
Ambiguous text
Special characters
Long documents
Malformed inputs
```

Example:

```json
{
    "name": "Unknown",
    "age": null,
    "occupation": "Engineer"
}
```

Your Pydantic schema should then explicitly allow `None`:

```python
class Person(BaseModel):
    name: str | None
    age: int | None
    occupation: str | None
```

---

# 23. What I would do in a real production system

I would use:

```text
Training
────────

High-quality instruction dataset
        +
Consistent JSON schema
        +
Chat template
        +
SFT with LoRA/QLoRA
        +
Validation set


Inference
─────────

Structured prompt
        +
Low-temperature decoding
        +
Schema-constrained generation
        +
JSON parsing
        +
Pydantic validation
        +
Retry/repair


Monitoring
──────────

JSON validity rate
Schema validity rate
Field accuracy
Hallucination rate
Latency
Retry rate
```

---

# Strong interview answer

> **To fine-tune a model for structured JSON output, I would first define the output contract using a JSON Schema or Pydantic model. I would create a high-quality instruction dataset where every training example consistently maps an input to a valid JSON object with stable field names and correct data types. I would format the data using the base model's chat template and fine-tune using SFT, usually with LoRA or QLoRA for efficiency.**
>
> **During evaluation, I would not rely only on loss. I would measure JSON validity rate, schema validity rate, field-level accuracy, and task correctness. In production, I would add schema-constrained decoding when reliability is critical, then parse and validate every response with Pydantic. Invalid outputs would go through a bounded retry or repair flow.**
>
> **The important principle is that fine-tuning increases the probability of producing valid structured output, but the production system should enforce correctness through constrained decoding and schema validation.**

## One-line answer

```text
Fine-tune for the behavior,
constrain generation for reliability,
and validate every output before returning it.
```
