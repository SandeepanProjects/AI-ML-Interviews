# How would you fine-tune an LLM for a specific company's terminology?

This is a common real-world enterprise LLM use case.

Suppose a company uses terminology that has special internal meanings:

| Company term         | Actual meaning                                 |
| -------------------- | ---------------------------------------------- |
| **Active Customer**  | Customer with a purchase in the last 90 days   |
| **Churned Customer** | Customer with no purchase in the last 180 days |
| **Premium User**     | User with annual spend > ₹50,000               |
| **GMV**              | Total value of orders before refunds           |
| **Net Revenue**      | GMV − refunds − discounts                      |
| **Escalation**       | Ticket requiring Level 2 support               |

A general LLM may know the words, but not the **company-specific definitions**.

---

# 1. The key question: Fine-tuning or RAG?

Before fine-tuning, ask:

> Is this terminology stable, or does it change frequently?

### If terminology changes frequently

Use:

```text
Company glossary
       ↓
RAG
       ↓
LLM
```

Example:

```text
Active Customer = purchase in last 90 days

Later changes to:

Active Customer = purchase in last 60 days
```

RAG is better because you update the document.

---

### If terminology is stable and appears repeatedly

Fine-tuning can help the model internalize:

* terminology
* preferred wording
* abbreviations
* domain-specific concepts
* relationships between concepts
* company communication style

The best enterprise solution is often:

```text
Stable terminology
        ↓
Fine-tuning
        +
Current terminology
        ↓
RAG
        ↓
LLM
```

---

# 2. What does the training data look like?

The model should see terminology in realistic contexts.

Bad:

```text
Input:
What does AC mean?

Output:
Active Customer
```

This teaches only a glossary lookup.

Better:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are an internal company assistant."
    },
    {
      "role": "user",
      "content": "How many active customers did we have last quarter?"
    },
    {
      "role": "assistant",
      "content": "An Active Customer is defined as a customer who made at least one purchase during the previous 90 days."
    }
  ]
}
```

Now the model learns:

```text
Company term
+
Real business question
+
Correct interpretation
```

---

# 3. Create a terminology dataset

Let's define company terminology.

```python
company_glossary = {
    "Active Customer": (
        "A customer who made at least one "
        "purchase in the previous 90 days."
    ),

    "Churned Customer": (
        "A customer who has not made a "
        "purchase in the previous 180 days."
    ),

    "Premium User": (
        "A customer whose annual spending "
        "is greater than 50000 INR."
    ),

    "GMV": (
        "The total value of orders before "
        "refunds and discounts."
    ),

    "Net Revenue": (
        "GMV minus refunds and discounts."
    )
}
```

---

# 4. Don't train only on dictionary definitions

A poor dataset would look like:

```text
Active Customer → Customer who purchased in last 90 days
Churned Customer → No purchase in last 180 days
GMV → Total order value
```

Instead, create multiple types of examples.

---

## Type 1: Definition questions

```python
examples = [
    {
        "instruction":
            "What is an Active Customer?",

        "response":
            "An Active Customer is a customer "
            "who made at least one purchase "
            "in the previous 90 days."
    }
]
```

---

## Type 2: Interpretation

```python
{
    "instruction":
        "Did customer C123 qualify as active if "
        "their last purchase was 70 days ago?",

    "response":
        "Yes. Customer C123 qualifies as an "
        "Active Customer because they made a "
        "purchase within the previous 90 days."
}
```

---

## Type 3: Business reasoning

```python
{
    "instruction":
        "A customer made their last purchase "
        "100 days ago. Are they active?",

    "response":
        "No. The customer is not an Active "
        "Customer because their last purchase "
        "was more than 90 days ago."
}
```

---

## Type 4: SQL generation using terminology

```python
{
    "instruction":
        "Show all Active Customers.",

    "response":
        """
SELECT DISTINCT customer_id
FROM orders
WHERE created_at >=
CURRENT_DATE - INTERVAL '90 days';
"""
}
```

This is especially useful because the model learns:

```text
Company terminology
        ↓
Business meaning
        ↓
Correct technical action
```

---

# 5. Use conversational fine-tuning format

For an instruction-tuned model, use messages.

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
                    "You are an internal company "
                    "assistant. Use the company's "
                    "official terminology."
                )
            },

            {
                "role": "user",
                "content": (
                    "What is an Active Customer?"
                )
            },

            {
                "role": "assistant",
                "content": (
                    "An Active Customer is a "
                    "customer who made at least "
                    "one purchase during the "
                    "previous 90 days."
                )
            }
        ]
    },


    {
        "messages": [

            {
                "role": "system",
                "content": (
                    "You are an internal company "
                    "assistant."
                )
            },

            {
                "role": "user",
                "content": (
                    "Is a customer active if their "
                    "last purchase was 60 days ago?"
                )
            },

            {
                "role": "assistant",
                "content": (
                    "Yes. The customer is an "
                    "Active Customer because "
                    "their last purchase occurred "
                    "within the previous 90 days."
                )
            }
        ]
    },


    {
        "messages": [

            {
                "role": "system",
                "content": (
                    "You are an internal company "
                    "assistant."
                )
            },

            {
                "role": "user",
                "content": (
                    "What is the difference between "
                    "GMV and Net Revenue?"
                )
            },

            {
                "role": "assistant",
                "content": (
                    "GMV is the total value of "
                    "orders before refunds and "
                    "discounts. Net Revenue is "
                    "GMV after subtracting refunds "
                    "and discounts."
                )
            }
        ]
    }
]
```

Save:

```python
with open(
    "company_terminology.jsonl",
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

# 6. Automatically generate terminology training examples

For a large company glossary, you may have thousands of terms.

Suppose:

```python
glossary = {
    "Active Customer": {
        "definition": (
            "A customer who made at least "
            "one purchase in the last 90 days."
        ),

        "aliases": [
            "active user",
            "engaged customer"
        ]
    },

    "GMV": {
        "definition": (
            "Total value of orders before "
            "refunds and discounts."
        ),

        "aliases": [
            "gross merchandise value"
        ]
    }
}
```

Generate definition examples:

```python
def create_definition_examples(
    glossary: dict
):

    examples = []

    for term, data in glossary.items():

        example = {

            "messages": [

                {
                    "role": "system",

                    "content": (
                        "You are an internal "
                        "company assistant. "
                        "Use official terminology."
                    )
                },

                {
                    "role": "user",

                    "content":
                        f"What is {term}?"
                },

                {
                    "role": "assistant",

                    "content":
                        data["definition"]
                }
            ]
        }

        examples.append(
            example
        )

    return examples
```

Usage:

```python
training_examples = (
    create_definition_examples(
        glossary
    )
)
```

However, automatically generated examples should be reviewed. Do not blindly generate thousands of nearly identical synthetic examples.

---

# 7. Create paraphrases

Users will not always use the exact company term.

For example:

```text
Active Customer
```

Users may say:

```text
Who are our active users?
Who has recently purchased?
Which customers are currently engaged?
Show customers who are still active.
```

You need examples covering this variation.

```python
examples = [
    {
        "question":
            "Who counts as an active user?",

        "answer":
            "An Active Customer is someone "
            "who made at least one purchase "
            "in the previous 90 days."
    },

    {
        "question":
            "What does customer activity mean "
            "in our reports?",

        "answer":
            "In company reporting, an Active "
            "Customer is a customer with at "
            "least one purchase in the "
            "previous 90 days."
    }
]
```

This teaches semantic variation.

---

# 8. Add negative examples

This is extremely important.

Suppose:

```text
Active Customer = purchase in 90 days
```

The model should not confuse this with:

```text
Customer created in 90 days
```

Example:

```python
{
    "messages": [

        {
            "role": "user",

            "content": (
                "A customer created an account "
                "30 days ago but never purchased "
                "anything. Are they active?"
            )
        },

        {
            "role": "assistant",

            "content": (
                "No. Under the company's "
                "definition, an Active Customer "
                "must have made at least one "
                "purchase in the previous "
                "90 days. Account creation "
                "alone does not qualify."
            )
        }
    ]
}
```

Negative examples teach boundaries.

---

# 9. Validate the terminology dataset

Before fine-tuning:

```text
Raw Glossary
      ↓
Normalize
      ↓
Remove duplicates
      ↓
Check contradictions
      ↓
Expert review
      ↓
Training dataset
```

For example:

```python
def normalize_text(
    text: str
) -> str:

    return (
        text
        .lower()
        .strip()
    )
```

Detect duplicate definitions:

```python
def find_duplicates(
    glossary: dict
):

    seen = {}

    duplicates = []

    for term, definition in glossary.items():

        normalized = normalize_text(
            definition
        )

        if normalized in seen:

            duplicates.append(
                (
                    term,
                    seen[normalized]
                )
            )

        else:

            seen[
                normalized
            ] = term

    return duplicates
```

---

# 10. Detect contradictory terminology

Suppose two sources say:

```text
Active Customer = purchase in 90 days
```

Another says:

```text
Active Customer = purchase in 60 days
```

This is dangerous.

A simple data structure:

```python
definitions = [
    {
        "term": "Active Customer",
        "definition":
            "Purchase in last 90 days",
        "source": "Finance"
    },

    {
        "term": "Active Customer",
        "definition":
            "Purchase in last 60 days",
        "source": "Marketing"
    }
]
```

Group definitions:

```python
from collections import defaultdict
```

```python
def group_by_term(
    definitions
):

    grouped = defaultdict(list)

    for item in definitions:

        grouped[
            item["term"]
        ].append(
            item
        )

    return grouped
```

Then flag terms with conflicting definitions for human review.

> Do not allow the model to learn contradictory answers.

---

# 11. Split training and validation data correctly

Do not put nearly identical examples into both sets.

Bad:

```text
TRAIN:
What is GMV?

VALIDATION:
Explain GMV.
```

The model may simply memorize.

Better:

```text
TRAIN:
Definition questions
Common use cases

VALIDATION:
New paraphrases
Edge cases
Business scenarios
Ambiguous questions
```

Example:

```python
from datasets import Dataset
```

```python
dataset = Dataset.from_list(
    examples
)
```

For a simple split:

```python
split = dataset.train_test_split(
    test_size=0.2,
    seed=42
)
```

For production, use a semantic split:

```text
Train
 ├── Common questions
 ├── Basic definitions
 └── Standard scenarios

Validation
 ├── Unseen phrasing
 ├── Boundary cases
 └── Multi-term reasoning
```

---

# 12. Load the base model

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
tokenizer = (
    AutoTokenizer
    .from_pretrained(
        MODEL_NAME
    )
)
```

```python
model = (
    AutoModelForCausalLM
    .from_pretrained(
        MODEL_NAME
    )
)
```

---

# 13. Apply the model's chat template

This is important.

Different models expect different formats.

```python
def format_example(
    example
):

    text = (
        tokenizer
        .apply_chat_template(

            example["messages"],

            tokenize=False,

            add_generation_prompt=False
        )
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

Conceptually:

```text
SYSTEM:
You are an internal company assistant.

USER:
What is an Active Customer?

ASSISTANT:
An Active Customer is a customer who made at least
one purchase during the previous 90 days.
```

---

# 14. Fine-tune using LoRA

For terminology adaptation, LoRA is often enough.

```python
from peft import (
    LoraConfig,
    get_peft_model
)
```

Configuration:

```python
lora_config = LoraConfig(

    r=8,

    lora_alpha=16,

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

Check:

```python
model.print_trainable_parameters()
```

Output conceptually:

```text
trainable params:
~0.1% - 1%

all base parameters:
frozen
```

---

# 15. Train with supervised fine-tuning

```python
from trl import (
    SFTTrainer,
    SFTConfig
)
```

Configuration:

```python
training_args = SFTConfig(

    output_dir="./company_terminology",

    num_train_epochs=2,

    learning_rate=1e-4,

    per_device_train_batch_size=2,

    gradient_accumulation_steps=8,

    max_length=1024,

    warmup_ratio=0.05,

    lr_scheduler_type="cosine",

    logging_steps=10,

    eval_strategy="steps",

    eval_steps=100,

    bf16=True,

    report_to="none"
)
```

Trainer:

```python
trainer = SFTTrainer(

    model=model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=validation_dataset,

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
    "./company_terminology_adapter"
)
```

---

# 16. Inference

```python
messages = [

    {
        "role": "system",

        "content": (
            "You are an internal company "
            "assistant. Use official company "
            "terminology."
        )
    },

    {
        "role": "user",

        "content": (
            "A customer last purchased "
            "75 days ago. Are they active?"
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

    max_new_tokens=200,

    do_sample=False
)
```

Decode:

```python
new_tokens = outputs[
    0,
    inputs.shape[1]:
]

answer = tokenizer.decode(
    new_tokens,
    skip_special_tokens=True
)

print(answer)
```

Expected:

```text
Yes. The customer qualifies as an Active Customer because
their last purchase occurred within the previous 90 days.
```

---

# 17. How should you evaluate terminology fine-tuning?

You need more than training loss.

## Metric 1: Definition accuracy

```text
Question:
What is Active Customer?

Expected:
Purchase within 90 days

Generated:
Purchase within 90 days

✓ Correct
```

---

## Metric 2: Terminology consistency

Ask:

```text
What is Active Customer?
Who counts as active?
What does active mean in reporting?
```

All should produce consistent definitions.

---

## Metric 3: Boundary accuracy

```text
Purchase 89 days ago → Active
Purchase 90 days ago → Depends on exact policy wording
Purchase 91 days ago → Not Active
```

Boundary examples are very important.

---

## Metric 4: Hallucination rate

Check:

```text
Does the model invent company policies?
Does it invent definitions?
Does it confuse similar terms?
```

---

# 18. A simple automated evaluator

```python
def evaluate_active_customer(
    last_purchase_days: int,
    model_answer: str
):

    expected = (
        last_purchase_days <= 90
    )

    answer = model_answer.lower()

    predicted = (
        "yes" in answer
        or "active customer" in answer
    )

    return {

        "expected": expected,

        "predicted": predicted,

        "correct":
            expected == predicted
    }
```

A better production evaluator should use structured labels rather than searching for `"yes"`.

For example:

```json
{
  "classification": "active",
  "reason": "Purchase occurred 75 days ago"
}
```

Then evaluate deterministically.

---

# 19. Production architecture: Fine-tuning + RAG

This is usually the best architecture.

```text
                    User
                     │
                     ▼
              Internal Question
                     │
                     ▼
              Terminology Search
                     │
          ┌──────────┴───────────┐
          ▼                      ▼
    Current Glossary        Current Policies
          │                      │
          └──────────┬───────────┘
                     ▼
             Fine-tuned LLM
                     │
                     ▼
                   Answer
                     │
                     ▼
                Validation
```

Example:

```python
def answer_question(
    question: str
):

    # 1. Retrieve current terminology
    context = retrieve_glossary(
        question
    )

    # 2. Build prompt
    prompt = f"""
Use the official terminology below.

{context}

Question:
{question}
"""

    # 3. Generate with fine-tuned model
    return generate(prompt)
```

Why both?

### Fine-tuning teaches:

```text
How the company communicates
How terminology is naturally used
Stable domain concepts
Reasoning patterns
```

### RAG provides:

```text
Latest definitions
Policy changes
New terminology
Versioned information
Source attribution
```

---

# 20. Important production issue: versioning terminology

Suppose:

```text
2025:
Active Customer = 90 days
```

Then:

```text
2026:
Active Customer = 60 days
```

If you only fine-tuned:

```text
Model still remembers:
90 days
```

This is why you need a source of truth.

Example:

```python
class GlossaryEntry:

    def __init__(
        self,
        term: str,
        definition: str,
        version: str
    ):

        self.term = term
        self.definition = definition
        self.version = version
```

Store:

```python
glossary_entries = [
    GlossaryEntry(
        term="Active Customer",
        definition=(
            "Customer with a purchase "
            "in the last 90 days"
        ),
        version="2025"
    ),

    GlossaryEntry(
        term="Active Customer",
        definition=(
            "Customer with a purchase "
            "in the last 60 days"
        ),
        version="2026"
    )
]
```

Your retrieval system should return the latest approved definition.

---

# Interview-ready answer

> **To fine-tune an LLM for a company's terminology, I would first determine whether the terminology is stable. For stable, frequently used concepts, I would create a supervised fine-tuning dataset containing official definitions, realistic business questions, paraphrases, boundary cases, negative examples, and domain-specific tasks such as report interpretation or SQL generation.**
>
> **I would clean the data, remove duplicates, detect conflicting definitions, and have domain experts validate the source-of-truth answers. I would typically use LoRA or QLoRA for efficient adaptation.**
>
> **However, I would not rely only on fine-tuning for terminology that changes frequently. In production, I would combine fine-tuning for stable behavior and communication patterns with RAG over a versioned, authoritative company glossary and policy repository for current definitions.**
>
> **I would evaluate definition accuracy, terminology consistency, boundary-case accuracy, hallucination rate, and performance on unseen business questions.**

## Final mental model

```text
COMPANY KNOWLEDGE
        │
        ├── Stable behavior
        │       ↓
        │   Fine-tuning
        │
        └── Changing definitions
                ↓
               RAG
```

The strongest enterprise approach is:

> **Fine-tuned model + versioned company glossary + RAG + validation**, rather than trying to store all company knowledge permanently inside the model.
