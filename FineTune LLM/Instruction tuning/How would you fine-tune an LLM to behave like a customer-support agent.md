# How would you fine-tune an LLM to behave like a customer-support agent?

This is a very common **LLM/AI Engineer interview question**.

The key idea is:

> I would not fine-tune an LLM primarily to memorize changing customer information. I would fine-tune it to learn the **behavior, tone, workflow, formatting, and support policies**, and use **RAG and tools/APIs** for dynamic information such as order status, account data, and current policies.

A production architecture would look like:

```text
                    CUSTOMER
                        │
                        ▼
                "Where is my order?"
                        │
                        ▼
                Intent / Safety Layer
                        │
                        ▼
          ┌─────────────┴─────────────┐
          │                           │
          ▼                           ▼
    Customer Context              RAG Search
    Order / Account API           Policies / FAQs
          │                           │
          └─────────────┬─────────────┘
                        ▼
              Fine-tuned LLM
                        │
                        ▼
             Policy-compliant response
```

---

# 1. First define what should be fine-tuned

Suppose we have a customer-support agent.

I would separate requirements into three categories.

## A. Fine-tune

Use fine-tuning for stable behavior:

```text
Tone
Empathy
Professional language
Response structure
Escalation behavior
Instruction following
Company-specific support workflow
```

Example:

```text
Customer:
My order is late!

Desired behavior:

1. Apologize
2. Show empathy
3. Ask for required information
4. Do not invent order status
5. Offer next action
```

---

## B. Use RAG

Use RAG for information that changes:

```text
Refund policies
Return policies
Product documentation
Support articles
Terms and conditions
FAQs
```

---

## C. Use tools/APIs

Use APIs for customer-specific data:

```text
Order status
Account information
Payment information
Subscription status
Shipping tracking
Refund status
```

Example:

```text
User:
Where is order #12345?

LLM
  │
  ▼
Order API
  │
  ▼
{
  "status": "shipped",
  "expected_delivery": "Tomorrow"
}
  │
  ▼
LLM generates response
```

You should **not** try to memorize all this information through fine-tuning.

---

# 2. Design the training objective

We want to teach the model:

```text
Customer message
       +
Support context
       ↓
Correct support behavior
       ↓
High-quality response
```

Example:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a customer support assistant. Be polite, empathetic, concise, and never invent account or order information."
    },
    {
      "role": "user",
      "content": "My package is late and I am really frustrated."
    },
    {
      "role": "assistant",
      "content": "I'm sorry your package is delayed, and I understand how frustrating that can be. Please share your order number so I can help check the next steps."
    }
  ]
}
```

This is a **conversational instruction-tuning dataset**.

---

# 3. Define the support behavior before collecting data

A production support assistant needs a clear policy.

For example:

```python
SUPPORT_RULES = {
    "tone": [
        "professional",
        "empathetic",
        "clear",
        "concise"
    ],

    "must_do": [
        "acknowledge the customer's issue",
        "provide accurate next steps",
        "ask for clarification when required",
        "escalate when necessary"
    ],

    "must_not_do": [
        "invent account information",
        "invent order status",
        "promise refunds without authorization",
        "request unnecessary sensitive information"
    ]
}
```

This becomes part of your:

```text
Dataset design
Prompt
System instructions
Evaluation criteria
```

---

# 4. What should the training dataset look like?

A good dataset should cover many real support situations.

For example:

```text
Order tracking
Delayed delivery
Refund requests
Subscription cancellation
Login problems
Payment failures
Product issues
Account issues
Angry customers
Confused customers
Escalation requests
Out-of-scope requests
```

Dataset distribution:

```text
100,000 conversations

Order tracking             20%
Refunds                    20%
Account issues             15%
Technical issues           15%
Billing                    10%
Cancellation               10%
Escalation                  5%
Edge cases                  5%
```

The exact percentages depend on real production traffic.

---

# 5. Example dataset

Let's create:

```text
customer_support.jsonl
```

Example 1:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a professional customer support agent. Be empathetic and helpful. Do not invent information."
    },
    {
      "role": "user",
      "content": "My order is three days late. Where is it?"
    },
    {
      "role": "assistant",
      "content": "I'm sorry your order is delayed. I understand how frustrating that can be. Please share your order number, and I can help you check the available tracking information and next steps."
    }
  ]
}
```

Example 2:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a professional customer support agent. Never promise a refund unless eligibility is confirmed."
    },
    {
      "role": "user",
      "content": "I want my money back immediately!"
    },
    {
      "role": "assistant",
      "content": "I'm sorry you're having this experience. I can help you review the refund process. Please share the relevant order details, and we'll check your eligibility according to the applicable refund policy."
    }
  ]
}
```

Example 3:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful support agent."
    },
    {
      "role": "user",
      "content": "I can't log into my account."
    },
    {
      "role": "assistant",
      "content": "I'm sorry you're having trouble logging in. Please first try resetting your password using the 'Forgot Password' option. If the issue continues, let me know the error message you see, without sharing your password."
    }
  ]
}
```

---

# 6. Create the dataset with Python

```python
import json
```

Create examples:

```python
examples = [

    {
        "messages": [

            {
                "role": "system",

                "content": (
                    "You are a professional customer "
                    "support agent. Be empathetic, helpful, "
                    "and concise. Never invent customer data."
                )
            },

            {
                "role": "user",

                "content": (
                    "My package is late. "
                    "I'm very frustrated."
                )
            },

            {
                "role": "assistant",

                "content": (
                    "I'm sorry your package is delayed. "
                    "I understand how frustrating that can be. "
                    "Please share your order number so I can "
                    "help you check the available next steps."
                )
            }
        ]
    },

    {
        "messages": [

            {
                "role": "system",

                "content": (
                    "You are a professional customer "
                    "support agent. Do not invent order "
                    "or account information."
                )
            },

            {
                "role": "user",

                "content": (
                    "Can you tell me when order 123 "
                    "will arrive?"
                )
            },

            {
                "role": "assistant",

                "content": (
                    "I can help you check the order status. "
                    "I'll need to look up the latest tracking "
                    "information before confirming an expected "
                    "delivery date."
                )
            }
        ]
    }
]
```

Save:

```python
with open(
    "customer_support.jsonl",
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

# 7. Clean and validate the dataset

Before training, validate every example.

```python
def validate_example(example):

    messages = example.get(
        "messages",
        []
    )

    if len(messages) < 2:

        return False

    valid_roles = {
        "system",
        "user",
        "assistant"
    }

    for message in messages:

        if (
            message.get("role")
            not in valid_roles
        ):
            return False

        if not message.get(
            "content"
        ):
            return False

    return True
```

Use:

```python
clean_examples = [

    example

    for example in examples

    if validate_example(
        example
    )
]
```

---

# 8. Remove sensitive customer data

This is critical in real-world customer-support datasets.

Historical conversations may contain:

```text
Email addresses
Phone numbers
Credit card information
Addresses
Account IDs
Passwords
Personal identifiers
```

Never blindly fine-tune on raw production conversations.

A simplified redaction example:

```python
import re
```

```python
def redact_sensitive_data(text):

    # Email
    text = re.sub(

        r"[A-Za-z0-9._%+-]+@"
        r"[A-Za-z0-9.-]+\.[A-Za-z]{2,}",

        "[EMAIL]",

        text
    )

    # Phone numbers
    text = re.sub(

        r"\b\d{10}\b",

        "[PHONE]",

        text
    )

    # Example card pattern
    text = re.sub(

        r"\b(?:\d[ -]*?){13,16}\b",

        "[CARD_NUMBER]",

        text
    )

    return text
```

Apply:

```python
for example in clean_examples:

    for message in example["messages"]:

        message["content"] = (
            redact_sensitive_data(
                message["content"]
            )
        )
```

In production, use stronger, domain-specific PII detection and review rather than relying only on regex.

---

# 9. Remove duplicate conversations

Why?

Suppose you have:

```text
"My order is late"
```

repeated:

```text
10,000 times
```

The model can become biased toward that pattern.

A simple deduplication approach:

```python
import hashlib
import json
```

```python
def conversation_hash(example):

    content = json.dumps(

        example["messages"],

        sort_keys=True
    )

    return hashlib.sha256(

        content.encode(
            "utf-8"
        )

    ).hexdigest()
```

Remove duplicates:

```python
seen = set()

deduplicated = []

for example in clean_examples:

    example_id = conversation_hash(
        example
    )

    if example_id not in seen:

        seen.add(
            example_id
        )

        deduplicated.append(
            example
        )
```

For very large datasets, you would typically use more scalable distributed deduplication and also semantic-near-duplicate detection.

---

# 10. Split train and validation data

```python
from datasets import Dataset
```

Create dataset:

```python
dataset = Dataset.from_list(
    deduplicated
)
```

Split:

```python
dataset_split = dataset.train_test_split(

    test_size=0.2,

    seed=42
)
```

Then split the held-out portion:

```python
validation_test = (
    dataset_split["test"]
    .train_test_split(
        test_size=0.5,
        seed=42
    )
)
```

Final:

```python
train_dataset = (
    dataset_split["train"]
)

validation_dataset = (
    validation_test["train"]
)

test_dataset = (
    validation_test["test"]
)
```

Result:

```text
Train        80%
Validation   10%
Test         10%
```

---

# 11. Load a base model

For demonstration, use a placeholder:

```python
MODEL_NAME = "your-base-instruct-model"
```

Load:

```python
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)
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

If required:

```python
if tokenizer.pad_token is None:

    tokenizer.pad_token = (
        tokenizer.eos_token
    )
```

---

# 12. Use the model's chat template

This is important.

Don't manually invent a prompt format unless the model requires it.

Example:

```python
def format_conversation(
    example
):

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
train_dataset = train_dataset.map(
    format_conversation
)

validation_dataset = (
    validation_dataset.map(
        format_conversation
    )
)
```

Conceptually:

```text
System
   ↓
You are a professional support agent

User
   ↓
My order is late

Assistant
   ↓
I'm sorry to hear that...
```

---

# 13. Fine-tune using LoRA

For many practical projects, I would start with LoRA rather than full fine-tuning.

Install:

```bash
pip install transformers datasets peft trl accelerate
```

Imports:

```python
from peft import (
    LoraConfig,
    get_peft_model
)
```

LoRA configuration:

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

Check:

```python
model.print_trainable_parameters()
```

Conceptually:

```text
Base Model
7 billion parameters
        │
        │ Frozen
        ▼
LoRA adapters
        │
        │ Trainable
        ▼
Customer support behavior
```

---

# 14. Training configuration

```python
from trl import (
    SFTConfig,
    SFTTrainer
)
```

```python
training_args = SFTConfig(

    output_dir="./customer_support_model",

    num_train_epochs=3,

    learning_rate=2e-4,

    per_device_train_batch_size=2,

    gradient_accumulation_steps=8,

    max_length=1024,

    warmup_ratio=0.05,

    lr_scheduler_type="cosine",

    logging_steps=10,

    eval_strategy="steps",

    eval_steps=100,

    save_strategy="steps",

    save_steps=100,

    bf16=True,

    report_to="none"
)
```

Create trainer:

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
    "./customer_support_adapter"
)
```

Save tokenizer:

```python
tokenizer.save_pretrained(
    "./customer_support_adapter"
)
```

---

# 15. What is actually being learned?

Suppose training data contains:

```text
Customer:
My payment failed.

Good response:
I'm sorry you're having trouble with your payment.
Please check whether your payment method is valid and try again.
If the problem continues, I can guide you through the next steps.
```

The model gradually learns patterns like:

```text
Negative customer emotion
        ↓
Acknowledge

Problem
        ↓
Troubleshooting

Cannot solve
        ↓
Escalate

Missing information
        ↓
Ask a clarification question
```

It is learning a **behavior policy**, not simply memorizing one response.

---

# 16. Handle angry customers

This should be represented in training data.

Example:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Your company is terrible! I have been waiting for a week!"
    },
    {
      "role": "assistant",
      "content": "I'm sorry you've had to wait so long, and I understand why you're frustrated. I'd like to help resolve this. Please share your order reference so the latest status can be checked."
    }
  ]
}
```

The model learns:

```text
Anger
 ↓
Do not become defensive
 ↓
Acknowledge frustration
 ↓
Focus on resolution
```

---

# 17. Train the model to escalate

Not every problem should be solved by the LLM.

Example:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Someone accessed my account and changed my email address."
    },
    {
      "role": "assistant",
      "content": "I'm sorry this happened. Because this may involve account security, please use the official account-recovery process or contact the security support team. Do not share your password or verification codes in chat."
    }
  ]
}
```

Teach the model:

```text
Normal issue
    ↓
LLM can assist

High-risk issue
    ↓
Escalate / secure workflow
```

---

# 18. But don't rely only on fine-tuning

This is a key production principle.

Suppose the user asks:

```text
Where is order #A123?
```

Fine-tuning cannot know the current answer.

Use a tool.

Example tool:

```python
def get_order_status(
    order_id: str
):

    # In production:
    # Call database or order API

    return {

        "order_id": order_id,

        "status": "SHIPPED",

        "estimated_delivery":
            "2026-08-28"
    }
```

Then:

```text
User
  │
  ▼
"Where is my order?"
  │
  ▼
Fine-tuned model recognizes:
Need order lookup
  │
  ▼
Order API
  │
  ▼
Current status
  │
  ▼
LLM creates natural response
```

---

# 19. Example FastAPI support endpoint

Since you are learning FastAPI and production AI architecture, this is how the pieces could fit together.

```python
from fastapi import (
    FastAPI,
    HTTPException
)

from pydantic import (
    BaseModel
)
```

Create the application:

```python
app = FastAPI()
```

Request model:

```python
class SupportRequest(
    BaseModel
):

    customer_id: str

    message: str

    order_id: str | None = None
```

Response model:

```python
class SupportResponse(
    BaseModel
):

    answer: str

    escalated: bool
```

Simple intent detection:

```python
def detect_order_request(
    message: str
):

    keywords = [

        "order",

        "delivery",

        "package",

        "tracking"
    ]

    message = message.lower()

    return any(

        keyword in message

        for keyword in keywords
    )
```

Order service:

```python
async def get_order_status(
    order_id: str
):

    # Example only
    return {

        "status":
            "Shipped",

        "estimated_delivery":
            "2026-08-28"
    }
```

LLM generation placeholder:

```python
async def generate_llm_response(
    customer_message: str,
    context: str
):

    # Replace with actual model inference

    return (
        "I'm sorry you're experiencing this issue. "
        "Based on the available information: "
        + context
    )
```

Endpoint:

```python
@app.post(
    "/support",
    response_model=SupportResponse
)
async def support(
    request: SupportRequest
):

    context = ""

    # Check whether we need order data
    if (

        request.order_id

        and

        detect_order_request(
            request.message
        )
    ):

        order = await get_order_status(
            request.order_id
        )

        context = (
            f"Order status: "
            f"{order['status']}. "
            f"Estimated delivery: "
            f"{order['estimated_delivery']}."
        )

    answer = await generate_llm_response(

        customer_message=
            request.message,

        context=context
    )

    return SupportResponse(

        answer=answer,

        escalated=False
    )
```

Request:

```json
{
  "customer_id": "customer_123",
  "message": "Where is my order?",
  "order_id": "order_456"
}
```

Response:

```json
{
  "answer": "I'm sorry for the uncertainty. Your order has been shipped and the estimated delivery date is August 28, 2026.",
  "escalated": false
}
```

---

# 20. Add RAG for support policies

Suppose a user asks:

```text
Can I return a product after 30 days?
```

The current policy may change.

Pipeline:

```text
User Question
      │
      ▼
Embed Question
      │
      ▼
Vector Search
      │
      ▼
Current Return Policy
      │
      ▼
Fine-tuned Support LLM
      │
      ▼
Answer
```

Pseudo-code:

```python
async def retrieve_policy(
    query: str
):

    results = await vector_store.search(

        query=query,

        limit=3
    )

    return results
```

Then:

```python
async def answer_support_question(
    message: str
):

    policy_documents = await retrieve_policy(
        message
    )

    context = "\n".join(

        document["content"]

        for document in policy_documents
    )

    prompt = f"""
Use only the following support policy.

POLICY:
{context}

CUSTOMER:
{message}
"""

    return await llm.generate(
        prompt
    )
```

In a production system, you would add relevance thresholds, document permissions, citations/grounding, and fallback behavior when no reliable context is retrieved.

---

# 21. Evaluate the fine-tuned model

Don't evaluate only with:

```text
Loss
```

Create a held-out support benchmark.

Example:

```python
evaluation_examples = [

    {
        "input":
            "I'm extremely angry. My order hasn't arrived.",

        "expected_behaviors": [

            "empathetic",

            "professional",

            "does_not_blame_customer",

            "provides_next_step"
        ]
    },

    {
        "input":
            "Give me someone else's order details.",

        "expected_behaviors": [

            "protects_privacy",

            "refuses_unauthorized_access"
        ]
    }
]
```

Evaluation function concept:

```python
def evaluate_response(
    response: str
):

    score = {

        "empathy": 0,

        "accuracy": 0,

        "policy_compliance": 0,

        "helpfulness": 0
    }

    return score
```

For production, evaluation can combine:

```text
Automated evaluation
+
LLM-as-judge
+
Rule-based checks
+
Human review
+
Business metrics
```

---

# 22. Useful metrics

For a customer-support model, I would track:

```text
Response quality
Policy compliance
Hallucination rate
Escalation accuracy
Tool-selection accuracy
Retrieval groundedness
Customer satisfaction
Resolution rate
Average handling time
Containment rate
Human escalation rate
```

Example:

```python
metrics = {

    "policy_compliance":
        0.98,

    "hallucination_rate":
        0.02,

    "escalation_accuracy":
        0.95,

    "customer_satisfaction":
        0.91
}
```

---

# 23. Production architecture

A more realistic architecture is:

```text
                     ┌──────────────┐
                     │   Customer   │
                     └──────┬───────┘
                            │
                            ▼
                    ┌───────────────┐
                    │    FastAPI    │
                    └──────┬────────┘
                           │
                           ▼
                  ┌──────────────────┐
                  │ Intent / Router  │
                  └─────┬───────┬────┘
                        │       │
            ┌───────────┘       └────────────┐
            ▼                                ▼
     ┌─────────────┐                   ┌──────────────┐
     │ RAG System  │                   │ Tools / APIs │
     │ Policies    │                   │ Order/Account│
     └──────┬──────┘                   └──────┬───────┘
            │                                 │
            └──────────────┬──────────────────┘
                           ▼
                 ┌──────────────────┐
                 │ Fine-tuned LLM   │
                 │ Support behavior │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │ Safety / Guard   │
                 └────────┬─────────┘
                          │
                          ▼
                     Response
```

---

# 24. The interview answer

If asked:

> **How would you fine-tune an LLM to behave like a customer-support agent?**

A strong answer:

> **First, I would define the desired support behavior, including tone, empathy, response structure, escalation rules, and actions the model must not perform. I would collect high-quality historical support conversations, remove or redact sensitive customer information, remove duplicates, correct poor responses, and convert the data into a conversational instruction-tuning format with user and ideal assistant responses. I would create train, validation, and test splits carefully to avoid customer or conversation leakage.**
>
> **For training, I would usually start with supervised fine-tuning using LoRA or QLoRA to reduce cost and preserve the base model. I would use the base model's official chat template and train the model primarily to learn support behavior rather than changing facts.**
>
> **For dynamic information, I would combine the fine-tuned model with RAG for policies and documentation and tools/APIs for customer-specific information such as orders and subscriptions. Finally, I would evaluate the model using policy compliance, hallucination rate, escalation accuracy, groundedness, customer satisfaction, and human review.**

# Final mental model

```text
Fine-tuning
    ↓
HOW the model behaves

RAG
    ↓
WHAT current knowledge the model can access

Tools
    ↓
WHAT real-time actions/data the model can use
```

For a production customer-support system:

> **Fine-tune the behavior, retrieve the knowledge, and call APIs for customer-specific truth.**
