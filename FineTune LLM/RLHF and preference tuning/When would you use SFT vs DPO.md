# When would you use SFT vs DPO?

This is an important LLM fine-tuning interview question.

The short answer is:

> **Use SFT when you want to teach the model what to do. Use DPO when you want to teach the model which response is better.**

---

# 1. First understand the fundamental difference

## SFT — Supervised Fine-Tuning

Your dataset looks like:

```text
Prompt
   ↓
Correct / Desired Answer
```

Example:

```python
{
    "prompt": "What is RAG?",
    "response": "RAG retrieves relevant information and provides it as context to an LLM."
}
```

SFT teaches:

```text
Question → Desired Answer
```

The objective is roughly:

[
\text{maximize } P(\text{desired response} \mid \text{prompt})
]

---

## DPO — Direct Preference Optimization

Your dataset looks like:

```text
Prompt
   │
   ├── Chosen response ✓
   │
   └── Rejected response ✗
```

Example:

```python
{
    "prompt": "How should I respond to an angry customer?",

    "chosen": (
        "I understand your frustration. Let me help resolve "
        "the issue and check what happened."
    ),

    "rejected": (
        "Calm down. This is not our problem."
    )
}
```

DPO teaches:

```text
Chosen response
       >
Rejected response
```

The objective is conceptually:

[
P_\theta(y_{chosen}|x)

>

P_\theta(y_{rejected}|x)
]

---

# 2. The easiest way to remember

```text
SFT:
"What should the model say?"

DPO:
"Which of these possible answers is better?"
```

---

# 3. When should you use SFT?

Use **SFT** when you have high-quality examples of the exact behavior you want.

For example:

```text
Customer question
       ↓
Perfect agent response
```

---

## Example 1: Teaching a customer-support model

Dataset:

```python
sft_dataset = [
    {
        "prompt": "My payment failed.",

        "response": (
            "I'm sorry your payment failed. Please verify your "
            "payment details and available balance. If the problem "
            "continues, please share the error message."
        )
    },
    {
        "prompt": "How do I reset my password?",

        "response": (
            "Click 'Forgot Password' on the login page and follow "
            "the instructions sent to your registered email."
        )
    }
]
```

You know exactly what you want the model to learn.

Use:

```text
SFT ✓
```

---

# 4. When should you use DPO?

Use **DPO** when you have **preferences** rather than one universally correct answer.

For example:

```text
Prompt:
Respond to an angry customer.
```

Both answers might be technically correct:

```text
Response A:
"Sorry. Your refund is being processed."

Response B:
"I understand this delay is frustrating. Your refund is being
processed, and I can help check its current status."
```

Both are acceptable, but:

```text
Response B > Response A
```

This is perfect for DPO.

Dataset:

```python
dpo_dataset = [
    {
        "prompt": "My refund is taking too long.",

        "chosen": (
            "I understand the delay is frustrating. I can help "
            "check the current status of your refund."
        ),

        "rejected": (
            "Refunds can take time. Please wait."
        )
    }
]
```

Use:

```text
DPO ✓
```

---

# 5. SFT teaches capability

Suppose your base model does not know how to generate your company's API calls.

You create:

```python
{
    "prompt": (
        "Get customer details using customer ID 123"
    ),

    "response": """
    GET /api/v1/customers/123
    """
}
```

Train with SFT.

The model learns:

```text
Natural Language
       ↓
Company API Format
```

This is **capability learning**.

DPO is usually not your first choice here.

Why?

Because before asking:

```text
Which answer is better?
```

the model must first know:

```text
How do I perform the task?
```

---

# 6. DPO teaches preference and behavior

Suppose the model already knows customer support.

Now you want it to:

* be polite
* be concise
* avoid blaming customers
* show empathy
* avoid unnecessary apologies
* follow company tone

These are often preference problems.

Example:

```python
{
    "prompt": "My account is locked!",

    "chosen": (
        "I understand how frustrating that is. I can help you "
        "regain access. Please use the password reset option first."
    ),

    "rejected": (
        "Your account was locked because of too many failed attempts."
    )
}
```

DPO teaches:

```text
Helpful + empathetic
        >
Technically correct but poor tone
```

---

# 7. SFT loss with code

Let's see a simplified implementation.

Suppose we have:

```text
Prompt:
What is RAG?

Target:
RAG retrieves external information.
```

The model predicts tokens.

```python
import torch
import torch.nn.functional as F
```

Simplified SFT loss:

```python
def sft_loss(
    logits: torch.Tensor,
    labels: torch.Tensor
):

    """
    logits:
        Model predictions.

        Shape:
        [batch_size, sequence_length, vocabulary_size]

    labels:
        Correct token IDs.

        Shape:
        [batch_size, sequence_length]
    """

    loss = F.cross_entropy(

        logits.view(
            -1,
            logits.size(-1)
        ),

        labels.view(-1),

        ignore_index=-100
    )

    return loss
```

Training:

```python
optimizer.zero_grad()

outputs = model(
    input_ids=input_ids,
    labels=labels
)

loss = outputs.loss

loss.backward()

optimizer.step()
```

The model is learning:

```text
Prompt
   ↓
Generate target response
```

---

# 8. DPO loss intuition with code

For DPO, we have:

```text
Prompt
   │
   ├── Chosen
   │
   └── Rejected
```

We calculate:

```text
How much does the model prefer chosen?

vs

How much does the model prefer rejected?
```

A simplified preference loss is:

[
L =
-\log
\sigma
(
\log P(chosen)
--------------

\log P(rejected)
)
]

Python:

```python
import torch
import torch.nn.functional as F


def simplified_dpo_loss(
    chosen_log_prob,
    rejected_log_prob
):

    preference_score = (
        chosen_log_prob
        -
        rejected_log_prob
    )

    loss = (
        -F.logsigmoid(
            preference_score
        )
    ).mean()

    return loss
```

Example:

```python
chosen_log_prob = torch.tensor([
    -2.0
])

rejected_log_prob = torch.tensor([
    -5.0
])

loss = simplified_dpo_loss(
    chosen_log_prob,
    rejected_log_prob
)

print(loss)
```

Because:

```text
-2 > -5
```

The chosen answer is more likely.

DPO encourages:

```text
P(chosen) ↑

P(rejected) ↓
```

**Note:** Real DPO also compares the policy against a reference policy; the simplified formula above is only for intuition.

---

# 9. Real-world SFT example using Hugging Face

A typical instruction dataset:

```python
from datasets import Dataset

dataset = Dataset.from_list([
    {
        "instruction": "Explain RAG",
        "response": (
            "RAG retrieves relevant documents and provides "
            "them as context to an LLM."
        )
    },
    {
        "instruction": "Explain LoRA",
        "response": (
            "LoRA fine-tunes a model using low-rank trainable "
            "adapter matrices."
        )
    }
])
```

Format it:

```python
def format_example(example):

    text = f"""
### Instruction:
{example["instruction"]}

### Response:
{example["response"]}
"""

    return {
        "text": text
    }


dataset = dataset.map(
    format_example
)
```

Load a model:

```python
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

model_name = "your-base-model"

tokenizer = AutoTokenizer.from_pretrained(
    model_name
)

model = AutoModelForCausalLM.from_pretrained(
    model_name
)
```

Tokenize:

```python
def tokenize(example):

    return tokenizer(
        example["text"],
        truncation=True,
        max_length=1024
    )


tokenized_dataset = dataset.map(
    tokenize
)
```

Then train with an SFT trainer or regular Transformers training loop.

Conceptually:

```text
Instruction + Correct Answer
              ↓
         Cross Entropy
              ↓
          Update Model
```

---

# 10. Real-world DPO dataset

Now imagine we want to improve response quality.

```python
dpo_dataset = Dataset.from_list([
    {
        "prompt": (
            "Customer: My order has not arrived.\n"
            "Assistant:"
        ),

        "chosen": (
            "I'm sorry your order has not arrived. I can help "
            "check its shipping status. Please provide your order ID."
        ),

        "rejected": (
            "The order may be delayed."
        )
    },

    {
        "prompt": (
            "Customer: I want a refund.\n"
            "Assistant:"
        ),

        "chosen": (
            "I can help with your refund request. Please provide "
            "your order ID so I can check eligibility and next steps."
        ),

        "rejected": (
            "You need to contact support."
        )
    }
])
```

The model already knows language.

We are refining:

```text
Response quality
```

---

# 11. DPO training conceptually

```text
                Prompt
                   │
          ┌────────┴────────┐
          ▼                 ▼
       Chosen            Rejected
          │                 │
          ▼                 ▼
    Policy Model      Policy Model
          │                 │
          ▼                 ▼
     Log Probability   Log Probability
          │                 │
          └────────┬────────┘
                   ▼
           Preference Difference
                   │
                   ▼
              DPO Loss
                   │
                   ▼
             Update Policy
```

The objective teaches:

```text
Chosen probability ↑

Rejected probability ↓
```

---

# 12. Practical DPO with TRL

You can use a dedicated DPO trainer. [Hugging Face TRL DPO Trainer documentation](https://huggingface.co/docs/trl/dpo_trainer?utm_source=chatgpt.com)

Conceptually:

```python
from trl import (
    DPOTrainer,
    DPOConfig
)
```

Configuration:

```python
training_args = DPOConfig(

    output_dir="./dpo-output",

    learning_rate=5e-7,

    per_device_train_batch_size=2,

    gradient_accumulation_steps=8,

    num_train_epochs=1,

    logging_steps=10
)
```

Trainer:

```python
trainer = DPOTrainer(

    model=model,

    ref_model=reference_model,

    args=training_args,

    train_dataset=dpo_dataset,

    processing_class=tokenizer
)
```

Train:

```python
trainer.train()
```

Conceptually:

```text
Chosen > Rejected
       ↓
     DPO Loss
       ↓
 Update Model
```

---

# 13. A real-world example: Customer support LLM

Imagine you are building:

```text
Company Customer Support LLM
```

## Stage 1: SFT

Train the model using:

```text
Customer Question
       ↓
Correct Answer
```

Dataset:

```python
sft_examples = [

    {
        "prompt": "How do I change my email?",

        "response": (
            "Go to Settings → Account → Email and follow "
            "the verification instructions."
        )
    },

    {
        "prompt": "How do I cancel my subscription?",

        "response": (
            "Go to Settings → Billing → Subscription and "
            "select Cancel Subscription."
        )
    }
]
```

Result:

```text
Model learns company support tasks
```

---

## Stage 2: DPO

Now improve behavior.

```python
dpo_examples = [

    {
        "prompt": "I am very angry about your service!",

        "chosen": (
            "I'm sorry you've had this experience. I understand "
            "your frustration and would like to help resolve the issue."
        ),

        "rejected": (
            "Please calm down and explain the issue."
        )
    }
]
```

Result:

```text
Model learns preferred communication style
```

Final pipeline:

```text
Base Model
    │
    ▼
   SFT
    │
    ▼
Task-Capable Model
    │
    ▼
   DPO
    │
    ▼
Preference-Aligned Model
```

This is a very common mental model.

---

# 14. When SFT is better than DPO

Use SFT when:

### 1. You have clear ground truth

```text
Input:
2 + 2

Output:
4
```

There is an exact answer.

---

### 2. Teaching a new task

```text
English
   ↓
SQL
```

```text
English
   ↓
Python Code
```

```text
Document
   ↓
Structured JSON
```

You want:

```text
Task capability ↑
```

---

### 3. Domain adaptation

For example:

```text
Medical terminology
Legal terminology
Company APIs
Internal documentation style
```

High-quality demonstrations are useful.

---

### 4. Small or medium training dataset

SFT is generally simpler to start with.

---

# 15. When DPO is better than SFT

Use DPO when:

### 1. Multiple answers can be valid

Example:

```text
Question:
How do I politely reject a meeting?

Answer A → Good
Answer B → Better
```

DPO learns:

```text
Better > Good
```

---

### 2. You care about style and preference

For example:

```text
More helpful
More concise
More empathetic
Less verbose
Safer
More professional
```

---

### 3. You have human preference data

Example:

```text
Prompt
   ↓
Human sees two responses
   ↓
Human chooses better one
```

Dataset:

```python
{
    "prompt": "...",
    "chosen": "...",
    "rejected": "..."
}
```

Perfect for DPO.

---

### 4. You want to align an already capable model

```text
Capable Model
       ↓
Improve behavior
       ↓
DPO
```

---

# 16. The biggest practical difference

Consider training a model to generate SQL.

## Case A: Model cannot generate SQL

Input:

```text
Get all customers from Bangalore
```

Desired:

```sql
SELECT *
FROM customers
WHERE city = 'Bangalore';
```

Use:

```text
SFT
```

Because the model needs to learn:

```text
Natural language → SQL
```

---

## Case B: Model can generate SQL, but quality varies

Generated responses:

```sql
SELECT *
FROM customers
WHERE city = 'Bangalore';
```

vs:

```sql
SELECT *
FROM customer
WHERE city = Bangalore;
```

You can create:

```text
Chosen SQL
     >
Rejected SQL
```

Use:

```text
DPO
```

Because the model already has the capability, and you are improving preference/quality.

---

# 17. SFT vs DPO comparison table

| Feature                     | SFT                 | DPO                        |
| --------------------------- | ------------------- | -------------------------- |
| Main goal                   | Teach task          | Teach preference           |
| Dataset                     | Input → target      | Prompt + chosen + rejected |
| Needs rejected answer       | No                  | Yes                        |
| Good for exact outputs      | Excellent           | Less direct                |
| Good for behavior alignment | Limited             | Excellent                  |
| Human preference data       | Optional            | Very useful                |
| Complexity                  | Lower               | Higher                     |
| Reference model             | No                  | Yes, in standard DPO       |
| Typical use                 | Capability learning | Alignment                  |
| Example                     | English → SQL       | Better SQL > worse SQL     |

---

# 18. Can you use SFT and DPO together?

**Yes. In fact, this is often a strong pipeline.**

```text
            Base LLM
               │
               ▼
             SFT
               │
       Learns the task
               │
               ▼
          SFT Model
               │
               ▼
             DPO
               │
       Learns preferences
               │
               ▼
         Final Aligned Model
```

Example:

```text
Base Model
   │
   ▼
SFT:
Learn customer support
   │
   ▼
DPO:
Learn helpfulness + tone
   │
   ▼
Production Model
```

---

# 19. Complete decision framework

Use this decision tree:

```text
Do you need to teach the model a new capability?
              │
        ┌─────┴─────┐
        │           │
       YES          NO
        │           │
       SFT     Do you have
                preference pairs?
                     │
               ┌─────┴─────┐
               │           │
              YES          NO
               │           │
              DPO    Start collecting
                     preference data
```

Another version:

```text
Do you have:

Prompt → Correct Answer
        │
        ▼
       SFT


Prompt → Better Answer + Worse Answer
        │
        ▼
       DPO
```

---

# 20. A realistic production pipeline

Suppose you are building an enterprise AI assistant.

```text
                   Base LLM
                      │
                      ▼
        ┌─────────────────────────┐
        │        SFT Stage        │
        │                         │
        │ Company terminology     │
        │ API calling             │
        │ JSON format             │
        │ Domain tasks            │
        └────────────┬────────────┘
                     │
                     ▼
                SFT Model
                     │
                     ▼
        ┌─────────────────────────┐
        │        DPO Stage        │
        │                         │
        │ Helpful vs unhelpful    │
        │ Safe vs unsafe          │
        │ Good tone vs bad tone   │
        │ Accurate vs weak        │
        └────────────┬────────────┘
                     │
                     ▼
              Aligned Model
                     │
                     ▼
                    RAG
                     │
                     ▼
             Production System
```

One important distinction:

* **SFT/DPO** change model behavior and learned patterns.
* **RAG** provides up-to-date or private factual context at inference time.

---

# Interview-ready answer

> **I would use SFT when I have high-quality input-output examples and want to teach the model a new capability, task, format, or domain behavior. For example, natural language to SQL, customer-support workflows, API calling, or structured JSON generation.**
>
> **I would use DPO when the model already has the basic capability but I want to optimize preferences between alternative responses. The dataset contains a prompt, a chosen response, and a rejected response. DPO is particularly useful for aligning helpfulness, tone, conciseness, safety, and response quality.**
>
> **In a real production pipeline, I would often first perform SFT to teach the capability, then DPO to align the behavior using human or high-quality preference data.**

## Final one-line rule

```text
SFT = Teach the model WHAT to do.

DPO = Teach the model WHICH answer is better.

SFT + DPO = Capability + Alignment.
```
