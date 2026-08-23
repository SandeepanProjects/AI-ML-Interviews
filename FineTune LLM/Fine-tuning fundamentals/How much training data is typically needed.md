# What Kind of Dataset Is Required for Fine-Tuning?

The most important point is:

> **Fine-tuning data should teach the model the behavior you want—not simply give it random information.**

The dataset depends on **what you are trying to improve**.

---

# 1. General structure of a fine-tuning dataset

For instruction fine-tuning, each training example usually looks like:

```text
Input / Instruction
        ↓
Expected Ideal Output
```

For chat-based LLMs, this is often represented as messages:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a customer support assistant."
    },
    {
      "role": "user",
      "content": "I was charged twice for my subscription."
    },
    {
      "role": "assistant",
      "content": "I’m sorry about that. I can help you investigate the duplicate charge..."
    }
  ]
}
```

The model learns:

```text
Given this type of input
        ↓
Produce this type of output
```

---

# 2. Types of datasets used for fine-tuning

## A. Instruction-following dataset

Used to improve how well a model follows instructions.

```json
{
  "instruction": "Summarize the following text in three bullet points.",
  "input": "Long document...",
  "output": "- Point 1\n- Point 2\n- Point 3"
}
```

Use this when you want the model to learn:

* How to follow instructions
* A response format
* A particular tone
* A specific workflow

---

# 3. Question-answer dataset

Example:

```json
{
  "question": "How do I reset my password?",
  "answer": "Go to Settings, select Security, and choose Reset Password."
}
```

This can be useful for teaching a task pattern, but be careful:

> If the answers represent frequently changing company knowledge, **RAG is often better than fine-tuning**.

---

# 4. Classification dataset

Example:

```json
{
  "input": "I was charged twice.",
  "output": "duplicate_charge"
}
```

Or structured output:

```json
{
  "input": "I was charged twice.",
  "output": {
    "intent": "duplicate_charge",
    "priority": "high",
    "department": "billing"
  }
}
```

This is a strong fine-tuning use case.

Common examples:

```text
Support Ticket → Category
Email → Priority
Document → Classification
Message → Intent
Text → Sentiment
```

---

# 5. Structured-output dataset

Suppose your application always needs JSON.

```json
{
  "input": "My application crashes when I log in.",
  "output": {
    "category": "technical",
    "severity": "high",
    "requires_human": true
  }
}
```

Fine-tuning can help the model learn a consistent output pattern.

However, in production you should still validate the output using:

* Pydantic
* JSON Schema
* Structured output / tool calling, where supported

Do not assume fine-tuning alone guarantees valid JSON.

---

# 6. Domain-specific dataset

Suppose you are building a legal AI system.

Your dataset might contain:

```text
Legal Question
      ↓
Expert Legal Response
```

Example:

```json
{
  "question": "What are the termination conditions?",
  "answer": "Based on the provided contract clauses..."
}
```

Or a financial system:

```json
{
  "input": "Revenue: $10M, Debt: $8M, Cash: $1M",
  "output": {
    "risk": "high",
    "reason": "High debt relative to cash reserves."
  }
}
```

The critical requirement is that outputs should ideally come from **domain experts or a carefully reviewed labeling process**.

---

# 7. What makes a good fine-tuning dataset?

Data quality is usually more important than simply collecting a huge number of examples.

A good dataset should be:

### 1. Correct

Bad example:

```text
Input:
What is 2 + 2?

Output:
5
```

The model learns bad behavior.

---

### 2. Consistent

Suppose you want JSON output.

Bad:

```json
{
  "input": "Issue A",
  "output": "Billing problem"
}
```

Then:

```json
{
  "input": "Issue B",
  "output": {
    "category": "billing"
  }
}
```

Then:

```text
Input: Issue C
Output: This appears to be a billing issue.
```

This inconsistency makes the desired behavior less clear.

Better:

```json
{
  "input": "Issue A",
  "output": {
    "category": "billing"
  }
}
```

```json
{
  "input": "Issue B",
  "output": {
    "category": "technical"
  }
}
```

---

### 3. Representative of production traffic

This is extremely important.

Suppose production data looks like:

```text
80% normal requests
15% ambiguous requests
5% difficult edge cases
```

Your training dataset should not contain only easy examples.

Include:

* Normal examples
* Edge cases
* Ambiguous inputs
* Short inputs
* Long inputs
* Different writing styles
* Misspellings, if users produce them
* Difficult examples

The model should be trained on data similar to what it will see in production.

---

### 4. Diverse

Bad dataset:

```text
I need a refund.
I want a refund.
Please give me a refund.
Can I get a refund?
```

All examples are almost identical.

Better:

```text
I was charged for something I didn't buy.
The transaction appears twice.
Please cancel my subscription and return the payment.
I want my money back because the service didn't work.
```

The model learns the broader pattern rather than memorizing one phrasing.

---

### 5. Clean

Remove:

```text
Duplicate records
Broken JSON
Incorrect labels
Sensitive information
PII where not needed
Conflicting examples
Low-quality outputs
```

For enterprise data, also ensure that you have appropriate authorization and governance for using the data in training.

---

# 8. How much training data is needed?

There is **no single magic number**.

It depends on:

* Task complexity
* Base model capability
* Model size
* Data quality
* Task diversity
* Number of classes
* Whether you use full fine-tuning or LoRA/QLoRA

But here are useful practical ranges.

---

## Scenario 1: Simple style or format adaptation

Example:

```text
Input → Output in a particular company style
```

You might start experimenting with:

```text
100–1,000 high-quality examples
```

The key is not the exact number but whether the examples cover the expected inputs.

---

## Scenario 2: Classification

Example:

```text
Support Message → Intent
```

A practical starting point might be:

```text
A few hundred examples per class
```

For a small number of simple, well-separated classes, fewer may work. For overlapping or difficult classes, you may need thousands or more.

Example:

```text
10 Intent Classes
      ↓
500 examples per class
      ↓
5,000 total examples
```

---

## Scenario 3: Domain-specific instruction tuning

Example:

```text
Complex Financial Input
        ↓
Expert Analysis
```

A practical experimentation range could be:

```text
1,000–10,000+ high-quality examples
```

More complex and diverse tasks generally need more coverage.

---

## Scenario 4: Large-scale specialized training

For highly complex applications:

```text
50,000
100,000
Millions of examples
```

may be useful, particularly for broad multi-task adaptation.

But remember:

> **100,000 low-quality examples can be worse than 5,000 excellent examples.**

---

# 9. A useful rule of thumb

```text
Simple behavior / format
        ↓
Hundreds of examples

Classification
        ↓
Hundreds per class to thousands overall

Domain-specific task
        ↓
Thousands to tens of thousands

Broad capability adaptation
        ↓
Tens of thousands to millions
```

These are **starting ranges**, not guarantees.

You should validate them experimentally.

---

# 10. Train / validation / test split

Never evaluate only on your training data.

A typical setup is:

```text
Dataset
   │
   ├── Training → 80%
   │
   ├── Validation → 10%
   │
   └── Test → 10%
```

For example:

```text
10,000 examples

8,000 → Training
1,000 → Validation
1,000 → Test
```

### Training set

Used to update model weights.

### Validation set

Used during experimentation to:

* Compare checkpoints
* Tune hyperparameters
* Detect overfitting

### Test set

Used for final evaluation.

Important:

> The test set should not be used to repeatedly tune the model.

Otherwise you start indirectly overfitting to the test set.

---

# 11. Example dataset for a production use case

Suppose you're building a support-ticket classifier.

### Training example

```json
{
  "messages": [
    {
      "role": "system",
      "content": "Classify the customer issue and return valid JSON."
    },
    {
      "role": "user",
      "content": "I was charged twice for my subscription."
    },
    {
      "role": "assistant",
      "content": "{\"intent\":\"duplicate_charge\",\"priority\":\"high\",\"department\":\"billing\"}"
    }
  ]
}
```

Another:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "Classify the customer issue and return valid JSON."
    },
    {
      "role": "user",
      "content": "The application crashes when I try to log in."
    },
    {
      "role": "assistant",
      "content": "{\"intent\":\"application_crash\",\"priority\":\"high\",\"department\":\"technical\"}"
    }
  ]
}
```

Your `.jsonl` file would contain one JSON object per line:

```text
example1
example2
example3
example4
...
```

---

# 12. Dataset preparation pipeline

In a real project:

```text
Raw Data
   ↓
Remove PII / Sensitive Data
   ↓
Data Cleaning
   ↓
Deduplication
   ↓
Quality Validation
   ↓
Human Review
   ↓
Convert to Training Format
   ↓
Train / Validation / Test Split
   ↓
Fine-Tuning
   ↓
Evaluation
   ↓
Deployment
```

---

# 13. How do you know you have enough data?

This is the correct engineering answer:

> **You don't decide based only on the number of examples. You decide based on evaluation performance.**

For example:

```text
Experiment 1
500 examples
Accuracy: 78%

Experiment 2
2,000 examples
Accuracy: 87%

Experiment 3
5,000 examples
Accuracy: 90%

Experiment 4
10,000 examples
Accuracy: 90.5%
```

At some point, additional data may provide diminishing returns.

You can also identify failure cases:

```text
Model Errors
     ↓
Analyze Failures
     ↓
Find Missing Data Patterns
     ↓
Collect More Examples
     ↓
Retrain
```

This is much better than blindly collecting millions of examples.

---

# 14. Very important: Data diversity vs data quantity

Suppose you have 100,000 examples:

```text
I need a refund.
I need a refund!
Please refund me.
I want a refund.
```

These may not add much new information.

Compare that with 10,000 diverse examples:

```text
I was charged for an order I cancelled.
The transaction appears twice.
The service was unavailable after payment.
My child accidentally purchased this.
The wrong product was delivered.
```

The second dataset may be more useful.

### Rule:

> **More unique, representative examples are often more valuable than repetitive examples.**

---

# 15. Fine-tuning dataset vs RAG dataset

This distinction is important in interviews.

## Fine-tuning dataset

Used to teach:

```text
Input → Desired Behavior
```

Example:

```text
Customer Issue
       ↓
Correct Classification
```

---

## RAG dataset

Used to provide:

```text
Documents → Searchable Knowledge
```

Example:

```text
Company policy PDF
Financial report
Product documentation
Internal wiki
```

You normally don't convert every company document into fine-tuning examples.

Instead:

```text
Documents
   ↓
Chunking
   ↓
Embedding
   ↓
Vector Database
   ↓
RAG
```

---

# Interview answer

If asked:

> **What kind of dataset is required for fine-tuning, and how much data do you need?**

You can answer:

> "The dataset should consist of high-quality examples that represent the behavior I want the model to learn. For instruction tuning, this usually means instruction or input paired with the ideal response. The data should be correct, consistent, diverse, representative of production traffic, and include important edge cases. There is no fixed amount of data required. For simple formatting or style adaptation, I might start with hundreds of high-quality examples. For classification or domain-specific tasks, I would typically experiment with thousands or more, depending on task complexity and diversity. Rather than choosing a dataset size arbitrarily, I would use a held-out validation and test set, measure performance, analyze failure cases, and iteratively add high-quality examples."

## Final memory trick

```text
Fine-tuning dataset
      =
Examples of how the model
should behave

RAG dataset
      =
Documents containing
information the model
should retrieve
```

**Data quality + diversity + representativeness are usually more important than simply maximizing the number of examples.**


# What kind of dataset is required for fine-tuning?

The most important rule is:

> **Fine-tuning data should contain examples of the exact behavior you want the model to learn.**

In general:

```text
Input / Instruction
        +
Expected High-Quality Output
        ↓
   Fine-tuning Dataset
```

For LLMs, this is usually called **supervised fine-tuning (SFT) data**.

---

# 1. Basic dataset structure

A simple dataset might look like this:

```json
{
  "instruction": "Classify the customer issue",
  "input": "I was charged twice for my subscription",
  "output": "billing"
}
```

Or:

```json
{
  "instruction": "Extract information from the text",
  "input": "John purchased 5 laptops for $5000.",
  "output": {
    "customer": "John",
    "product": "laptop",
    "quantity": 5,
    "amount": 5000
  }
}
```

The model learns:

```text
Desired Input
      ↓
Desired Output
```

---

# 2. Chat-style dataset

Most modern LLM fine-tuning uses conversation-style examples.

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a customer support assistant."
    },
    {
      "role": "user",
      "content": "I was charged twice."
    },
    {
      "role": "assistant",
      "content": "I’m sorry about that. I can help you investigate the duplicate charge."
    }
  ]
}
```

Multiple examples form the dataset:

```text
Example 1
Example 2
Example 3
Example 4
...
```

Usually stored as **JSONL**:

```text
train.jsonl
validation.jsonl
test.jsonl
```

Example:

```json
{"messages":[{"role":"user","content":"I was charged twice"},{"role":"assistant","content":"billing"}]}
{"messages":[{"role":"user","content":"I cannot log in"},{"role":"assistant","content":"account_access"}]}
```

Each line is one training example.

---

# 3. What makes a good fine-tuning dataset?

## A. High-quality labels

Bad:

```text
Input:
The app crashes.

Output:
Maybe technical issue.
```

Good:

```text
Input:
The application crashes immediately after login on iOS.

Output:
{
  "category": "technical",
  "sub_category": "application_crash",
  "priority": "high"
}
```

The model learns from what you give it.

> **Bad data → bad fine-tuned model**

---

## B. Consistent output format

If half your dataset looks like:

```json
{
  "category": "billing"
}
```

and the other half:

```text
This seems to be a billing-related issue.
```

the model receives conflicting signals.

Better:

```json
{
  "category": "billing"
}
```

for every example.

---

## C. Representative data

Suppose your production data contains:

```text
50% billing
30% technical
20% account issues
```

But your training data contains:

```text
90% billing
5% technical
5% account
```

Your model may become biased toward billing.

Your dataset should approximately represent real-world usage—or intentionally rebalance classes when the task requires it.

---

## D. Diversity

Don't train on 10,000 nearly identical examples.

Bad:

```text
I was charged twice.
I was charged twice!
I got charged twice.
I was charged two times.
```

Better diversity:

```text
I was charged twice.
My card has two identical transactions.
I see a duplicate payment on my statement.
The subscription fee was deducted two times.
```

Also include:

* Short inputs
* Long inputs
* Ambiguous inputs
* Edge cases
* Different writing styles
* Typos
* Production-like language

---

## E. Correct and verified answers

For enterprise applications, examples should ideally be:

```text
Raw Data
   ↓
Human Annotation / SME Review
   ↓
Quality Validation
   ↓
Training Dataset
```

For example, in a financial application:

```text
Financial Data
       ↓
Financial Expert Review
       ↓
Approved Output
```

This is much better than automatically collecting random LLM-generated examples without validation.

---

# 4. Different datasets for different fine-tuning goals

## A. Instruction following

Goal:

> Make the model follow instructions better.

Dataset:

```json
{
  "instruction": "Summarize the text in three bullet points",
  "input": "Long document...",
  "output": "- Point 1\n- Point 2\n- Point 3"
}
```

---

## B. Classification

Goal:

> Categorize text.

```json
{
  "input": "I was charged twice",
  "output": "duplicate_payment"
}
```

---

## C. Information extraction

Goal:

> Convert unstructured text into structured data.

```json
{
  "input": "Invoice INV-1001 is for $5000 and is due on August 30.",
  "output": {
    "invoice_id": "INV-1001",
    "amount": 5000,
    "due_date": "2026-08-30"
  }
}
```

---

## D. Domain-specific assistant

Goal:

> Teach specialized response patterns.

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Explain EBITDA."
    },
    {
      "role": "assistant",
      "content": "EBITDA is a measure of..."
    }
  ]
}
```

---

## E. Code generation

```json
{
  "instruction": "Create a FastAPI endpoint for creating a user",
  "output": "@router.post('/users')..."
}
```

---

## F. Style or tone

```json
{
  "input": "Explain machine learning",
  "output": "Machine learning is a way for computers to learn patterns from data..."
}
```

Here the training examples teach:

* Tone
* Vocabulary
* Structure
* Writing style

---

# 5. How much training data is typically needed?

There is **no universal number**.

The answer depends on:

* Task complexity
* Model size
* Quality of data
* Diversity
* How different the task is from the base model
* Desired accuracy
* Whether you use full fine-tuning or LoRA/QLoRA

But here are useful practical ranges.

## Small task: 100–1,000 examples

Useful for:

* Simple formatting
* Narrow classification
* Response style
* Proof of concept

Example:

```text
Input → Category
```

For a narrow task, a few hundred excellent examples may outperform thousands of poor examples.

---

## Medium task: 1,000–10,000 examples

Useful for:

* Customer support
* Information extraction
* Domain-specific Q&A behavior
* Structured outputs
* Specialized instructions

Example:

```text
Customer Message
       ↓
Intent + Priority + Department
```

This is a common starting range for many practical fine-tuning experiments.

---

## Large/complex task: 10,000–100,000+ examples

Useful for:

* Complex domain adaptation
* Many task variations
* Large classification taxonomies
* Complex code behavior
* High-volume specialized assistants

Example:

```text
Complex Financial Input
          ↓
Detailed Expert Analysis
```

---

# 6. A practical table

| Use case                 | Rough starting dataset size |
| ------------------------ | --------------------------: |
| Simple style adaptation  |                   100–1,000 |
| Output formatting        |                   100–1,000 |
| Simple classification    |                   500–5,000 |
| Information extraction   |                1,000–10,000 |
| Customer support         |                1,000–20,000 |
| Domain-specific behavior |               5,000–50,000+ |
| Complex code generation  |             10,000–100,000+ |

These are **starting heuristics, not guarantees**.

---

# 7. Quality vs quantity

This is extremely important.

Imagine Dataset A:

```text
100,000 examples
30% incorrect labels
Inconsistent outputs
Duplicate examples
```

Dataset B:

```text
5,000 examples
Expert-reviewed
Diverse
Correct
Consistent
```

Dataset B may be significantly better.

Think of it like:

```text
Quality
   ×
Diversity
   ×
Correctness
   ×
Representativeness
```

Not simply:

```text
More data = better model
```

---

# 8. Train/validation/test split

Do not train and evaluate on the same data.

A typical split is:

```text
Total Dataset
      │
      ├── Train: 70–90%
      │
      ├── Validation: 5–15%
      │
      └── Test: 5–15%
```

Example:

```text
10,000 examples

Training:    8,000
Validation:  1,000
Test:        1,000
```

### Training set

Used to update the model:

```text
Train Data
    ↓
Loss
    ↓
Backpropagation
    ↓
Update Parameters
```

### Validation set

Used during development to check:

* Overfitting
* Hyperparameters
* Checkpoint selection

### Test set

Used for final evaluation.

The model should ideally not have been trained or tuned against the test examples.

---

# 9. Example dataset preparation pipeline

In a real project:

```text
Raw Data
   │
   ▼
Data Collection
   │
   ▼
Remove PII / Sensitive Data
   │
   ▼
Remove Duplicates
   │
   ▼
Quality Validation
   │
   ▼
Normalize Format
   │
   ▼
Human / SME Review
   │
   ▼
Train / Validation / Test Split
   │
   ▼
Fine-tuning
   │
   ▼
Evaluation
```

For example, a customer-support system:

```text
Historical Tickets
       │
       ▼
Remove names/emails/PII
       │
       ▼
Remove incorrect tickets
       │
       ▼
Label Intent
       │
       ▼
Review by Experts
       │
       ▼
Convert to JSONL
       │
       ▼
Fine-tune Model
```

---

# 10. Example Python code to create a fine-tuning dataset

Suppose your raw CSV contains:

```text
customer_message,category
"I was charged twice",billing
"The app crashes",technical
"I forgot my password",account
```

You can convert it into JSONL:

```python
import csv
import json

input_file = "support_tickets.csv"
output_file = "train.jsonl"

with open(input_file, "r", encoding="utf-8") as csv_file, \
     open(output_file, "w", encoding="utf-8") as jsonl_file:

    reader = csv.DictReader(csv_file)

    for row in reader:
        example = {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a customer support classifier. "
                        "Return only the issue category."
                    )
                },
                {
                    "role": "user",
                    "content": row["customer_message"]
                },
                {
                    "role": "assistant",
                    "content": row["category"]
                }
            ]
        }

        jsonl_file.write(json.dumps(example) + "\n")
```

Generated file:

```text
train.jsonl
```

Containing:

```json
{"messages":[{"role":"system","content":"You are a customer support classifier. Return only the issue category."},{"role":"user","content":"I was charged twice"},{"role":"assistant","content":"billing"}]}
```

---

# 11. How would I decide how much data to collect?

I would use an **iterative approach**.

### Step 1: Start with a baseline

Use the base model with prompting:

```text
Prompt + Examples
       ↓
Evaluate
```

Suppose:

```text
Accuracy = 78%
```

---

### Step 2: Collect failure cases

```text
Production / Evaluation Requests
              ↓
Identify failures
              ↓
Group by failure type
```

Example:

```text
Model Failure
     │
     ├── Incorrect classification
     ├── Invalid JSON
     ├── Domain terminology
     └── Edge cases
```

---

### Step 3: Create high-quality examples for those gaps

Instead of randomly collecting 100,000 examples:

```text
Failure Cases
      ↓
Curated Examples
      ↓
Fine-tuning Dataset
```

This is often much more efficient.

---

### Step 4: Fine-tune and evaluate

```text
Base Model
    ↓
Fine-tune
    ↓
Evaluation Dataset
    ↓
Compare with Base Model
```

Measure task-specific metrics.

For classification:

```text
Accuracy
Precision
Recall
F1 Score
```

For generation:

```text
Human evaluation
Task success rate
Format compliance
Faithfulness
Safety metrics
```

---

# Best interview answer

If an interviewer asks:

> **What kind of dataset is required for fine-tuning, and how much data is needed?**

You can answer:

> "For supervised fine-tuning, I need examples that represent the exact behavior I want the model to learn, typically in instruction-input-output or chat-message format. The dataset should be high quality, correctly labeled, diverse, representative of production traffic, and consistent in its expected outputs. I would also keep separate training, validation, and test sets. The amount of data depends on task complexity. For a narrow task, hundreds of high-quality examples can be useful, while more complex domain adaptation may require thousands or tens of thousands of examples. I wouldn't choose the dataset size upfront based only on a number—I would start with a strong baseline, fine-tune on curated data, evaluate on a held-out test set, analyze failure cases, and iteratively add high-value examples."

## Key takeaway

```text
Fine-tuning Dataset =
High Quality
+ Correct Labels
+ Diversity
+ Production-like Examples
+ Consistent Outputs
+ Good Evaluation Data
```

And remember:

> **1,000 excellent examples can be more valuable than 100,000 noisy examples.**
