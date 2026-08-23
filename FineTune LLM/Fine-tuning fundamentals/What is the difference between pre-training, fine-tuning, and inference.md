# Pre-training vs Fine-tuning vs Inference

These are three different stages in the lifecycle of an LLM.

## The simplest explanation

> **Pre-training = Teach the model general knowledge**
> **Fine-tuning = Specialize the model for a task or behavior**
> **Inference = Use the trained model to generate predictions/answers**

---

# 1. Complete LLM lifecycle

```text
Massive Raw Data
      │
      ▼
┌─────────────────┐
│   PRE-TRAINING  │
│ Learn language, │
│ patterns, etc.  │
└────────┬────────┘
         │
         ▼
     Base Model
         │
         ▼
┌─────────────────┐
│   FINE-TUNING   │
│ Learn a specific│
│ task/behavior   │
└────────┬────────┘
         │
         ▼
 Specialized Model
         │
         ▼
┌─────────────────┐
│    INFERENCE    │
│ Use the model   │
│ to answer users │
└────────┬────────┘
         │
         ▼
      Output
```

---

# 2. What is pre-training?

## Definition

**Pre-training** is the initial training of an LLM on a massive amount of general data.

The goal is to teach the model:

* Language
* Grammar
* Reasoning patterns
* Programming
* Mathematics
* General world knowledge
* Relationships between concepts
* Patterns in text

For example, a model may be trained on a huge corpus containing:

```text
Books
Wikipedia-like content
Web pages
Documentation
Research papers
Code
Other licensed/training data
```

The exact training data varies by model and provider.

---

## How does pre-training work?

A common objective for an autoregressive LLM is:

```text
Input:
"The capital of France is"

Target:
"Paris"
```

The model predicts the next token.

```text
"The capital of France is" → ?
```

Initially, it might predict:

```text
London
```

Then calculate the error:

```text
Loss = Difference between prediction and expected token
```

Then:

```text
Forward Pass
      ↓
Prediction
      ↓
Calculate Loss
      ↓
Backpropagation
      ↓
Update Billions of Parameters
      ↓
Repeat
```

This process happens across enormous amounts of data.

### Output of pre-training

You get a **base model**:

```text
Raw Data
    ↓
Pre-training
    ↓
Base LLM
```

Examples of model families include models from companies and open-source organizations such as [OpenAI](https://openai.com/?utm_source=chatgpt.com) and [Meta AI](https://ai.meta.com/?utm_source=chatgpt.com).

---

## Characteristics of pre-training

| Feature            | Pre-training                             |
| ------------------ | ---------------------------------------- |
| Data size          | Extremely large                          |
| Compute            | Very expensive                           |
| Duration           | Days to months                           |
| GPUs               | Large clusters                           |
| Parameters updated | Usually all trainable model parameters   |
| Goal               | General capabilities                     |
| Example            | Learn language and next-token prediction |

---

# 3. What is fine-tuning?

## Definition

**Fine-tuning** takes an already pre-trained model and trains it further on a smaller, specialized dataset.

```text
Base Model
    +
Domain/Task Dataset
    ↓
Fine-tuning
    ↓
Specialized Model
```

The goal is not to teach the model everything from scratch.

Instead, it adapts the model to:

* A specific task
* A domain
* A response style
* A classification problem
* A particular output format
* Better instruction following

---

## Example

Suppose we have a general LLM.

We want it to classify customer-support messages.

Training data:

```json
{
  "input": "I was charged twice",
  "output": {
    "category": "billing",
    "priority": "high"
  }
}
```

Another example:

```json
{
  "input": "The application crashes when I log in",
  "output": {
    "category": "technical",
    "priority": "high"
  }
}
```

After fine-tuning:

```text
Base LLM
   ↓
Fine-tuning Data
   ↓
Customer Support LLM
```

---

## Fine-tuning methods

### Full fine-tuning

Updates many or all model parameters.

```text
Base Model Weights
       ↓
Training
       ↓
Updated Model Weights
```

This is expensive.

---

### Parameter-Efficient Fine-Tuning (PEFT)

Instead of modifying the entire model:

```text
Base Model (Frozen)
        +
Small Trainable Parameters
        ↓
Fine-tuned Model
```

Popular approaches include:

* LoRA
* QLoRA
* Adapters

For example:

```text
Original Model: 7 Billion parameters
             ↓
Only train small adapter matrices
             ↓
Lower GPU memory and training cost
```

This is commonly used for adapting open-weight models.

---

# 4. What is inference?

## Definition

**Inference** is the process of using a trained model to make predictions or generate output.

For example:

```text
User:
"What is RAG?"

        ↓

Trained LLM

        ↓

Response:
"RAG stands for Retrieval-Augmented Generation..."
```

During normal inference:

> The model is generally **not learning**.

Its parameters are typically fixed.

---

## Inference flow

```text
User Input
     ↓
Tokenizer
     ↓
Input Tokens
     ↓
Model Forward Pass
     ↓
Probability Distribution
     ↓
Select Next Token
     ↓
Generate Token
     ↓
Repeat
     ↓
Final Response
```

Example:

```text
Input:
"What is the capital of India?"

Model generates:

"The"
   ↓
"capital"
   ↓
"of"
   ↓
"India"
   ↓
"is"
   ↓
"New"
   ↓
"Delhi"
```

The model generates tokens sequentially.

---

# 5. Training vs inference

This distinction is extremely important.

## Training

```text
Input
  ↓
Model
  ↓
Prediction
  ↓
Compare with Correct Answer
  ↓
Calculate Loss
  ↓
Backpropagation
  ↓
Update Weights
```

The model **learns**.

---

## Inference

```text
Input
  ↓
Model
  ↓
Prediction
  ↓
Output
```

The model generally **does not update weights**.

---

# 6. Main differences

| Feature          | Pre-training                           | Fine-tuning                   | Inference             |
| ---------------- | -------------------------------------- | ----------------------------- | --------------------- |
| Purpose          | Build general model                    | Specialize model              | Use the model         |
| Starting point   | Random/initialized model or checkpoint | Pre-trained model             | Trained model         |
| Dataset          | Massive general corpus                 | Smaller specialized dataset   | User input/context    |
| Updates weights? | Yes                                    | Yes, fully or partially       | Normally no           |
| Compute cost     | Extremely high                         | Moderate to high              | Per-request cost      |
| Duration         | Weeks/months                           | Minutes/days                  | Milliseconds/seconds  |
| Example          | Learn language                         | Learn customer classification | Answer customer query |

---

# 7. Real-world example

Imagine building a **Financial AI Assistant**.

## Stage 1: Pre-training

A model is trained on broad data.

```text
Books
Code
Math
General finance concepts
Language
```

It learns general capabilities.

```text
                PRE-TRAINING
                       │
                       ▼
              General Base Model
```

---

## Stage 2: Fine-tuning

You specialize it using examples:

```text
Financial Input
       ↓
Expert Analysis
```

For example:

```text
Input:
Revenue = $10M
Debt = $8M
Cash = $1M

Expected Output:
Risk level: High
Reason: High debt-to-cash ratio
```

Now:

```text
General Model
      +
Financial Examples
      ↓
Fine-tuning
      ↓
Financial Specialized Model
```

---

## Stage 3: Inference

A real user asks:

```text
Analyze this company's financial risk.
```

At runtime:

```text
User Query
     ↓
Financial Data / RAG
     ↓
Fine-tuned or Base Model
     ↓
Inference
     ↓
Answer
```

---

# 8. Where does RAG fit?

RAG is usually not a training stage.

It works mainly during **inference**:

```text
                    TRAINING
              ┌────────────────┐
              │                │
              ▼                ▼
         Pre-training      Fine-tuning
              │                │
              └───────┬────────┘
                      ▼
                 Trained LLM
                      │
                      │
                INFERENCE TIME
                      │
                      ▼
                 User Question
                      │
                      ▼
                     RAG
                      │
                      ▼
             Retrieve Information
                      │
                      ▼
             Context + User Query
                      │
                      ▼
                    LLM
                      │
                      ▼
                   Response
```

This is a very important distinction:

> **Pre-training and fine-tuning are training processes. RAG usually provides additional context during inference.**

---

# 9. Interview answer

If asked:

> **What is the difference between pre-training, fine-tuning, and inference?**

You can say:

> "Pre-training is the initial large-scale training phase where an LLM learns general language patterns, knowledge, and capabilities from massive datasets. Fine-tuning starts with that pre-trained model and further trains it on smaller, task-specific or domain-specific data to specialize its behavior. Inference is the runtime phase where we use the trained model to process new inputs and generate predictions without normally updating its weights. So, pre-training builds the general model, fine-tuning specializes it, and inference uses it."

## Final memory trick

```text
PRE-TRAINING
Learn everything general
        ↓
FINE-TUNING
Learn a specific job
        ↓
INFERENCE
Do the job
```
