## 1. What is fine-tuning of an LLM?

**Fine-tuning** means taking an already pre-trained Large Language Model (LLM) and training it further on a **specific dataset** so that it becomes better at a particular task, domain, style, or behavior.

### Simple idea

A general LLM:

> Pre-trained LLM → knows language + general knowledge

Fine-tuned LLM:

> Pre-trained LLM + your domain-specific training examples → specialized model

For example, a base model might know general English and programming. You can fine-tune it to become better at:

* Customer support
* Medical document classification
* Legal document extraction
* Financial analysis
* Generating company-specific responses
* Structured JSON generation
* Code generation in a particular codebase/style

### Example

Suppose you have a customer-support application.

Without fine-tuning, you prompt:

```text
You are a customer support agent.
Always be polite.
Classify the issue as:
- billing
- technical
- account
Return JSON.
```

You must send these instructions repeatedly.

With fine-tuning, you train the model on examples:

```json
{
  "input": "I was charged twice for my subscription",
  "output": {
    "category": "billing",
    "priority": "high"
  }
}
```

After seeing thousands of similar examples, the model learns the desired pattern.

---

# 2. How does fine-tuning technically work?

During pre-training, an LLM learns by predicting the next token:

```text
The capital of France is → Paris
```

The model has billions of parameters:

```text
weights / parameters
```

During fine-tuning, we continue training using a smaller, task-specific dataset.

```text
Base LLM
    ↓
Domain-specific examples
    ↓
Calculate prediction error (loss)
    ↓
Backpropagation
    ↓
Update model parameters
    ↓
Fine-tuned LLM
```

Conceptually:

```text
Pre-trained model
        +
Your training dataset
        ↓
Fine-tuning
        ↓
Specialized model
```

---

# 3. Example of a fine-tuning dataset

Suppose we want to fine-tune a model for sentiment analysis.

### Training examples

```json
[
  {
    "input": "The product is amazing",
    "output": "positive"
  },
  {
    "input": "The product stopped working",
    "output": "negative"
  },
  {
    "input": "The delivery was okay",
    "output": "neutral"
  }
]
```

For instruction tuning, data often looks like:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a sentiment classifier."
    },
    {
      "role": "user",
      "content": "The product is amazing."
    },
    {
      "role": "assistant",
      "content": "positive"
    }
  ]
}
```

The model learns:

```text
Input → Expected Output
```

---

# 4. Why fine-tune instead of just using prompting?

This is one of the most important interview questions.

The answer is:

> **Prompting tells the model what to do at inference time. Fine-tuning changes the model's learned behavior through training.**

Let's compare them.

| Prompt Engineering            | Fine-Tuning                             |
| ----------------------------- | --------------------------------------- |
| Instructions given at runtime | Model behavior is trained               |
| No weight updates             | Some model parameters are updated       |
| Fast to implement             | Requires training data                  |
| Flexible                      | More specialized                        |
| Instructions consume tokens   | Less repeated instruction may be needed |
| Good for changing tasks       | Good for repeated stable tasks          |
| Cheap initially               | Training has upfront cost               |
| General behavior              | Consistent specialized behavior         |

---

# 5. When is prompting better?

Suppose you want the model to behave differently depending on the request.

### Prompt

```text
You are an AI assistant.

Task:
Summarize the following document.

Rules:
1. Use bullet points.
2. Do not hallucinate.
3. Keep the answer under 100 words.
```

This is a good use of prompting.

Why?

Because:

* Requirements may change
* No training data is required
* The task is simple
* The model already has the necessary capability

You should **not fine-tune a model just because prompting exists**.

In real-world AI engineering, the usual approach is:

```text
Prompting
   ↓
RAG if knowledge is missing
   ↓
Fine-tuning if behavior still isn't reliable
```

---

# 6. When is fine-tuning better?

## Case 1: You need consistent output

Suppose an enterprise application requires this JSON:

```json
{
  "intent": "refund_request",
  "priority": "high",
  "department": "billing"
}
```

You could repeatedly send a long prompt:

```text
You are a classifier.
Analyze the user message.
Return exactly this schema.
Use one of these 20 intents.
Follow these business rules...
```

If this happens millions of times, fine-tuning may help produce more consistent behavior.

---

## Case 2: Domain-specific style or behavior

Suppose your company has thousands of examples:

```text
Customer question → Expert company response
```

For example:

```text
Question:
How do I reset my account?

Expected answer:
To reset your account, follow these steps...
```

Fine-tuning can teach the model:

* Tone
* Writing style
* Response format
* Task-specific reasoning patterns
* Classification behavior

---

## Case 3: You want to use a smaller model

Imagine:

```text
Large Model
70B parameters
```

with a huge prompt:

```text
10,000 tokens of instructions
```

This may be expensive.

Instead:

```text
Small Model
8B parameters
+
Fine-tuning
```

could perform a narrow task efficiently.

This can reduce:

* Latency
* Inference cost
* Prompt size
* Infrastructure requirements

For high-volume production systems, this can be important.

---

# 7. Important: Fine-tuning is NOT the solution for adding knowledge

This is a very common interview question.

Suppose your company wants an LLM to answer:

> What was our company's revenue last quarter?

You generally should **not fine-tune the model every quarter**.

Why?

Because the information changes.

Instead, use:

```text
User Question
      ↓
Retriever
      ↓
Latest Company Documents / Database
      ↓
Relevant Context
      ↓
LLM
      ↓
Answer
```

This is **RAG**.

### Example

For frequently changing data:

```text
Company policies
Financial reports
Product inventory
Customer records
Current documentation
```

Use:

> **RAG**

For learning behavior:

```text
Response style
Classification
Output format
Domain-specific patterns
Task behavior
```

Consider:

> **Fine-tuning**

---

# 8. Prompting vs RAG vs Fine-tuning

This is the best way to explain it in an interview.

### Prompting

Use when:

> The model already knows how to perform the task, but you need to tell it how to behave.

Example:

```text
Summarize this document in 5 bullet points.
```

---

### RAG

Use when:

> The model needs external, private, or frequently changing knowledge.

Example:

```text
What is the latest company leave policy?
```

Flow:

```text
Question
   ↓
Search Company Documents
   ↓
Retrieve Relevant Context
   ↓
LLM
   ↓
Answer
```

---

### Fine-tuning

Use when:

> The model needs to learn a specialized behavior consistently.

Example:

```text
Customer Message
       ↓
Fine-tuned Model
       ↓
Intent + Priority + Department
```

---

# 9. Real-world example

Imagine you're building an **Enterprise Financial Advisor Copilot**.

A user asks:

```text
Which investments in my portfolio are high risk?
```

You might use all three techniques.

### Step 1: RAG

Retrieve:

```text
Portfolio data
Investment documents
Market data
Risk policies
```

### Step 2: Prompting

```text
You are a financial analysis assistant.

Analyze only the provided data.
Do not invent information.
Explain risk clearly.
```

### Step 3: Fine-tuning

You may fine-tune the model using thousands of examples of:

```text
Financial Input
      ↓
Expected Structured Analysis
```

The architecture becomes:

```text
                    ┌──────────────┐
User Query ────────►│     RAG      │
                    │ Retrieval    │
                    └──────┬───────┘
                           ↓
                    ┌──────────────┐
                    │    Prompt    │
                    │  Engineering │
                    └──────┬───────┘
                           ↓
                    ┌──────────────┐
                    │ Fine-Tuned   │
                    │     LLM      │
                    └──────┬───────┘
                           ↓
                       Response
```

---

# 10. Simple interview answer

If an interviewer asks:

> **Why would you fine-tune an LLM instead of using prompting?**

You can answer:

> "Prompting is used to guide an LLM's behavior at inference time without changing the model weights. Fine-tuning is useful when I need the model to consistently perform a specialized task, follow a particular style, produce a specific output format, or learn patterns from a large number of domain-specific examples. I would usually start with prompting because it's faster and cheaper. If the task requires frequently changing knowledge, I would use RAG rather than fine-tuning. I would choose fine-tuning when prompt engineering and retrieval are insufficient and I have enough high-quality labeled data to teach stable behavior."

### One-line memory trick

> **Prompting = tell the model what to do.**
> **RAG = give the model knowledge.**
> **Fine-tuning = teach the model how to behave.**
