# Fine-tuning vs RAG vs Prompt Engineering — When Would You Use Each?

This is one of the **most important GenAI interview topics**. The key is to understand that these techniques solve **different problems**.

## The simplest way to remember

> **Prompt Engineering = Tell the model what to do**
> **RAG = Give the model the information it needs**
> **Fine-tuning = Teach the model how to behave**

---

# 1. High-level comparison

| Technique          | Main purpose                         | Changes model weights? | Uses external knowledge? |
| ------------------ | ------------------------------------ | ---------------------: | -----------------------: |
| Prompt Engineering | Control behavior for a request       |                   ❌ No |                 Optional |
| RAG                | Provide relevant, current knowledge  |                   ❌ No |                    ✅ Yes |
| Fine-tuning        | Specialize model behavior/capability |                  ✅ Yes |            Not primarily |

---

# 2. Prompt Engineering

## What is it?

Prompt engineering means designing effective instructions that guide the LLM.

Example:

```text
You are a financial assistant.

Rules:
1. Explain in simple language.
2. Do not make up facts.
3. Return the answer in JSON.
4. If information is missing, say "Insufficient information".
```

The model's weights do not change.

Every time you call the LLM:

```text
Prompt + User Input → LLM → Response
```

## When should you use it?

Use prompt engineering when:

* The base model already has the required capability.
* You only need to control behavior.
* Requirements change frequently.
* You want to experiment quickly.
* You don't have enough training data.
* The task doesn't require new knowledge.
* A few instructions/examples can solve the problem.

### Example

You want:

```text
User: Summarize this document.

Output:
- Maximum 5 bullet points
- Simple English
- Include risks
```

You don't need fine-tuning.

A prompt is enough.

## Few-shot prompting

You can also provide examples:

```text
Input: "I was charged twice"
Output: Billing

Input: "My account is locked"
Output: Account

Input: "The app crashes"
Output: Technical

Input: "I cannot login"
Output:
```

The model learns the pattern **within the current context**, but its weights are not permanently changed.

---

# 3. RAG — Retrieval-Augmented Generation

## What problem does RAG solve?

LLMs have limited or outdated knowledge.

Suppose you ask:

> What is my company's current leave policy?

The LLM probably does not know your private company policy.

You could put the entire policy into a prompt:

```text
Prompt:
[100 pages of company documents]

Question:
What is the leave policy?
```

But this becomes expensive and inefficient.

Instead, use RAG.

## RAG architecture

```text
                    ┌─────────────────────┐
                    │ Company Documents   │
                    │ PDFs / DB / Wiki    │
                    └──────────┬──────────┘
                               │
                               ▼
                         Chunk Documents
                               │
                               ▼
                           Embeddings
                               │
                               ▼
                          Vector Database
                         (Qdrant, etc.)
                               │
                               │
User Question ──────────────────┤
                               ▼
                           Retriever
                               │
                               ▼
                      Relevant Documents
                               │
                               ▼
                     Prompt + Retrieved Data
                               │
                               ▼
                              LLM
                               │
                               ▼
                            Answer
```

The key idea:

> Instead of changing the model's knowledge, we retrieve relevant information at runtime.

---

## When should you use RAG?

Use RAG when information is:

### 1. Frequently changing

For example:

```text
Latest product prices
Current company policies
Today's inventory
Latest financial reports
```

You don't want to retrain your model every time data changes.

Instead:

```text
Update Documents
      ↓
Update Vector Database
      ↓
RAG automatically retrieves new information
```

---

### 2. Private

Example:

```text
Internal company documents
Employee policies
Customer information
Private contracts
Enterprise knowledge bases
```

You don't need to put private information inside model weights.

---

### 3. Too large to include in every prompt

Suppose your company has:

```text
10,000 PDFs
100 million documents
Multiple databases
```

You retrieve only:

```text
Top 5 relevant chunks
```

Instead of sending everything to the LLM.

---

# 4. Fine-tuning

## What problem does fine-tuning solve?

Fine-tuning helps when you want the model to consistently learn a **specific behavior or task**.

Example:

You have 100,000 historical customer-support interactions:

```text
Customer Input
       ↓
Correct Classification
       ↓
Correct Response
```

Example:

```text
Input:
"I was charged twice"

Output:
{
    "intent": "duplicate_charge",
    "priority": "high",
    "team": "billing"
}
```

You can train the model on thousands of examples.

The model then learns patterns from your data.

---

## When should you fine-tune?

### 1. Highly repetitive tasks

Suppose you always want:

```text
Input → Specific Output
```

For example:

```text
Customer email → Classification
Medical text → Structured fields
Legal document → Contract categories
Support ticket → Intent + priority
```

Fine-tuning can work very well.

---

### 2. Consistent output behavior is required

Suppose you need:

```json
{
  "risk_level": "HIGH",
  "reason": "...",
  "recommended_action": "..."
}
```

Prompting might work 95% of the time.

But if you need very consistent behavior at scale, fine-tuning can improve reliability—assuming you have enough high-quality, representative training data.

---

### 3. Domain-specific language or style

Examples:

```text
Legal language
Financial terminology
Company-specific writing style
Specialized coding patterns
Customer support tone
```

---

### 4. You have a large amount of high-quality labeled data

For example:

```text
50,000 examples

Input:
Customer issue

Output:
Correct resolution
```

Good fine-tuning data can improve specialized performance.

---

# 5. The biggest mistake: using fine-tuning for knowledge updates

Suppose your company policy changes:

```text
Old policy:
20 vacation days

New policy:
25 vacation days
```

Should you fine-tune?

Usually, **no**.

Why?

Because you would need to:

```text
Collect data
      ↓
Prepare dataset
      ↓
Fine-tune model
      ↓
Evaluate model
      ↓
Deploy model
```

Instead:

```text
Update policy document
      ↓
Re-index it
      ↓
RAG retrieves the new policy
```

This is why:

> **RAG is generally better for knowledge. Fine-tuning is generally better for behavior.**

---

# 6. Real-world decision table

| Situation                       | Best approach                             | Why                               |
| ------------------------------- | ----------------------------------------- | --------------------------------- |
| Change response format          | Prompting                                 | Simple instruction                |
| Summarize a document            | Prompting                                 | Model already knows summarization |
| Company internal knowledge      | RAG                                       | Private/external knowledge        |
| Frequently changing data        | RAG                                       | No retraining required            |
| Latest financial data           | RAG + tools                               | Need current information          |
| Consistent classification       | Fine-tuning                               | Learn stable patterns             |
| Specific writing style          | Fine-tuning or prompting                  | Depends on consistency needed     |
| New task with little data       | Prompting                                 | Fast and cheap                    |
| Millions of repetitive requests | Fine-tuning                               | Can reduce prompt overhead        |
| Need company document answers   | RAG                                       | Retrieve relevant documents       |
| Need exact output schema        | Prompting first, fine-tuning if necessary | Start simple                      |

---

# 7. A practical decision flow

Use this in an interview:

```text
                 What is the problem?
                         │
                         ▼
              Does the model lack knowledge?
                   /              \
                 Yes               No
                  │                 │
                  ▼                 ▼
          Is knowledge current,    Do I only need
          private or changing?     instructions?
                  │                 │
                 Yes               Yes
                  │                 │
                  ▼                 ▼
                 RAG             Prompting
                  │                 │
                  │                 ▼
                  │         Is performance
                  │         consistent enough?
                  │                 │
                  │            Yes / No
                  │                 │
                  │                 ▼
                  │              Fine-tuning
                  │
                  ▼
        Combine RAG + Fine-tuning
        if both knowledge and
        specialized behavior are needed
```

---

# 8. Real-world example: Enterprise Financial AI Assistant

Suppose you're building an AI financial assistant.

User asks:

> What are the risks in my investment portfolio?

## Step 1: Prompt engineering

```text
You are a financial analysis assistant.

Rules:
- Use only the provided data.
- Do not invent financial information.
- Clearly explain risk.
- Return structured JSON.
```

This controls behavior.

---

## Step 2: RAG

Retrieve:

```text
Portfolio documents
Risk policies
Financial reports
Investment research
```

The model receives:

```text
User Question
+
Relevant Portfolio Data
+
Relevant Financial Documents
```

This provides knowledge.

---

## Step 3: Fine-tuning

Suppose you have 100,000 examples:

```text
Portfolio Data
       ↓
Expert Risk Analysis
```

You can fine-tune for:

* Risk classification patterns
* Response style
* Structured output
* Domain-specific task behavior

The complete system becomes:

```text
                         USER
                           │
                           ▼
                     API / FastAPI
                           │
             ┌─────────────┴─────────────┐
             ▼                           ▼
       Prompt Engineering              RAG
             │                           │
             │                     Retriever
             │                           │
             │                     Vector DB
             │                           │
             └─────────────┬─────────────┘
                           │
                           ▼
                    Fine-Tuned LLM
                           │
                           ▼
                        Response
```

---

# 9. Can you combine all three?

Absolutely. In production systems, this is common.

```text
                    ┌─────────────────────┐
                    │ Prompt Engineering  │
                    │ Behavior + Rules    │
                    └──────────┬──────────┘
                               │
                               ▼
User Query ───────────────► Application
                               │
                               ▼
                    ┌─────────────────────┐
                    │ RAG                 │
                    │ Retrieve Knowledge  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Fine-Tuned Model    │
                    │ Specialized Skills  │
                    └──────────┬──────────┘
                               │
                               ▼
                            Response
```

Think of them as complementary:

```text
Prompting
    ↓
Controls behavior

RAG
    ↓
Provides knowledge

Fine-tuning
    ↓
Improves specialized behavior
```

---

# 10. What I would do in a real production project

My approach would usually be:

### Phase 1: Start with prompting

```text
System prompt
+
Few-shot examples
+
Structured output
```

Evaluate:

```text
Accuracy
Latency
Cost
Hallucination rate
Format compliance
```

---

### Phase 2: Add RAG if knowledge is the problem

```text
Documents
   ↓
Chunking
   ↓
Embeddings
   ↓
Vector DB
   ↓
Hybrid Retrieval
   ↓
Reranking
   ↓
LLM
```

Evaluate:

```text
Context precision
Context recall
Faithfulness
Answer relevance
```

---

### Phase 3: Fine-tune if behavior remains the problem

Only after identifying a repeatable failure pattern.

For example:

```text
Prompting accuracy: 80%
       ↓
Prompt improvement: 86%
       ↓
RAG: not relevant because knowledge is not the issue
       ↓
Fine-tuning with 50,000 high-quality examples
       ↓
Accuracy: potentially improves
```

The exact improvement must always be measured on a held-out evaluation set; fine-tuning is not guaranteed to improve every task.

---

# 11. Best interview answer

If an interviewer asks:

> **Fine-tuning vs RAG vs prompt engineering — when would you use each?**

You can answer:

> "I see them as solving different problems. Prompt engineering is my first choice when the base model already has the required capability and I simply need to guide its behavior. I use RAG when the problem is missing, private, large, or frequently changing knowledge, because RAG retrieves the latest relevant information without retraining the model. I use fine-tuning when I need consistent specialized behavior and I have enough high-quality examples of the task. In practice, I usually start with prompting, add RAG when the model needs external knowledge, and consider fine-tuning only when there is a measurable, repeatable behavior gap that prompting and retrieval cannot solve."

## Final memory trick

> **Prompt Engineering → Instructions**
> **RAG → Knowledge**
> **Fine-tuning → Learned behavior**

A good production AI engineer does **not automatically choose fine-tuning**. They first identify whether the actual problem is **instruction-following, missing knowledge, or specialized behavior**.
