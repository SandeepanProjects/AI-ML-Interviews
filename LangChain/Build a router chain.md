This is a **very common Senior AI Engineer interview question**.

Interviewers want to evaluate whether you understand:

* Multi-LLM architectures
* Cost optimization
* Latency optimization
* Model routing
* Fallback strategies
* Production AI system design

In real companies, **one model is rarely used for every request**.

Examples:

* Simple FAQs → Small, inexpensive model
* Coding → GPT-4.1
* Long document summarization → Claude
* Vision → GPT-4.1 Vision
* Translation → Specialized model

---

# Problem Statement

Build an AI assistant that intelligently chooses the best model.

Example

```text
User: What is 2+2?

↓

Cheap model

↓

Answer
```

---

```text
User: Write a distributed consensus algorithm in Python

↓

GPT-4.1

↓

Answer
```

---

```text
User: Summarize this 500-page PDF

↓

Claude

↓

Answer
```

---

# High-Level Architecture

```text
                    User
                      │
                      ▼
                Router Chain
          ┌───────────┼────────────┐
          ▼           ▼            ▼
      GPT-4.1     GPT-4.1-mini   Claude
          │           │            │
          └───────────┼────────────┘
                      ▼
                Final Response
```

The router's responsibility is to determine which model is most appropriate.

---

# Option 1: Rule-Based Router

The simplest approach is to use rules.

```python
from langchain_openai import ChatOpenAI

cheap_llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0
)

smart_llm = ChatOpenAI(
    model="gpt-4.1",
    temperature=0
)

def route(question: str):

    question = question.lower()

    if "python" in question:
        return smart_llm

    if "algorithm" in question:
        return smart_llm

    if len(question) < 50:
        return cheap_llm

    return smart_llm
```

Invoke

```python
llm = route(user_question)

response = llm.invoke(user_question)
```

---

# Execution Flow

```text
Question

↓

Router

↓

Is it coding?

↓

Yes

↓

GPT-4.1

↓

Answer
```

Otherwise

```text
Question

↓

Router

↓

Simple?

↓

Yes

↓

GPT-4.1-mini

↓

Answer
```

---

# Option 2: LLM-as-a-Router

Instead of hardcoded rules, use an LLM to classify requests.

Architecture

```text
User

↓

Router LLM

↓

Returns

coding

↓

GPT-4.1
```

Router Prompt

```text
You are a routing assistant.

Return exactly one option.

Options:

- cheap
- coding
- summarization

Question:

{question}
```

Implementation

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

router_prompt = ChatPromptTemplate.from_template("""
You are a routing assistant.

Choose only one category:

cheap
coding
summary

Question:
{question}
""")

router_chain = (
    router_prompt
    | cheap_llm
    | StrOutputParser()
)
```

Routing

```python
category = router_chain.invoke(
    {
        "question": user_question
    }
)

if category == "coding":
    llm = smart_llm

elif category == "summary":
    llm = claude

else:
    llm = cheap_llm
```

---

# Option 3: LCEL Router

LangChain Expression Language makes routing composable.

```python
from langchain_core.runnables import RunnableLambda

def choose_llm(inputs):

    q = inputs["question"]

    if "python" in q.lower():
        return smart_llm

    return cheap_llm

router = RunnableLambda(choose_llm)
```

Invoke

```python
selected_model = router.invoke(
    {
        "question": question
    }
)

response = selected_model.invoke(question)
```

---

# Option 4: Cost-Based Router

Many companies optimize for cost first.

Example pricing (illustrative)

```text
gpt-4.1-mini     ₹
gpt-4.1          ₹₹₹₹
Claude           ₹₹₹
```

Routing policy

```text
Simple Questions

↓

Cheap model
```

Complex

```text
Reasoning

↓

Expensive model
```

Architecture

```text
                  Question
                      │
                      ▼
            Complexity Estimator
                      │
          ┌───────────┴───────────┐
          ▼                       ▼
      Complexity < 4         Complexity > 4
          ▼                       ▼
    GPT-4.1-mini             GPT-4.1
```

---

# Option 5: Latency-Based Router

Some applications prioritize response time.

```text
Need answer in under 1 second?

↓

Small model

↓

Done
```

If not

```text
Need highest quality?

↓

Large model
```

---

# Option 6: Multi-Step Routing

Example

User

```text
Analyze this image and summarize it.
```

Flow

```text
Vision Model

↓

Image Description

↓

Claude

↓

Summary
```

Another example

```text
Translate this document

↓

Translation Model

↓

Grammar Model

↓

Final
```

---

# Fallback Router

Production systems should handle model failures.

```python
def ask(question):

    try:
        return gpt4.invoke(question)

    except Exception:

        return llama.invoke(question)
```

Architecture

```text
            User
              │
              ▼
          GPT-4.1
              │
       Success?
      ┌─────┴─────┐
     Yes         No
      │           │
      ▼           ▼
   Response     Llama
                  │
                  ▼
              Response
```

---

# Production Router

```python
def select_model(question: str):

    if is_math(question):
        return math_model

    if is_code(question):
        return code_model

    if is_translation(question):
        return translation_model

    if is_long_document(question):
        return claude

    return gpt41
```

---

# Enterprise Architecture

```text
                    User
                      │
                      ▼
               API Gateway
                      │
                      ▼
             Request Router
                      │
      ┌───────────────┼─────────────────┐
      ▼               ▼                 ▼
 GPT-4.1-mini      GPT-4.1          Claude
      │               │                 │
      └───────────────┼─────────────────┘
                      ▼
             Response Aggregator
                      ▼
                 Final Answer
```

---

# LangGraph Router

Routing is especially clean in LangGraph because it supports conditional edges.

State

```python
from typing import TypedDict

class AgentState(TypedDict):
    question: str
    route: str
    answer: str
```

Router Node

```python
def router_node(state):

    question = state["question"].lower()

    if "python" in question:
        return {
            "route": "code"
        }

    return {
        "route": "general"
    }
```

Conditional Routing

```text
            Router
               │
      ┌────────┴────────┐
      ▼                 ▼
  Code Model      General Model
      │                 │
      └────────┬────────┘
               ▼
            Finish
```

---

# Adding Monitoring

Track routing decisions in production.

```python
logger.info({
    "question": question,
    "selected_model": model_name,
    "latency_ms": latency,
    "tokens": token_count,
    "cost": cost
})
```

Useful metrics

* Selected model
* Router confidence
* Input/output tokens
* End-to-end latency
* Cost per request
* Failure rate
* Fallback count
* User satisfaction (if available)

---

# Interview Follow-Up Questions

## 1. Why not always use the most powerful model?

Because it is often:

* More expensive
* Higher latency
* Unnecessary for simple requests
* Lower throughput under heavy load

Routing improves both cost and scalability.

---

## 2. How do you classify a query?

Common approaches:

* Rule-based (keywords, length)
* Lightweight classifier model
* LLM-as-a-router
* Embedding similarity against predefined intent examples
* Fine-tuned intent classifier for high-volume systems

---

## 3. What if the router chooses the wrong model?

Mitigation strategies:

* Retry with a stronger model when confidence is low.
* Escalate if the answer fails validation.
* Use confidence thresholds.
* Collect feedback to improve routing logic.

---

## 4. Can one request use multiple models?

Yes. For example:

```text
User uploads a financial report

        │
        ▼
OCR Model
        │
        ▼
LLM extracts KPIs
        │
        ▼
Reasoning model analyzes trends
        │
        ▼
Smaller model formats the report
```

This is common in enterprise AI pipelines.

---

# Production Best Practices

A production-grade router typically includes:

* **Intent detection** to understand the task.
* **Complexity estimation** (reasoning depth, context size).
* **Cost-aware routing** to minimize unnecessary spend.
* **Latency-aware routing** for interactive applications.
* **Capability-based routing** (code, vision, multilingual, long context).
* **Health checks and fallback models** when a provider is unavailable.
* **Observability** with traces, token usage, latency, and routing decisions.
* **A/B testing** to compare routing policies and model performance over time.

In senior AI engineering interviews, a strong answer goes beyond code: explain **why** the router exists, **how** routing decisions are made, **how** failures are handled, and **how** you monitor and continuously improve routing in production.
