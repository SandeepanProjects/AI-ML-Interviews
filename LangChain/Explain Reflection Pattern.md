The **Reflection Pattern** is one of the most important reasoning patterns used in **Agentic AI**. It is widely used in production systems to improve answer quality by allowing an AI to **critique and improve its own output before returning it to the user**.

Companies like OpenAI, Anthropic, Microsoft, Google, and many enterprise AI teams use variations of this pattern for:

* Code generation
* RAG systems
* Report generation
* Financial analysis
* Legal document drafting
* SQL generation
* Multi-agent systems

---

# What is Reflection?

Instead of generating one answer and immediately returning it:

```text
Question

↓

LLM

↓

Answer
```

The Reflection Pattern introduces a review step:

```text
Question

↓

Generate Draft

↓

Critique Draft

↓

Improve Draft

↓

Final Answer
```

The model (or another model) acts as its own reviewer.

---

# Why Do We Need Reflection?

Suppose the user asks:

> Write a Python function to reverse a linked list.

Without reflection:

```text
Question

↓

LLM

↓

Code

↓

Return
```

The code may contain:

* Bugs
* Missing edge cases
* Poor variable names
* Incorrect complexity
* Missing documentation

With reflection:

```text
Question

↓

Draft Code

↓

Review Code

↓

Improve Code

↓

Return Better Code
```

---

# Real-World Example

User asks:

> Explain LangGraph.

First draft:

```text
LangGraph is a framework for AI workflows.
```

Reflection:

```text
Problems:

- Too short
- No architecture
- No examples
- Doesn't explain state
```

Improved answer:

```text
LangGraph is a framework for building
stateful AI applications.

It supports:

• Multi-agent workflows
• Conditional routing
• Checkpointing
• Human approval
• Persistent memory

Example:
Planner → Retriever → LLM → Human → END
```

The second answer is much more useful.

---

# Reflection Workflow

```text
            User Question
                  │
                  ▼
           Generator Agent
                  │
                  ▼
              Draft Answer
                  │
                  ▼
          Reflection Agent
                  │
      Good? ──────┴─────── No
       │                   │
       ▼                   ▼
 Return Answer      Improve Draft
                          │
                          ▼
                   Final Answer
```

---

# Reflection vs Simple Prompting

Without reflection:

```python
answer = llm.invoke(question)
```

With reflection:

```python
draft = llm.invoke(question)

feedback = llm.invoke(
    f"Critique this:\n{draft}"
)

final = llm.invoke(
    f"""
Improve this answer.

Draft:
{draft}

Feedback:
{feedback}
"""
)
```

Now the answer is reviewed before being returned.

---

# Reflection Prompts

### Generator Prompt

```text
You are an expert software engineer.

Answer the user's question.
```

---

### Reflection Prompt

```text
You are a senior reviewer.

Review the answer.

Check:

• Correctness

• Missing details

• Hallucinations

• Clarity

• Security

• Best practices

List all issues.
```

---

### Improvement Prompt

```text
Improve the answer using the review.

Return only the improved answer.
```

---

# LangGraph Implementation

## State

```python
from typing import TypedDict

class ReflectionState(TypedDict):
    question: str
    draft: str
    feedback: str
    final_answer: str
```

---

## Generator Node

```python
def generate(state):

    draft = llm.invoke(state["question"])

    return {
        "draft": draft.content
    }
```

---

## Reflection Node

```python
def reflect(state):

    prompt = f"""
Review this answer.

{state['draft']}
"""

    feedback = llm.invoke(prompt)

    return {
        "feedback": feedback.content
    }
```

---

## Improve Node

```python
def improve(state):

    prompt = f"""
Improve:

{state['draft']}

Feedback:

{state['feedback']}
"""

    answer = llm.invoke(prompt)

    return {
        "final_answer": answer.content
    }
```

---

## LangGraph

```text
START

↓

Generate

↓

Reflect

↓

Improve

↓

END
```

---

# Conditional Reflection

Sometimes the first draft is already good.

```text
Generate

↓

Reflection Score

↓

> 0.9 ?

↓

Yes

↓

Return

------------

No

↓

Improve
```

Router:

```python
def router(state):

    if state["score"] > 0.9:
        return "end"

    return "improve"
```

This saves cost and latency.

---

# Multi-Agent Reflection

Instead of one model reviewing itself:

```text
Generator Agent

↓

Reviewer Agent

↓

Security Agent

↓

Final Editor

↓

Return
```

Example:

* Generator → writes SQL
* Security agent → checks for SQL injection
* Reviewer → checks correctness
* Editor → formats output

---

# Reflection in RAG

```text
Question

↓

Retriever

↓

Answer

↓

Reflection

↓

Enough Evidence?

↓

No

↓

Retrieve Again

↓

Generate Again
```

This reduces hallucinations by ensuring the answer is supported by retrieved documents.

---

# Reflection for Code Generation

```text
Prompt

↓

Write Code

↓

Run Tests

↓

Tests Failed

↓

Reflect

↓

Fix Code

↓

Run Tests

↓

Success
```

This is a common pattern in coding agents.

---

# Reflection for SQL

```text
Question

↓

Generate SQL

↓

Review SQL

↓

Execute

↓

Correct?

↓

Return
```

The reviewer checks for:

* Missing filters
* Inefficient joins
* Syntax issues
* Security problems

---

# Production Architecture

```text
                    User
                      │
                      ▼
                 LangGraph
                      │
                      ▼
               Generator Agent
                      │
                      ▼
                 Draft Answer
                      │
                      ▼
               Reflection Agent
          ┌───────────┴───────────┐
          ▼                       ▼
     High Quality          Needs Improvement
          │                       │
          ▼                       ▼
     Return Answer          Improve Answer
                                     │
                                     ▼
                               Final Response
```

---

# Advantages

* Higher answer quality
* Fewer hallucinations
* Better code generation
* Better reasoning
* More complete responses
* Stronger adherence to instructions
* Better factual consistency

---

# Disadvantages

* Higher latency
* More token usage
* Increased cost
* More complex orchestration
* Risk of over-editing if the reviewer is too aggressive

---

# Reflection vs ReAct

| Reflection                   | ReAct                          |
| ---------------------------- | ------------------------------ |
| Reviews its own answer       | Chooses tools and actions      |
| Focuses on quality           | Focuses on task execution      |
| Improves drafts              | Solves problems step by step   |
| Often uses no external tools | Frequently uses external tools |

Many production systems combine both.

---

# Reflection vs Planning

| Planning           | Reflection              |
| ------------------ | ----------------------- |
| Before execution   | After generation        |
| Decides what to do | Evaluates what was done |
| Creates a strategy | Improves the result     |

---

# Interview Follow-Up Questions

## 1. Can the same model perform reflection?

Yes. A single model can generate, critique, and improve. However, some systems use a smaller model for review to reduce cost, or a stronger model for reviewing high-value outputs.

---

## 2. How many reflection rounds should you allow?

Usually **one or two**.

Too many rounds:

* Increase latency
* Increase token cost
* May lead to diminishing returns

Always enforce a maximum iteration count.

---

## 3. How do you know whether reflection is needed?

Use:

* A quality score from a reviewer
* Confidence thresholds
* Rule-based checks (e.g., missing citations)
* Test execution (for code)
* Human review for critical workflows

---

## 4. Where is reflection most useful?

Reflection provides the greatest value in:

* Code generation
* Long-form writing
* RAG
* SQL generation
* Financial reports
* Legal analysis
* Multi-step planning

---

## 5. How would you make reflection production-ready?

A mature implementation includes:

* Maximum reflection iterations
* Structured review output (for example, using Pydantic)
* Quality scoring
* Cost and latency budgets
* Retry and timeout handling
* Observability and tracing
* Human escalation for high-risk tasks

---

# Complete Enterprise Reflection Architecture

```text
                      User Question
                            │
                            ▼
                     Planner (Optional)
                            │
                            ▼
                     Generator Agent
                            │
                            ▼
                      Draft Response
                            │
                            ▼
                     Reflection Agent
                            │
                 Quality Score / Review
                  ┌─────────┴─────────┐
                  ▼                   ▼
            High Quality      Needs Improvement
                  │                   │
                  ▼                   ▼
           Return Answer     Improvement Agent
                                      │
                                      ▼
                              Final Validation
                                      │
                                      ▼
                                 User Response
```

## Senior AI Engineer interview tip

When explaining the Reflection Pattern, don't describe it as simply "asking the model twice." Emphasize that it is a **quality assurance workflow**:

1. Generate an initial draft.
2. Critique it against explicit quality criteria.
3. Revise the draft using the critique.
4. Stop after a bounded number of iterations or when a quality threshold is met.

In production systems, reflection is often combined with **RAG**, **tool use**, **human approval**, and **automated validation** (such as unit tests or schema validation) to produce more reliable AI applications.
