This is one of the **most frequently asked LangGraph interview questions** for Senior AI Engineer and Staff AI Engineer roles.

Interviewers want to evaluate whether you understand:

* State management
* Nodes
* Edges
* Conditional routing
* Cycles
* Multi-agent workflows
* Production orchestration

Unlike LangChain, **LangGraph is a state machine**. Every node receives a state, updates it, and passes it to the next node.

---

# Problem Statement

Build a workflow where:

* If the retrieved documents are good enough → Answer the user
* Otherwise → Retrieve more documents
* Repeat until good context is found

This is a common production RAG architecture.

---

# Workflow

```text
                 User Question
                       │
                       ▼
                  Retrieve Docs
                       │
                       ▼
                  Grade Context
                 /             \
        Good Context      Poor Context
             │                 │
             ▼                 ▼
       Generate Answer    Retrieve Again
             │                 │
             └─────────────────┘
```

This loop is **not possible with a simple LangChain chain**, but it is natural in LangGraph.

---

# Step 1: Install

```bash
pip install langgraph
pip install langchain-openai
```

---

# Step 2: Define State

Every node receives and returns this state.

```python
from typing import TypedDict, List

class AgentState(TypedDict):
    question: str
    documents: List[str]
    answer: str
    context_score: float
```

Example state

```python
{
    "question": "What is LangGraph?",
    "documents": [],
    "answer": "",
    "context_score": 0.0
}
```

---

# Step 3: Retrieval Node

```python
def retrieve(state: AgentState):

    docs = [
        "LangGraph supports stateful workflows.",
        "It enables conditional routing."
    ]

    return {
        "documents": docs
    }
```

Input

```python
{
    "question":"What is LangGraph?"
}
```

Output

```python
{
    "documents":[
        "LangGraph supports stateful workflows.",
        "It enables conditional routing."
    ]
}
```

---

# Step 4: Grading Node

Suppose we evaluate whether the retrieved documents are sufficient.

```python
def grade_documents(state: AgentState):

    docs = state["documents"]

    score = 0.9 if len(docs) >= 2 else 0.4

    return {
        "context_score": score
    }
```

---

# Step 5: Generation Node

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-4.1",
    temperature=0
)

def generate_answer(state: AgentState):

    context = "\n".join(state["documents"])

    prompt = f"""
Context:

{context}

Question:

{state['question']}
"""

    response = llm.invoke(prompt)

    return {
        "answer": response.content
    }
```

---

# Step 6: Conditional Function

This function decides which node executes next.

```python
def decide_next_node(state: AgentState):

    if state["context_score"] > 0.8:
        return "generate"

    return "retrieve"
```

If the score is high:

```text
Grade

↓

Generate
```

Otherwise

```text
Grade

↓

Retrieve Again
```

---

# Step 7: Build the Graph

```python
from langgraph.graph import StateGraph, END

workflow = StateGraph(AgentState)

workflow.add_node("retrieve", retrieve)
workflow.add_node("grade", grade_documents)
workflow.add_node("generate", generate_answer)
```

---

# Step 8: Connect Nodes

```python
workflow.set_entry_point("retrieve")

workflow.add_edge(
    "retrieve",
    "grade"
)
```

Conditional edge

```python
workflow.add_conditional_edges(
    "grade",
    decide_next_node,
    {
        "generate": "generate",
        "retrieve": "retrieve"
    }
)
```

Finish

```python
workflow.add_edge(
    "generate",
    END
)
```

Compile

```python
graph = workflow.compile()
```

---

# Step 9: Execute

```python
result = graph.invoke(
    {
        "question": "What is LangGraph?"
    }
)

print(result["answer"])
```

---

# Execution Flow

## First iteration

```text
Question

↓

Retrieve

↓

Grade

↓

Score = 0.4

↓

Retrieve Again
```

---

## Second iteration

```text
Retrieve

↓

More Documents

↓

Grade

↓

0.95

↓

Generate

↓

Finish
```

---

# Complete Workflow

```text
                 User Question
                       │
                       ▼
                Retrieve Node
                       │
                       ▼
                 Grade Node
                 /         \
        score > 0.8    score <= 0.8
             │               │
             ▼               ▼
       Generate Node    Retrieve Node
             │
             ▼
            END
```

---

# Internal State Evolution

### Initial

```python
{
    "question":"What is LangGraph?"
}
```

---

### After Retrieve

```python
{
    "question":"What is LangGraph?",
    "documents":[
        "LangGraph supports stateful workflows."
    ]
}
```

---

### After Grade

```python
{
    "question":"What is LangGraph?",
    "documents":[...],
    "context_score":0.45
}
```

---

### Second Retrieval

```python
{
    "documents":[
        "...",
        "...",
        "..."
    ]
}
```

---

### Final

```python
{
    "answer":"LangGraph is a framework for stateful workflows..."
}
```

---

# Multi-Agent Branching Example

Suppose we have different specialist agents.

```text
                    User
                      │
                      ▼
                   Router
          ┌───────────┼───────────┐
          ▼           ▼           ▼
     Code Agent  Finance Agent  HR Agent
          │           │           │
          └───────────┼───────────┘
                      ▼
                 Final Response
```

Router

```python
def router(state):

    q = state["question"].lower()

    if "python" in q:
        return "coder"

    if "invoice" in q:
        return "finance"

    return "general"
```

Conditional edges

```python
workflow.add_conditional_edges(
    "router",
    router,
    {
        "coder": "code_agent",
        "finance": "finance_agent",
        "general": "general_agent"
    }
)
```

---

# Human-in-the-Loop Branch

Production AI systems often require approval before high-risk actions.

```text
                Generate Answer
                       │
                       ▼
             Confidence > 0.9?
                 /           \
               Yes            No
               │              │
               ▼              ▼
              END     Human Approval
                              │
                              ▼
                            END
```

Decision function

```python
def confidence_router(state):

    if state["confidence"] > 0.9:
        return "finish"

    return "human_review"
```

---

# Retry Pattern

Another common production workflow.

```text
             API Call
                │
                ▼
           Success?
         /          \
       Yes          No
       │             │
       ▼             ▼
      END         Retry
                     │
                     ▼
                 Max Retries?
                 /          \
               No            Yes
               │             │
               ▼             ▼
           API Call      Fallback
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
              LangGraph
                    │
      ┌─────────────┼─────────────┐
      ▼             ▼             ▼
 Retriever      Tool Agent    Planner
      │             │             │
      └─────────────┼─────────────┘
                    ▼
               Evaluator
               /        \
        Approved      Retry
             │            │
             ▼            ▼
            END      Planner
```

---

# Interview Follow-Up Questions

## 1. Why use LangGraph instead of LangChain?

| LangChain                 | LangGraph                         |
| ------------------------- | --------------------------------- |
| Linear execution          | Graph execution                   |
| Sequential steps          | Branching and loops               |
| Limited orchestration     | Complex workflows                 |
| Minimal state             | Shared state across nodes         |
| Best for simple pipelines | Best for agents and orchestration |

---

## 2. Why is state important?

Every node receives the same evolving state.

Example

```python
{
    "question": "...",
    "documents": [...],
    "answer": "...",
    "score": 0.91
}
```

Each node updates only the fields it is responsible for.

---

## 3. Can nodes execute in parallel?

Yes. Independent branches can execute concurrently.

Example

```text
           User Question
                 │
         ┌───────┴────────┐
         ▼                ▼
  Search Database    Search Vector DB
         │                │
         └───────┬────────┘
                 ▼
             Merge Results
```

Parallel execution reduces latency when tasks are independent.

---

## 4. How do you prevent infinite loops?

Use safeguards such as:

* Maximum iteration count
* Retry counters in state
* Timeouts
* Confidence thresholds
* Exit conditions
* Checkpointing to resume safely

Example:

```python
class AgentState(TypedDict):
    question: str
    retries: int
    documents: list[str]
```

Then route:

```python
def decide_next_node(state):
    if state["context_score"] > 0.8:
        return "generate"

    if state["retries"] >= 3:
        return "generate"  # or a fallback node

    return "retrieve"
```

---

## 5. What would a production-grade version include?

A robust enterprise LangGraph workflow typically includes:

* Query rewriting before retrieval
* Hybrid search (BM25 + vector search)
* Cross-encoder reranking
* Conditional retrieval loops
* Structured output with Pydantic
* Tool-calling nodes
* Human approval for sensitive actions
* Retry and fallback branches
* Streaming responses
* Checkpointing for recovery
* Observability with LangSmith or OpenTelemetry
* Authentication and authorization around external tools

---

# Complete Production Workflow

```text
                    User Question
                           │
                           ▼
                   Query Rewriter
                           │
                           ▼
               Hybrid Retrieval Layer
                           │
                           ▼
                      Reranker
                           │
                           ▼
                  Context Evaluator
                  /                \
         Good Context         Poor Context
              │                    │
              ▼                    ▼
         Answer Generator     Retrieve Again
              │
              ▼
      Structured Output Check
              │
       Valid? / \ Invalid
            ▼     ▼
          Finish  Retry / Human Review
```

This pattern is widely used in production RAG and agent systems because it combines **stateful execution**, **conditional branching**, **loops**, and **recoverable workflows**—capabilities that are difficult to implement cleanly with a simple linear chain.
