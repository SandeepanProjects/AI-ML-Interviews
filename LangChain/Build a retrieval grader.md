This is one of the **most common production RAG interview coding questions** for Senior AI Engineer, Staff AI Engineer, and AI Architect roles.

Companies like **OpenAI, Microsoft, Google, Anthropic, Databricks, Snowflake, Amazon, EY, Deloitte, and Accenture** build retrieval systems that **don't trust the first retrieval**. Instead, they evaluate the retrieved context and continue searching until the context is sufficient.

This pattern is called **Retrieval Grading** or **Self-Corrective RAG**.

---

# Problem Statement

Build a workflow that:

1. Retrieves documents.
2. Grades whether the retrieved context is sufficient.
3. If insufficient:

   * rewrite the query,
   * retrieve again.
4. Repeat until:

   * enough context is found, or
   * maximum retries reached.

---

# Production Architecture

```text
                  User Question
                        │
                        ▼
                  Retrieve Documents
                        │
                        ▼
                 Retrieval Grader
                 /             \
        Good Context      Poor Context
             │                 │
             ▼                 ▼
      Generate Answer    Rewrite Query
                               │
                               ▼
                        Retrieve Again
                               │
                     (Loop until success)
```

---

# Why Do This?

Suppose the user asks:

> Explain LangGraph interrupts.

Initial retrieval returns:

```text
LangChain is a framework.

Prompt engineering tips.

Vector databases.
```

The answer would be poor.

Instead:

```
Retriever

↓

Grader

↓

Not Relevant

↓

Rewrite Query

↓

Retrieve Again

↓

LangGraph interrupts documentation

↓

Answer
```

---

# Step 1: Define Graph State

```python
from typing import TypedDict, List

class RAGState(TypedDict):
    question: str
    rewritten_question: str
    documents: List[str]
    score: float
    retries: int
    answer: str
```

---

# Step 2: Retrieval Node

```python
def retrieve(state):

    query = (
        state["rewritten_question"]
        if state["rewritten_question"]
        else state["question"]
    )

    docs = vectorstore.similarity_search(
        query,
        k=5
    )

    return {
        "documents": docs
    }
```

---

# Step 3: Retrieval Grader

Instead of blindly trusting retrieval, ask an LLM to judge it.

Prompt

```text
You are evaluating retrieval quality.

Question:

{question}

Retrieved Documents:

{documents}

Does the retrieved context contain enough
information to answer the question?

Return only:

YES

or

NO
```

---

## Implementation

```python
grader_llm = ChatOpenAI(
    model="gpt-4.1",
    temperature=0
)
```

```python
def grade_documents(state):

    prompt = f"""
Question:
{state['question']}

Documents:
{state['documents']}

Answer YES or NO.
"""

    response = grader_llm.invoke(prompt)

    if "YES" in response.content.upper():

        return {
            "score": 1.0
        }

    return {
        "score": 0.0
    }
```

---

# Step 4: Query Rewriter

If retrieval failed:

```python
rewrite_llm = ChatOpenAI(
    model="gpt-4.1"
)
```

```python
def rewrite_query(state):

    prompt = f"""
Rewrite the question for better retrieval.

Question:

{state['question']}
"""

    response = rewrite_llm.invoke(prompt)

    return {
        "rewritten_question": response.content,
        "retries": state["retries"] + 1
    }
```

Example

Input

```text
LangGraph memory
```

Output

```text
How does LangGraph implement persistent
conversation memory using checkpoints?
```

---

# Step 5: Generate Answer

```python
generator = ChatOpenAI(
    model="gpt-4.1"
)
```

```python
def generate(state):

    context = "\n".join(
        str(doc.page_content)
        for doc in state["documents"]
    )

    prompt = f"""
Context:

{context}

Question:

{state['question']}
"""

    response = generator.invoke(prompt)

    return {
        "answer": response.content
    }
```

---

# Step 6: Conditional Router

```python
MAX_RETRIES = 3

def router(state):

    if state["score"] > 0.8:
        return "generate"

    if state["retries"] >= MAX_RETRIES:
        return "generate"

    return "rewrite"
```

Logic

```
Enough Context?

↓

Yes

↓

Generate

-------------

No

↓

Retries Left?

↓

Yes

↓

Rewrite

↓

Retrieve Again

-------------

No

↓

Generate with Best Available Context
```

---

# LangGraph

```python
from langgraph.graph import StateGraph, END

workflow = StateGraph(RAGState)

workflow.add_node(
    "retrieve",
    retrieve
)

workflow.add_node(
    "grade",
    grade_documents
)

workflow.add_node(
    "rewrite",
    rewrite_query
)

workflow.add_node(
    "generate",
    generate
)
```

Edges

```python
workflow.set_entry_point("retrieve")

workflow.add_edge(
    "retrieve",
    "grade"
)

workflow.add_conditional_edges(
    "grade",
    router,
    {
        "generate": "generate",
        "rewrite": "rewrite"
    }
)

workflow.add_edge(
    "rewrite",
    "retrieve"
)

workflow.add_edge(
    "generate",
    END
)
```

---

# Execution Example

Initial Question

```
Explain LangGraph memory.
```

---

### First Retrieval

```
Prompt engineering

Transformers

Embeddings
```

Grader

```
NO
```

---

### Query Rewrite

```
How does LangGraph use checkpoints
for conversation memory?
```

---

### Second Retrieval

```
LangGraph checkpoints

Interrupts

Memory persistence

Resume execution
```

Grader

```
YES
```

---

### Generation

Produces the final answer using the improved context.

---

# Complete Execution Flow

```text
              User Question
                    │
                    ▼
             Retrieve Documents
                    │
                    ▼
              Grade Retrieval
             /               \
        Good Context      Poor Context
             │                 │
             ▼                 ▼
     Generate Answer     Rewrite Query
                               │
                               ▼
                       Retrieve Again
                               │
                               └───────────┐
                                           │
                                           ▼
                                  Grade Retrieval
```

---

# Better Retrieval Grading

Instead of returning only YES/NO, return a confidence score.

```python
class RetrievalGrade(BaseModel):

    score: float

    reason: str
```

Example

```python
grader = llm.with_structured_output(
    RetrievalGrade
)
```

Output

```python
RetrievalGrade(
    score=0.92,
    reason="Documents directly answer the question."
)
```

Advantages:

* Easier threshold tuning
* Better observability
* Useful for dashboards
* Supports analytics over retrieval quality

---

# Production Improvements

## 1. Hybrid Retrieval

Instead of only vector search:

```text
Question

↓

Vector Search

+

BM25

↓

Merge Results

↓

Grade
```

Improves recall.

---

## 2. Reranking

```text
Retriever

↓

20 Documents

↓

Cross-Encoder

↓

Top 5 Documents

↓

Grader
```

Reduces noise before grading.

---

## 3. Multi-Retriever

```text
User

↓

Router

↓

FAQ

Vector DB

SQL

Knowledge Graph

↓

Merge

↓

Grade
```

Each retriever specializes in a different source.

---

## 4. Confidence Threshold

Instead of:

```python
if score > 0.8
```

Use configurable thresholds.

Example

```
0.95 → Excellent

0.80 → Good

0.60 → Retry

0.30 → Escalate
```

---

## 5. Human Review

If retrieval repeatedly fails:

```text
Retriever

↓

Grade

↓

Failed

↓

Human

↓

Answer
```

Useful in regulated environments.

---

# Enterprise Architecture

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
                 Cross Encoder
                        │
                        ▼
             Retrieval Grader LLM
              /               \
        Good Context      Poor Context
             │                 │
             ▼                 ▼
      Answer Generator   Rewrite Query
             │                 │
             └─────────────────┘
                        │
                        ▼
                 Final Response
```

---

# Interview Follow-Up Questions

## 1. Why use a retrieval grader?

Without a grader:

* The LLM answers using poor context.
* Hallucination rates increase.
* Irrelevant documents reduce answer quality.

The grader acts as a quality gate before generation.

---

## 2. Why rewrite the query?

Users often ask vague questions.

Example:

```
memory
```

Better query:

```
How does LangGraph persist conversation state using checkpoints?
```

This improves retrieval recall and precision.

---

## 3. Should the loop run forever?

No.

Always enforce limits such as:

* Maximum retries
* Minimum confidence threshold
* Timeout budget

This prevents infinite loops and excessive cost.

---

## 4. Should the grader use the same model as the generator?

Not necessarily.

A common production setup is:

* Small model → grading (fast, inexpensive)
* Large model → final generation (higher quality)

This balances cost and performance.

---

## 5. How do production RAG systems improve this further?

A mature implementation often includes:

* Query rewriting before retrieval
* Hybrid search (BM25 + embeddings)
* Metadata filtering (tenant, permissions, document type)
* Cross-encoder reranking
* Structured retrieval grading (Pydantic schema)
* Retry limits and adaptive thresholds
* Fallback to web search or another knowledge source
* Human escalation when retrieval repeatedly fails
* Tracing and evaluation (retrieval precision, recall, latency)

---

# Complete Production Self-Corrective RAG

```text
                    User Question
                          │
                          ▼
                   Query Rewriter
                          │
                          ▼
          Hybrid Retriever (BM25 + Vector)
                          │
                          ▼
                     Reranker
                          │
                          ▼
                Retrieval Grader
                 /             \
        High Confidence    Low Confidence
              │                  │
              ▼                  ▼
      Answer Generator    Rewrite Query
              │                  │
              └──────────────────┘
                          │
                Max Retries Reached?
                   /             \
                 No              Yes
                 │                │
                 ▼                ▼
          Retrieve Again   Generate Best-Effort
                               or Human Review
```

This **retrieve → grade → rewrite → retrieve** loop is the foundation of **self-corrective RAG**. It significantly reduces hallucinations, improves answer quality, and is a common design pattern in production AI systems where retrieval quality cannot be assumed on the first attempt.
