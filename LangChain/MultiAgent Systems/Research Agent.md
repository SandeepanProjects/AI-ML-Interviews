The **Research Agent → Writer Agent** architecture is one of the most common **multi-agent design patterns** used in enterprise AI systems.

It separates **information gathering** from **content generation**.

Instead of asking one LLM to search, analyze, and write everything, we split responsibilities:

* **Research Agent** → Finds and verifies information
* **Writer Agent** → Writes the final response using only the research

This architecture is used by:

* Enterprise RAG systems
* Financial copilots
* Legal assistants
* Medical assistants
* Report generation systems
* AI search engines

It is a common **Senior AI Engineer interview** topic.

---

# Why Not Use One Agent?

Suppose a user asks:

> Explain how transformers work and include the latest research.

A single agent must:

* Search documents
* Read documents
* Remove duplicates
* Rank sources
* Summarize
* Write the article

```text
                User
                  │
                  ▼
            One Large Agent
                  │
      Search + Analyze + Write
```

Problems:

* Very large prompt
* Hallucinations
* Difficult to debug
* Hard to evaluate retrieval separately
* Cannot reuse research

---

# Better Architecture

Split responsibilities.

```text
                     User
                       │
                       ▼
                Research Agent
                       │
        Search + Retrieve + Rank
                       │
                       ▼
                Research Notes
                       │
                       ▼
                  Writer Agent
                       │
               Final Response
```

The writer never searches.

The researcher never writes.

---

# Responsibilities

## Research Agent

Responsible for

* Searching
* Retrieving documents
* Reranking
* Fact extraction
* Source selection
* Citation collection

Output

```text
Research Notes
```

---

## Writer Agent

Responsible for

* Reading notes
* Organizing information
* Writing naturally
* Following formatting instructions
* Producing final output

Output

```text
Final Report
```

---

# Graph State

```python
from typing import TypedDict

class AgentState(TypedDict):

    query: str

    retrieved_docs: list[str]

    research_notes: str

    draft: str

    final_answer: str
```

State evolves throughout the workflow.

Initial state

```python
{
    "query":"Explain LangGraph",
    "retrieved_docs":[],
    "research_notes":"",
    "draft":"",
    "final_answer":""
}
```

---

# Step 1 — Research Agent

Suppose we have a retriever.

```python
def research_agent(state):

    docs = retriever.invoke(
        state["query"]
    )

    notes = "\n".join(
        doc.page_content
        for doc in docs
    )

    return {

        "retrieved_docs": docs,

        "research_notes": notes

    }
```

Output

```python
{

"research_notes":

"""
LangGraph is a graph-based framework.

Supports checkpoints.

Supports stateful workflows.

"""

}
```

---

# Step 2 — Writer Agent

The writer receives only the research.

```python
def writer_agent(state):

    prompt = f"""
You are a technical writer.

Research:

{state["research_notes"]}

Write a detailed article.
"""

    response = llm.invoke(prompt)

    return {

        "final_answer":

        response.content

    }
```

Notice:

The writer never calls the retriever.

---

# Complete LangGraph

```python
from langgraph.graph import (
    StateGraph,
    START,
    END
)

builder = StateGraph(AgentState)

builder.add_node(
    "research",
    research_agent
)

builder.add_node(
    "writer",
    writer_agent
)

builder.add_edge(
    START,
    "research"
)

builder.add_edge(
    "research",
    "writer"
)

builder.add_edge(
    "writer",
    END
)

graph = builder.compile()
```

Execution

```python
result = graph.invoke({

    "query":

    "Explain LangGraph"

})
```

---

# Workflow

```text
START

↓

Research Agent

↓

Retrieve Documents

↓

Extract Notes

↓

Writer Agent

↓

Write Report

↓

END
```

---

# Real Production Example

User

> Write a report comparing GPT and Claude.

Research Agent

```text
Search GPT documentation

↓

Search Claude documentation

↓

Extract facts

↓

Remove duplicates

↓

Rank information
```

Output

```text
GPT

- Large context window
- Tool calling

Claude

- Long context
- Constitutional AI
```

Writer Agent

```text
Writes

Comparison Report
```

---

# Using RAG

Research Agent

```python
def research_agent(state):

    docs = vector_store.similarity_search(
        state["query"],
        k=5
    )

    notes = []

    for doc in docs:

        notes.append(
            doc.page_content
        )

    return {

        "research_notes":

        "\n".join(notes)

    }
```

---

# Research Agent with Reranking

Instead of using every document

```text
Retriever

↓

20 Documents

↓

Reranker

↓

Top 5 Documents

↓

Research Notes
```

```python
def research_agent(state):

    docs = retriever.invoke(
        state["query"]
    )

    ranked = reranker.rank(docs)

    notes = [

        d.page_content

        for d in ranked[:5]

    ]

    return {

        "research_notes":

        "\n".join(notes)

    }
```

---

# Writer with Structured Output

Instead of plain text

```python
from pydantic import BaseModel

class Report(BaseModel):

    title:str

    summary:str

    conclusion:str
```

Writer

```python
structured_llm = llm.with_structured_output(
    Report
)

report = structured_llm.invoke(prompt)

return {

    "final_answer":

    report.model_dump()
}
```

Output

```json
{
"title":"LangGraph",

"summary":"...",

"conclusion":"..."
}
```

---

# Reflection Pattern

Many production systems insert a reviewer.

```text
Research

↓

Writer

↓

Reviewer

↓

Needs Changes?

↓

Yes

↓

Writer Again

↓

END
```

Reviewer

```python
def reviewer(state):

    if "citation" not in state["draft"]:

        return {

            "needs_revision":True

        }

    return {

        "needs_revision":False

    }
```

---

# Parallel Research

Multiple research agents can run simultaneously.

```text
                     User

                       │

                       ▼

               Parallel Research

      ┌────────────┼─────────────┐

      ▼            ▼             ▼

 Web Search   Vector DB      SQL Database

      │            │             │

      └────────────┼─────────────┘

                   ▼

             Merge Research

                   ▼

             Writer Agent
```

Merge

```python
def merge(state):

    notes = "\n".join([

        state["web_notes"],

        state["db_notes"],

        state["sql_notes"]

    ])

    return {

        "research_notes":

        notes

    }
```

---

# Why This Is Better

Suppose the final report is wrong.

You inspect

```text
Research Notes

↓

Correct?
```

If

No

Problem

```text
Retriever
```

If

Yes

Problem

```text
Writer
```

This makes debugging much easier.

---

# Enterprise Architecture

```text
                    User

                      │

                      ▼

               Supervisor Agent

                      │

                      ▼

               Research Agent

        ┌──────────┼────────────┐

        ▼          ▼            ▼

   Web Search   Vector DB   SQL Query

        │          │            │

        └──────────┼────────────┘

                   ▼

            Research Notes

                   ▼

              Writer Agent

                   ▼

             Reviewer Agent

                   ▼

        Human Approval (optional)

                   ▼

                  END
```

---

# Production Enhancements

## Cache research

```python
cache_key = state["query"]

cached = redis.get(cache_key)

if cached:
    return {
        "research_notes": cached
    }
```

---

## Evaluate research quality

```python
if len(state["retrieved_docs"]) < 3:

    return {
        "retry": True
    }
```

Route back to the research agent if retrieval is insufficient.

---

## Checkpoint after research

```text
Research Completed

↓

Save Checkpoint

↓

Writer Starts
```

If the writer fails, you don't repeat the research.

---

## Add tracing

Log:

* Retrieval latency
* Number of documents
* Tokens
* Cost
* Writer latency
* Draft quality

---

# Research Agent vs Writer Agent

| Research Agent                     | Writer Agent                             |
| ---------------------------------- | ---------------------------------------- |
| Retrieves information              | Generates content                        |
| Uses search, vector DBs, SQL, APIs | Uses LLM for writing                     |
| Produces research notes            | Produces final answer                    |
| Focuses on factual grounding       | Focuses on clarity and structure         |
| Can rerank and filter documents    | Can format as markdown, HTML, JSON, etc. |

---

# Interview Questions

### Why separate research from writing?

Because retrieval quality and writing quality are different problems. Separating them makes the system more modular, easier to evaluate, easier to debug, and reduces hallucinations by grounding the writer on curated research.

---

### Can the writer search for new information?

In this pattern, **no**. The writer should only use the research notes it receives. If more information is needed, control should return to the research agent instead of allowing the writer to retrieve additional data.

---

### How do you improve reliability?

* Add reranking before creating research notes.
* Evaluate retrieval quality (for example, context precision and recall).
* Cache research results.
* Checkpoint after research.
* Add a reviewer agent that checks citations, completeness, and factual consistency before returning the final answer.

---

# Senior AI Engineer Interview Answer

> **The Research Agent–Writer Agent pattern separates information gathering from content generation. The Research Agent retrieves, filters, reranks, and summarizes relevant information into structured research notes. Those notes become part of the shared graph state and are passed to the Writer Agent, which generates the final response without performing additional retrieval. This separation improves grounding, reduces hallucinations, makes retrieval and generation independently testable, and allows production features such as checkpointing, caching, reflection, and human review to be added without tightly coupling the two responsibilities.**
