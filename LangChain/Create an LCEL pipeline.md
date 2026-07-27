This is one of the **most common Senior AI Engineer interview coding questions**.

Interviewers want to know if you understand:

* **LCEL (LangChain Expression Language)**
* Prompt composition
* Retrieval
* RAG pipelines
* Output parsing
* Modular pipeline design
* Production architecture

A good answer demonstrates how to build a **Retrieval-Augmented Generation (RAG)** pipeline using LCEL.

---

# Problem Statement

Build a question-answering system over company documents.

Documents

```text
Doc1:
LangChain is a framework for building LLM applications.

Doc2:
LangGraph is used for stateful agent workflows.

Doc3:
OpenAI provides GPT models.
```

User asks

```text
What is LangGraph?
```

Pipeline

```text
Question

↓

Retriever

↓

Relevant Documents

↓

Prompt

↓

LLM

↓

Parser

↓

Final Answer
```

---

# What is LCEL?

LCEL (LangChain Expression Language) allows you to compose chains using the `|` operator.

Instead of:

```python
prompt = prompt_template.invoke(inputs)

response = llm.invoke(prompt)

answer = parser.invoke(response)
```

You write:

```python
chain = prompt | llm | parser
```

Each component is called a **Runnable**.

---

# Overall Architecture

```text
                 User Question
                       │
                       ▼
                 Vector Retriever
                       │
                       ▼
              Retrieved Documents
                       │
                       ▼
               Prompt Template
                       │
                       ▼
                    GPT-4.1
                       │
                       ▼
                Output Parser
                       │
                       ▼
                 Final Answer
```

---

# Step 1: Install

```bash
pip install langchain
pip install langchain-openai
pip install langchain-community
pip install faiss-cpu
```

---

# Step 2: Create Documents

```python
from langchain_core.documents import Document

docs = [
    Document(
        page_content="LangChain is a framework for building LLM applications."
    ),
    Document(
        page_content="LangGraph is used for stateful multi-agent workflows."
    ),
    Document(
        page_content="OpenAI develops GPT models."
    )
]
```

---

# Step 3: Build a Vector Store

```python
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

embeddings = OpenAIEmbeddings()

db = FAISS.from_documents(
    docs,
    embeddings
)
```

Now every document has an embedding.

---

# Step 4: Create a Retriever

```python
retriever = db.as_retriever(
    search_kwargs={
        "k": 2
    }
)
```

Question

```text
What is LangGraph?
```

Retriever returns

```text
LangGraph is used for stateful multi-agent workflows.
```

---

# Step 5: Create Prompt

```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template(
"""
Answer using only the provided context.

Context:
{context}

Question:
{question}
"""
)
```

---

# Step 6: Create LLM

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-4.1",
    temperature=0
)
```

---

# Step 7: Output Parser

```python
from langchain_core.output_parsers import StrOutputParser

parser = StrOutputParser()
```

This converts the LLM response into a plain Python string.

---

# Step 8: Build LCEL Pipeline

We first need to pass the retrieved documents into the prompt.

```python
from langchain_core.runnables import RunnablePassthrough

chain = (
    {
        "context": retriever,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | parser
)
```

This is the key LCEL pattern:

```text
User Question
      │
      ├───────────────┐
      ▼               ▼
 Retriever       Original Question
      │               │
      └──────┬────────┘
             ▼
         Prompt
             ▼
           LLM
             ▼
         Parser
```

---

# Step 9: Invoke

```python
answer = chain.invoke(
    "What is LangGraph?"
)

print(answer)
```

Output

```text
LangGraph is used for stateful multi-agent workflows.
```

---

# What Happens Internally?

## Step 1

User

```text
What is LangGraph?
```

---

## Step 2

Retriever

```python
retriever.invoke(question)
```

Returns

```text
LangGraph is used for stateful multi-agent workflows.
```

---

## Step 3

Prompt becomes

```text
Context

LangGraph is used for stateful multi-agent workflows.

Question

What is LangGraph?
```

---

## Step 4

GPT generates

```text
LangGraph is used for stateful multi-agent workflows.
```

---

## Step 5

Parser returns

```python
str
```

---

# Complete Execution Flow

```text
                    Question
                        │
                        ▼
                 RunnablePassthrough
                        │
          ┌─────────────┴──────────────┐
          ▼                            ▼
     Retriever                   Original Question
          │                            │
          └─────────────┬──────────────┘
                        ▼
                  Prompt Template
                        ▼
                    GPT-4.1
                        ▼
                 String Parser
                        ▼
                  Final Answer
```

---

# Using Structured Output Instead

Instead of returning plain text, we can return validated objects.

```python
from pydantic import BaseModel

class Answer(BaseModel):
    answer: str
    confidence: float
```

```python
structured_llm = llm.with_structured_output(Answer)

chain = (
    {
        "context": retriever,
        "question": RunnablePassthrough()
    }
    | prompt
    | structured_llm
)
```

Output

```python
Answer(
    answer="LangGraph is used for stateful multi-agent workflows.",
    confidence=0.97
)
```

---

# Adding Document Formatting

Retrievers return `Document` objects, not strings. A formatter converts them into prompt-friendly text.

```python
from langchain_core.runnables import RunnableLambda

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

chain = (
    {
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | parser
)
```

This is the pattern you'll commonly see in production RAG systems.

---

# Adding Conversation Memory

Memory can be included before retrieval.

```text
Question

↓

Conversation Memory

↓

Retriever

↓

Prompt

↓

LLM

↓

Parser
```

The prompt now contains:

* Conversation history
* Retrieved documents
* Current question

---

# Enterprise Architecture

```text
                     User
                       │
                       ▼
                API Gateway
                       │
                       ▼
                 Memory Service
                       │
                       ▼
                 Vector Retriever
                       │
                       ▼
            Document Formatter
                       │
                       ▼
              Prompt Template
                       │
                       ▼
                  GPT-4.1
                       │
                       ▼
             Structured Parser
                       │
                       ▼
                Business Logic
                       │
                       ▼
                  Final Answer
```

---

# Production Enhancements

### 1. Reranking

```text
Retriever

↓

Top 20 Documents

↓

Reranker

↓

Top 5 Documents

↓

LLM
```

Improves answer quality by filtering the most relevant context.

---

### 2. Hybrid Search

```text
Question

↓

BM25

+

Vector Search

↓

Merged Results

↓

LLM
```

Combines keyword and semantic search.

---

### 3. Caching

Cache:

* Embeddings
* Retrieval results
* LLM responses

to reduce latency and cost.

---

### 4. Observability

Track:

* Retrieval latency
* Retrieved documents
* Prompt size
* Token usage
* LLM latency
* Output quality
* Hallucination rate

---

# Interview Follow-Up Questions

## Q1. Why use LCEL instead of calling `.invoke()` repeatedly?

LCEL makes pipelines:

* More readable
* Easier to compose
* Reusable
* Stream-friendly
* Async-friendly

---

## Q2. What is `RunnablePassthrough`?

It forwards the original input unchanged.

Example:

Input

```text
What is LangGraph?
```

Output

```text
What is LangGraph?
```

This lets the question travel alongside the retrieved context.

---

## Q3. Why is document formatting necessary?

Retrievers return `Document` objects with metadata. Prompt templates expect strings. Formatting extracts the relevant text and optionally includes metadata such as titles or sources.

---

## Q4. Can LCEL support parallel execution?

Yes. Independent branches can run concurrently.

Example:

```text
          User Question
                │
        ┌───────┴────────┐
        ▼                ▼
   Retriever A      Retriever B
        │                │
        └───────┬────────┘
                ▼
            Merge Results
                ▼
               LLM
```

This is useful for hybrid search or querying multiple knowledge sources.

---

## Q5. How would you make this production-ready?

A senior-level pipeline typically includes:

* Query rewriting for ambiguous questions
* Hybrid retrieval (BM25 + vector search)
* Metadata filtering (tenant, permissions, document type)
* Cross-encoder reranking
* Conversation memory
* Structured output with Pydantic
* Retries and fallback models
* Streaming responses
* Tracing and metrics (for example, LangSmith or OpenTelemetry)
* Guardrails against prompt injection and unauthorized document access

---

# Complete Production RAG Pipeline

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
                Document Formatter
                         │
                         ▼
      Prompt + Conversation Memory
                         │
                         ▼
                     GPT-4.1
                         │
                         ▼
            Pydantic / String Parser
                         │
                         ▼
               Business Logic Layer
                         │
                         ▼
                  Final Response
```

This LCEL pattern—**retrieval → prompt → LLM → parser**—is the foundation of most production RAG systems built with LangChain. Senior AI engineers are expected to understand not only how to write the pipeline, but also how to extend it with reranking, memory, structured outputs, observability, and security for real-world deployments.
