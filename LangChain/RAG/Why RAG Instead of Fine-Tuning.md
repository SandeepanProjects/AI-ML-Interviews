# Why RAG Instead of Fine-Tuning?

This is one of the **most frequently asked Senior AI Engineer interview questions**.

A common misconception is:

> "If I have company documents, I should fine-tune the model."

**In most enterprise applications, this is the wrong approach.**

Fine-tuning and RAG solve **different problems**.

* **RAG** gives the model **access to external knowledge**.
* **Fine-tuning** changes **how the model behaves**.

In production, many systems actually use **both**.

---

# Simple Analogy

Imagine interviewing an employee.

### RAG

You allow the employee to use the latest company handbook.

```text
Employee

+

Company Handbook

↓

Answers questions
```

The handbook can be updated every day.

---

### Fine-Tuning

You train the employee for six months.

```text
Employee

↓

Training

↓

Remembers knowledge
```

Updating knowledge requires retraining.

---

# What Problem Does RAG Solve?

Suppose today your HR uploads:

```text
LeavePolicy.pdf

Updated Today
```

Question:

> How many maternity leave days are available?

With RAG:

```text
Question

↓

Retrieve LeavePolicy.pdf

↓

LLM

↓

180 Days
```

The model does **not** need retraining.

---

# What Problem Does Fine-Tuning Solve?

Suppose you want the model to always respond like your company.

Example:

Normal GPT:

```text
How can I help you today?
```

Company Style:

```text
Hello!

Welcome to ABC Bank.

I'm your financial assistant.

How may I assist you?
```

This is **behavior**, not knowledge.

Fine-tuning is ideal for this.

---

# RAG Architecture

```text
                 Question
                     │
                     ▼
             Retrieve Documents
                     │
                     ▼
             Relevant Chunks
                     │
                     ▼
                  Prompt
                     │
                     ▼
                    LLM
                     │
                     ▼
                  Answer
```

Knowledge stays outside the model.

---

# Fine-Tuning Architecture

```text
Training Data

↓

Model Training

↓

New Model

↓

Inference
```

Knowledge is embedded into model weights.

---

# Example 1 — HR Policy

Policy changes weekly.

Question:

> What is the travel reimbursement policy?

---

### Fine-Tuning

```text
Policy Changed

↓

Need New Training

↓

Deploy New Model
```

Expensive.

---

### RAG

```text
Policy Changed

↓

Upload PDF

↓

Done
```

No retraining.

---

# Example 2 — Banking

Interest rate today:

```text
Home Loan

8.2%
```

Tomorrow:

```text
8.6%
```

Would you fine-tune every day?

No.

RAG retrieves the latest policy.

---

# Example 3 — Product Documentation

Company releases

```text
API v2.4
```

Next week

```text
API v2.5
```

RAG automatically retrieves v2.5 documentation.

---

# RAG Code

Suppose we have:

```text
leave_policy.pdf
```

Load it.

```python
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("leave_policy.pdf")

docs = loader.load()
```

Split.

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = splitter.split_documents(docs)
```

Embed.

```python
from langchain_openai import OpenAIEmbeddings

embedding = OpenAIEmbeddings()
```

Store.

```python
from langchain_qdrant import QdrantVectorStore

vectorstore = QdrantVectorStore.from_documents(
    chunks,
    embedding,
    url="http://localhost:6333",
    collection_name="policies"
)
```

Retrieve.

```python
retriever = vectorstore.as_retriever()

documents = retriever.invoke(
    "How many maternity leave days?"
)
```

Generate.

```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("""
Answer only using the context.

Question:
{question}

Context:
{context}
""")

context = "\n\n".join(doc.page_content for doc in documents)

response = (prompt | llm).invoke({
    "question": "How many maternity leave days?",
    "context": context,
})

print(response.content)
```

Updating the PDF is enough.

---

# Fine-Tuning Workflow

```text
Documents

↓

Create Q&A Dataset

↓

Thousands of Examples

↓

Training

↓

Validation

↓

Deploy New Model
```

Fine-tuning requires a curated dataset, compute resources, evaluation, and deployment.

---

# Cost Comparison

Suppose HR updates one policy.

### RAG

```text
Upload PDF

↓

Done
```

Time:

```text
30 seconds
```

---

### Fine-Tuning

```text
Collect Data

↓

Train

↓

Evaluate

↓

Deploy
```

Time:

```text
Hours to days
```

---

# Accuracy Comparison

Suppose the company updates:

```text
Leave

180 Days

↓

210 Days
```

### Fine-Tuned Model

```text
Still Answers

180 Days
```

Until retrained.

---

### RAG

```text
Retrieve Latest PDF

↓

210 Days
```

---

# Memory Usage

### Fine-Tuning

Knowledge is stored inside model weights.

```text
Model

↓

Parameters

↓

Knowledge
```

---

### RAG

Knowledge stays in the database.

```text
Model

+

Vector DB

↓

Answer
```

---

# Security

Suppose a confidential finance document.

With RAG:

```text
Finance PDF

↓

RBAC

↓

Retriever

↓

LLM
```

Only authorized users retrieve the document.

With fine-tuning, once confidential data is learned by the model, separating access becomes much harder. This is one reason enterprises are cautious about fine-tuning on sensitive internal documents.

---

# Scalability

Suppose:

```
10 million documents
```

Would you fine-tune?

No.

Store them in:

* Qdrant
* Pinecone
* Milvus

Retrieve only what's needed.

---

# When Should You Use Fine-Tuning?

Fine-tune when you want to improve **behavior**, not inject changing knowledge.

Examples:

* Company tone
* Email style
* JSON formatting consistency
* Domain-specific terminology
* Tool-calling reliability
* Reducing verbosity
* Following organization-specific instructions

Example:

```text
User

↓

Model

↓

Always Returns

Valid JSON
```

---

# When Should You Use RAG?

Use RAG when knowledge changes.

Examples:

* HR policies
* Banking regulations
* Product documentation
* Medical guidelines
* Legal documents
* Company wiki
* Support manuals
* Jira tickets
* Confluence pages
* SharePoint documents

---

# Enterprise Architecture

```text
                    Employee
                        │
                        ▼
                  FastAPI API
                        │
                        ▼
                 LangGraph Workflow
                        │
               Query Rewriting
                        │
                        ▼
              Hybrid Retrieval
          (Vector + BM25 Search)
                        │
                        ▼
                  Reranking
                        │
                        ▼
              Context Compression
                        │
                        ▼
                    LLM
                        │
                        ▼
              Grounded Answer
```

The LLM remains unchanged while the knowledge base evolves independently.

---

# Can We Combine Both?

Yes—and many production systems do.

```text
            Fine-Tuned Model
                  │
                  ▼
            Better Behavior
                  │
                  ▼
          Retrieval-Augmented
              Generation
                  │
                  ▼
           Latest Knowledge
```

For example:

* Fine-tune the model to produce your company's preferred response format.
* Use RAG to retrieve the latest internal documentation.

---

# RAG vs Fine-Tuning

| Feature                          | RAG     | Fine-Tuning             |
| -------------------------------- | ------- | ----------------------- |
| Adds new knowledge               | ✅       | ❌ (requires retraining) |
| Handles frequently changing data | ✅       | ❌                       |
| Updates instantly                | ✅       | ❌                       |
| Changes model behavior           | Limited | ✅                       |
| Custom response style            | Limited | ✅                       |
| Uses external database           | ✅       | ❌                       |
| Lower update cost                | ✅       | ❌                       |
| Best for enterprise documents    | ✅       | Usually no              |
| Best for company tone and format | ⚠️      | ✅                       |

---

# Common Interview Questions

### Why not fine-tune on company documents?

Because company knowledge changes frequently. RAG separates knowledge from the model, allowing updates by re-indexing documents instead of retraining.

---

### Can fine-tuning reduce hallucinations?

It can improve instruction following and domain behavior, but it does not guarantee up-to-date factual knowledge. Grounding responses with RAG is generally more effective for changing information.

---

### When would you choose fine-tuning over RAG?

When the objective is to change how the model responds—for example, enforcing a specific tone, output schema, or specialized reasoning style—rather than providing access to evolving knowledge.

---

### Can you use both together?

Yes. A common enterprise pattern is to fine-tune (or otherwise customize) a model for behavior and combine it with RAG for current, organization-specific knowledge.

---

# Senior AI Engineer Interview Answer

> **RAG and fine-tuning address different problems. I use RAG when the challenge is providing the model with current, external knowledge such as company documents, policies, or product manuals. Documents are indexed in a vector database and retrieved at query time, so updates require only re-indexing, not model retraining. I use fine-tuning when I need to change the model's behavior—for example, enforcing a company-specific tone, improving structured outputs, or adapting to a specialized task. In enterprise systems, I often combine both approaches: a customized model for consistent behavior together with RAG for accurate, up-to-date information. This provides better maintainability, lower operational cost, and more reliable answers than relying on fine-tuning alone.
