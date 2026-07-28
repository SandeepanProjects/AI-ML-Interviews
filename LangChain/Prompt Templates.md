**Prompt Templates** are one of the core building blocks in LangChain. Instead of hardcoding prompts as strings, Prompt Templates let you create **dynamic, reusable, and parameterized prompts**.

Think of them like **Python f-strings or Jinja templates**, but designed for LLMs.

---

# Why Do We Need Prompt Templates?

Without Prompt Templates:

```python
question = "Explain Transformers"

prompt = f"""
You are an AI expert.

Answer the following question:

{question}
"""

response = llm.invoke(prompt)
```

Problems:

* Duplicate prompt text
* Hard to maintain
* Difficult to reuse
* Error-prone
* Hard to version

Instead, use a Prompt Template.

---

# Prompt Template Concept

```text
Template

↓

Fill Variables

↓

Final Prompt

↓

LLM
```

Example:

Template:

```text
Explain the following topic:

{topic}
```

Input:

```text
topic = "LangGraph"
```

Generated Prompt:

```text
Explain the following topic:

LangGraph
```

---

# Basic PromptTemplate

```python
from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate.from_template(
    """
    Explain {topic} in simple terms.
    """
)
```

Using it:

```python
formatted = prompt.invoke(
    {
        "topic": "Transformers"
    }
)

print(formatted)
```

Output:

```text
Explain Transformers in simple terms.
```

---

# Multiple Variables

```python
prompt = PromptTemplate.from_template(
"""
You are a {role}.

Explain {topic}

Audience:

{audience}
"""
)
```

Invoke:

```python
prompt.invoke(
{
    "role":"Senior AI Engineer",
    "topic":"LangGraph",
    "audience":"Software Developers"
}
)
```

Generated prompt:

```text
You are a Senior AI Engineer.

Explain LangGraph.

Audience:

Software Developers
```

---

# Using with an LLM

```python
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(model="gpt-4.1")

chain = (
    prompt
    | llm
    | StrOutputParser()
)

response = chain.invoke(
{
    "topic":"Vector Databases"
}
)

print(response)
```

Pipeline:

```text
Variables

↓

Prompt Template

↓

LLM

↓

Parser

↓

Answer
```

---

# ChatPromptTemplate

Modern chat models use structured messages instead of one long string.

```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages(
[
    ("system",
     "You are a helpful AI tutor."),

    ("human",
     "Explain {topic}")
]
)
```

Invoke:

```python
prompt.invoke(
{
    "topic":"Embeddings"
}
)
```

Internally it creates:

```text
System:
You are a helpful AI tutor.

Human:
Explain Embeddings
```

---

# System, Human, and AI Messages

A chat prompt can include different roles.

```python
prompt = ChatPromptTemplate.from_messages(
[
    ("system",
     "You are a financial advisor."),

    ("human",
     "{question}"),

    ("ai",
     "Let's solve this carefully.")
]
)
```

This gives the model conversational context.

---

# MessagesPlaceholder

Useful for conversation history.

```python
from langchain_core.prompts import (
    MessagesPlaceholder,
    ChatPromptTemplate
)

prompt = ChatPromptTemplate.from_messages(
[
    ("system",
     "You are a chatbot."),

    MessagesPlaceholder("history"),

    ("human",
     "{question}")
]
)
```

Input:

```python
{
    "history": chat_history,
    "question":"What's my name?"
}
```

Prompt:

```text
System:
You are a chatbot.

Human:
My name is John.

AI:
Nice to meet you.

Human:
What's my name?
```

---

# Partial Variables

Some variables never change.

```python
prompt = PromptTemplate(
    template="""
You are a {role}

Explain {topic}
""",
    input_variables=["topic"],
    partial_variables={
        "role":"Senior AI Engineer"
    }
)
```

Now only:

```python
prompt.invoke(
{
    "topic":"RAG"
}
)
```

---

# Prompt Composition

Prompt Templates are Runnables.

```python
chain = (
    prompt
    | llm
    | parser
)
```

Execution:

```text
Input

↓

Prompt Template

↓

LLM

↓

Parser

↓

Output
```

---

# Prompt Templates in RAG

```text
Question

↓

Retriever

↓

Documents

↓

Prompt Template

↓

LLM
```

Example:

```python
template = """
Answer the question using
only the provided context.

Context:
{context}

Question:
{question}
"""
```

This grounds the answer in retrieved documents.

---

# Prompt Templates in Agents

Example agent prompt:

```text
You are an AI assistant.

Available tools:

{tools}

Question:

{input}

Previous reasoning:

{agent_scratchpad}
```

Here:

* `{tools}` lists available tools.
* `{input}` is the user's request.
* `{agent_scratchpad}` contains intermediate reasoning and observations.

---

# Best Practices

### 1. Keep system instructions separate

Instead of:

```text
You are an expert.

Question:

Explain AI
```

Use:

```python
("system",
"You are an AI expert.")

("human",
"{question}")
```

---

### 2. Use variables

Bad:

```text
Explain Transformers.
```

Good:

```text
Explain {topic}
```

---

### 3. Avoid prompt duplication

Create one template and reuse it across the application.

---

### 4. Add clear instructions

Example:

```text
Return JSON only.

Do not explain.

Maximum 100 words.
```

Specific instructions generally produce more reliable outputs.

---

# Common Interview Questions

## 1. PromptTemplate vs ChatPromptTemplate

| PromptTemplate                      | ChatPromptTemplate                           |
| ----------------------------------- | -------------------------------------------- |
| Produces a single text prompt       | Produces structured chat messages            |
| Suitable for text-completion models | Designed for chat models                     |
| Simpler                             | Supports system, human, AI, and placeholders |

---

## 2. Why use Prompt Templates?

Benefits:

* Reusable prompts
* Dynamic variables
* Easier maintenance
* Better readability
* Versioning
* Integration with LCEL
* Consistent prompting across an application

---

## 3. Can Prompt Templates be chained?

Yes.

```python
chain = prompt | llm | parser
```

Prompt templates implement the Runnable interface.

---

## 4. How do Prompt Templates help RAG?

They inject retrieved context into a consistent structure.

Example:

```text
Context:

{context}

Question:

{question}
```

This encourages the model to answer using the retrieved evidence.

---

## 5. How do Prompt Templates help Agents?

They provide the agent with:

* System instructions
* Available tools
* User input
* Conversation history
* Previous reasoning (`agent_scratchpad`)

This structured prompt guides the agent's decision-making.

---

# Production Architecture

```text
                  User Question
                        │
                        ▼
                 Prompt Template
            (system + variables)
                        │
                        ▼
                 Filled Prompt
                        │
                        ▼
                      LLM
                        │
                        ▼
                 Output Parser
                        │
                        ▼
                  Final Response
```

---

# PromptTemplate vs Chain vs Agent

| Feature                 | Prompt Template | Chain      | Agent   |
| ----------------------- | --------------- | ---------- | ------- |
| Stores prompt structure | ✅               | ❌          | ❌       |
| Accepts variables       | ✅               | Sometimes  | Yes     |
| Calls an LLM            | ❌               | ✅          | ✅       |
| Calls tools             | ❌               | Fixed only | Dynamic |
| Makes decisions         | ❌               | ❌          | ✅       |
| Reusable                | ✅               | ✅          | ✅       |

---

# Senior AI Engineer Interview Answer

If asked **"What are Prompt Templates?"**, a strong answer is:

> **Prompt Templates are reusable, parameterized prompt definitions that separate prompt logic from application code. They allow developers to inject variables such as user questions, retrieved context, conversation history, and tool descriptions into a consistent prompt structure. In LangChain, `PromptTemplate` is used for text prompts, while `ChatPromptTemplate` creates structured messages for chat models. Prompt Templates are Runnables, so they can be composed with LLMs, retrievers, and output parsers using LCEL to build modular, maintainable AI pipelines.**
