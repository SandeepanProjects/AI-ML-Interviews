Managing the **context window** is one of the most important skills for a Senior AI Engineer.

In interviews, you'll often be asked:

* **What is a context window?**
* **What happens when it is exceeded?**
* **How do you handle long conversations?**
* **How do ChatGPT, Claude, and Gemini manage memory?**
* **How do you optimize context in production?**

A good answer is **not** "use a larger context window." Production systems use multiple techniques together.

---

# What is a Context Window?

The context window is the **maximum number of tokens** an LLM can process in a single request.

```
Context Window

+------------------------------+
| System Prompt                |
| User Messages                |
| Chat History                 |
| Retrieved Documents          |
| Tool Outputs                |
| Current Question             |
+------------------------------+

Total Tokens <= Model Limit
```

For example:

```
Model Context Window = 128,000 tokens

Current Prompt = 95,000 tokens

Response Budget = 2,000 tokens

Remaining = 31,000 tokens
```

If you exceed the limit:

```
150,000 tokens

↓

Model Limit = 128,000

↓

Request Fails
```

or older context must be removed before sending the request.

---

# Why Does Context Become Large?

Imagine a RAG application.

```
System Prompt
      500

Chat History
   20,000

Retrieved Documents
   40,000

Tool Outputs
   10,000

Current Question
      200

------------------

Total
70,700 Tokens
```

As conversations continue:

```
Message 1

↓

Message 2

↓

Message 3

↓

...

↓

1000 Messages
```

The prompt keeps growing.

---

# Strategy 1: Sliding Window (Most Common)

Keep only the most recent messages.

Instead of:

```
100 Messages
```

Keep:

```
Last 10 Messages
```

### Code

```python
from langchain_core.messages import BaseMessage

MAX_MESSAGES = 10


def trim_messages(messages: list[BaseMessage]):

    return messages[-MAX_MESSAGES:]
```

Usage:

```python
messages = trim_messages(messages)

response = llm.invoke(messages)
```

---

## Example

Before:

```
Message 1

Message 2

...

Message 100
```

After trimming:

```
Message 91

...

Message 100
```

---

## Pros

* Very simple
* Very fast
* Low latency

Cons:

* Older information is lost

---

# Strategy 2: Token-Based Trimming

Message count is misleading.

Example:

```
Message 1
10 tokens

Message 2
5000 tokens
```

Instead, trim by tokens.

---

### Install

```bash
pip install tiktoken
```

---

### Count Tokens

```python
import tiktoken

encoding = tiktoken.encoding_for_model("gpt-4.1")


def count_tokens(text: str):

    return len(
        encoding.encode(text)
    )
```

---

### Trim Until Budget Fits

```python
MAX_TOKENS = 6000


def trim_token_budget(messages):

    total = 0
    selected = []

    for msg in reversed(messages):

        tokens = count_tokens(msg.content)

        if total + tokens > MAX_TOKENS:
            break

        selected.insert(0, msg)

        total += tokens

    return selected
```

Now the prompt always stays below the token budget.

---

# Strategy 3: Conversation Summary

Instead of:

```
100 Messages
```

Summarize:

```
Conversation Summary

+

Recent Messages
```

---

### Flow

```
Old Messages

↓

Summary LLM

↓

Summary

↓

Delete Old Messages
```

---

### Code

```python
from langchain_openai import ChatOpenAI

summary_llm = ChatOpenAI(model="gpt-4.1")


def summarize(messages):

    prompt = f"""
Summarize this conversation.

{messages}
"""

    summary = summary_llm.invoke(prompt)

    return summary.content
```

---

Suppose:

```
80 messages
```

becomes

```
Summary:

The user works at EY,
prefers Python,
is building an AI platform.
```

Instead of thousands of tokens:

```
300 Tokens
```

---

# Strategy 4: Retrieval Instead of Full History

Don't send everything.

Store conversations in a vector database.

```
Conversation

↓

Embeddings

↓

Vector DB
```

When a user asks:

```
Tell me about my project.
```

Retrieve only relevant memories.

---

### Code

```python
query_embedding = embed_model.encode(
    user_question
)

results = vector_db.search(
    embedding=query_embedding,
    top_k=3
)
```

Instead of:

```
Entire Chat
```

Prompt contains:

```
3 Relevant Memories
```

---

# Strategy 5: Context Compression

Suppose retrieved documents are huge.

```
10 PDFs

↓

150 Pages
```

Don't send all of them.

Compress them.

---

### Code

```python
compression_prompt = f"""
Compress this document
while preserving key facts.

{document}
"""

compressed = llm.invoke(
    compression_prompt
)
```

Now:

```
150 Pages

↓

2 Pages
```

---

# Strategy 6: Reranking

Suppose retrieval returns:

```
20 Documents
```

Don't send all 20.

Rerank.

```
Retriever

↓

20 Docs

↓

Reranker

↓

Top 3

↓

LLM
```

Example:

```python
top_docs = reranker.rank(
    query,
    docs
)[:3]
```

---

# Strategy 7: Chunk Large Documents

Never send:

```
Entire PDF

↓

LLM
```

Instead:

```
PDF

↓

Chunks

↓

Retrieve

↓

LLM
```

Example:

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(

    chunk_size=500,

    chunk_overlap=100

)

chunks = splitter.split_text(document)
```

---

# Strategy 8: Store User Profile Separately

Don't keep repeating:

```
User likes Python

Works at EY

Timezone IST
```

Store separately.

```
Profile DB

↓

Load Profile

↓

Prompt
```

Example:

```python
profile = {

    "language": "Python",

    "company": "EY"

}
```

Prompt:

```python
prompt = f"""
User Profile

{profile}

Question

{question}
"""
```

---

# Strategy 9: Token Budget Allocation

Suppose your model supports:

```
128k tokens
```

Don't use all 128k for one component.

Allocate a budget.

```
System Prompt
2k

History
20k

Documents
40k

User
1k

Response
2k

Remaining Buffer
63k
```

Example:

```python
MAX_HISTORY = 20000

MAX_DOCS = 40000

MAX_SYSTEM = 2000
```

Each section gets its own limit.

---

# Strategy 10: LangGraph Context Manager

State:

```python
from typing import TypedDict
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):

    messages: list[BaseMessage]

    summary: str

    retrieved_docs: list[str]
```

Context node:

```python
MAX_MESSAGES = 8


def prepare_context(state):

    recent = state["messages"][-MAX_MESSAGES:]

    docs = state["retrieved_docs"][:3]

    return {

        "messages": recent,

        "summary": state["summary"],

        "retrieved_docs": docs

    }
```

The LLM receives:

```
Summary

+

Recent Messages

+

Top Documents
```

instead of the full history.

---

# Complete Production Pipeline

```
                    User

                      │

                      ▼

             Recent Messages

                      │

                      ▼

         Conversation Summary

                      │

                      ▼

          Vector Retrieval

                      │

                      ▼

              Reranking

                      │

                      ▼

        Token Budget Manager

                      │

                      ▼

                 Prompt Builder

                      │

                      ▼

                     LLM
```

---

# Full Example

```python
MAX_MESSAGES = 10
MAX_DOCS = 3


def build_context(state):

    messages = state["messages"][-MAX_MESSAGES:]

    docs = state["retrieved_docs"][:MAX_DOCS]

    prompt = f"""
Conversation Summary:

{state["summary"]}

Recent Messages:

{messages}

Relevant Documents:

{docs}

Question:

{messages[-1].content}
"""

    return prompt
```

Now invoke:

```python
response = llm.invoke(
    build_context(state)
)
```

---

# Best Practices

### Don't send the entire conversation

Instead:

```
Summary

+

Recent Messages
```

---

### Retrieve only relevant documents

Avoid including every retrieved document.

---

### Measure tokens before calling the LLM

```python
tokens = count_tokens(prompt)

if tokens > MAX_ALLOWED:
    raise ValueError(
        "Prompt exceeds token budget"
    )
```

---

### Reserve space for the response

If the model supports 128k tokens:

```
Prompt

126k

Response

2k
```

Avoid consuming the entire context window with the prompt.

---

### Use multiple memory layers

```
Recent Chat

+

Summary

+

User Profile

+

Retrieved Knowledge
```

This is how most production conversational systems are designed.

---

# Interview Questions

### Why not always use a larger context window?

Larger context windows increase latency, cost, and may still include irrelevant information. Carefully selected context is usually more effective than simply sending more tokens.

---

### Why use summaries?

They preserve the important facts from older conversations while keeping prompts compact and inexpensive.

---

### Why use vector retrieval?

Instead of sending the entire conversation history, semantic search retrieves only the memories relevant to the user's current question.

---

### How do ChatGPT-like systems manage long conversations?

Typically by combining:

* A recent message window
* Conversation summaries
* Long-term memory retrieval
* User profile data
* Relevant external documents (RAG)
* Token-budget management

---

# Senior AI Engineer Interview Answer

> **In production, I manage context windows using a layered strategy rather than relying on the model's maximum context size. I maintain a sliding window of recent messages, summarize older conversations, retrieve only semantically relevant long-term memories and documents, rerank retrieved content, and enforce token budgets before every LLM call. I also reserve tokens for the model's response and monitor prompt size programmatically. This approach keeps latency and costs low while ensuring the model receives the most relevant context instead of the largest possible context.**
