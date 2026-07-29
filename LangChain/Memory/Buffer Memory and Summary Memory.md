`Buffer Memory` and `Summary Memory` are two classic conversation memory strategies that were widely used in LangChain. Although LangChain's memory APIs have evolved (newer versions encourage using **message history** and **LangGraph state** instead of the older memory classes), the concepts are still extremely important for interviews and production system design.

---

# Why Do We Need Conversation Memory?

Suppose the conversation is:

```text
User: My name is Alice.

Assistant: Nice to meet you.

User: I work at EY.

Assistant: Great!

User: What's my name?
```

Without memory:

```text
LLM

↓

Only sees:

What's my name?

↓

"I don't know."
```

With memory:

```text
LLM

↓

Conversation History

↓

"My name is Alice"

↓

Answer:
Your name is Alice.
```

The question is:

**How should we store the conversation history?**

---

# Buffer Memory

## Definition

Buffer Memory stores **the complete conversation history**.

Nothing is removed or summarized.

Architecture:

```text
User

↓

Conversation Buffer

↓

Entire Chat History

↓

LLM
```

---

## Example

Conversation:

```text
User: Hi

Assistant: Hello

User: My name is Alice

Assistant: Nice to meet you

User: I work at EY

Assistant: Great
```

Buffer Memory contains:

```text
Hi

Hello

My name is Alice

Nice to meet you

I work at EY

Great
```

Everything is preserved.

---

# Example Using LangChain

```python
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

llm = ChatOpenAI(model="gpt-4.1")

memory = ConversationBufferMemory()

conversation = ConversationChain(
    llm=llm,
    memory=memory
)

print(conversation.predict(
    input="My name is Alice."
))

print(conversation.predict(
    input="I work at EY."
))

print(conversation.predict(
    input="What's my name?"
))
```

Output:

```text
Your name is Alice.
```

---

## Internal Memory

After three messages:

```python
print(memory.buffer)
```

Output:

```text
Human: My name is Alice

AI: Nice to meet you.

Human: I work at EY.

AI: Great!

Human: What's my name?
```

Everything is stored.

---

# Problem with Buffer Memory

Imagine:

```text
1000 Messages

↓

Buffer

↓

LLM
```

Prompt size:

```text
450,000 Tokens
```

Problems:

* Expensive
* Slow
* Context window overflow

Buffer memory does **not scale**.

---

# Summary Memory

Instead of storing every message, Summary Memory stores a **compressed summary**.

Architecture:

```text
Conversation

↓

Summarizer LLM

↓

Summary

↓

LLM
```

---

## Example

Original conversation:

```text
User:
Hello

Assistant:
Hi

User:
I like Python

Assistant:
Great

User:
I work at EY

Assistant:
Nice
```

Summary:

```text
User prefers Python.

User works at EY.
```

The detailed conversation is replaced by a compact summary.

---

# LangChain Example

```python
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationChain
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4.1")

memory = ConversationSummaryMemory(
    llm=llm
)

conversation = ConversationChain(
    llm=llm,
    memory=memory
)

conversation.predict(
    input="My name is Alice."
)

conversation.predict(
    input="I work at EY."
)

conversation.predict(
    input="I like Python."
)
```

---

Inspect the summary:

```python
print(memory.buffer)
```

Example output:

```text
The user is Alice.

The user works at EY.

The user prefers Python.
```

Notice:

The entire chat history has been compressed.

---

# How Summary Memory Works

Internally:

```text
New Message

↓

Existing Summary

↓

Summarizer LLM

↓

Updated Summary
```

Example:

Current summary:

```text
User works at EY.
```

New message:

```text
User likes Kubernetes.
```

Updated summary:

```text
User works at EY.

User likes Kubernetes.
```

Only the summary is kept.

---

# Visual Comparison

## Buffer Memory

```text
Message 1

↓

Message 2

↓

Message 3

↓

Message 4

↓

LLM
```

Everything goes into the prompt.

---

## Summary Memory

```text
Message 1

↓

Summarize

↓

Message 2

↓

Update Summary

↓

Message 3

↓

Update Summary

↓

LLM
```

Only the summary is sent.

---

# Token Usage Example

Suppose each message is 100 tokens.

Conversation:

```text
100 messages
```

Buffer Memory:

```text
100 × 100

=

10,000 Tokens
```

Summary Memory:

```text
Conversation Summary

=

500 Tokens
```

Huge savings.

---

# Buffer vs Summary

| Feature     | Buffer Memory              | Summary Memory        |
| ----------- | -------------------------- | --------------------- |
| Stores      | Entire conversation        | Conversation summary  |
| Token usage | Grows continuously         | Mostly constant       |
| Speed       | Slower over time           | Faster                |
| Cost        | Increases with chat length | Lower for long chats  |
| Accuracy    | Preserves every detail     | May lose fine details |
| Best for    | Short conversations        | Long conversations    |

---

# Real Example

Conversation:

```text
User:
I'm Alice.

↓

User:
I work at EY.

↓

User:
I'm building an AI agent.

↓

User:
I like Python.

↓

User:
I use Kubernetes.
```

### Buffer Memory

Prompt contains:

```text
I'm Alice.

I work at EY.

I'm building an AI agent.

I like Python.

I use Kubernetes.
```

---

### Summary Memory

Prompt contains:

```text
Alice works at EY.

She is developing AI agents.

She prefers Python.

She uses Kubernetes.
```

Much shorter.

---

# Production Architecture

Modern systems often combine both.

```text
               User

                 │

                 ▼

        Recent Messages (Buffer)

                 │

                 ▼

     Older Messages (Summary)

                 │

                 ▼

      Combined Context

                 │

                 ▼

                LLM
```

Example:

Keep:

```text
Last 10 Messages
```

Plus:

```text
Conversation Summary
```

The model gets:

```text
Summary

+

Recent Chat
```

This balances detail and efficiency.

---

# Modern LangGraph Approach

Instead of using `ConversationBufferMemory`, LangGraph typically stores messages in the graph state.

```python
from typing import TypedDict
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    messages: list[BaseMessage]
    summary: str
```

A summarization node periodically compresses old messages.

```python
def summarize(state):
    summary = summary_llm.invoke(
        f"""
        Existing summary:
        {state['summary']}

        New messages:
        {state['messages']}
        """
    )

    return {
        "summary": summary.content,
        "messages": state["messages"][-10:]  # Keep only recent messages
    }
```

The graph state now contains:

```text
Summary

+

Last 10 Messages
```

which is the pattern commonly used in production.

---

# Which Should You Use?

| Use Case            | Recommendation                                |
| ------------------- | --------------------------------------------- |
| Short chatbot       | Buffer Memory                                 |
| Customer support    | Summary + Recent Messages                     |
| AI coding assistant | Recent Messages + Summary                     |
| Long-running agent  | Summary + Vector Memory + User Profile        |
| Enterprise AI       | Summary + Recent Buffer + Long-term Retrieval |

---

# Senior AI Engineer Interview Answer

> **Buffer Memory stores the complete conversation history and provides maximum conversational fidelity, but its token usage grows linearly with the length of the conversation, making it expensive and eventually exceeding the model's context window. Summary Memory periodically uses an LLM to compress older conversations into a concise summary, dramatically reducing token usage while preserving the key facts. In production, I typically combine a short buffer of recent messages with a running conversation summary, allowing the model to retain recent conversational details while keeping prompts small and efficient.**
