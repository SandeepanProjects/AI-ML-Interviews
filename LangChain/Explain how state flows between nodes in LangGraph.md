Understanding **how state flows between nodes** is the **most important concept in LangGraph**.

If you understand state flow, you understand:

* ✅ Multi-agent systems
* ✅ Tool calling
* ✅ Memory
* ✅ Loops
* ✅ Human-in-the-loop
* ✅ Conditional routing

A common interview question is:

> **"Explain how state flows between nodes in LangGraph."**

---

# High-Level Idea

Unlike traditional programming where functions call each other directly, **LangGraph passes a shared state object between nodes**.

```text
                Initial State

{
    question,
    documents,
    answer
}

        │
        ▼

    Retrieve Node

        │
        ▼

Updated State

{
    question,
    documents,
    answer
}

        │
        ▼

    Generate Node

        │
        ▼

Updated State

{
    question,
    documents,
    answer
}

        │
        ▼

        END
```

Notice:

* Nodes **do not call each other**
* Nodes **only receive state**
* Nodes **return updates**
* LangGraph merges the updates automatically

---

# Step 1: Define the Graph State

```python
from typing import TypedDict

class AgentState(TypedDict):
    question: str
    documents: list[str]
    answer: str
```

Initially:

```python
state = {
    "question": "What is LangGraph?",
    "documents": [],
    "answer": ""
}
```

---

# Step 2: First Node Receives State

```python
def retrieve(state: AgentState):

    print(state)

    docs = [

        "LangGraph is a framework.",

        "It supports workflows."

    ]

    return {

        "documents": docs

    }
```

Input:

```python
{

    "question": "What is LangGraph?",

    "documents": [],

    "answer": ""

}
```

Output:

```python
{

    "documents": [

        "LangGraph is a framework.",

        "It supports workflows."

    ]

}
```

Notice the node **does not return the whole state**.

It only returns:

```python
{

    "documents": docs

}
```

---

# Step 3: LangGraph Merges the State

Before merge:

```python
{

    "question": "What is LangGraph?",

    "documents": [],

    "answer": ""

}
```

Node update:

```python
{

    "documents": [

        "LangGraph is a framework.",

        "It supports workflows."

    ]

}
```

Merged state:

```python
{

    "question": "What is LangGraph?",

    "documents": [

        "LangGraph is a framework.",

        "It supports workflows."

    ],

    "answer": ""

}
```

Now this updated state goes to the next node.

---

# Step 4: Second Node

```python
def generate(state: AgentState):

    context = "\n".join(

        state["documents"]

    )

    answer = f"""

Question:

{state["question"]}

Context:

{context}

"""

    return {

        "answer": answer

    }
```

Input:

```python
{

    "question": "What is LangGraph?",

    "documents": [

        "LangGraph is a framework.",

        "It supports workflows."

    ],

    "answer": ""

}
```

Output:

```python
{

    "answer": "...generated answer..."

}
```

---

# Step 5: Merge Again

Before:

```python
{

    question,

    documents,

    answer=""
}
```

Update:

```python
{

    answer="Generated"
}
```

After:

```python
{

    question,

    documents,

    answer="Generated"
}
```

Then the workflow ends.

---

# Complete Flow

```text
Initial State

{
 question,
 documents=[],
 answer=""
}

        │
        ▼

Retrieve Node

Receives

{
 question,
 documents=[],
 answer=""
}

Returns

{
 documents=[...]
}

        │
        ▼

Merged State

{
 question,
 documents=[...],
 answer=""
}

        │
        ▼

Generate Node

Receives

{
 question,
 documents=[...],
 answer=""
}

Returns

{
 answer="..."
}

        │
        ▼

Merged State

{
 question,
 documents=[...],
 answer="..."
}

        │
        ▼

END
```

---

# Complete Working Code

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

# -----------------------
# State
# -----------------------

class AgentState(TypedDict):
    question: str
    documents: list[str]
    answer: str


# -----------------------
# Node 1
# -----------------------

def retrieve(state: AgentState):

    print("\n=== Retrieve Node ===")
    print("Received:", state)

    docs = [
        "LangGraph is a workflow framework.",
        "It uses shared graph state."
    ]

    return {
        "documents": docs
    }


# -----------------------
# Node 2
# -----------------------

def generate(state: AgentState):

    print("\n=== Generate Node ===")
    print("Received:", state)

    answer = (
        f"Question: {state['question']}\n\n"
        f"Context:\n" +
        "\n".join(state["documents"])
    )

    return {
        "answer": answer
    }


# -----------------------
# Build Graph
# -----------------------

builder = StateGraph(AgentState)

builder.add_node("retrieve", retrieve)
builder.add_node("generate", generate)

builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "generate")
builder.add_edge("generate", END)

graph = builder.compile()

# -----------------------
# Execute
# -----------------------

result = graph.invoke({

    "question": "Explain LangGraph",

    "documents": [],

    "answer": ""

})

print("\nFinal State")
print(result)
```

---

# Console Output

```text
=== Retrieve Node ===

Received:

{
    question='Explain LangGraph',
    documents=[],
    answer=''
}

=============================

=== Generate Node ===

Received:

{
    question='Explain LangGraph',

    documents=[
        'LangGraph is a workflow framework.',
        'It uses shared graph state.'
    ],

    answer=''
}

=============================

Final State

{
    question='Explain LangGraph',

    documents=[
        'LangGraph is a workflow framework.',
        'It uses shared graph state.'
    ],

    answer='Question: Explain LangGraph ...'
}
```

Notice how the second node automatically received the `documents` produced by the first node.

---

# State Flow in an Agent

A production agent usually has many nodes.

```text
                    START
                      │
                      ▼
                 Planner Node
                      │
         ┌────────────┴────────────┐
         ▼                         ▼
   Tool Selection            Direct Answer
         │
         ▼
     Tool Execution
         │
         ▼
    Tool Result Added
         │
         ▼
     LLM Generation
         │
         ▼
          END
```

State evolves at each step:

| Step      | State                                            |
| --------- | ------------------------------------------------ |
| Start     | `{question}`                                     |
| Planner   | `{question, selected_tool}`                      |
| Tool      | `{question, selected_tool, tool_result}`         |
| Generator | `{question, selected_tool, tool_result, answer}` |

Each node **adds information** to the shared state.

---

# Example with Retry Loop

Suppose retrieval returns too few documents.

State:

```python
class AgentState(TypedDict):
    question: str
    documents: list[str]
    retry_count: int
```

Retriever:

```python
def retrieve(state):

    docs = search(state["question"])

    return {
        "documents": docs,
        "retry_count": state["retry_count"] + 1
    }
```

Router:

```python
def route(state):

    if len(state["documents"]) < 3:
        return "retrieve"

    return "generate"
```

State flow:

```text
START

↓

Retrieve

retry = 1

↓

Too Few Docs

↓

Retrieve

retry = 2

↓

Enough Docs

↓

Generate

↓

END
```

Notice that `retry_count` stays in the state throughout the workflow.

---

# State with Messages

Most conversational agents keep chat history in the state.

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage

class ChatState(TypedDict):
    messages: Annotated[list, add_messages]
```

Node:

```python
def chatbot(state):

    return {
        "messages": [
            AIMessage(
                content="Hello!"
            )
        ]
    }
```

Current state:

```python
messages = [

    HumanMessage("Hi")
]
```

Returned update:

```python
{

    "messages": [

        AIMessage("Hello!")

    ]
}
```

Reducer (`add_messages`) automatically merges them:

```python
messages = [

    HumanMessage("Hi"),

    AIMessage("Hello!")

]
```

The node never manually appends messages.

---

# Production Example

```text
User Question
      │
      ▼
Retrieve Documents
      │
      ▼
documents added to state
      │
      ▼
Grade Retrieval
      │
      ▼
retrieval_score added
      │
      ▼
If score < 0.7
      │
      ├──────────────► Retrieve Again
      │
      ▼
Generate Answer
      │
      ▼
answer added
      │
      ▼
Store Conversation
      │
      ▼
messages updated
      │
      ▼
END
```

Each node contributes a small, focused update, and the state becomes richer as it moves through the graph.

---

# Best Practices

* Keep the graph state small and focused.
* Return only changed fields from each node.
* Use reducers (such as `add_messages`) for fields that multiple nodes update.
* Avoid mutating the incoming state directly; treat it as immutable input.
* Store intermediate results (tool outputs, retrieval scores, planner decisions) in the state so later nodes can reuse them.
* Use counters in the state to control retries and prevent infinite loops.

---

# Senior AI Engineer Interview Answer

> **In LangGraph, state flows through the workflow as a shared immutable object. Each node receives the latest merged state, performs one responsibility, and returns only the fields it wants to update. LangGraph merges those updates and passes the resulting state to the next node. This incremental state propagation enables modular workflows, conditional routing, retries, loops, tool execution, and multi-agent coordination without requiring nodes to call each other directly. For fields like chat history that are updated by multiple nodes, I use reducers such as `add_messages` to merge updates safely instead of replacing the existing value.`
