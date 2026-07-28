Reducers are one of the **most advanced and most misunderstood concepts in LangGraph**.

They answer one important question:

> **What should happen when multiple nodes update the same state field?**

Without reducers, LangGraph **overwrites** the value.

With reducers, LangGraph can **append**, **merge**, **sum**, or apply any custom logic.

---

# Why Do We Need Reducers?

Suppose your state looks like this:

```python
from typing import TypedDict

class AgentState(TypedDict):
    messages: list[str]
```

Initial state:

```python
{
    "messages": []
}
```

---

## Node A

```python
def planner(state):

    return {
        "messages": [
            "Planning..."
        ]
    }
```

State becomes

```python
{
    "messages": [
        "Planning..."
    ]
}
```

---

## Node B

```python
def retriever(state):

    return {
        "messages": [
            "Retrieved documents"
        ]
    }
```

What should happen now?

### Option 1 (Default)

Replace the old value.

Result:

```python
{
    "messages": [
        "Retrieved documents"
    ]
}
```

The planner message is lost.

---

### Option 2

Append.

Result:

```python
{
    "messages": [
        "Planning...",
        "Retrieved documents"
    ]
}
```

This is what reducers enable.

---

# What is a Reducer?

A reducer is simply a **function** that combines:

```text
Old Value

+

New Value

↓

Updated Value
```

Mathematically,

```text
Reducer(old, new) → updated
```

---

# Without Reducers

Suppose

```python
state = {
    "count": 10
}
```

Node returns

```python
{
    "count": 20
}
```

LangGraph performs

```python
state["count"] = 20
```

Old value disappears.

---

# With Reducers

Reducer

```python
def reducer(old, new):

    return old + new
```

Now

```python
Old = 10

New = 20

↓

30
```

Instead of replacement.

---

# Reducers in LangGraph

Reducers are declared using `typing.Annotated`.

```python
from typing import TypedDict, Annotated
from operator import add

class AgentState(TypedDict):

    messages: Annotated[list[str], add]
```

The important part is

```python
Annotated[list[str], add]
```

This tells LangGraph:

> When this field is updated, don't replace it.
>
> Use `operator.add`.

---

# How `operator.add` Works

For lists

```python
from operator import add

old = ["A"]

new = ["B"]

print(add(old, new))
```

Output

```python
["A", "B"]
```

It concatenates the lists.

---

# Complete Example

```python
from typing import TypedDict, Annotated
from operator import add

from langgraph.graph import StateGraph


class AgentState(TypedDict):

    messages: Annotated[list[str], add]


def planner(state):

    return {

        "messages": [

            "Planning"

        ]
    }


def retriever(state):

    return {

        "messages": [

            "Retrieving"

        ]
    }


builder = StateGraph(AgentState)

builder.add_node("planner", planner)

builder.add_node("retriever", retriever)

builder.set_entry_point("planner")

builder.add_edge("planner", "retriever")

builder.set_finish_point("retriever")

graph = builder.compile()

result = graph.invoke(

    {

        "messages": []

    }

)

print(result)
```

Output

```python
{
    "messages": [

        "Planning",

        "Retrieving"

    ]
}
```

Without the reducer, only `"Retrieving"` would remain.

---

# Reducers for Numbers

Reducers can sum values.

```python
from operator import add

class AgentState(TypedDict):

    retries: Annotated[int, add]
```

Initial

```python
{
    "retries": 0
}
```

Node

```python
return {

    "retries": 1
}
```

Another node

```python
return {

    "retries": 1
}
```

Final

```python
{
    "retries": 2
}
```

Without reducer

```python
{
    "retries": 1
}
```

---

# Reducers for Dictionaries

Suppose

```python
state = {

    "tool_results": {}
}
```

Planner

```python
return {

    "tool_results": {

        "weather": "30°C"
    }
}
```

Calculator

```python
return {

    "tool_results": {

        "math": "42"
    }
}
```

Without reducer

```python
{
    "tool_results": {

        "math": "42"
    }
}
```

Weather disappears.

---

A custom reducer can merge dictionaries.

```python
def merge_dicts(old, new):

    merged = old.copy()

    merged.update(new)

    return merged
```

State

```python
class AgentState(TypedDict):

    tool_results: Annotated[
        dict,
        merge_dicts
    ]
```

Final

```python
{
    "tool_results": {

        "weather": "30°C",

        "math": "42"
    }
}
```

---

# Reducers for Chat Messages

LangGraph frequently uses reducers with messages.

```python
from typing import Annotated
from langgraph.graph.message import add_messages

class AgentState(TypedDict):

    messages: Annotated[list, add_messages]
```

Why not `operator.add`?

Because chat messages have metadata.

Example

```python
HumanMessage(...)

AIMessage(...)

ToolMessage(...)
```

`add_messages` correctly merges these message objects while preserving their metadata and order.

---

# Visual Example

Without reducer

```text
State

Messages

↓

[]

↓

Planner

↓

["Planning"]

↓

Retriever

↓

["Retrieving"]
```

Planner output is lost.

---

With reducer

```text
[]

↓

Planner

↓

["Planning"]

↓

Retriever

↓

["Planning",

 "Retrieving"]
```

Everything is preserved.

---

# Parallel Execution

Reducers become even more important when multiple branches execute in parallel.

Graph

```text
             Planner

           /          \

          /            \

   Search Agent    SQL Agent

          \            /

           \          /

            Generator
```

Search

```python
return {

    "documents": [

        "Search Result"
    ]
}
```

SQL

```python
return {

    "documents": [

        "Database Result"
    ]
}
```

Without reducer

```python
documents = [

    "Database Result"
]
```

One result overwrites the other.

---

With

```python
documents: Annotated[
    list,
    add
]
```

Result

```python
documents = [

    "Search Result",

    "Database Result"
]
```

Both branches contribute.

---

# Custom Reducer Example

Suppose you want the highest confidence score.

```python
def max_confidence(old, new):

    return max(old, new)
```

State

```python
class AgentState(TypedDict):

    confidence: Annotated[
        float,
        max_confidence
    ]
```

Planner

```python
0.72
```

Reviewer

```python
0.91
```

Final

```python
0.91
```

---

# Production Example

```python
from typing import TypedDict, Annotated
from operator import add
from langgraph.graph.message import add_messages

class AgentState(TypedDict):

    messages: Annotated[list, add_messages]

    retrieved_docs: Annotated[list, add]

    tool_results: Annotated[
        dict,
        merge_dicts
    ]

    retries: int

    answer: str
```

Each field has behavior appropriate to its type:

* `messages` → append chat messages safely.
* `retrieved_docs` → concatenate lists.
* `tool_results` → merge dictionaries.
* `retries` → overwrite (or use a numeric reducer if you want accumulation).
* `answer` → replace with the latest answer.

---

# Common Reducer Patterns

| Data Type     | Reducer           | Result                |
| ------------- | ----------------- | --------------------- |
| `list`        | `operator.add`    | Append items          |
| Chat messages | `add_messages`    | Merge message objects |
| `dict`        | Custom merge      | Combine keys          |
| `int`         | `operator.add`    | Sum values            |
| `float`       | Custom `max`      | Keep highest score    |
| `set`         | Custom union      | Remove duplicates     |
| `str`         | Default overwrite | Replace text          |

---

# When Should You Use Reducers?

Use reducers when:

* Multiple nodes update the same field.
* Parallel branches contribute results.
* You want to accumulate information instead of replacing it.
* You are building multi-agent systems where several agents write to shared state.

If only one node owns a field (for example, only the generator writes `answer`), you typically don't need a reducer.

---

# Interview Questions

### Why not always use `operator.add`?

Because different data types require different merge semantics:

* Lists should concatenate.
* Chat messages should preserve metadata (`add_messages`).
* Dictionaries should merge by key.
* Confidence scores might keep the maximum.
* Answers usually should be overwritten.

---

### Are reducers called on every state update?

Only for fields declared with a reducer using `Annotated`. Other fields use the default behavior: the new value replaces the old value.

---

### Can I write my own reducer?

Yes. A reducer is any callable with the signature:

```python
def my_reducer(old_value, new_value):
    # merge logic
    return merged_value
```

LangGraph invokes it automatically whenever that field is updated.

---

# Senior AI Engineer Interview Answer

> **A reducer in LangGraph defines how updates to a state field are combined. By default, LangGraph overwrites a field with the latest value. If multiple nodes need to contribute to the same field—such as appending retrieved documents, accumulating chat messages, or merging tool outputs—we declare that field with `typing.Annotated` and provide a reducer function. Built-in reducers like `operator.add` concatenate lists, while `add_messages` correctly merges chat message objects. We can also implement custom reducers for dictionaries, counters, or confidence scores. Reducers are especially important in parallel and multi-agent workflows, where several nodes may update the same state concurrently.**
