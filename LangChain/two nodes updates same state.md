This is one of the **most frequently asked LangGraph interview questions**.

The answer depends on **whether a reducer is defined for that state field**.

There are **three scenarios**:

1. **Sequential updates (no reducer)** → Latest value overwrites the previous one.
2. **Sequential updates (with reducer)** → Values are combined using the reducer.
3. **Parallel updates** → A reducer is usually required; otherwise LangGraph cannot safely combine concurrent updates.

Let's examine each case.

---

# Scenario 1: Two Nodes Update the Same Field (No Reducer)

State:

```python
from typing import TypedDict

class AgentState(TypedDict):
    answer: str
```

Initial state:

```python
{
    "answer": ""
}
```

---

## Node A

```python
def node_a(state):

    return {
        "answer": "First Answer"
    }
```

State becomes

```python
{
    "answer": "First Answer"
}
```

---

## Node B

```python
def node_b(state):

    return {
        "answer": "Second Answer"
    }
```

State becomes

```python
{
    "answer": "Second Answer"
}
```

The first value is overwritten.

---

## Flow

```text
Initial

answer = ""

      │

      ▼

Node A

↓

answer = "First Answer"

      │

      ▼

Node B

↓

answer = "Second Answer"
```

Final value:

```python
{
    "answer": "Second Answer"
}
```

This is the default behavior.

---

# Scenario 2: Sequential Updates with a Reducer

Now suppose we want to keep every answer.

State:

```python
from typing import TypedDict, Annotated
from operator import add

class AgentState(TypedDict):

    answers: Annotated[list[str], add]
```

Initial

```python
{
    "answers": []
}
```

---

Node A

```python
def node_a(state):

    return {
        "answers": ["First"]
    }
```

State

```python
{
    "answers": ["First"]
}
```

---

Node B

```python
def node_b(state):

    return {
        "answers": ["Second"]
    }
```

LangGraph performs

```python
add(
    ["First"],
    ["Second"]
)
```

Result

```python
{
    "answers": [
        "First",
        "Second"
    ]
}
```

---

## Flow

```text
[]

↓

Node A

↓

["First"]

↓

Node B

↓

Reducer

↓

["First", "Second"]
```

---

# Scenario 3: Parallel Nodes

This is where reducers become essential.

Graph

```text
           Planner

         /          \

        /            \

 Search Node     SQL Node

        \            /

         \          /

        Generator
```

Both nodes execute independently.

---

Search node

```python
return {
    "documents": [
        "Search Result"
    ]
}
```

SQL node

```python
return {
    "documents": [
        "Database Result"
    ]
}
```

Without a reducer, LangGraph has no deterministic way to combine these updates.

With

```python
from typing import Annotated
from operator import add

class AgentState(TypedDict):

    documents: Annotated[list[str], add]
```

LangGraph merges them:

```python
add(
    ["Search Result"],
    ["Database Result"]
)
```

Result:

```python
{
    "documents": [
        "Search Result",
        "Database Result"
    ]
}
```

---

# What Happens Internally?

Suppose the state is:

```python
state = {

    "documents": []
}
```

Node A returns

```python
{

    "documents": ["A"]
}
```

Node B returns

```python
{

    "documents": ["B"]
}
```

Without reducer

```python
state["documents"] = ["A"]

state["documents"] = ["B"]
```

Final

```python
["B"]
```

The first update is lost.

---

With reducer

```python
state["documents"] =
add(
    ["A"],
    ["B"]
)
```

Final

```python
["A", "B"]
```

---

# Dictionary Example

State

```python
class AgentState(TypedDict):

    tool_results: dict
```

Weather node

```python
return {

    "tool_results": {

        "weather": "30°C"
    }
}
```

Calculator node

```python
return {

    "tool_results": {

        "math": 42
    }
}
```

Without reducer

```python
{
    "tool_results": {
        "math": 42
    }
}
```

Weather disappears.

---

Custom reducer

```python
def merge_dicts(old, new):

    merged = old.copy()

    merged.update(new)

    return merged
```

State

```python
from typing import Annotated

class AgentState(TypedDict):

    tool_results: Annotated[
        dict,
        merge_dicts
    ]
```

Result

```python
{
    "tool_results": {

        "weather": "30°C",

        "math": 42
    }
}
```

---

# Chat Messages

For messages, don't use `operator.add`.

Use the built-in reducer.

```python
from typing import Annotated
from langgraph.graph.message import add_messages

class AgentState(TypedDict):

    messages: Annotated[
        list,
        add_messages
    ]
```

This preserves message metadata and ordering.

---

# Why Doesn't LangGraph Automatically Merge Everything?

Consider this state:

```python
{
    "answer": "..."
}
```

If two nodes produce:

```python
{
    "answer": "Answer A"
}
```

and

```python
{
    "answer": "Answer B"
}
```

Should LangGraph:

* Keep A?
* Keep B?
* Concatenate them?
* Choose the longer one?
* Keep the highest confidence?

There is no universally correct answer.

That's why **the developer specifies the merge strategy using reducers**.

---

# Production Example

```python
from typing import TypedDict, Annotated
from operator import add
from langgraph.graph.message import add_messages

def merge_tool_results(old, new):
    merged = old.copy()
    merged.update(new)
    return merged

class AgentState(TypedDict):

    messages: Annotated[list, add_messages]

    retrieved_docs: Annotated[list, add]

    tool_results: Annotated[
        dict,
        merge_tool_results
    ]

    answer: str
```

Behavior:

* Multiple agents writing `messages` → append.
* Multiple retrievers writing `retrieved_docs` → concatenate.
* Multiple tools writing `tool_results` → merge keys.
* Generator writing `answer` → overwrite.

---

# Interview Question

**Q:** What happens if two nodes update the same state field?

**Answer:**

> By default, LangGraph overwrites the existing value with the latest update. If the field is annotated with a reducer, LangGraph calls the reducer to combine the old and new values instead. This is especially important in parallel or multi-agent workflows, where multiple nodes may update the same field concurrently. Reducers provide deterministic merge behavior, such as appending lists, merging dictionaries, or accumulating counters.
