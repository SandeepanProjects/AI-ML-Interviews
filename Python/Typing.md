# Python Typing (Senior Python / AI Engineer Interview)

Modern Python interviews increasingly test **static typing** because production systems need:

* Better maintainability
* Fewer runtime bugs
* Better IDE support
* Safer refactoring
* Clear API contracts

Python is dynamically typed:

```python
def add(a, b):
    return a + b
```

Python allows:

```python
add(10,20)
```

and:

```python
add("AI","Engineer")
```

Both run.

Problem:

Large codebases become difficult to maintain.

Typing adds **type hints**.

---

# 1. Basic Type Hints

Example:

```python
def add(
    a: int,
    b: int
) -> int:

    return a + b
```

Meaning:

```text
a should be int

b should be int

return should be int
```

Important:

Python does NOT enforce this at runtime.

```python
add("hello", "world")
```

still executes.

A type checker catches it.

---

# 2. Optional

## Purpose

Represents:

> A value can be a specific type OR None.

Before Python 3.10:

```python
from typing import Optional
```

Example:

```python
from typing import Optional


def find_user(
    user_id:int
) -> Optional[str]:

    if user_id == 1:
        return "John"

    return None
```

Meaning:

```python
Optional[str]
```

equals:

```python
Union[str, None]
```

---

Without Optional:

```python
def find_user(id:int)->str:
```

Problem:

Function says:

"I always return string"

but actually:

```python
return None
```

Type mismatch.

---

# Modern Python Syntax

Python 3.10+:

```python
def find_user(
    id:int
) -> str | None:

    pass
```

Equivalent:

```python
Optional[str]
```

---

# AI Example

RAG retrieval:

```python
def get_document(
    doc_id:str
) -> Document | None:

    pass
```

Because:

Document may not exist.

---

# 3. Union

## Purpose

A variable can have multiple possible types.

Example:

```python
from typing import Union


def process(
    value: Union[int,str]
):

    pass
```

Allowed:

```python
process(10)

process("hello")
```

---

Modern syntax:

```python
def process(
    value:int | str
):
    pass
```

---

# Example

API response:

```python
from typing import Union


Response = Union[
    dict,
    str
]
```

Function:

```python
def call_api() -> Response:

    pass
```

Can return:

```python
{
"status":"ok"
}
```

or:

```python
"error"
```

---

# Optional vs Union

Optional:

```python
Optional[int]
```

means:

```python
int OR None
```

Union:

```python
Union[int,str]
```

means:

```python
int OR str
```

---

# 4. Generic

## Problem

Suppose:

```python
def first(items):

    return items[0]
```

What is return type?

Could be:

```python
int
```

or:

```python
str
```

or:

```python
User
```

---

Generic allows:

> Type remains flexible but consistent.

---

Example:

```python
from typing import TypeVar, Generic


T = TypeVar("T")


def first(
    items:list[T]
) -> T:

    return items[0]
```

---

Usage:

```python
numbers=[
1,2,3
]

x = first(numbers)
```

Type:

```python
x:int
```

---

Strings:

```python
names=[
"John",
"Bob"
]

x=first(names)
```

Type:

```python
x:str
```

---

# 5. TypeVar

## Purpose

Create a placeholder type.

Example:

```python
from typing import TypeVar


T = TypeVar("T")
```

Means:

"Some unknown type"

---

Example:

```python
T = TypeVar("T")


def identity(
    value:T
)->T:

    return value
```

---

Usage:

```python
x = identity(10)
```

Type:

```python
int
```

---

```python
x = identity("AI")
```

Type:

```python
str
```

---

# Why not Any?

Bad:

```python
def identity(
    value:Any
)->Any:
```

Because:

Everything is allowed.

You lose type safety.

---

Generic:

```python
T -> T
```

preserves relationship.

---

# Generic Class Example

Very common interview question.

Create a typed cache:

```python
from typing import Generic, TypeVar


T = TypeVar("T")


class Cache(Generic[T]):

    def __init__(self):

        self.value:T | None = None


    def set(
        self,
        value:T
    ):

        self.value=value


    def get(self)->T:

        return self.value
```

---

Usage:

```python
cache = Cache[int]()

cache.set(100)

value = cache.get()
```

Now:

```python
value:int
```

---

AI Example:

Embedding cache:

```python
Cache[list[float]]
```

Stores:

```python
[
0.12,
0.55,
0.89
]
```

---

# 6. Protocol

## Purpose

Protocol enables **structural typing**.

Meaning:

> If an object behaves like something, it can be used.

Also called:

"duck typing with static checking"

---

Traditional inheritance:

```python
class Animal:

    def speak(self):
        pass


class Dog(Animal):
    pass
```

Dog must inherit.

---

Protocol:

No inheritance required.

---

Example:

```python
from typing import Protocol


class Logger(Protocol):

    def log(
        self,
        message:str
    ):
        ...
```

---

Any class with:

```python
log()
```

works.

Example:

```python
class FileLogger:

    def log(self,message):

        print(message)
```

No inheritance.

Still valid:

```python
def write_log(
    logger:Logger
):

    logger.log("hello")
```

---

# Why Protocol?

Because Python follows:

> "If it behaves like a duck, treat it as a duck."

---

# AI Example

Multiple LLM providers:

```python
class LLMProvider(Protocol):

    async def generate(
        self,
        prompt:str
    )->str:
        ...
```

Implementations:

```python
class OpenAIClient:

    async def generate(self,prompt):
        return "answer"
```

---

```python
class ClaudeClient:

    async def generate(self,prompt):
        return "answer"
```

Both satisfy protocol.

No inheritance required.

---

# 7. mypy

## What is mypy?

`mypy` is a static type checker.

Install:

```bash
pip install mypy
```

---

Example:

File:

```python
# test.py


def add(
    a:int,
    b:int
)->int:

    return a+b



add(
    "hello",
    "world"
)
```

Run:

```bash
mypy test.py
```

Output:

```
error:
Argument 1 has incompatible type "str";
expected "int"
```

---

# Python Runtime vs mypy

Python:

```python
add("a","b")
```

runs.

mypy:

```text
❌ Error
```

before execution.

---

# Why Use mypy in Production?

Large systems:

```
10 developers
1 million lines
100 services
```

Without typing:

```text
Change API
 |
 |
Runtime failure
```

With typing:

```text
Change API
 |
 |
mypy catches issues
```

---

# AI Production Example

Suppose:

Embedding function:

```python
def embed(
    text:str
)->list[float]:

    pass
```

Someone changes:

```python
return "embedding"
```

mypy detects:

```
Expected list[float]
Got str
```

---

# Typing in FastAPI

Very common.

Example:

```python
from pydantic import BaseModel


class Query(BaseModel):

    text:str

    limit:int


@app.post("/search")
async def search(
    query:Query
):

    return query
```

Benefits:

* Validation
* Documentation
* IDE support

---

# Comparison

| Feature  | Purpose                       |
| -------- | ----------------------------- |
| Optional | Type or None                  |
| Union    | Multiple possible types       |
| Generic  | Reusable type-safe code       |
| TypeVar  | Placeholder type              |
| Protocol | Interface without inheritance |
| mypy     | Static type checking          |

---

# Senior AI Engineer Interview Answers

## Why use typing in Python?

> Typing improves readability, prevents bugs, enables IDE intelligence, supports large-scale refactoring, and creates clear contracts between components.

---

## Why use Generic instead of Any?

> Generic preserves relationships between input and output types, while Any disables type checking.

---

## Why use Protocol?

> Protocol enables structural typing, allowing different implementations to satisfy the same interface without explicit inheritance.

---

## Why use mypy?

> mypy performs static analysis before runtime and catches type errors early, which is valuable in large production Python systems.

---

# Real AI System Example

```text
FastAPI

   |
   |
Typed Request Model

   |
   |
RAG Service Protocol

   |
   |
Embedding Provider Generic

   |
   |
Vector Store Interface

   |
   |
LLM Provider Protocol
```

Typing helps maintain large AI platforms with many interchangeable components.
