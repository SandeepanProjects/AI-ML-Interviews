# Dataclasses Explained Like a Senior AI Engineer

`@dataclass` is one of the most commonly asked Python interview topics for:

* Senior AI Engineer
* ML Engineer
* Backend Engineer
* LLM Infrastructure Engineer

Interviewers typically ask:

* What is a dataclass?
* Why use it instead of a normal class?
* How does it work internally?
* What methods does it generate?
* What is `frozen=True`?
* What is `default_factory`?
* Dataclass vs NamedTuple?
* Dataclass vs Pydantic?
* Where is it used in AI systems?

---

# The Problem

Suppose we want to represent a User.

Without dataclass:

```python
class User:

    def __init__(self, name, age):

        self.name = name
        self.age = age

    def __repr__(self):

        return f"User(name='{self.name}', age={self.age})"

    def __eq__(self, other):

        return (
            self.name == other.name and
            self.age == other.age
        )
```

Usage

```python
u1 = User("Alice", 25)
u2 = User("Alice", 25)

print(u1)
print(u1 == u2)
```

Output

```text
User(name='Alice', age=25)

True
```

Problem:

Most of this code is boilerplate.

---

# What is a Dataclass?

Python automatically generates boilerplate code.

```python
from dataclasses
```

# `@dataclass` in Python (Senior AI Engineer Interview)

`@dataclass` is one of the most common Python interview topics. It is widely used in:

* AI/ML applications
* FastAPI services
* Configuration management
* API request/response models
* Data pipelines

The goal of a dataclass is simple:

> **Automatically generate boilerplate code for classes that primarily store data.**

---

# Why was dataclass introduced?

Suppose you create a normal class.

```python
class User:

    def __init__(self, name, age):

        self.name = name
        self.age = age
```

Create object:

```python
u = User("John", 30)

print(u)
```

Output:

```text
<__main__.User object at 0x10f2345>
```

Not very useful.

Now compare two objects:

```python
u1 = User("John", 30)
u2 = User("John", 30)

print(u1 == u2)
```

Output

```text
False
```

Why?

Because Python compares object identity by default.

```text
u1 ----------> Object A

u2 ----------> Object B
```

Different objects.

---

# Solution: dataclass

```python
from dataclasses import dataclass


@dataclass
class User:

    name: str
    age: int
```

Now:

```python
u = User("John", 30)

print(u)
```

Output

```text
User(name='John', age=30)
```

Much nicer.

---

Compare objects:

```python
u1 = User("John", 30)
u2 = User("John", 30)

print(u1 == u2)
```

Output

```text
True
```

Because dataclass automatically compares fields.

---

# What does `@dataclass` generate?

This:

```python
@dataclass
class User:

    name: str
    age: int
```

is roughly equivalent to writing:

```python
class User:

    def __init__(self, name, age):

        self.name = name
        self.age = age


    def __repr__(self):

        return f"User(name={self.name}, age={self.age})"


    def __eq__(self, other):

        return (
            self.name == other.name
            and
            self.age == other.age
        )
```

You get these methods automatically.

---

# Automatically Generated Methods

By default dataclass generates:

```text
✓ __init__()

✓ __repr__()

✓ __eq__()
```

Optional:

```text
✓ __hash__()

✓ __lt__()

✓ __gt__()

✓ __le__()

✓ __ge__()
```

---

# Example

```python
from dataclasses import dataclass


@dataclass
class Employee:

    id: int
    name: str
    salary: float
```

Create:

```python
e = Employee(
    1,
    "Alice",
    90000
)

print(e)
```

Output

```text
Employee(id=1, name='Alice', salary=90000)
```

---

# Equality

```python
e1 = Employee(
    1,
    "Alice",
    90000
)

e2 = Employee(
    1,
    "Alice",
    90000
)

print(e1 == e2)
```

Output

```text
True
```

---

# `field()`

Used to customize fields.

Example:

```python
from dataclasses import dataclass, field


@dataclass
class User:

    name: str

    skills: list = field(default_factory=list)
```

---

Why not:

```python
skills=[]
```

Exactly the same reason as:

```python
def add(x=[]):
```

Mutable default arguments are shared.

Instead:

```python
field(default_factory=list)
```

creates a new list for every object.

---

Example:

```python
u1 = User("Alice")
u2 = User("Bob")

u1.skills.append("Python")

print(u1.skills)
print(u2.skills)
```

Output

```text
['Python']

[]
```

Each object has its own list.

---

# `__post_init__`

Runs immediately after `__init__`.

Example:

```python
from dataclasses import dataclass


@dataclass
class Employee:

    salary: float

    tax: float = 0


    def __post_init__(self):

        self.tax = self.salary * 0.3
```

Usage:

```python
e = Employee(100000)

print(e.tax)
```

Output

```text
30000
```

Useful when some fields depend on others.

---

# Frozen Dataclass

Immutable object.

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class Point:

    x: int
    y: int
```

Now:

```python
p = Point(1,2)

p.x = 10
```

Output

```text
FrozenInstanceError
```

Similar to:

```text
tuple
```

instead of

```text
list
```

---

# Ordering

```python
@dataclass(order=True)
class Product:

    price: int
```

Now:

```python
p1 = Product(10)
p2 = Product(20)

print(p1 < p2)
```

Output

```text
True
```

Comparison methods are generated automatically.

---

# Slots

Python objects normally store attributes in `__dict__`.

```text
Object

|

__dict__

|

name

age

salary
```

Memory usage increases.

Dataclass:

```python
@dataclass(slots=True)
class User:

    name: str
    age: int
```

Benefits:

* Less memory
* Faster attribute access
* Good for millions of objects

Common in ML pipelines.

---

# Dataclass vs Normal Class

Normal class:

```python
class User:

    def __init__(self, name, age):

        self.name = name
        self.age = age
```

Dataclass:

```python
@dataclass
class User:

    name: str
    age: int
```

Much shorter.

---

# Dataclass vs Class

| Feature         | Normal Class | Dataclass     |
| --------------- | ------------ | ------------- |
| **init**        | Manual       | Automatic     |
| **repr**        | Manual       | Automatic     |
| **eq**          | Manual       | Automatic     |
| Boilerplate     | High         | Very low      |
| Readability     | Lower        | Higher        |
| Data containers | Possible     | Excellent     |
| Business logic  | Excellent    | Also possible |

---

# AI Example

Configuration object:

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:

    model_name: str

    temperature: float

    max_tokens: int
```

Usage:

```python
config = ModelConfig(
    "gpt-4",
    0.2,
    4096
)
```

Cleaner than dictionaries:

```python
config["temperature"]
```

Instead:

```python
config.temperature
```

---

# FastAPI Example

Service configuration:

```python
from dataclasses import dataclass


@dataclass
class DatabaseConfig:

    host: str

    port: int

    username: str

    password: str
```

Passed throughout the application.

---

# ML Pipeline Example

Represent training metrics:

```python
from dataclasses import dataclass


@dataclass
class Metrics:

    accuracy: float

    precision: float

    recall: float

    f1: float
```

Easy to log and compare.

---

# When should you use a dataclass?

Use a dataclass when:

* The object mainly stores data.
* You want automatic `__init__`, `__repr__`, and `__eq__`.
* You want cleaner, more maintainable code.
* You don't need complex initialization logic (or you can use `__post_init__`).

---

# When should you use a normal class?

Use a normal class when:

* The object has significant business logic.
* Initialization is highly customized.
* You need fine-grained control over object behavior.
* You're implementing complex inheritance or metaclass behavior.

---

# Interview Questions

### Why use `@dataclass`?

A good answer:

> A dataclass reduces boilerplate by automatically generating methods like `__init__`, `__repr__`, and `__eq__`. It makes data-centric classes more readable, maintainable, and less error-prone.

---

### Why use `default_factory`?

> Mutable default values like `[]` or `{}` are shared across instances. `default_factory` creates a new object for each instance, preventing unintended shared state.

---

### Why use `frozen=True`?

> It creates immutable objects, making them safer to share across threads and preventing accidental modification.

---

### Why use `slots=True`?

> `slots=True` removes the per-instance `__dict__`, reducing memory usage and improving attribute access speed. This is especially useful when creating large numbers of objects, such as records in ML or data-processing pipelines.

---

# Senior AI Engineer Summary

| Use Case                          | Recommendation                                         |
| --------------------------------- | ------------------------------------------------------ |
| Configuration objects             | ✅ `@dataclass(frozen=True)`                            |
| API request/response data         | ✅ `@dataclass` (or Pydantic when validation is needed) |
| ML metrics/results                | ✅ `@dataclass`                                         |
| Large datasets with many objects  | ✅ `@dataclass(slots=True)`                             |
| Classes with heavy business logic | ✅ Normal class                                         |
| Immutable shared configuration    | ✅ `@dataclass(frozen=True)`                            |

**One important distinction in modern AI applications:** use **`dataclass`** when you simply need structured Python objects, but use **Pydantic models** (e.g., in FastAPI) when you also need runtime validation, parsing, and serialization of external data such as API requests or configuration files.


