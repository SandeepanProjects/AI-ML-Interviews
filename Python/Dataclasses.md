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



