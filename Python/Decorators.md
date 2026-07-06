# Decorators Explained Like a Senior AI Engineer

**Decorators** are one of the most frequently asked Python interview topics for **Senior AI Engineer**, **ML Engineer**, **Backend Engineer**, and **LLM Infrastructure Engineer** roles.

Interviewers usually ask:

* What is a decorator?
* Why do we need decorators?
* How do they work internally?
* What are closures?
* What is `@` syntax?
* How do you write your own decorator?
* Where are decorators used in AI systems?

Understanding decorators requires understanding **functions as first-class objects** and **closures** first.

---

# Step 1: Functions are Objects

In Python, functions are **first-class objects**.

This means a function can be:

* Assigned to a variable
* Passed as an argument
* Returned from another function
* Stored in data structures

Example

```python
def greet():
    print("Hello")

say_hello = greet

say_hello()
```

Output

```
Hello
```

Memory

```
greet ----------+

                |
                ↓
        Function Object

                ↑

say_hello ------+
```

Both variables point to the same function object.

---

# Functions Can Be Passed

```python
def greet():
    print("Hello")

def execute(func):
    func()

execute(greet)
```

Output

```
Hello
```

Flow

```
execute()

↓

Receives Function

↓

Calls Function
```

---

# Functions Can Be Returned

```python
def outer():

    def inner():
        print("Inside")

    return inner

f = outer()

f()
```

Output

```
Inside
```

Notice

```
outer()

↓

Creates inner()

↓

Returns inner()

↓

Later executed
```

This ability is the foundation of decorators.

---

# What Problem Do Decorators Solve?

Suppose you have three functions.

```python
def add():
    print("Adding")

def subtract():
    print("Subtracting")

def multiply():
    print("Multiplying")
```

Now suppose every function needs logging.

Bad approach

```python
def add():

    print("Starting")

    print("Adding")

    print("Finished")
```

Repeat for every function.

Problems

* Duplicate code
* Hard to maintain
* Easy to forget

Instead:

```
Logging Logic

↓

Wrap Function

↓

Reuse Everywhere
```

That's exactly what decorators do.

---

# First Decorator

```python
def logger(func):

    def wrapper():

        print("Starting")

        func()

        print("Finished")

    return wrapper
```

Use it

```python
def hello():
    print("Hello")

hello = logger(hello)

hello()
```

Output

```
Starting

Hello

Finished
```

---

# What Happened Internally?

Original

```
hello

↓

Function Object
```

After

```python
hello = logger(hello)
```

Memory

```
logger()

↓

Creates wrapper()

↓

Returns wrapper

↓

hello now points to wrapper
```

The original function is still available through the closure inside `wrapper`.

---

# Visual Flow

```
Call hello()

↓

wrapper()

↓

Print Starting

↓

Call original hello()

↓

Print Finished
```

The caller never knows that the function was wrapped.

---

# The @ Syntax

Instead of writing

```python
hello = logger(hello)
```

Python provides syntactic sugar:

```python
@logger
def hello():
    print("Hello")
```

This is exactly equivalent to:

```python
def hello():
    print("Hello")

hello = logger(hello)
```

---

# Decorators with Arguments

Suppose

```python
def add(a, b):
    print(a + b)
```

The previous decorator fails because `wrapper()` takes no arguments.

Solution

```python
def logger(func):

    def wrapper(*args, **kwargs):

        print("Starting")

        result = func(*args, **kwargs)

        print("Finished")

        return result

    return wrapper
```

Usage

```python
@logger
def add(a, b):
    return a + b

print(add(5, 3))
```

Output

```
Starting

Finished

8
```

---

# Why `*args` and `**kwargs`?

Without them:

```python
def wrapper():
```

Only functions with no parameters can be decorated.

With:

```python
def wrapper(*args, **kwargs):
```

The decorator works for:

```python
f()
```

```python
f(1)
```

```python
f(1, 2)
```

```python
f(name="Alice")
```

It's a production best practice.

---

# Closures

A decorator works because of **closures**.

Example

```python
def outer():

    message = "Hello"

    def inner():
        print(message)

    return inner
```

Usage

```python
f = outer()

f()
```

Output

```
Hello
```

Question

Why does `message` still exist after `outer()` has finished?

Because the inner function **closes over** the variable.

Memory

```
inner

↓

Closure

↓

message="Hello"
```

Decorators rely on this behavior to remember the original function.

---

# Timing Decorator

Very common interview question.

```python
import time

def timer(func):

    def wrapper(*args, **kwargs):

        start = time.time()

        result = func(*args, **kwargs)

        end = time.time()

        print(f"Execution Time: {end-start:.3f}s")

        return result

    return wrapper
```

Usage

```python
@timer
def work():

    time.sleep(2)

work()
```

Output

```
Execution Time: 2.001s
```

Used extensively in production to monitor latency.

---

# Authentication Decorator

```python
def authenticate(func):

    def wrapper(user, *args, **kwargs):

        if not user.get("authenticated"):
            raise PermissionError("Access denied")

        return func(user, *args, **kwargs)

    return wrapper
```

Usage

```python
@authenticate
def view_profile(user):
    return f"Welcome {user['name']}"

user = {"name": "Alice", "authenticated": True}

print(view_profile(user))
```

Output

```
Welcome Alice
```

---

# Retry Decorator

Very common in AI systems because external APIs can fail.

```python
import time

def retry(max_attempts=3):

    def decorator(func):

        def wrapper(*args, **kwargs):

            for attempt in range(max_attempts):

                try:
                    return func(*args, **kwargs)

                except Exception:

                    print(f"Retry {attempt+1}")

                    time.sleep(1)

            raise Exception("Maximum retries exceeded")

        return wrapper

    return decorator
```

Usage

```python
@retry(max_attempts=3)
def call_api():
    raise RuntimeError("Temporary failure")

call_api()
```

Output

```
Retry 1

Retry 2

Retry 3

Exception: Maximum retries exceeded
```

---

# Caching Decorator

Instead of recomputing expensive work:

```python
from functools import wraps

def cache(func):

    memory = {}

    @wraps(func)
    def wrapper(n):

        if n in memory:
            return memory[n]

        result = func(n)

        memory[n] = result

        return result

    return wrapper
```

Usage

```python
@cache
def square(x):
    print("Computing...")
    return x * x

print(square(5))
print(square(5))
```

Output

```
Computing...
25
25
```

The second call uses the cached value.

---

# Why `functools.wraps`?

Without `wraps`:

```python
print(square.__name__)
```

Output

```
wrapper
```

The original function metadata is lost.

With:

```python
from functools import wraps

@wraps(func)
```

Output

```
square
```

`wraps` preserves:

* function name
* docstring
* annotations
* module information

Always use it in production decorators.

---

# Multiple Decorators

```python
@timer
@logger
def train():
    print("Training model")
```

Equivalent to:

```python
train = timer(logger(train))
```

Execution order:

```
timer

↓

logger

↓

train

↓

logger

↓

timer
```

Decorators are applied from the bottom up.

---

# Real AI Example

Suppose every LLM request should:

* log latency
* authenticate
* retry on transient failures

```python
@retry(max_attempts=3)
@timer
@authenticate
def generate(user, prompt):
    return "LLM response"
```

Execution:

```
Client

↓

Retry

↓

Timer

↓

Authentication

↓

LLM

↓

Response
```

Each decorator adds one responsibility while keeping the business logic clean.

---

# Decorators in Popular AI Frameworks

### FastAPI

```python
@app.get("/health")
async def health():
    return {"status": "ok"}
```

`@app.get` registers the function as an HTTP endpoint.

---

### PyTorch

```python
import torch

@torch.no_grad()
def predict(model, x):
    return model(x)
```

This decorator disables gradient tracking during inference.

---

### Dataclasses

```python
from dataclasses import dataclass

@dataclass
class User:
    name: str
    age: int
```

The decorator automatically generates methods like `__init__`, `__repr__`, and `__eq__`.

---

# Common Interview Questions

### Q1. What is a decorator?

A decorator is a function that takes another function, extends or modifies its behavior, and returns a new function without changing the original function's source code.

---

### Q2. Why use decorators?

They separate cross-cutting concerns such as logging, authentication, caching, retries, and timing from the core business logic, improving code reuse and maintainability.

---

### Q3. What is the role of a closure?

The wrapper function keeps a reference to the original function through a closure, allowing it to call the original function even after the decorator has finished executing.

---

### Q4. Why use `*args` and `**kwargs`?

To make the decorator work with functions that accept any combination of positional and keyword arguments.

---

### Q5. Why use `functools.wraps`?

To preserve the original function's metadata, such as its name, docstring, annotations, and module.

---

# Senior AI Engineer Takeaway

| Use Case                           | Decorator                         |
| ---------------------------------- | --------------------------------- |
| API latency measurement            | `@timer`                          |
| Authentication                     | `@authenticate`                   |
| Retry failed LLM requests          | `@retry`                          |
| Caching expensive computations     | `@cache` or `functools.lru_cache` |
| API routing                        | `@app.get`, `@app.post`           |
| Disable gradients during inference | `@torch.no_grad()`                |
| Automatic class generation         | `@dataclass`                      |

## Key Insight

Think of decorators like layers around a function:

```text
Client
   │
   ▼
Retry Decorator
   │
   ▼
Logging Decorator
   │
   ▼
Authentication Decorator
   │
   ▼
Original Function
```

The original function stays focused on its core task, while decorators add reusable behavior around it. This pattern is heavily used in production AI systems to implement observability, authentication, retries, caching, validation, and framework features without duplicating code.
