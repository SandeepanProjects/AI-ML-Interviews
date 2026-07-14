# Multiple Inheritance in Python

Multiple inheritance means:

> A class can inherit from **more than one parent class**.

Example:

```python
class A:
    def feature_a(self):
        print("Feature A")


class B:
    def feature_b(self):
        print("Feature B")


class C(A, B):
    pass


obj = C()

obj.feature_a()
obj.feature_b()
```

Output:

```
Feature A
Feature B
```

Here:

```
        A       B
         \     /
           C
```

`C` gets behavior from both `A` and `B`.

---

# Why Multiple Inheritance?

Imagine an AI system.

You may have separate reusable capabilities:

```python
class LoggingMixin:
    def log(self):
        print("Logging")


class CacheMixin:
    def cache(self):
        print("Caching")


class Model:
    def predict(self):
        print("Prediction")
```

Combine them:

```python
class ProductionModel(Model, LoggingMixin, CacheMixin):
    pass
```

Now:

```python
model = ProductionModel()

model.predict()
model.log()
model.cache()
```

Output:

```
Prediction
Logging
Caching
```

This pattern is called **Mixin-based design**.

---

# Types of Multiple Inheritance

## 1. Independent Parent Classes

```python
class A:
    pass

class B:
    pass

class C(A, B):
    pass
```

No relationship between A and B.

---

## 2. Diamond Inheritance

Most interview questions focus here.

```
        A
       / \
      B   C
       \ /
        D
```

Example:

```python
class A:
    def show(self):
        print("A")


class B(A):
    pass


class C(A):
    pass


class D(B, C):
    pass
```

Python uses **MRO** to solve this.

```python
print(D.mro())
```

Output:

```
D
B
C
A
object
```

---

# Pros of Multiple Inheritance

---

# 1. Code Reuse

Without multiple inheritance:

```python
class Logger:

    def log(self):
        print("log")


class Model:

    def predict(self):
        print("predict")
```

You may duplicate logging code everywhere.

With inheritance:

```python
class AIModel(Model, Logger):
    pass
```

Reuse existing behavior.

---

# 2. Mixins (Most Common Real Use)

Python frameworks heavily use mixins.

Example:

```python
class JSONMixin:

    def to_json(self):
        return "{}"


class DatabaseModel:

    def save(self):
        print("Saving")


class User(DatabaseModel, JSONMixin):
    pass
```

Now:

```python
u = User()

u.save()
u.to_json()
```

One class gets multiple capabilities.

---

# 3. Separation of Responsibilities

Instead of creating one huge class:

Bad:

```python
class AIModel:

    def predict():
        pass

    def log():
        pass

    def cache():
        pass

    def authenticate():
        pass
```

Better:

```python
class Predictor:
    pass


class Logger:
    pass


class Cache:
    pass


class SecureModel(Predictor, Logger, Cache):
    pass
```

Each class has one responsibility.

---

# 4. Flexible Design

You can compose features.

Example:

```python
class GPUModel(Model, GPUAccelerated):
    pass


class CPUModel(Model, CPUOptimized):
    pass
```

Different combinations can be created.

---

# Cons of Multiple Inheritance

---

# 1. Diamond Problem

The biggest issue.

Example:

```python
class A:

    def hello(self):
        print("A")


class B(A):

    def hello(self):
        print("B")


class C(A):

    def hello(self):
        print("C")


class D(B,C):
    pass
```

Question:

Which `hello()`?

```
B?
C?
A?
```

Python solves this using MRO.

```python
D.mro()
```

Result:

```
D
B
C
A
object
```

So:

```python
D().hello()
```

Output:

```
B
```

---

# 2. Tight Coupling

Multiple inheritance creates strong dependencies.

Example:

```python
class A:
    def process(self):
        pass


class B(A):
    pass


class C(A):
    pass
```

Changing `A` can unexpectedly affect many subclasses.

---

# 3. Difficult Method Resolution

Consider:

```python
class A:
    def run(self):
        print("A")


class B:
    def run(self):
        print("B")


class C(A,B):
    pass
```

Which runs?

Answer:

```python
C().run()
```

Output:

```
A
```

because:

```
C
↓
A
↓
B
```

The order matters.

---

# 4. Harder Debugging

A method call:

```python
obj.process()
```

may search:

```
Class
 |
Parent 1
 |
Parent 2
 |
Parent 3
 |
object
```

Finding where the method actually comes from can be harder.

---

# 5. Constructor Complexity

Example:

```python
class A:

    def __init__(self):
        print("A")


class B(A):

    def __init__(self):
        print("B")


class C(A):

    def __init__(self):
        print("C")


class D(B,C):
    pass
```

Calling:

```python
D()
```

Output:

```
B
```

Only `B.__init__()` runs.

`C` and `A` are skipped.

This can create bugs.

---

# `super()` in Multiple Inheritance

Many developers misunderstand:

> `super()` does NOT mean "call my parent".

It means:

> Call the next class in the MRO.

---

Example:

```python
class A:

    def hello(self):
        print("A")


class B(A):

    def hello(self):
        print("B")
        super().hello()


class C(A):

    def hello(self):
        print("C")
        super().hello()


class D(B,C):

    def hello(self):
        print("D")
        super().hello()
```

Check MRO:

```python
print(D.mro())
```

Output:

```
[
D,
B,
C,
A,
object
]
```

---

Now:

```python
d = D()

d.hello()
```

Execution:

```
D.hello()

super()

↓

B.hello()

super()

↓

C.hello()

super()

↓

A.hello()
```

Output:

```
D
B
C
A
```

---

# Why Use `super()`?

Without `super()`:

```python
class B(A):

    def hello(self):
        A.hello(self)
```

Problem:

In diamond inheritance:

```
        A
       / \
      B   C
       \ /
        D
```

`A` may execute multiple times.

---

With `super()`:

```python
class B(A):

    def hello(self):
        super().hello()
```

Python follows MRO:

```
D
↓
B
↓
C
↓
A
```

Each class runs once.

---

# Cooperative Multiple Inheritance

For `super()` to work correctly:

Every class should call `super()`.

Example:

```python
class Logger:

    def process(self):
        print("Logging")
        super().process()


class Cache:

    def process(self):
        print("Caching")
        super().process()


class Model:

    def process(self):
        print("Model")
```

Now:

```python
class AIService(Logger, Cache, Model):
    pass
```

MRO:

```
AIService
Logger
Cache
Model
object
```

Call:

```python
AIService().process()
```

Output:

```
Logging
Caching
Model
```

---

# When Should You Use Multiple Inheritance?

Good use cases:

✅ Mixins

Example:

```python
class APIView(AuthenticationMixin, LoggingMixin):
    pass
```

✅ Small reusable behaviors

```python
class SerializableMixin
class TimestampMixin
class ValidationMixin
```

✅ Framework design

---

# When Avoid It?

Avoid when:

❌ Classes represent real-world entities with complex relationships

❌ Multiple parents have overlapping responsibilities

❌ You need complicated constructor chains

❌ Composition is simpler

---

# Multiple Inheritance vs Composition

## Inheritance

"IS-A relationship"

Example:

```
Dog IS-A Animal
```

```python
class Dog(Animal):
    pass
```

---

## Composition

"HAS-A relationship"

Example:

```
Car HAS-A Engine
```

```python
class Car:

    def __init__(self):
        self.engine = Engine()
```

---

Usually:

> Prefer composition over inheritance when possible.

---

# Interview Answers

## Q: What are advantages of multiple inheritance?

Answer:

> Multiple inheritance improves code reuse, enables mixin-based designs, allows composition of independent behaviors, and reduces duplication.

---

## Q: What are disadvantages?

Answer:

> It can introduce ambiguity through the diamond problem, make method resolution harder to understand, increase coupling, and complicate initialization.

---

## Q: How does Python solve diamond inheritance?

Answer:

> Python uses the C3 Linearization algorithm to create a consistent Method Resolution Order (MRO), ensuring each class appears once and methods are resolved predictably.

---

## Q: What does `super()` do?

Answer:

> `super()` delegates method calls to the next class in the MRO, not necessarily the direct parent. It enables cooperative multiple inheritance by ensuring each class participates exactly once.

---

# Senior AI Engineer Practical Example

A production AI pipeline may use:

```python
class LoggingMixin:
    pass

class MetricsMixin:
    pass

class RetryMixin:
    pass

class LLMClient(
    LoggingMixin,
    MetricsMixin,
    RetryMixin
):
    pass
```

The final client gets:

* logging
* monitoring
* retry behavior

without putting everything into one massive class.

This is the main reason multiple inheritance exists in Python production systems.
