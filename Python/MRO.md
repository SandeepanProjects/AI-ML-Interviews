This is one of the **most frequently asked Senior Python interview questions**, especially for **Senior AI Engineer**, **Backend Engineer**, and **Python Developer** roles.

Interviewers usually ask:

* What is MRO?
* How does Python resolve methods?
* What is Diamond Inheritance?
* Why doesn't Python call the parent class twice?
* What is the C3 Linearization algorithm?

Let's understand it from the basics.

---

# What is MRO?

**MRO (Method Resolution Order)** is the order in which Python searches classes when looking for a method or attribute.

Suppose:

```python
class A:
    def show(self):
        print("A")

class B(A):
    pass

b = B()
b.show()
```

Question:

Where is `show()`?

Python searches:

```
B

↓

A

↓

object
```

Since `A` has `show()`:

Output

```
A
```

This search order is called the **Method Resolution Order (MRO)**.

---

# Why Do We Need MRO?

Imagine multiple inheritance.

```python
class A:
    pass

class B:
    pass

class C(A, B):
    pass
```

Now if both `A` and `B` have the same method:

```python
class A:
    def hello(self):
        print("A")

class B:
    def hello(self):
        print("B")

class C(A, B):
    pass

C().hello()
```

Question:

Should Python call

```
A.hello()

or

B.hello()
```

MRO answers that question.

---

# Checking MRO

```python
print(C.mro())
```

Output

```python
[
 C,
 A,
 B,
 object
]
```

Python searches exactly in this order.

---

# Single Inheritance

```python
class Animal:
    def eat(self):
        print("eat")

class Dog(Animal):
    pass
```

MRO

```
Dog

↓

Animal

↓

object
```

---

# Multiple Inheritance

```python
class A:
    pass

class B:
    pass

class C(A, B):
    pass
```

MRO

```
C

↓

A

↓

B

↓

object
```

Because

```python
class C(A, B):
```

lists `A` first.

---

# The Diamond Problem

This is the famous interview question.

Consider

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

Hierarchy

```
          A
         / \
        /   \
       B     C
        \   /
         \ /
          D
```

This is called **Diamond Inheritance**.

---

# The Big Question

Suppose

```python
d = D()

d.show()
```

Question:

How should Python search?

Possible paths:

```
D

↓

B

↓

A
```

OR

```
D

↓

C

↓

A
```

Should Python visit `A` twice?

```
D

↓

B

↓

A

↓

C

↓

A
```

That would execute the same class twice.

Not good.

---

# Python's Solution

Python uses **C3 Linearization**.

Result:

```python
print(D.mro())
```

Output

```python
[
 D,
 B,
 C,
 A,
 object
]
```

Search order

```
D

↓

B

↓

C

↓

A

↓

object
```

Notice

`A`

appears only once.

---

# Example

```python
class A:
    def show(self):
        print("A")

class B(A):
    pass

class C(A):
    pass

class D(B,C):
    pass

d = D()

d.show()
```

Search

```
D

↓

B

↓

show?

No

↓

C

↓

show?

No

↓

A

↓

Found
```

Output

```
A
```

---

# Why Not Visit A Twice?

Suppose

```python
class A:

    def __init__(self):
        print("A")
```

If Python visited twice

Output would become

```
A

A
```

Initialization happens twice.

That would be wrong.

So Python guarantees each class appears **only once** in the MRO.

---

# MRO with Overriding

```python
class A:
    def show(self):
        print("A")

class B(A):
    def show(self):
        print("B")

class C(A):
    def show(self):
        print("C")

class D(B,C):
    pass
```

Now

```python
D().show()
```

Search

```
D

↓

B

↓

Found
```

Output

```
B
```

Python stops at the first match.

---

# super() and MRO

This is the most important interview topic.

Example

```python
class A:
    def hello(self):
        print("A")

class B(A):

    def hello(self):
        print("B")
        super().hello()

class C(B):
    pass

C().hello()
```

Output

```
B

A
```

Why?

`super()` does **not** mean "call my parent."

It means:

> Call the **next class in the MRO**.

For `C`, the MRO is:

```
C

↓

B

↓

A

↓

object
```

Inside `B`, `super()` goes to the next class in that MRO, which is `A`.

---

# `super()` in Diamond Inheritance

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

D().hello()
```

MRO

```python
print(D.mro())
```

```
D

↓

B

↓

C

↓

A

↓

object
```

Execution

```
D

↓

B

↓

C

↓

A
```

Output

```
D

B

C

A
```

Notice:

`A`

runs exactly once.

---

# Why is `super()` Better Than Calling Parent Directly?

Bad

```python
class B(A):

    def hello(self):

        A.hello(self)
```

In Diamond Inheritance

```
A
```

may execute multiple times.

Good

```python
super().hello()
```

Because `super()` respects the MRO and ensures cooperative multiple inheritance.

---

# C3 Linearization

Interviewers may ask:

> **Which algorithm does Python use?**

Answer:

**C3 Linearization**

Goals:

* Preserve left-to-right order
* Preserve parent-child relationships
* Visit every class exactly once
* Produce a consistent order

You usually don't need to compute it manually in interviews, but you should know its purpose.

---

# Real AI Example

```python
class Logger:

    def predict(self):
        print("Logging")
        super().predict()

class Cache:

    def predict(self):
        print("Cache")
        super().predict()

class BaseModel:

    def predict(self):
        print("Model")

class AIModel(Logger, Cache, BaseModel):
    pass
```

MRO

```
AIModel

↓

Logger

↓

Cache

↓

BaseModel

↓

object
```

Execution

```python
AIModel().predict()
```

Output

```
Logging

Cache

Model
```

Each layer adds behavior before delegating to the next class.

This "cooperative multiple inheritance" pattern is one reason Python's MRO exists.

---

# Interview Questions

## Q1. What is MRO?

MRO (Method Resolution Order) is the order in which Python searches classes for methods and attributes. You can inspect it using:

```python
ClassName.mro()
```

or

```python
ClassName.__mro__
```

---

## Q2. What is Diamond Inheritance?

```
      A
     / \
    B   C
     \ /
      D
```

A common ancestor (`A`) is inherited through multiple paths.

Python resolves this using the MRO so that `A` appears only once.

---

## Q3. Why does Python use MRO?

To provide a deterministic and consistent search order, avoid ambiguity in multiple inheritance, and ensure each class is visited only once.

---

## Q4. What is C3 Linearization?

It is the algorithm Python uses to compute the MRO. It guarantees:

* Local precedence order (left-to-right bases are respected where possible)
* Monotonicity (subclass MROs remain consistent with parent MROs)
* Each class appears only once

---

## Q5. Does `super()` call the parent class?

**No.**

This is a common misconception.

`super()` calls the **next class in the current class's MRO**, not necessarily the direct parent. That's why it works correctly with multiple inheritance.

---

# Summary

| Concept             | Meaning                                                        |
| ------------------- | -------------------------------------------------------------- |
| MRO                 | The order Python uses to search for methods and attributes     |
| `Class.mro()`       | Returns the method resolution order                            |
| Diamond Inheritance | A class inherits from two classes that share a common ancestor |
| `super()`           | Calls the next class in the MRO, not simply the parent         |
| C3 Linearization    | The algorithm Python uses to compute a consistent MRO          |

### Senior AI Engineer Interview Tip

If you're asked:

> **"Why does Python use C3 Linearization?"**

A concise, strong answer is:

> Python uses C3 Linearization to produce a deterministic method resolution order that respects inheritance relationships, maintains a consistent left-to-right ordering where possible, and ensures every class in a multiple-inheritance hierarchy is visited exactly once. This makes cooperative multiple inheritance with `super()` reliable and predictable.
