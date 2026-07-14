These are **very common Senior Python interview questions** because they test whether you understand **descriptors, method binding, and object-oriented design** rather than just syntax.

---

# 1. `@property`

## Problem Without `@property`

Suppose you have a class:

```python
class Employee:

    def __init__(self, salary):
        self.salary = salary
```

Usage:

```python
e = Employee(50000)

print(e.salary)
```

Output

```text
50000
```

Everything looks fine.

---

## New Requirement

Later your manager says:

> Salary should never be negative.

Without `@property`, you have a problem.

```python
e.salary = -100
```

Now

```python
print(e.salary)
```

Output

```text
-100
```

Invalid data entered the object.

---

# Solution 1 (Not Good)

You might create methods.

```python
class Employee:

    def __init__(self, salary):
        self._salary = salary

    def get_salary(self):
        return self._salary

    def set_salary(self, salary):

        if salary < 0:
            raise ValueError

        self._salary = salary
```

Usage

```python
e.set_salary(10000)

print(e.get_salary())
```

This works.

But it's not Pythonic.

---

# Solution 2 : `@property`

```python
class Employee:

    def __init__(self, salary):
        self._salary = salary

    @property
    def salary(self):
        return self._salary

    @salary.setter
    def salary(self, value):

        if value < 0:
            raise ValueError("Salary cannot be negative")

        self._salary = value
```

Usage

```python
e = Employee(50000)

print(e.salary)

e.salary = 60000
```

Looks exactly like attribute access.

But methods are executing.

---

# Internally

When Python sees

```python
print(e.salary)
```

it actually does

```python
Employee.salary.__get__(e)
```

which eventually calls

```python
salary()
```

Likewise,

```python
e.salary = 100
```

becomes

```python
Employee.salary.__set__(e,100)
```

which invokes the setter.

The user thinks they are reading or writing an attribute, but Python is actually calling methods.

---

# Why Use `@property`?

It lets you:

* Validate data
* Compute values lazily
* Keep a stable public API
* Add logging
* Add caching
* Preserve backward compatibility

Example

```python
class Circle:

    def __init__(self,r):
        self.radius = r

    @property
    def area(self):
        return 3.14*self.radius*self.radius
```

Usage

```python
c = Circle(5)

print(c.area)
```

Notice

```python
c.area
```

looks like an attribute,

but is actually

```python
c.area()
```

behind the scenes.

---

# Read-only Property

```python
class Student:

    def __init__(self,name):
        self._name = name

    @property
    def name(self):
        return self._name
```

Now

```python
s.name = "Bob"
```

raises

```text
AttributeError
```

because no setter exists.

---

# 2. `@staticmethod`

Suppose

```python
class Math:

    def add(a,b):
        return a+b
```

Calling

```python
Math.add(3,4)
```

fails because Python expects `self`.

---

Instead

```python
class Math:

    @staticmethod
    def add(a,b):
        return a+b
```

Now

```python
Math.add(3,4)
```

Output

```text
7
```

---

## Why?

Static methods receive

* no `self`
* no `cls`

They behave like normal functions that happen to live inside a class.

Memory

```text
Math Class

↓

add()

No object required
```

---

# Example

```python
class Temperature:

    @staticmethod
    def c_to_f(c):
        return c*9/5+32
```

Usage

```python
Temperature.c_to_f(20)
```

Output

```text
68
```

No object required.

---

# When to Use Static Methods

Use when the method

* belongs logically to the class
* doesn't need instance data
* doesn't need class data

Examples

* validation
* conversion
* helper utilities
* math functions

---

# 3. `@classmethod`

Unlike static methods,

class methods receive

```python
cls
```

instead of

```python
self
```

Example

```python
class Employee:

    company = "OpenAI"

    @classmethod
    def show_company(cls):
        print(cls.company)
```

Usage

```python
Employee.show_company()
```

Output

```text
OpenAI
```

---

Internally

Python converts

```python
Employee.show_company()
```

into

```python
Employee.show_company(Employee)
```

where

```python
cls = Employee
```

---

# Why Use Class Methods?

Access

* class variables
* alternative constructors
* factory methods

---

# Factory Method Example

Instead of

```python
Employee("Alice",25)
```

Suppose data comes from CSV.

```python
class Employee:

    def __init__(self,name,age):
        self.name=name
        self.age=age

    @classmethod
    def from_string(cls,text):

        name,age=text.split(",")

        return cls(name,int(age))
```

Usage

```python
e = Employee.from_string("Alice,25")
```

Output

```text
Employee("Alice",25)
```

Very common in production code.

---

# Another Example

```python
class Date:

    def __init__(self,y,m,d):
        self.y=y
        self.m=m
        self.d=d

    @classmethod
    def today(cls):

        return cls(2026,7,14)
```

Usage

```python
d = Date.today()
```

No constructor details exposed.

---

# Internal Difference

## Instance Method

```python
class A:

    def show(self):
        ...
```

Calling

```python
a.show()
```

Python does

```python
A.show(a)
```

---

## Class Method

```python
class A:

    @classmethod
    def show(cls):
        ...
```

Calling

```python
A.show()
```

Python does

```python
A.show(A)
```

---

## Static Method

```python
class A:

    @staticmethod
    def show():
        ...
```

Calling

```python
A.show()
```

Python does

```python
show()
```

No automatic argument is passed.

---

# Memory Diagram

```text
                Class

                  |

      -------------------------

      |           |          |

Instance      Class       Static

Method        Method      Method

      |           |          |

self          cls        nothing
```

---

# Real AI Example

```python
class LLM:

    DEFAULT_MODEL = "gpt"

    def __init__(self,name):
        self.name=name

    @property
    def model(self):
        return self.name.upper()

    @staticmethod
    def validate(prompt):
        return len(prompt)>0

    @classmethod
    def default(cls):
        return cls(cls.DEFAULT_MODEL)
```

Usage

```python
m = LLM.default()

print(m.model)

print(LLM.validate("Hello"))
```

Output

```text
GPT
True
```

---

# Interview Questions

## Q1. Why use `@property` instead of getter/setter methods?

A `@property` lets you expose an attribute-like API while retaining the ability to validate, compute, cache, or log access. It also preserves backward compatibility if you later change an attribute into computed logic without changing client code.

---

## Q2. Difference between instance, static, and class methods?

| Feature                   | Instance Method                                                               | `@classmethod`                       | `@staticmethod`                                |
| ------------------------- | ----------------------------------------------------------------------------- | ------------------------------------ | ---------------------------------------------- |
| First argument            | `self`                                                                        | `cls`                                | None                                           |
| Access instance variables | ✅                                                                             | ❌                                    | ❌                                              |
| Access class variables    | ✅                                                                             | ✅                                    | ❌ (unless referenced explicitly by class name) |
| Can create objects        | ✅                                                                             | ✅ (commonly used as factory methods) | ❌                                              |
| Requires an object        | Usually yes (though it can be called via the class with an explicit instance) | No                                   | No                                             |

---

## Q3. When should you use each?

* **Instance method**: When you need to read or modify the state of a specific object.
* **Class method**: When working with class-level state or providing alternative constructors/factory methods.
* **Static method**: When the logic belongs conceptually to the class but doesn't need access to either the instance or the class.

---

# Summary

| Decorator       | Receives | Main Purpose                                    | Typical Use                                                    |
| --------------- | -------- | ----------------------------------------------- | -------------------------------------------------------------- |
| `@property`     | `self`   | Attribute-style access with getter/setter logic | Validation, computed values, read-only attributes              |
| `@classmethod`  | `cls`    | Operate on the class itself                     | Factory methods, alternative constructors, class configuration |
| `@staticmethod` | Nothing  | Utility function grouped with the class         | Validation, conversions, helper functions                      |

## Senior AI Engineer Interview Tip

A common follow-up question is:

> **Why is `from_pretrained()` in libraries like Hugging Face often a class method?**

A strong answer is:

> `from_pretrained()` is typically implemented as a **class method** because it acts as an alternative constructor. It creates and returns an instance of the class based on external resources (such as model weights and configuration) without requiring the caller to instantiate the object first. Using `cls(...)` instead of a hard-coded class name also allows subclasses to inherit the method and correctly construct instances of the subclass.
