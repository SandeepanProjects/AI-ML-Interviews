These are **very common Senior Python interview questions** because they test whether you understand **Python's data model**.

The key idea is:

> **Magic methods (also called dunder methods, because of the double underscores) allow your objects to behave like Python's built-in objects.**

For example:

```python
len(my_object)
```

works because Python internally calls

```python
my_object.__len__()
```

Similarly,

```python
print(my_object)
```

internally calls

```python
my_object.__str__()
```

These methods are how Python implements operators and built-in functions.

---

# How Python Creates an Object

Consider

```python
class Person:

    def __new__(cls):
        print("__new__")
        return super().__new__(cls)

    def __init__(self):
        print("__init__")

p = Person()
```

Output

```text
__new__
__init__
```

Python performs these steps internally.

```text
Person()

↓

Call __new__()

↓

Create Object

↓

Return Object

↓

Call __init__()

↓

Return Initialized Object
```

So,

```text
__new__()

↓

creates object
```

while

```text
__init__()

↓

initializes object
```

---

# 1. `__new__`

## Purpose

Responsible for **creating** a new object.

Signature

```python
class A:

    def __new__(cls):
        print("Creating object")
        return super().__new__(cls)

    def __init__(self):
        print("Initializing object")

a = A()
```

Output

```text
Creating object
Initializing object
```

---

## Internal Flow

Python executes

```python
A()
```

Equivalent to

```python
obj = A.__new__(A)

A.__init__(obj)
```

So

```text
__new__()

↓

memory allocation
```

---

## Why Override `__new__`?

Rarely needed, but useful for:

* Immutable classes
* Singleton pattern
* Object caching
* Custom object creation

Example (Singleton)

```python
class Singleton:

    _instance = None

    def __new__(cls):

        if cls._instance is None:
            cls._instance = super().__new__(cls)

        return cls._instance
```

Now

```python
a = Singleton()
b = Singleton()

print(a is b)
```

Output

```text
True
```

Only one object exists.

---

# 2. `__init__`

After object creation,

Python calls

```python
__init__()
```

Example

```python
class Student:

    def __init__(self,name):
        self.name = name
```

```python
s = Student("Alice")
```

Memory

```text
Student Object

↓

name

↓

Alice
```

---

## Important Interview Question

Does `__init__` create the object?

**No.**

Many beginners think so.

Actually

```text
__new__

↓

creates object

↓

__init__

↓

initializes object
```

---

# 3. `__str__`

Used for **human-readable** representation.

Example

```python
class Student:

    def __str__(self):
        return "Student Object"

s = Student()

print(s)
```

Output

```text
Student Object
```

Without `__str__`

Output

```text
<__main__.Student object at 0x...>
```

---

## Internally

Python converts

```python
print(s)
```

into

```python
print(s.__str__())
```

---

# 4. `__repr__`

Developer-friendly representation.

Example

```python
class Student:

    def __repr__(self):
        return "Student('Alice')"

s = Student()

print(repr(s))
```

Output

```text
Student('Alice')
```

---

## Difference

```python
print(obj)
```

uses

```text
__str__()
```

while

```python
repr(obj)
```

uses

```text
__repr__()
```

If `__str__` is missing,

Python falls back to `__repr__`.

---

### Best Practice

`__repr__` should ideally return an unambiguous representation that could recreate the object if possible.

Example

```python
Student("Alice")
```

---

# 5. `__len__`

Controls

```python
len(obj)
```

Example

```python
class Team:

    def __init__(self):
        self.members = ["A","B","C"]

    def __len__(self):
        return len(self.members)

team = Team()

print(len(team))
```

Output

```text
3
```

Internally

```python
len(team)
```

becomes

```python
team.__len__()
```

---

# 6. `__iter__`

Makes an object iterable.

Example

```python
class Numbers:

    def __iter__(self):
        return iter([1,2,3])
```

Now

```python
for i in Numbers():
    print(i)
```

Output

```text
1
2
3
```

---

## Internally

Python executes

```text
for x in obj
```

as

```python
iterator = obj.__iter__()

while True:
    value = next(iterator)
```

---

# 7. `__next__`

Represents the **next value** of an iterator.

Example

```python
class Counter:

    def __init__(self):
        self.i = 1

    def __iter__(self):
        return self

    def __next__(self):

        if self.i > 5:
            raise StopIteration

        value = self.i
        self.i += 1

        return value
```

Usage

```python
for x in Counter():
    print(x)
```

Output

```text
1
2
3
4
5
```

---

## Internal Flow

Python performs

```python
iterator = Counter().__iter__()

next(iterator)

next(iterator)

next(iterator)
```

When

```python
raise StopIteration
```

occurs,

the loop ends.

---

## Memory

```text
Counter Object

↓

i =1

↓

next()

↓

i =2

↓

next()

↓

i =3
```

The iterator stores its own state.

---

# 8. `__call__`

Allows an object to behave like a function.

Example

```python
class Multiply:

    def __call__(self,x,y):
        return x*y

m = Multiply()

print(m(3,4))
```

Output

```text
12
```

Internally

```python
m(3,4)
```

becomes

```python
m.__call__(3,4)
```

---

## Why Useful?

Widely used in

* PyTorch (`nn.Module`)
* TensorFlow layers
* FastAPI dependencies
* AI pipelines
* Callable objects

Example

```python
class Model:

    def __call__(self,text):
        return text.upper()

model = Model()

print(model("hello"))
```

Output

```text
HELLO
```

Looks like a function,

but it's an object.

---

# Complete Example

```python
class Counter:

    def __init__(self,n):
        self.n = n

    def __len__(self):
        return self.n

    def __str__(self):
        return f"Counter({self.n})"

    def __call__(self):
        return self.n * 2

c = Counter(5)

print(c)

print(len(c))

print(c())
```

Output

```text
Counter(5)

5

10
```

Python internally calls

```python
print(c)
```

↓

```python
c.__str__()
```

---

```python
len(c)
```

↓

```python
c.__len__()
```

---

```python
c()
```

↓

```python
c.__call__()
```

---

# Interview Questions

### Q1. Difference between `__new__` and `__init__`?

| `__new__`               | `__init__`             |
| ----------------------- | ---------------------- |
| Creates the object      | Initializes the object |
| Runs first              | Runs after `__new__`   |
| Must return an instance | Returns `None`         |
| Rarely overridden       | Commonly overridden    |

---

### Q2. Difference between `__str__` and `__repr__`?

| `__str__`                          | `__repr__`                              |
| ---------------------------------- | --------------------------------------- |
| Human-readable                     | Developer/debug representation          |
| Used by `print()`                  | Used by `repr()` and interactive shells |
| Falls back to `__repr__` if absent | Preferred to be unambiguous             |

---

### Q3. Difference between `__iter__` and `__next__`?

| `__iter__`                            | `__next__`                            |
| ------------------------------------- | ------------------------------------- |
| Returns an iterator                   | Returns the next item                 |
| Called once at the start of iteration | Called repeatedly during iteration    |
| Enables `for` loops                   | Raises `StopIteration` when exhausted |

---

### Q4. What does `__call__` do?

It makes an instance **callable**, so you can invoke it like a function:

```python
obj()
```

instead of

```python
obj.method()
```

---

# Summary

| Magic Method | Triggered By                   | Purpose                               |
| ------------ | ------------------------------ | ------------------------------------- |
| `__new__`    | `Class()`                      | Create a new object                   |
| `__init__`   | After `__new__`                | Initialize object state               |
| `__str__`    | `print(obj)`, `str(obj)`       | Human-readable string                 |
| `__repr__`   | `repr(obj)`, interactive shell | Developer/debug representation        |
| `__len__`    | `len(obj)`                     | Return object length                  |
| `__iter__`   | `iter(obj)`, `for` loop        | Return an iterator                    |
| `__next__`   | `next(iterator)`               | Return the next value                 |
| `__call__`   | `obj()`                        | Make an object behave like a function |

## Senior AI Engineer Interview Tip

A common follow-up question is:

> **Why do frameworks like PyTorch use `__call__`?**

A good answer is:

> `__call__` lets model instances behave like functions (`output = model(input)`), while still allowing the class to manage internal state, hooks, logging, validation, caching, and preprocessing. Internally, `model(input)` invokes `model.__call__(input)`, which can perform extra work before and after delegating to the actual computation method (such as `forward()` in PyTorch). This provides a clean API without sacrificing flexibility.
