These are among the **most frequently asked Python interview topics** for **Senior AI Engineer**, **Senior Backend Engineer**, and **Senior Python Developer** roles.

Interviewers don't just want definitions—they want you to explain **why OOP exists, how Python implements it internally, and where it is used in production AI systems**.

---

# What is OOP?

**Object-Oriented Programming (OOP)** is a programming paradigm where you model your application as a collection of **objects** that encapsulate both **data (attributes)** and **behavior (methods)**.

Instead of writing:

```python
name = "Alice"
age = 30
```

you model a person as an object:

```python
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def introduce(self):
        print(f"My name is {self.name}")
```

Now the data and operations belong together.

---

# Why OOP?

Imagine building an AI platform.

Without OOP:

```python
load_model()
predict()
save_model()
```

As the project grows, code becomes difficult to organize.

With OOP:

```python
class Model:
    def load(self):
        ...

    def predict(self):
        ...

    def save(self):
        ...
```

Everything related to the model is grouped together.

---

# 1. Class

## Definition

A **class** is a blueprint for creating objects.

It defines:

* attributes (data)
* methods (behavior)

Example

```python
class Car:

    def __init__(self, brand):
        self.brand = brand

    def drive(self):
        print("Driving")
```

Think of a class as an architectural blueprint.

```text
Blueprint

Car

↓

brand

drive()
```

Nothing exists yet.

No memory for a car has been allocated.

---

## Internally

When Python executes

```python
class Car:
    ...
```

Python creates a **class object**.

```text
Car

↓

Class Object
```

Even classes are objects in Python.

Check

```python
print(type(Car))
```

Output

```text
<class 'type'>
```

---

# 2. Object

An object is an **instance of a class**.

```python
car1 = Car("Tesla")
```

Memory

```text
car1

↓

Object

brand="Tesla"
```

Create another

```python
car2 = Car("BMW")
```

Memory

```text
car1

↓

Tesla


car2

↓

BMW
```

Both use the same class.

Different objects.

---

## Internal Flow

```python
car = Car("Tesla")
```

Python does

1. Allocate memory
2. Create object
3. Call `__init__`
4. Return reference

Equivalent

```text
allocate memory

↓

initialize attributes

↓

return object
```

---

# self

Interview favorite.

```python
class Car:

    def drive(self):
        print(self)
```

```python
car = Car()

car.drive()
```

Python internally converts

```python
car.drive()
```

into

```python
Car.drive(car)
```

So

```python
self
```

is simply the current object.

---

# 3. Encapsulation

## Definition

Encapsulation means:

> **Bundling data and methods together while controlling access to internal state.**

Example

```python
class BankAccount:

    def __init__(self):
        self.balance = 1000

    def deposit(self, amount):
        self.balance += amount
```

The object manages its own state.

---

## Python Access Levels

### Public

```python
self.balance
```

Accessible everywhere.

---

### Protected (by convention)

```python
self._balance
```

Means:

> "Internal use. Please don't access directly."

Python does **not** enforce it.

---

### Private

```python
self.__balance
```

Python performs **name mangling**.

Example

```python
class A:

    def __init__(self):
        self.__x = 10
```

Internally Python renames it to

```text
_A__x
```

So

```python
obj.__x
```

fails

but

```python
obj._A__x
```

works.

This is **not true security**, just collision avoidance.

---

## Example

```python
class Account:

    def __init__(self):
        self.__balance = 100

    def deposit(self, amount):
        self.__balance += amount

    def get_balance(self):
        return self.__balance
```

Users cannot accidentally modify the internal state through `__balance` without bypassing the convention.

---

# 4. Inheritance

Inheritance allows one class to reuse another class.

Example

```python
class Animal:

    def speak(self):
        print("Sound")
```

```python
class Dog(Animal):

    def bark(self):
        print("Woof")
```

```python
dog = Dog()

dog.speak()
dog.bark()
```

Output

```text
Sound

Woof
```

---

## Memory

```text
Dog Object

↓

Dog Class

↓

Animal Class

↓

object
```

Python searches methods using the **Method Resolution Order (MRO)**.

---

## MRO

```python
class A:
    pass

class B(A):
    pass

class C(B):
    pass
```

Search order

```text
C

↓

B

↓

A

↓

object
```

Check

```python
print(C.mro())
```

---

# 5. Abstraction

Abstraction means

> Expose only what is necessary and hide implementation details.

Example

You drive a car.

You use

```text
Steering

Brake

Accelerator
```

You don't need to know how the engine works internally.

---

## Python Example

```python
from abc import ABC, abstractmethod

class Animal(ABC):

    @abstractmethod
    def speak(self):
        pass
```

Child

```python
class Dog(Animal):

    def speak(self):
        print("Woof")
```

Without implementing `speak()`, you cannot instantiate `Dog`.

---

## Why?

The abstract base class defines the **contract** that subclasses must follow.

This is useful when building pluggable systems (for example, different LLM providers all implementing a common interface).

---

# 6. Polymorphism

Polymorphism means

> One interface, many implementations.

Example

```python
class Dog:

    def speak(self):
        print("Woof")
```

```python
class Cat:

    def speak(self):
        print("Meow")
```

```python
animals = [

    Dog(),

    Cat()

]

for animal in animals:
    animal.speak()
```

Output

```text
Woof

Meow
```

Same method name.

Different behavior.

---

## Duck Typing

Python uses **duck typing**.

It doesn't care about the class.

It only cares whether the object supports the required operation.

Example

```python
class Duck:

    def fly(self):
        print("Duck")
```

```python
class Plane:

    def fly(self):
        print("Plane")
```

```python
def start(x):
    x.fly()

start(Duck())
start(Plane())
```

Output

```text
Duck

Plane
```

Python never checks whether `Duck` and `Plane` share a common base class.

If the object has a `fly()` method, it works.

---

# OOP in AI Systems

These principles are heavily used in production AI systems.

```python
class BaseLLM:
    def generate(self, prompt):
        raise NotImplementedError
```

Different implementations:

```python
class OpenAIModel(BaseLLM):
    ...

class ClaudeModel(BaseLLM):
    ...

class LlamaModel(BaseLLM):
    ...
```

Application code:

```python
def answer(model: BaseLLM, prompt):
    return model.generate(prompt)
```

The caller doesn't need to know which model is being used. This is **polymorphism** and **abstraction** working together.

---

# Interview Questions

### What is the difference between a class and an object?

| Class                          | Object                   |
| ------------------------------ | ------------------------ |
| Blueprint                      | Instance of a class      |
| Defines attributes and methods | Holds actual data        |
| Created once                   | Many instances can exist |

---

### What is encapsulation?

Grouping related data and methods inside a class while providing controlled access to internal state.

---

### What is inheritance?

A mechanism where a subclass reuses and extends the behavior of a parent class.

---

### What is abstraction?

Providing a simplified interface while hiding implementation details. In Python, abstract base classes (`abc.ABC`) are commonly used to define required methods.

---

### What is polymorphism?

Using the same interface (for example, a method like `generate()` or `speak()`) with different implementations depending on the object.

---

# Summary

| Concept       | Meaning                                   | Production Example                              |
| ------------- | ----------------------------------------- | ----------------------------------------------- |
| Class         | Blueprint for objects                     | `LLM`, `VectorStore`, `Model`                   |
| Object        | Instance of a class                       | `OpenAIModel()`, `FAISSStore()`                 |
| Encapsulation | Bundle data and behavior, control access  | `BankAccount`, `Model` managing its state       |
| Inheritance   | Reuse and extend a base class             | `OpenAIModel(BaseLLM)`                          |
| Abstraction   | Define a contract, hide implementation    | `BaseLLM.generate()`                            |
| Polymorphism  | Same interface, different implementations | `generate()` on OpenAI, Claude, or Llama models |

## Senior AI Engineer Interview Tip

A common follow-up question is:

> **"Python doesn't have true private members. How does encapsulation work?"**

A strong answer is:

> Python relies on conventions (`_attribute`) and name mangling (`__attribute`) rather than strict access modifiers. Encapsulation in Python is primarily about designing clear interfaces and protecting object state through methods and properties, rather than enforcing access restrictions at the language level.
