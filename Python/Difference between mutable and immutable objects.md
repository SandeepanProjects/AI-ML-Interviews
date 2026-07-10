This is one of the most frequently asked **Senior AI Engineer Python interview questions** because it affects memory usage, function behavior, multithreading, caching, and performance.

---

# What are Mutable and Immutable Objects?

The difference is simple:

* **Mutable object** → Can be modified after it is created.
* **Immutable object** → Cannot be modified after it is created. Any "change" creates a new object.

Think of it like this:

* **Mutable** = Whiteboard (erase and write again)
* **Immutable** = Printed paper (you need a new sheet to make changes)

---

# Example

## Immutable

```python
x = 10

print(id(x))

x = x + 5

print(id(x))
```

Output

```
140225340122
140225340500
```

Notice the IDs are different.

Python created a **new integer object** instead of modifying the old one.

Memory

```
Before

x
 |
 v
10

After

x
 |
 v
15

10 still exists until garbage collected.
```

---

## Mutable

```python
numbers = [1,2,3]

print(id(numbers))

numbers.append(4)

print(id(numbers))
```

Output

```
43820400
43820400
```

Same object.

Python modified the existing list.

Memory

```
Before

numbers
   |
   v
[1,2,3]

After

numbers
   |
   v
[1,2,3,4]
```

---

# Which Python Objects are Immutable?

| Type      | Immutable                         |
| --------- | --------------------------------- |
| int       | ✅                                 |
| float     | ✅                                 |
| bool      | ✅                                 |
| complex   | ✅                                 |
| str       | ✅                                 |
| tuple     | ✅ (if all elements are immutable) |
| bytes     | ✅                                 |
| frozenset | ✅                                 |

---

# Which Objects are Mutable?

| Type                        | Mutable |
| --------------------------- | ------- |
| list                        | ✅       |
| dict                        | ✅       |
| set                         | ✅       |
| bytearray                   | ✅       |
| most custom class instances | ✅       |

---

# Why Strings are Immutable?

Example

```python
name = "John"

print(id(name))

name += " Doe"

print(id(name))
```

Python creates

```
"John"

↓

"John Doe"
```

instead of modifying

```
John
```

This enables:

* Hashing
* Thread safety
* Dictionary keys
* String interning

---

# Why Lists are Mutable?

```python
nums = [1,2,3]

nums.append(4)
```

Python changes

```
[1,2,3]

↓

[1,2,3,4]
```

instead of creating another list.

This saves memory when frequent updates are needed.

---

# id() Demonstration

```python
x = [1,2]

print(id(x))

x.append(3)

print(id(x))
```

Same object.

Now compare

```python
x = "AI"

print(id(x))

x += " Engineer"

print(id(x))
```

Different object.

---

# Mutable Inside Immutable

A tuple itself is immutable, but it can contain mutable objects.

```python
t = ([1,2], 10)

t[0].append(3)

print(t)
```

Output

```
([1,2,3], 10)
```

Why?

The tuple's references don't change, but the list referenced by the tuple is modified.

---

# Function Arguments

## Mutable Example

```python
def add_item(lst):
    lst.append(100)

nums = [1,2]

add_item(nums)

print(nums)
```

Output

```
[1,2,100]
```

The original list changes because the function receives a reference to the same object.

---

## Immutable Example

```python
def increment(x):
    x += 1

num = 5

increment(num)

print(num)
```

Output

```
5
```

The integer is immutable, so `x += 1` creates a new integer object local to the function.

---

# Common Interview Trap: Mutable Default Arguments

Bad

```python
def add(item, lst=[]):
    lst.append(item)
    return lst

print(add(1))
print(add(2))
```

Output

```
[1]
[1,2]
```

The default list is created once when the function is defined and reused across calls.

Correct

```python
def add(item, lst=None):
    if lst is None:
        lst = []

    lst.append(item)
    return lst

print(add(1))
print(add(2))
```

Output

```
[1]
[2]
```

---

# Copying Mutable Objects

## Assignment

```python
a = [1,2]

b = a

b.append(3)

print(a)
```

Output

```
[1,2,3]
```

Both variables point to the same list.

---

## Shallow Copy

```python
import copy

a = [[1],[2]]

b = copy.copy(a)

b[0].append(100)

print(a)
```

Output

```
[[1,100],[2]]
```

The outer list is copied, but the nested lists are shared.

---

## Deep Copy

```python
import copy

a = [[1],[2]]

b = copy.deepcopy(a)

b[0].append(100)

print(a)
```

Output

```
[[1],[2]]
```

A deep copy recursively duplicates nested objects.

---

# Why Immutability Matters in AI Systems

Immutability is valuable because immutable objects:

* Are inherently thread-safe.
* Can be used as dictionary keys and cache keys.
* Help avoid accidental side effects.
* Simplify reasoning about concurrent code.

Examples:

* Cache keys for embedding or prompt caches (`str`, `tuple`).
* Configuration objects that should never change during runtime.
* Model identifiers and version numbers.

Mutability is useful for:

* Batches of data being processed.
* Feature dictionaries that are updated.
* Training state (weights, gradients, optimizer state).
* Buffers and queues in streaming systems.

---

# Interview Follow-up: Why Can a Tuple Be a Dictionary Key but a List Cannot?

Dictionary keys must be **hashable**. Immutable objects generally have a stable hash because their contents cannot change.

```python
d = {}

d[(1, 2)] = "valid"

print(d)
```

This works because the tuple is immutable.

```python
d = {}

d[[1, 2]] = "invalid"
```

This raises:

```text
TypeError: unhashable type: 'list'
```

A list is mutable, so if it were allowed as a key, changing its contents would invalidate the dictionary's internal hash table.

---

# Senior AI Engineer Interview Answer (30–60 seconds)

> **Mutable objects** can be modified in place after creation, such as `list`, `dict`, and `set`. Their identity (`id`) remains the same when updated. **Immutable objects**, such as `int`, `float`, `str`, `tuple`, and `frozenset`, cannot be modified; any apparent change creates a new object with a different identity. Immutability provides predictable behavior, hashability, and thread safety, making immutable objects suitable for dictionary keys, caches, and configuration data. Mutability is preferred for data structures that require efficient in-place updates, such as lists, dictionaries, and model state during machine learning workflows.
