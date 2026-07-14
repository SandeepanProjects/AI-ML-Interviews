# Python Collections Module (Senior Interview Level)

The `collections` module provides specialized data structures that are optimized for common programming patterns.

Common interview questions:

* `defaultdict`
* `Counter`
* `deque`
* `namedtuple`
* `ChainMap`
* `OrderedDict`

These are frequently used in:

* Data processing
* ML pipelines
* Caching
* Streaming systems
* Configuration management

---

# 1. defaultdict

## Problem with normal dictionary

Example:

```python
data = {}

data["apple"].append(1)
```

Error:

```
KeyError: 'apple'
```

Because the key does not exist.

---

## defaultdict Solution

`defaultdict` automatically creates a default value when a key is missing.

Syntax:

```python
from collections import defaultdict
```

---

## Example: Grouping Data

Normal dictionary:

```python
students = {}

names = [
    ("AI", "John"),
    ("AI", "Bob"),
    ("ML", "Alice")
]


for course, name in names:

    if course not in students:
        students[course] = []

    students[course].append(name)
```

---

Using defaultdict:

```python
from collections import defaultdict


students = defaultdict(list)


for course, name in names:

    students[course].append(name)


print(students)
```

Output:

```python
{
 'AI':['John','Bob'],
 'ML':['Alice']
}
```

---

## How defaultdict Works Internally

```python
d = defaultdict(list)
```

means:

When missing key:

```python
d["new"]
```

Python executes:

```python
list()
```

creates:

```python
[]
```

and stores it.

---

# Common Default Factories

## List

```python
defaultdict(list)
```

Output:

```python
{
a:[1,2]
}
```

Use:

* Grouping
* Graph adjacency list

---

## Integer

```python
defaultdict(int)
```

Default:

```python
0
```

Example:

```python
counter = defaultdict(int)

counter["apple"] += 1

print(counter)
```

Output:

```
{'apple':1}
```

---

## Set

```python
defaultdict(set)
```

Useful for unique values.

---

# AI Example

Document indexing:

```python
index = defaultdict(list)


for word, doc_id in documents:

    index[word].append(doc_id)
```

Creates:

```
word -> documents
```

similar to search engines.

---

---

# 2. Counter

`Counter` counts occurrences.

Example:

```python
from collections import Counter


words = [
    "AI",
    "ML",
    "AI",
    "Python"
]


count = Counter(words)

print(count)
```

Output:

```
{
'AI':2,
'ML':1,
'Python':1
}
```

---

# Internally

Counter is a dictionary:

```python
Counter(
{
"A":3,
"B":2
}
)
```

Equivalent:

```python
{
"A":3,
"B":2
}
```

---

# Most Common Elements

Very common interview question.

```python
count.most_common(2)
```

Output:

```
[
('AI',2),
('ML',1)
]
```

---

# Counter Operations

Example:

```python
a = Counter(
    {"A":3,"B":2}
)

b = Counter(
    {"A":1,"B":5}
)
```

---

Addition:

```python
a+b
```

Result:

```
A:4
B:7
```

---

Intersection:

```python
a & b
```

Result:

```
A:1
B:2
```

---

# AI Example

Token frequency:

```python
tokens = [
"transformer",
"attention",
"transformer"
]


frequency = Counter(tokens)
```

Output:

```
transformer:2
attention:1
```

---

---

# 3. deque (Double Ended Queue)

A deque allows fast insertion/removal from both ends.

Import:

```python
from collections import deque
```

---

Normal list:

```python
list.append()
```

Fast:

```
O(1)
```

But:

```python
list.insert(0,x)
```

Slow:

```
O(n)
```

because elements shift.

---

Deque:

```python
d = deque()

d.append(1)

d.appendleft(0)
```

Result:

```
[0,1]
```

---

# Operations

## Add Right

```python
d.append(10)
```

---

## Add Left

```python
d.appendleft(5)
```

---

## Remove Right

```python
d.pop()
```

---

## Remove Left

```python
d.popleft()
```

---

# Example

```python
from collections import deque


queue = deque()


queue.append("Task1")
queue.append("Task2")


print(queue.popleft())
```

Output:

```
Task1
```

---

# Why deque over list?

Queue:

```text
FIFO

Task1
Task2
Task3
```

Need:

```
remove from front
```

Deque:

```
O(1)
```

List:

```
O(n)
```

---

# AI Example

Streaming tokens:

```
Token Stream

[A][B][C][D]

consume left
add right
```

Use:

```python
deque
```

---

# 4. namedtuple

A lightweight immutable object.

Problem:

Normal tuple:

```python
user = (
    "John",
    30,
    "AI Engineer"
)

print(user[0])
```

Problem:

Index-based access is confusing.

---

namedtuple:

```python
from collections import namedtuple


User = namedtuple(
    "User",
    [
        "name",
        "age",
        "role"
    ]
)


u = User(
    "John",
    30,
    "AI Engineer"
)


print(u.name)
```

Output:

```
John
```

---

# Internally

It is still a tuple.

```python
isinstance(u, tuple)
```

Output:

```
True
```

---

# Immutable

```python
u.age = 40
```

Error:

```
AttributeError
```

---

# When Use?

Good for:

* Configuration objects
* API responses
* Read-only records

---

# namedtuple vs class

| namedtuple  | Class            |
| ----------- | ---------------- |
| Lightweight | More flexible    |
| Immutable   | Mutable possible |
| Less code   | More code        |
| No methods  | Can have methods |

---

# 5. ChainMap

ChainMap combines multiple dictionaries into one view.

Example:

Configuration system:

```
Default config

        +
Environment config

        +
User config
```

---

Example:

```python
from collections import ChainMap


default = {
    "timeout":30,
    "host":"localhost"
}


user = {
    "timeout":10
}


config = ChainMap(
    user,
    default
)


print(config["timeout"])
```

Output:

```
10
```

---

Lookup order:

```
user
 |
default
```

First match wins.

---

Example:

```python
print(config["host"])
```

Output:

```
localhost
```

Because:

```
user
(no host)

default
(found)
```

---

# Real Production Example

Application configuration:

```
Priority:

Command line arguments

        ↓

Environment variables

        ↓

Default settings
```

ChainMap models this.

---

# 6. OrderedDict

Before Python 3.7:

Normal dictionary:

```python
dict
```

did not guarantee order.

So:

```python
OrderedDict
```

was used.

---

Example:

```python
from collections import OrderedDict


d = OrderedDict()


d["a"]=1
d["b"]=2
d["c"]=3


print(d)
```

Output:

```
a,b,c
```

---

# Important Methods

## Move item to end

```python
d.move_to_end("a")
```

Result:

```
b,c,a
```

---

## Move item to beginning

```python
d.move_to_end(
"a",
last=False
)
```

---

# OrderedDict vs dict

Modern Python:

```python
dict
```

preserves insertion order.

Example:

```python
d={
"a":1,
"b":2
}
```

Order maintained.

---

So why OrderedDict?

Special operations:

* move_to_end()
* order-sensitive equality
* older Python compatibility

---

# LRU Cache Example

OrderedDict is commonly used to implement LRU cache.

Example:

```
Capacity = 3

A B C

Access A

B C A

Add D

Evict B
```

Implementation uses:

```python
OrderedDict
```

---

# Comparison Table

| Collection  | Purpose               | Example Use        |
| ----------- | --------------------- | ------------------ |
| defaultdict | Missing key handling  | Grouping           |
| Counter     | Counting              | Frequency analysis |
| deque       | Fast queue            | Streaming          |
| namedtuple  | Immutable record      | API response       |
| ChainMap    | Multiple dictionaries | Config priority    |
| OrderedDict | Ordered operations    | LRU cache          |

---

# Senior AI Engineer Examples

## 1. RAG Document Grouping

```python
chunks = defaultdict(list)

chunks[document_id].append(chunk)
```

---

## 2. Token Statistics

```python
tokens = Counter(text.split())
```

---

## 3. Streaming LLM Tokens

```python
buffer = deque()

buffer.append(token)

token = buffer.popleft()
```

---

## 4. Model Configuration

```python
config = ChainMap(
    env_config,
    default_config
)
```

---

## 5. LRU Response Cache

```python
cache = OrderedDict()
```

---

# Interview Summary

### defaultdict

> Dictionary that automatically creates missing values using a factory function.

### Counter

> Dictionary specialized for counting hashable objects.

### deque

> Double-ended queue optimized for O(1) insertion and removal from both ends.

### namedtuple

> Immutable tuple with named fields for readable access.

### ChainMap

> Combines multiple dictionaries into one lookup chain.

### OrderedDict

> Dictionary with ordering operations such as moving keys, mainly used for specialized ordering and LRU caches.

These are frequently expected knowledge for **Senior Python, Backend, and AI Engineer roles**.
