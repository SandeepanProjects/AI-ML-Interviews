# Generators Explained Like a Senior AI Engineer

**Generators** are one of the most frequently asked Python interview topics for **Senior AI Engineer**, **ML Engineer**, **Backend Engineer**, and **LLM Infrastructure Engineer** roles.

Interviewers typically ask:

* What is a generator?
* Why do we need generators?
* How does `yield` work internally?
* Generator vs function?
* Generator vs iterator?
* Why are generators memory efficient?
* Where are generators used in AI systems?

Let's build the concept from first principles.

---

# Why Do We Need Generators?

Suppose you want numbers from **1 to 1 billion**.

Normal Python function:

```python
def numbers():

    result = []

    for i in range(1_000_000_000):
        result.append(i)

    return result
```

Question:

**What happens before the function returns?**

Python creates

```text
1 Billion Integers

↓

Store in List

↓

Return List
```

Memory

```text
+----------------------------------+

0

1

2

3

...

999999999

+----------------------------------+
```

This requires **huge memory**.

---

# Memory Problem

Suppose

```python
list(range(1_000_000_000))
```

Memory

```text
RAM

↓

Allocate

↓

Allocate

↓

Allocate

↓

Huge Memory Usage
```

Most of those numbers may never even be used.

---

# Generator Idea

Instead of creating

```text
1

2

3

4

...

1000000000
```

all at once,

create

```text
1

↓

Give User

↓

2

↓

Give User

↓

3

↓

Give User
```

One value at a time.

This is exactly what a **generator** does.

---

# First Generator

```python
def numbers():

    yield 1

    yield 2

    yield 3
```

Now

```python
g = numbers()

print(g)
```

Output

```text
<generator object numbers at ...>
```

Notice

Nothing executed yet.

---

# What Happens Internally?

Calling

```python
numbers()
```

does **not** execute the function.

Instead

```text
Create Generator Object

↓

Store Function State

↓

Return Generator
```

Execution has not started.

---

# next()

```python
g = numbers()

print(next(g))
```

Output

```text
1
```

Internally

```text
Generator

↓

Start Function

↓

yield 1

↓

Pause
```

Execution stops at `yield`.

---

Next call

```python
print(next(g))
```

Output

```text
2
```

Python resumes exactly where it stopped.

```text
yield 1

↓

Resume

↓

yield 2

↓

Pause
```

---

Third call

```python
print(next(g))
```

Output

```text
3
```

---

Fourth call

```python
next(g)
```

Output

```text
StopIteration
```

The generator is exhausted.

---

# Difference Between return and yield

Normal function

```python
def add():

    return 10

    return 20
```

Execution

```text
Start

↓

return

↓

Function Ends
```

Only one return.

---

Generator

```python
def add():

    yield 10

    yield 20
```

Execution

```text
Start

↓

yield 10

↓

Pause

↓

Resume

↓

yield 20

↓

Pause

↓

End
```

The function remembers where it stopped.

---

# Internal Memory

Suppose

```python
def example():

    x = 10

    yield x

    x += 1

    yield x
```

State

```text
Generator Object

Current Line

x = 10

Stack Frame

Local Variables
```

Everything is preserved.

---

# Generator Frame

Unlike a normal function

```text
Function

↓

Run

↓

Destroy Stack
```

Generator

```text
Run

↓

Pause

↓

Keep Stack

↓

Resume Later
```

This is why generators remember their state.

---

# Example

```python
def counter():

    x = 1

    while True:

        yield x

        x += 1
```

Usage

```python
g = counter()

print(next(g))
print(next(g))
print(next(g))
```

Output

```text
1

2

3
```

Notice

`x` is never reset.

---

# Infinite Generator

```python
def infinite():

    i = 0

    while True:

        yield i

        i += 1
```

Usage

```python
g = infinite()

for _ in range(5):
    print(next(g))
```

Output

```text
0
1
2
3
4
```

Impossible with a list because an infinite list cannot exist.

---

# for Loop Uses next()

Many people don't realize this.

```python
for x in numbers():

    print(x)
```

Internally, Python roughly does:

```python
g = numbers()

while True:

    try:

        value = next(g)

        print(value)

    except StopIteration:

        break
```

The `for` loop repeatedly calls `next()` until the generator is exhausted.

---

# Generator Expression

Instead of

```python
numbers = []

for i in range(10):

    numbers.append(i*i)
```

Write

```python
numbers = (i*i for i in range(10))
```

Notice

`()`

instead of

`[]`

List comprehension

```python
[i*i for i in range(10)]
```

creates

```text
Entire List
```

Generator expression

```python
(i*i for i in range(10))
```

creates

```text
Generator
```

No values computed until requested.

---

# Memory Comparison

List

```python
nums = [i for i in range(1_000_000)]
```

Memory

```text
Store

1

2

3

...

999999
```

Generator

```python
nums = (i for i in range(1_000_000))
```

Memory

```text
Generator

↓

One Number

↓

Next Number

↓

Next Number
```

Only one value is produced at a time.

---

# Large File Example

Bad

```python
with open("large.txt") as f:

    data = f.readlines()
```

Memory

```text
Entire File

↓

RAM
```

If the file is 20 GB, this is a problem.

---

Good

```python
with open("large.txt") as f:

    for line in f:

        print(line)
```

Python reads one line at a time.

This is generator-like behavior.

---

# Custom Generator

```python
def read_lines(filename):

    with open(filename) as f:

        for line in f:

            yield line
```

Usage

```python
for line in read_lines("large.txt"):

    process(line)
```

Even a very large file can be processed with low memory usage.

---

# Generator Pipeline

Example

```python
def numbers():

    for i in range(10):

        yield i

def squares(nums):

    for n in nums:

        yield n*n

def even(nums):

    for n in nums:

        if n%2==0:

            yield n
```

Usage

```python
result = even(squares(numbers()))

for x in result:

    print(x)
```

Pipeline

```text
Numbers

↓

Squares

↓

Even Filter

↓

Output
```

No intermediate lists are created.

---

# AI Example

Suppose you have

```text
10 Million Documents
```

Bad

```python
docs = load_all_documents()
```

Memory

```text
10 Million Documents

↓

RAM
```

Good

```python
def load_documents():

    for doc in database:

        yield doc
```

Processing

```python
for doc in load_documents():

    embedding = embed(doc)
```

Only one document is in memory at a time.

---

# Streaming LLM Output

Large language models often stream tokens.

Conceptually:

```python
def generate():

    yield "Hello"

    yield " "

    yield "World"
```

The client receives:

```text
Hello

↓

Space

↓

World
```

instead of waiting for the complete response.

---

# Generators vs Lists

| List                   | Generator                                  |
| ---------------------- | ------------------------------------------ |
| Stores all values      | Produces values on demand                  |
| High memory            | Low memory                                 |
| Faster random access   | Sequential access only                     |
| Can iterate many times | Usually exhausted after one full iteration |

---

# Generator vs Iterator

An **iterator** is any object that implements:

```python
__iter__()
__next__()
```

A generator is a special kind of iterator created automatically by using `yield`.

Example

```python
def numbers():

    yield 1
```

Python creates an iterator for you.

No need to manually implement

```python
class MyIterator:

    def __iter__(self):
        return self

    def __next__(self):
        ...
```

---

# send()

Generators can also receive values.

```python
def echo():

    value = yield

    print(value)
```

Usage

```python
g = echo()

next(g)      # Prime the generator

g.send("Hello")
```

Output

```text
Hello
```

`send()` is more advanced and is less common in day-to-day application code, but it is useful to know for interviews.

---

# Production AI Usage

Generators are commonly used for:

* Streaming LLM responses token by token
* Reading very large datasets
* Batch inference pipelines
* Data preprocessing
* ETL jobs
* Lazy loading documents
* Log processing
* Feature engineering pipelines

Example:

```python
def batches(items, batch_size):
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]

documents = ["doc1", "doc2", "doc3", "doc4", "doc5"]

for batch in batches(documents, 2):
    print(batch)
```

Output:

```text
['doc1', 'doc2']
['doc3', 'doc4']
['doc5']
```

This pattern is common when sending data to embedding models or inference services in manageable chunks.

---

# Common Interview Questions

### Q1. What is a generator?

A generator is a special type of iterator that produces values lazily, one at a time, using the `yield` keyword instead of returning all values at once.

---

### Q2. Why are generators memory efficient?

Because they generate values only when requested instead of storing the entire sequence in memory.

---

### Q3. What does `yield` do?

`yield` returns a value to the caller **and pauses execution**, preserving the function's local variables and execution state so it can resume later.

---

### Q4. Can a generator be reused?

Generally, no. Once a generator is exhausted, it raises `StopIteration`. To iterate again, you create a new generator instance.

---

### Q5. When should you use generators?

Use generators when:

* processing very large files
* reading data from databases incrementally
* streaming responses
* building data pipelines
* reducing memory usage
* creating infinite sequences

---

# Senior AI Engineer Takeaway

| Use Case                    | Why Generators?                           |
| --------------------------- | ----------------------------------------- |
| Reading large log files     | Avoid loading the entire file into memory |
| Training data pipelines     | Process one sample or batch at a time     |
| Streaming LLM tokens        | Send partial responses immediately        |
| ETL pipelines               | Lazy processing reduces memory footprint  |
| Batch inference             | Generate batches on demand                |
| Document processing for RAG | Handle millions of documents efficiently  |

## Key Insight

Think of the three approaches like this:

```text
List
────
Cook the entire buffet first,
then serve everyone.
```

```text
Generator
─────────
Cook one plate,
serve it,
cook the next plate.
```

```text
Iterator
────────
The waiter who keeps handing out the next plate.
```

A **generator** is Python's simplest way to build an iterator. It enables **lazy evaluation**, dramatically reduces memory usage for large datasets, and is heavily used in production AI systems for streaming data, preprocessing large corpora, and delivering token-by-token responses from language models.
