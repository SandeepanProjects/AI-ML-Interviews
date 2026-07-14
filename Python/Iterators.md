# Iterators in Python

This is a **very common Senior Python / AI Engineer interview topic** because iterators are the foundation of:

* Generators
* Data pipelines
* Streaming APIs
* Large dataset processing
* PyTorch DataLoader
* Token streaming in LLM applications

The key interview questions are:

1. What is an iterable?
2. What is an iterator?
3. Difference between them?
4. Implement your own iterator using `__iter__` and `__next__`.
5. How does a `for` loop work internally?

---

# 1. What is an Iterable?

An **iterable** is any object that can return an iterator.

It implements:

```python
__iter__()
```

Examples:

```python id="z6n9b1"
list
tuple
string
dict
set
range
```

Example:

```python id="6w0zqr"
numbers = [1,2,3]

iterator = iter(numbers)

print(iterator)
```

Output:

```text
<list_iterator object>
```

The list itself is not the iterator.

It creates an iterator.

---

# 2. What is an Iterator?

An **iterator** is an object that produces values one at a time.

It implements:

```python
__iter__()
```

and

```python
__next__()
```

Example:

```python id="0r1vzr"
numbers = [1,2,3]

it = iter(numbers)

print(next(it))
print(next(it))
print(next(it))
```

Output:

```text
1
2
3
```

After values are exhausted:

```python id="sbyzj8"
next(it)
```

Output:

```
StopIteration
```

---

# Iterable vs Iterator

| Feature          | Iterable  | Iterator      |
| ---------------- | --------- | ------------- |
| Has `__iter__()` | ✅         | ✅             |
| Has `__next__()` | ❌ Usually | ✅             |
| Produces values  | ❌         | ✅             |
| Maintains state  | ❌         | ✅             |
| Example          | list      | list_iterator |

---

## Example

List:

```python id="r4we1p"
numbers = [10,20,30]
```

is iterable.

Check:

```python id="0k1rsv"
print(hasattr(numbers,"__iter__"))
```

Output:

```text
True
```

But:

```python id="u9pq1c"
print(hasattr(numbers,"__next__"))
```

Output:

```text
False
```

---

Iterator:

```python id="r7p5o9"
it = iter(numbers)
```

Check:

```python id="i9x2e1"
print(hasattr(it,"__iter__"))
print(hasattr(it,"__next__"))
```

Output:

```text
True
True
```

---

# How Does a `for` Loop Work Internally?

When you write:

```python id="4a6p1y"
for x in [1,2,3]:
    print(x)
```

Python internally does:

```python id="zcb9r7"
iterator = iter([1,2,3])

while True:

    try:
        x = next(iterator)

        print(x)

    except StopIteration:
        break
```

So:

```text
for loop

↓

__iter__()

↓

__next__()

↓

StopIteration
```

---

# Implement Custom Iterator

Let's create:

```python id="c3zv8q"
Counter(1,5)
```

which produces:

```
1
2
3
4
5
```

---

# Implementation

```python id="h0e0lm"
class Counter:

    def __init__(self, start, end):
        self.current = start
        self.end = end


    def __iter__(self):
        return self


    def __next__(self):

        if self.current > self.end:
            raise StopIteration

        value = self.current

        self.current += 1

        return value
```

---

Usage:

```python id="1xw6dl"
counter = Counter(1,5)

for num in counter:
    print(num)
```

Output:

```
1
2
3
4
5
```

---

# How This Works Internally

Create object:

```python id="p8jr1x"
counter = Counter(1,5)
```

Memory:

```
Counter Object

current = 1
end = 5
```

---

## Step 1

For loop starts:

```python id="b8g8yo"
iter(counter)
```

Python calls:

```python id="f40byj"
counter.__iter__()
```

Returns:

```
counter object
```

---

## Step 2

Python calls:

```python id="n4u9r7"
next(counter)
```

Internally:

```python id="k6c5zj"
counter.__next__()
```

---

First call:

```
current = 1
```

Return:

```
1
```

Update:

```
current = 2
```

---

Second call:

```
current = 2
```

Return:

```
2
```

Update:

```
current = 3
```

---

Continue:

```
1
2
3
4
5
```

---

When:

```
current = 6
```

Condition:

```python
if self.current > self.end:
```

becomes true.

Python raises:

```python
raise StopIteration
```

The loop stops.

---

# Why Does Iterator Need State?

Because it remembers where it is.

Example:

```python id="5f1d7r"
it = Counter(1,5)

next(it)
```

Memory:

```
current = 2
```

Again:

```python
next(it)
```

Memory:

```
current = 3
```

The iterator stores progress.

---

# Iterable Creating New Iterators

Important interview point.

A list:

```python id="r6q7do"
numbers = [1,2,3]
```

can create multiple iterators:

```python id="2m4dsv"
a = iter(numbers)
b = iter(numbers)
```

Memory:

```
List

[1,2,3]


Iterator A

position=0


Iterator B

position=0
```

They are independent.

---

# Iterator is Usually Itself Iterable

Why does an iterator have `__iter__`?

Because:

```python id="p9kgcz"
it = iter([1,2,3])

iter(it) is it
```

Output:

```
True
```

Implementation:

```python
def __iter__(self):
    return self
```

---

# Common Interview Trap

Question:

Is a list an iterator?

Answer:

No.

Example:

```python id="u8u2bm"
lst = [1,2,3]

next(lst)
```

Error:

```
TypeError:
list object is not an iterator
```

Because list has no:

```python
__next__()
```

---

# Generator vs Iterator

A generator is a special type of iterator.

Example:

```python id="g4p3xa"
def counter():

    yield 1
    yield 2
    yield 3
```

Usage:

```python id="l9w7k0"
for x in counter():
    print(x)
```

Generators automatically implement:

```
__iter__()
__next__()
```

---

# Why Iterators Matter in AI Systems

## Large Dataset Processing

Bad:

```python
data = load_all_documents()
```

Memory:

```
10 million documents
```

---

Better:

```python
for doc in document_iterator:
    process(doc)
```

Only one document is loaded at a time.

---

## LLM Streaming

Example:

```python
for token in llm.stream(prompt):
    print(token)
```

Internally:

```
Iterator

token1
token2
token3
...
```

No need to wait for the full response.

---

## PyTorch DataLoader

```python
for batch in dataloader:
    train(batch)
```

DataLoader is iterator-based.

It loads batches lazily.

---

# Interview Questions

## Q1. Difference between iterable and iterator?

Strong answer:

> An iterable is an object that can return an iterator through `__iter__()`. An iterator is an object that maintains state and produces values one at a time using `__next__()`. Every iterator is iterable, but not every iterable is an iterator.

---

## Q2. Why raise StopIteration?

Because it signals to Python that the iterator has no more values.

The `for` loop catches it internally and stops.

---

## Q3. Why does `__iter__()` return self in an iterator?

Because the iterator itself represents the iteration state. Returning itself satisfies Python's iterable protocol.

---

## Q4. Iterator vs Generator?

| Iterator                                      | Generator               |
| --------------------------------------------- | ----------------------- |
| Manually implements `__iter__` and `__next__` | Uses `yield`            |
| More code                                     | Less code               |
| Full control                                  | Python manages state    |
| Can implement complex behavior                | Best for lazy sequences |

---

# Final Interview Summary

```text
Iterable
    |
    | __iter__()
    ↓
Iterator
    |
    | __next__()
    ↓
Values one by one
    |
    | StopIteration
    ↓
Loop ends
```

A senior-level answer:

> Python's iterator protocol separates the container from traversal logic. An iterable defines how to create an iterator, while an iterator maintains traversal state and produces values lazily through `__next__()`. This design enables memory-efficient processing of large datasets, streaming responses, and scalable AI pipelines.
