# Python Functional Programming Tools (Senior Interview)

Topics:

* `lru_cache`
* `partial`
* `reduce`
* Memoization

These are common in **Senior Python / AI Engineer interviews** because they test:

* Function optimization
* Functional programming
* Caching strategies
* Performance improvement

---

# 1. Memoization

## What is Memoization?

Memoization is an optimization technique where we:

> Store previously computed results and return the cached result when the same input appears again.

Instead of:

```text
Input → Compute → Result
```

Every time:

```text
5 → Calculate
5 → Calculate
5 → Calculate
```

We do:

```text
5 → Calculate → Store

5 → Return Cache

5 → Return Cache
```

---

# Example: Fibonacci Without Memoization

Mathematical definition:

```
F(n)=F(n-1)+F(n-2)
```

Code:

```python
def fibonacci(n):

    if n <= 1:
        return n

    return (
        fibonacci(n-1)
        +
        fibonacci(n-2)
    )


print(fibonacci(10))
```

---

Problem:

For:

```python
fibonacci(10)
```

Python calculates:

```
                 fib(10)

          fib(9)       fib(8)

      fib(8) fib(7)  fib(7) fib(6)

      ...
```

Many duplicate calculations.

Complexity:

```
O(2^n)
```

---

# Implement Memoization Manually

Using dictionary:

```python
def fibonacci():

    cache = {}


    def fib(n):

        if n in cache:
            return cache[n]


        if n <= 1:
            result = n

        else:
            result = (
                fib(n-1)
                +
                fib(n-2)
            )


        cache[n] = result

        return result


    return fib



fib = fibonacci()


print(fib(10))
```

Output:

```
55
```

---

## Flow

First call:

```
fib(10)

calculate

store:
{
0:0,
1:1,
2:1,
3:2
}
```

Second call:

```
fib(10)

lookup cache

return immediately
```

---

# Complexity

Without memoization:

```
O(2^n)
```

With memoization:

```
O(n)
```

Because each value is calculated once.

---

# 2. `lru_cache`

## What is LRU?

LRU:

> Least Recently Used

It automatically removes old cache entries when cache size is full.

Import:

```python
from functools import lru_cache
```

---

# Example

```python
from functools import lru_cache


@lru_cache(maxsize=100)
def fibonacci(n):

    if n <= 1:
        return n

    return (
        fibonacci(n-1)
        +
        fibonacci(n-2)
    )


print(fibonacci(50))
```

---

Internally:

First call:

```
fib(50)

calculate
store result
```

Next call:

```
fib(50)

cache hit

return instantly
```

---

# Check Cache Statistics

```python
print(
    fibonacci.cache_info()
)
```

Output:

```
CacheInfo(
 hits=48,
 misses=51,
 maxsize=100,
 currsize=51
)
```

Meaning:

```
hits:
returned from cache

misses:
calculated
```

---

# Clear Cache

```python
fibonacci.cache_clear()
```

---

# How LRU Works Internally

Example:

Cache size:

```
3
```

Operations:

```
A
B
C
```

Cache:

```
A B C
```

Access:

```
A
```

A becomes recent:

```
B C A
```

Add:

```
D
```

Remove least used:

```
C B D
```

---

# AI Example: Embedding Cache

Problem:

Same user query repeated:

```
"What is RAG?"
```

Embedding API called repeatedly.

Bad:

```
Request
 |
Embedding API
 |
Vector
```

Every time.

---

With LRU:

```python
from functools import lru_cache


@lru_cache(maxsize=10000)
def create_embedding(text):

    return embedding_model.encode(text)
```

Flow:

```
Query
 |
Check Cache
 |
--------------
|            |
Hit          Miss
|             |
Return       Generate
```

Benefits:

* Lower API cost
* Lower latency
* Less GPU usage

---

# 3. `partial`

## Purpose

`partial()` creates a new function by fixing some arguments.

Import:

```python
from functools import partial
```

---

Example:

Normal function:

```python
def multiply(a,b):

    return a*b
```

Calling:

```python
multiply(10,5)
```

Output:

```
50
```

---

Suppose we always multiply by 10.

Create:

```python
multiply_by_10
```

Using partial:

```python
from functools import partial


multiply_by_10 = partial(
    multiply,
    10
)


print(
    multiply_by_10(5)
)
```

Output:

```
50
```

---

Internally:

Before:

```python
multiply(10,5)
```

After:

```python
multiply_by_10(5)
```

It remembers:

```
a=10
```

---

# Another Example

Function:

```python
def power(base, exponent):

    return base ** exponent
```

Create square:

```python
square = partial(
    power,
    exponent=2
)


print(square(5))
```

Output:

```
25
```

---

# AI Example

LLM API:

```python
def call_llm(
    prompt,
    model,
    temperature
):

    pass
```

Create specialized functions:

```python
gpt4_call = partial(
    call_llm,
    model="gpt-4",
    temperature=0
)
```

Now:

```python
gpt4_call(
    "Explain RAG"
)
```

Internally:

```python
call_llm(
    prompt,
    model="gpt-4",
    temperature=0
)
```

---

# 4. `reduce`

## Purpose

`reduce()` repeatedly applies a function to items.

Import:

```python
from functools import reduce
```

---

Example:

Sum numbers.

Without reduce:

```python
numbers=[
1,2,3,4
]


total=0

for n in numbers:

    total += n
```

---

Using reduce:

```python
from functools import reduce


numbers=[
1,2,3,4
]


result = reduce(
    lambda x,y:x+y,
    numbers
)


print(result)
```

Output:

```
10
```

---

# How Reduce Works

Given:

```
[1,2,3,4]
```

Step 1:

```
1 + 2 = 3
```

Step 2:

```
3 + 3 = 6
```

Step 3:

```
6 + 4 = 10
```

Final:

```
10
```

---

# Reduce With Initial Value

```python
result = reduce(
    lambda x,y:x+y,
    [1,2,3],
    10
)
```

Execution:

```
10+1+2+3

=16
```

---

# Example: Maximum Value

```python
from functools import reduce


numbers=[
10,
5,
20,
15
]


maximum = reduce(
    lambda x,y:
    x if x>y else y,
    numbers
)


print(maximum)
```

Output:

```
20
```

---

# AI Example

Combine embeddings:

Suppose:

```
chunk embeddings:

[0.1,0.2]
[0.3,0.4]
[0.5,0.6]
```

Reduce can combine:

```python
reduce(
    merge_embeddings,
    embeddings
)
```

---

# lru_cache vs Memoization

| Memoization     | lru_cache           |
| --------------- | ------------------- |
| Concept         | Implementation      |
| Manual cache    | Built-in decorator  |
| Full control    | Automatic eviction  |
| Need dictionary | Uses internal cache |

---

# partial vs lambda

Both can create specialized functions.

Lambda:

```python
square=lambda x:x*x
```

Partial:

```python
square=partial(
    power,
    exponent=2
)
```

Partial is better when:

* Existing function
* Need fixed arguments
* Better readability

---

# reduce vs loop

Loop:

```python
total=0

for x in numbers:
    total+=x
```

Usually clearer.

Reduce:

```python
reduce(
    lambda a,b:a+b,
    numbers
)
```

Useful for:

* Function composition
* Folding operations
* Aggregations

---

# Senior AI Engineer Real Examples

## 1. LLM Response Cache

Use:

```python
@lru_cache
def generate_answer(prompt):
    return llm(prompt)
```

---

## 2. Model Configuration

Use:

```python
partial(
    inference,
    model="llama"
)
```

---

## 3. Data Aggregation

Use:

```python
reduce(
    merge_results,
    responses
)
```

---

# Interview Questions

## Q1. What is memoization?

> Memoization is an optimization technique where function results are cached based on input arguments so repeated calls return stored results instead of recomputing.

---

## Q2. How does lru_cache work?

> `lru_cache` stores function results in an internal dictionary and removes the least recently used entries when the cache reaches its maximum size.

---

## Q3. Difference between cache and LRU cache?

Simple cache:

```
Store forever
```

LRU:

```
Store limited entries
Remove least recently used
```

---

## Q4. Why is lru_cache useful in AI systems?

> It reduces repeated expensive computations such as embeddings, model metadata loading, feature calculations, and API calls, improving latency and reducing cost.

---

# Final Summary

```
Memoization
-------------
Store previous results


lru_cache
-------------
Automatic memoization + eviction


partial
-------------
Create specialized functions


reduce
-------------
Combine iterable into single result
```

These concepts are frequently tested in **Senior Python Backend, ML Engineer, and AI Engineer interviews**.
