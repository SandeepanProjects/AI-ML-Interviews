# Python `itertools` Module (Senior Python Interview)

The `itertools` module provides **memory-efficient iterator building blocks**.

Common interview topics:

* `product`
* `permutations`
* `combinations`
* `groupby`
* `cycle`
* `chain`

These are useful in:

* Data processing
* ML experiments
* Hyperparameter search
* Feature combinations
* Streaming pipelines

---

# 1. `product()`

## Purpose

`product()` creates the **Cartesian product** of iterables.

Meaning:

> Generate all possible combinations where order matters and repetition is allowed.

Import:

```python
from itertools import product
```

---

## Example

```python
from itertools import product


colors = ["Red", "Blue"]

sizes = ["S", "M", "L"]


result = product(colors, sizes)


print(list(result))
```

Output:

```python
[
 ('Red','S'),
 ('Red','M'),
 ('Red','L'),
 ('Blue','S'),
 ('Blue','M'),
 ('Blue','L')
]
```

---

## Internally

It creates:

```
colors × sizes


Red  × S
Red  × M
Red  × L

Blue × S
Blue × M
Blue × L
```

Total:

```
2 * 3 = 6 combinations
```

---

# `repeat` parameter

Example:

```python
list(product([1,2], repeat=2))
```

Output:

```python
[
(1,1),
(1,2),
(2,1),
(2,2)
]
```

Equivalent:

```
[1,2] × [1,2]
```

---

# AI Example: Hyperparameter Search

Suppose:

```python
models = [
    "LSTM",
    "Transformer"
]

learning_rates = [
    0.001,
    0.0001
]

batch_sizes = [
    32,
    64
]
```

Generate experiments:

```python
from itertools import product


experiments = product(
    models,
    learning_rates,
    batch_sizes
)


for e in experiments:
    print(e)
```

Output:

```
('LSTM',0.001,32)
('LSTM',0.001,64)
('Transformer',0.0001,32)
...
```

---

# 2. `permutations()`

## Purpose

Generate all possible arrangements.

Important:

> Order matters.

Example:

```python
from itertools import permutations


items = [1,2,3]


print(list(permutations(items)))
```

Output:

```python
[
(1,2,3),
(1,3,2),
(2,1,3),
(2,3,1),
(3,1,2),
(3,2,1)
]
```

---

Number of results:

For:

```
n items
```

Formula:

```
n!
```

Example:

```
3!

=3×2×1

=6
```

---

# Length parameter

```python
list(
    permutations(
        [1,2,3],
        2
    )
)
```

Output:

```
(1,2)
(1,3)
(2,1)
(2,3)
(3,1)
(3,2)
```

---

# Permutation vs Combination

Example:

Selecting:

```
A,B,C
```

Permutation:

```
AB
BA
```

Different.

---

# AI Example

Testing prompt ordering:

```python
prompts = [
    "context",
    "question",
    "instruction"
]


for order in permutations(prompts):
    test(order)
```

---

# 3. `combinations()`

## Purpose

Generate selections where order does NOT matter.

Example:

```python
from itertools import combinations


items = [1,2,3]


print(list(
    combinations(items,2)
))
```

Output:

```
[
(1,2),
(1,3),
(2,3)
]
```

---

Notice:

No:

```
(2,1)
```

because:

```
A+B == B+A
```

---

# Formula

Number of combinations:

```
nCr = n! / r!(n-r)!
```

Example:

```
3C2

=3
```

---

# Example

Feature selection:

Features:

```python
features=[
"age",
"income",
"location",
"education"
]
```

Choose 2 features:

```python
for combo in combinations(features,2):
    print(combo)
```

Output:

```
age,income
age,location
age,education
...
```

---

# Difference

|             | Order matters | Example |
| ----------- | ------------- | ------- |
| product     | No            | A×B     |
| permutation | Yes           | AB ≠ BA |
| combination | No            | AB = BA |

---

# 4. `groupby()`

## Purpose

Groups consecutive items based on a key.

Import:

```python
from itertools import groupby
```

---

Example:

```python
from itertools import groupby


data = [
    ("AI",90),
    ("AI",95),
    ("ML",80),
    ("ML",85)
]


for key, group in groupby(
    data,
    lambda x:x[0]
):

    print(
        key,
        list(group)
    )
```

Output:

```
AI [('AI',90),('AI',95)]

ML [('ML',80),('ML',85)]
```

---

# Important

`groupby()` requires sorted data.

Wrong:

```python
[
AI,
ML,
AI
]
```

It creates:

```
AI group
ML group
AI group
```

---

Correct:

```python
sorted(data)
```

---

# Internal Concept

It is similar to SQL:

```sql
GROUP BY
```

Example:

SQL:

```sql
SELECT department,
COUNT(*)
FROM employees
GROUP BY department
```

Python:

```python
groupby()
```

---

# AI Example

Grouping predictions:

```python
predictions=[
("cat",0.9),
("cat",0.8),
("dog",0.7)
]
```

Group by label.

---

# 5. `cycle()`

## Purpose

Creates an infinite iterator.

Example:

```python
from itertools import cycle


colors = cycle(
    ["Red","Blue"]
)


for i in range(5):

    print(next(colors))
```

Output:

```
Red
Blue
Red
Blue
Red
```

---

Internally:

```
Red
Blue
|
repeat forever
```

---

# Use Cases

## Round Robin Scheduling

Example:

Workers:

```
Worker1
Worker2
Worker3
```

Requests:

```
1 -> Worker1
2 -> Worker2
3 -> Worker3
4 -> Worker1
```

Implementation:

```python
workers = cycle(
    ["GPU1","GPU2","GPU3"]
)


for request in requests:

    worker = next(workers)

    assign(request,worker)
```

---

# AI Example

Distribute inference requests:

```
User requests

       |
       v

GPU1
GPU2
GPU3
GPU1
GPU2
```

---

# 6. `chain()`

## Purpose

Combine multiple iterables into one iterator.

Example:

```python
from itertools import chain


a=[1,2,3]

b=[4,5,6]


result = chain(a,b)


print(list(result))
```

Output:

```
[1,2,3,4,5,6]
```

---

Without chain:

```python
for x in a:
    yield x

for x in b:
    yield x
```

---

# Multiple Iterables

```python
chain(
    [1,2],
    [3,4],
    [5,6]
)
```

Result:

```
1 2 3 4 5 6
```

---

# AI Example

Combine datasets:

```python
train_data = [...]

validation_data = [...]

test_data = []


all_data = chain(
    train_data,
    validation_data,
    test_data
)
```

No extra memory copy.

---

# Important: Lazy Evaluation

All itertools functions return iterators.

Example:

```python
x = product(
    range(1000000),
    range(1000000)
)
```

It does NOT create:

```
1 trillion items
```

in memory.

It generates values on demand.

---

# Comparison Table

| Function     | Purpose                 | Order Matters? | Example       |
| ------------ | ----------------------- | -------------- | ------------- |
| product      | Cartesian product       | No             | A×B           |
| permutations | Arrangements            | Yes            | AB, BA        |
| combinations | Selection               | No             | AB only       |
| groupby      | Group consecutive items | N/A            | SQL GROUP BY  |
| cycle        | Repeat forever          | N/A            | Round robin   |
| chain        | Join iterables          | N/A            | Merge streams |

---

# Interview Questions

## Q1. Difference between permutations and combinations?

Answer:

> Permutations consider ordering, while combinations ignore ordering. For example AB and BA are different permutations but the same combination.

---

## Q2. Why use itertools instead of lists?

Answer:

> itertools uses lazy evaluation and generators, so it avoids storing the entire result in memory. It is suitable for processing large datasets and streams.

---

## Q3. Difference between chain and list concatenation?

List:

```python
a+b
```

creates a new list in memory.

Chain:

```python
chain(a,b)
```

creates a lazy iterator.

---

## Q4. Where is cycle used in production?

Answer:

> Cycle is useful for round-robin scheduling, load balancing, worker assignment, and rotating resources.

---

# Senior AI Engineer Examples

### Hyperparameter search

```
product()
```

---

### Feature combinations

```
combinations()
```

---

### Prompt order testing

```
permutations()
```

---

### Data grouping

```
groupby()
```

---

### GPU worker distribution

```
cycle()
```

---

### Streaming datasets

```
chain()
```

---

These `itertools` primitives are common in **Senior Python, ML Engineering, Data Engineering, and AI Platform interviews**.
