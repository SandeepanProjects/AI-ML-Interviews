# Pandas (Senior AI Engineer Interview)

Pandas is one of the most frequently tested libraries for **Senior AI/ML Engineers**, **Data Engineers**, and **Backend Engineers**.

Interviewers typically ask:

* How does Pandas work internally?
* Difference between `merge()` and `join()`
* How does `groupby()` work?
* What is a pivot table?
* Why is `apply()` slow?
* Why is vectorization faster?
* How do you handle missing values?
* How do you optimize a 100GB DataFrame?

A senior engineer is expected to explain not only **how** to use these APIs, but also **their performance implications**.

---

# Pandas Internals

Pandas is built on top of **NumPy**.

```text
Application

↓

Pandas DataFrame

↓

NumPy ndarray

↓

Contiguous Memory

↓

CPU
```

Each column is essentially a NumPy array.

Example

```python
import pandas as pd

df = pd.DataFrame({
    "age":[20,25,30],
    "salary":[1000,2000,3000]
})
```

Internally:

```text
DataFrame

age ----> ndarray

salary -> ndarray
```

Each column is stored separately.

This makes column operations very fast.

---

# 1. Merge

Most common interview question.

Suppose:

Customers

```text
id

1

2

3
```

Orders

```text
id

1

2

2
```

Merge:

```python
import pandas as pd

result = customers.merge(
    orders,
    on="id"
)
```

Output

```text
id

1

2

2
```

---

## SQL Equivalent

```sql
SELECT *

FROM customers

JOIN orders

ON customers.id = orders.id
```

Pandas merge is essentially SQL JOIN.

---

## Types of Merge

### Inner

```python
df1.merge(df2)
```

Common rows only.

---

### Left

```python
df1.merge(
    df2,
    how="left"
)
```

Keep all left rows.

---

### Right

```python
df1.merge(
    df2,
    how="right"
)
```

---

### Outer

```python
df1.merge(
    df2,
    how="outer"
)
```

Keep everything.

---

# How Merge Works Internally

For large datasets Pandas generally uses optimized join algorithms (such as hash joins for many cases).

```text
Key

↓

Hash Table

↓

Lookup

↓

Merge
```

Much faster than nested loops.

Complexity is typically close to **O(n + m)** for hash-based joins, rather than O(n × m).

---

# 2. Join

People confuse merge and join.

Suppose:

```python
df1.join(df2)
```

Join works primarily on the **index**.

Merge works on **columns**.

---

Example

Merge

```python
df.merge(df2,on="id")
```

Join

```python
df.join(df2)
```

Usually:

```text
Index

↓

Match

↓

Combine
```

---

# Merge vs Join

| Merge         | Join                     |
| ------------- | ------------------------ |
| Uses columns  | Uses index by default    |
| SQL-style     | Index-based              |
| More flexible | Simpler for indexed data |

---

# 3. GroupBy

Probably the most asked Pandas question.

Suppose:

```text
Department

Salary

IT

100

IT

200

HR

300
```

Code

```python
df.groupby("Department")["Salary"].mean()
```

Output

```text
IT 150

HR 300
```

---

## Internally

GroupBy performs:

```text
Rows

↓

Hash by key

↓

Buckets

↓

Aggregate
```

Example

```text
IT

↓

100

200

↓

Mean

↓

150
```

---

# Multiple Aggregations

```python
df.groupby("Department").agg({
    "Salary":["mean","max","count"]
})
```

Output

```text
Mean

Max

Count
```

One pass computes multiple aggregates.

---

# AI Example

LLM requests

```text
User

Tokens
```

Group:

```python
df.groupby("User")["Tokens"].sum()
```

Compute token usage per user.

---

# 4. Pivot

Suppose

```text
Year Product Sales

2024 A 10

2024 B 20

2025 A 15

2025 B 30
```

Pivot

```python
df.pivot(
    index="Year",
    columns="Product",
    values="Sales"
)
```

Output

```text
      A   B

2024 10 20

2025 15 30
```

Rows become columns.

---

# Pivot Table

Unlike `pivot()`, `pivot_table()` supports aggregation.

```python
df.pivot_table(
    index="Department",
    values="Salary",
    aggfunc="mean"
)
```

Useful when there are duplicate keys.

---

# 5. Apply

Very common interview trap.

Example

```python
df["salary"].apply(
    lambda x:x*2
)
```

Works.

But...

It's usually slower than vectorized operations because it executes a Python function for each element.

---

## Better

Instead of

```python
df["salary"].apply(
    lambda x:x*2
)
```

Use

```python
df["salary"] * 2
```

This is vectorized and runs through NumPy.

---

# Why is Apply Slower?

Apply

```text
Python

↓

Function

↓

Function

↓

Function
```

Vectorized

```text
NumPy

↓

C Loop

↓

SIMD
```

Much faster.

---

# 6. Vectorization

Instead of

```python
result=[]

for s in df["salary"]:
    result.append(s*2)
```

Use

```python
df["salary"] *= 2
```

Entire column processed in optimized native code.

---

# AI Example

Normalize embeddings

Instead of

```python
for row in df:
```

Use vectorized NumPy/Pandas operations.

---

# 7. Missing Values

Very common interview topic.

Check

```python
df.isna()
```

Count

```python
df.isna().sum()
```

---

Drop missing

```python
df.dropna()
```

---

Fill missing

```python
df.fillna(0)
```

---

Forward fill

```python
df.ffill()
```

---

Backward fill

```python
df.bfill()
```

---

Fill mean

```python
df["age"].fillna(
    df["age"].mean()
)
```

---

# Why Missing Values Matter

ML algorithms usually cannot train with missing values.

Typical preprocessing pipeline:

```text
CSV

↓

Missing Values

↓

Imputation

↓

Encoding

↓

Scaling

↓

Training
```

---

# 8. Optimizing Large DataFrames

Suppose:

```text
100 GB CSV
```

Can you do

```python
pd.read_csv(...)
```

No.

Memory explodes.

---

## Read in Chunks

```python
for chunk in pd.read_csv(
    "large.csv",
    chunksize=100000
):
    process(chunk)
```

Memory stays bounded.

---

## Use Correct Data Types

Bad

```text
int64
```

If values are:

```text
0-100
```

Use

```python
astype("int8")
```

Much smaller.

---

## Convert Strings to Category

Instead of

```text
Department

IT

IT

IT

HR
```

Store

```text
0

0

0

1
```

```python
df["Department"] = df["Department"].astype("category")
```

Can dramatically reduce memory for repeated values.

---

## Read Required Columns Only

Instead of

```python
pd.read_csv("big.csv")
```

Use

```python
pd.read_csv(
    "big.csv",
    usecols=["age","salary"]
)
```

Less I/O.

Less memory.

---

## Use Parquet Instead of CSV

CSV

```text
100 GB
```

Parquet

```text
30 GB
```

Better compression.

Columnar storage.

Faster reads.

---

## Avoid Object Dtype

Worst dtype:

```text
object
```

Slow.

Prefer

```text
category

int32

float32

datetime64
```

---

## Avoid Loops

Don't do

```python
for row in df.iterrows():
```

Slow.

Better

```python
df["salary"]*=2
```

---

## Use `itertuples()` When You Must Iterate

If iteration is unavoidable:

```python
for row in df.itertuples():
    ...
```

Faster than `iterrows()` because it avoids creating a Series object for each row.

---

# AI Example

Training pipeline

```text
100 GB CSV

↓

Chunk

↓

Clean

↓

Missing Value

↓

Feature Engineering

↓

Parquet

↓

Training
```

---

# Performance Comparison

| Operation      | Speed |
| -------------- | ----- |
| Python loop    | ⭐     |
| `iterrows()`   | ⭐     |
| `apply()`      | ⭐⭐    |
| `itertuples()` | ⭐⭐⭐   |
| Vectorization  | ⭐⭐⭐⭐⭐ |
| NumPy          | ⭐⭐⭐⭐⭐ |

---

# Interview Questions

## Difference between Merge and Join?

A good answer:

> `merge()` performs SQL-style joins using one or more columns (or indexes if specified), while `join()` is a convenience method that joins primarily on the DataFrame index. `merge()` is more flexible for relational operations.

---

## Why is Apply slow?

> `apply()` typically invokes a Python function for each row or element, introducing interpreter overhead. Vectorized operations execute in optimized native code and are significantly faster.

---

## How does GroupBy work?

> `groupby()` partitions rows by key (often using hash-based grouping), forms groups, and then applies aggregation functions such as `sum`, `mean`, or `count` to each group.

---

## How would you optimize a 100GB DataFrame?

A strong answer:

* Read with `chunksize`
* Select only required columns using `usecols`
* Use efficient dtypes (`int8`, `float32`, `category`)
* Convert repeated strings to `category`
* Prefer vectorized operations over loops
* Avoid `iterrows()` where possible
* Store and read data in Parquet instead of CSV for analytical workloads
* If the data exceeds a single machine's memory, consider distributed tools like Dask, Spark, or Polars depending on the workload

---

# Senior AI Engineer Summary

| Topic            | Best Practice                                                      |
| ---------------- | ------------------------------------------------------------------ |
| Merge            | Use SQL-style joins on columns                                     |
| Join             | Use for index-based joins                                          |
| GroupBy          | Aggregate with built-in functions                                  |
| Pivot            | Reshape tabular data                                               |
| Apply            | Avoid for simple arithmetic; prefer vectorization                  |
| Vectorization    | Use NumPy-backed operations whenever possible                      |
| Missing Values   | Detect, then impute or remove appropriately                        |
| Large DataFrames | Chunk processing, optimize dtypes, use Parquet, avoid Python loops |

The overarching principle is the same as with NumPy: **keep computation in optimized native code, minimize Python-level iteration, and reduce memory usage through efficient storage formats and data types.** These practices are essential for building scalable data pipelines and production AI systems.
