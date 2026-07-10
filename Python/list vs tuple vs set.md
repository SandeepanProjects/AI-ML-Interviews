These are **very common Senior Python interview questions**. Interviewers expect you to explain not only the differences, but also **why each data structure exists, when to use it, and the underlying performance characteristics**.

---

# 1. List vs Tuple

| Feature          | List            | Tuple                                |
| ---------------- | --------------- | ------------------------------------ |
| Syntax           | `[ ]`           | `( )`                                |
| Mutable          | ✅ Yes           | ❌ No                                 |
| Ordered          | ✅ Yes           | ✅ Yes                                |
| Indexing         | ✅ Yes           | ✅ Yes                                |
| Duplicate values | ✅ Allowed       | ✅ Allowed                            |
| Hashable         | ❌ No            | ✅ Yes (if all elements are hashable) |
| Dictionary key   | ❌ No            | ✅ Yes                                |
| Memory usage     | Higher          | Lower                                |
| Performance      | Slightly slower | Slightly faster                      |

---

## Example

### List

```python
numbers = [1, 2, 3]

numbers.append(4)

print(numbers)
```

Output

```python
[1, 2, 3, 4]
```

The list is modified in place.

---

### Tuple

```python
numbers = (1, 2, 3)

numbers.append(4)
```

Output

```text
AttributeError: 'tuple' object has no attribute 'append'
```

Tuples cannot be modified after creation.

---

## Memory

### List

```text
+---------------------+
| Dynamic Array       |
| Extra Capacity      |
+---------------------+
```

Lists allocate extra space to make appends efficient.

---

### Tuple

```text
+---------------------+
| Fixed Array         |
+---------------------+
```

Tuples have a fixed size, so they are generally more memory-efficient.

---

## Performance

```python
import timeit

print(timeit.timeit("(1,2,3)", number=10_000_000))
print(timeit.timeit("[1,2,3]", number=10_000_000))
```

Tuples are usually a bit faster to create and iterate because they don't support resizing.

---

## When to use a List

Use a list when:

* Data changes frequently.
* You need `append()`, `extend()`, `insert()`, or `remove()`.
* The collection grows or shrinks.

Examples:

* Training batches
* Prediction results
* Request queues
* User sessions

---

## When to use a Tuple

Use a tuple when:

* Data is fixed.
* You want read-only semantics.
* The object should be hashable.
* You need a dictionary key or set element.

Examples:

```python
point = (10, 20)

rgb = (255, 0, 0)

version = (3, 12)
```

---

# 2. List vs Set

| Feature           | List      | Set                                                      |
| ----------------- | --------- | -------------------------------------------------------- |
| Ordered           | ✅ Yes     | ❌ No (iteration order is not guaranteed by the language) |
| Duplicates        | ✅ Allowed | ❌ Not allowed                                            |
| Indexing          | ✅ Yes     | ❌ No                                                     |
| Mutable           | ✅ Yes     | ✅ Yes                                                    |
| Hash Table        | ❌ No      | ✅ Yes                                                    |
| Membership (`in`) | O(n)      | Average O(1)                                             |

---

## Example

### List

```python
numbers = [1, 2, 2, 3]

print(numbers)
```

Output

```python
[1, 2, 2, 3]
```

Duplicates are preserved.

---

### Set

```python
numbers = {1, 2, 2, 3}

print(numbers)
```

Output

```python
{1, 2, 3}
```

Duplicates are automatically removed.

---

## Membership Test

### List

```python
numbers = [1, 2, 3, 4]

print(4 in numbers)
```

Python checks elements one by one until it finds a match.

Average time complexity: **O(n)**.

---

### Set

```python
numbers = {1, 2, 3, 4}

print(4 in numbers)
```

Python uses a hash table.

Average time complexity: **O(1)**.

---

## Why is a Set Faster?

A set is implemented using a **hash table**.

Instead of scanning every element:

```text
1

↓

2

↓

3

↓

4
```

Python computes the hash of the value and looks directly in the appropriate bucket.

Conceptually:

```text
Hash(value)

↓

Bucket

↓

Found
```

---

## Removing Duplicates

List approach:

```python
numbers = [1, 2, 2, 3]

unique = list(set(numbers))

print(unique)
```

Output:

```python
[1, 2, 3]
```

Note that converting through a set does **not** preserve the original order.

If preserving insertion order matters (Python 3.7+), you can use:

```python
unique = list(dict.fromkeys(numbers))
```

---

## Set Operations

```python
a = {1, 2, 3}
b = {3, 4, 5}

print(a | b)   # Union
print(a & b)   # Intersection
print(a - b)   # Difference
```

Output:

```python
{1, 2, 3, 4, 5}
{3}
{1, 2}
```

These operations are efficient because sets are hash-based.

---

# Time Complexity

| Operation           | List           | Tuple | Set                  |
| ------------------- | -------------- | ----- | -------------------- |
| Indexing            | O(1)           | O(1)  | ❌                    |
| Append              | O(1) amortized | ❌     | O(1) average (`add`) |
| Insert at beginning | O(n)           | ❌     | ❌                    |
| Membership (`in`)   | O(n)           | O(n)  | O(1) average         |
| Remove              | O(n)           | ❌     | O(1) average         |
| Iteration           | O(n)           | O(n)  | O(n)                 |

---

# Real-world AI Examples

### List

```python
embeddings = []
embeddings.append(vector)
```

A dynamic collection of embeddings.

---

### Tuple

```python
cache_key = ("gpt-4.1", "temperature=0.2")
```

A fixed, hashable key for a cache.

---

### Set

```python
visited_documents = set()

visited_documents.add(doc_id)
```

Efficiently tracks processed documents without duplicates.

---

# When to Use Which?

| Scenario                         | Best Choice |
| -------------------------------- | ----------- |
| Dynamic collection               | List        |
| Fixed record                     | Tuple       |
| Dictionary key                   | Tuple       |
| Remove duplicates                | Set         |
| Fast membership checks           | Set         |
| Ordered sequence with duplicates | List        |
| Read-only configuration          | Tuple       |
| Tracking unique IDs              | Set         |

---

# Senior AI Engineer Interview Answer (60 seconds)

> A **list** is an ordered, mutable sequence that supports duplicates and is ideal for dynamic collections where elements are frequently added, removed, or updated. A **tuple** is also ordered and allows duplicates, but it is immutable, making it more memory-efficient and suitable for fixed records or hashable keys in dictionaries and sets. A **set** is an unordered collection of unique hashable elements implemented as a hash table, providing average **O(1)** membership tests and automatically eliminating duplicates. In AI systems, lists are commonly used for batches and predictions, tuples for immutable cache keys and configuration records, and sets for fast duplicate removal and tracking visited items or processed document IDs.
