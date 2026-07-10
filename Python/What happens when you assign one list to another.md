This is a **very common Senior Python / Senior AI Engineer interview question** because it tests your understanding of Python's object model and memory management.

---

# Question

**What happens when you assign one list to another?**

Example:

```python
a = [1, 2, 3]
b = a
```

Does Python create another list?

**No.**

Python creates **another reference** to the **same list object**.

---

# Memory Representation

Initially

```text
a
|
v
[1,2,3]
```

After

```python
b = a
```

Memory becomes

```text
a --------\
           \
            -----> [1,2,3]
           /
b --------/
```

Both variables point to the **same object**.

---

# Proof using `id()`

```python
a = [1, 2, 3]
b = a

print(id(a))
print(id(b))
```

Output

```text
43822152
43822152
```

The IDs are identical, confirming that `a` and `b` reference the same object.

---

# Modifying One Variable

```python
a = [1, 2, 3]
b = a

b.append(4)

print(a)
print(b)
```

Output

```text
[1, 2, 3, 4]
[1, 2, 3, 4]
```

Why?

Because there is only **one list** in memory, and both variables reference it.

---

# Changing an Element

```python
a = [10, 20]

b = a

b[0] = 100

print(a)
```

Output

```text
[100, 20]
```

Again, both names refer to the same list.

---

# What if We Reassign `b`?

```python
a = [1, 2, 3]
b = a

b = [10, 20]

print(a)
print(b)
```

Output

```text
[1, 2, 3]
[10, 20]
```

Memory changes from:

```text
a ----\
       \
        ---> [1,2,3]
       /
b ----/
```

to:

```text
a --------> [1,2,3]

b --------> [10,20]
```

The original list is unaffected because `b` now references a different object.

---

# Assignment vs Copy

## Assignment

```python
a = [1, 2, 3]
b = a
```

```text
One list
Two references
```

---

## Shallow Copy

```python
a = [1, 2, 3]

b = a.copy()
```

or

```python
b = list(a)
```

or

```python
b = a[:]
```

Memory

```text
a -----> [1,2,3]

b -----> [1,2,3]
```

These are two separate list objects.

Proof:

```python
print(id(a))
print(id(b))
```

Different IDs.

---

## Modifying the Copy

```python
a = [1, 2, 3]

b = a.copy()

b.append(4)

print(a)
print(b)
```

Output

```text
[1,2,3]
[1,2,3,4]
```

The original list remains unchanged.

---

# Nested Lists: Shallow Copy Pitfall

```python
a = [[1], [2]]

b = a.copy()

b[0].append(100)

print(a)
print(b)
```

Output

```text
[[1,100],[2]]
[[1,100],[2]]
```

Why?

The outer list was copied, but the inner lists are still shared.

Memory

```text
a ---> +---------+
       |   * ----|----> [1]
       |   * ----|----> [2]
       +---------+

b ---> +---------+
       |   * ----|----> [1]
       |   * ----|----> [2]
       +---------+
```

Both outer lists point to the same inner list objects.

---

# Deep Copy

```python
import copy

a = [[1], [2]]

b = copy.deepcopy(a)

b[0].append(100)

print(a)
print(b)
```

Output

```text
[[1],[2]]
[[1,100],[2]]
```

`deepcopy()` recursively creates independent copies of all nested objects.

---

# Time Complexity

| Operation          | Time Complexity                    |
| ------------------ | ---------------------------------- |
| `b = a`            | **O(1)** (copy the reference only) |
| `a.copy()`         | **O(n)** (copy all elements)       |
| `copy.deepcopy(a)` | **O(total size of object graph)**  |

---

# Why This Matters in AI Systems

Understanding reference semantics is important in production AI code:

* **Data preprocessing:** Accidentally modifying a shared dataset can corrupt training or evaluation data.
* **Model configurations:** Shared mutable dictionaries can introduce subtle bugs if updated in different parts of the code.
* **Caching:** Returning a mutable cached object lets callers modify the cached value unintentionally. Returning an immutable object or a copy is often safer.
* **Concurrency:** Multiple threads or async tasks sharing mutable objects require synchronization to avoid race conditions.

---

# Senior AI Engineer Interview Answer (30–45 seconds)

> When you assign one list to another using `b = a`, Python does **not** create a new list. It creates a new reference to the same list object, so both variables point to the same memory location. Any in-place modification, such as `append()` or item assignment, is visible through both references. If an independent copy is needed, use `list.copy()`, slicing (`a[:]`), or `list(a)` for a shallow copy, and `copy.deepcopy()` when nested mutable objects must also be copied. Understanding this distinction is essential for avoiding unintended side effects in production Python and AI systems.
