a = [1,2,3]
b = a

b.append(4)

print(a)
Questions

Why did a change?

How to avoid this?




This is a classic **Senior Python interview question**. It tests whether you understand **references, mutable objects, memory management, and copying**.

---

## Code

```python
a = [1, 2, 3]
b = a

b.append(4)

print(a)
```

Output:

```python
[1, 2, 3, 4]
```

---

# Question 1: Why did `a` change?

The statement:

```python
b = a
```

**does not create a new list**.

Instead, Python creates a **new reference** to the **same list object**.

### Memory before assignment

```text
a
|
v
[1, 2, 3]
```

### After

```python
b = a
```

Memory becomes:

```text
a --------\
           \
            -----> [1, 2, 3]
           /
b --------/
```

Both `a` and `b` point to the **same object**.

---

When you execute:

```python
b.append(4)
```

Python modifies that single shared list **in place**.

Memory becomes:

```text
a --------\
           \
            -----> [1, 2, 3, 4]
           /
b --------/
```

Therefore,

```python
print(a)
```

prints

```python
[1, 2, 3, 4]
```

---

## Proof with `id()`

```python
a = [1, 2, 3]
b = a

print(id(a))
print(id(b))
```

Example output:

```text
43782656
43782656
```

Both variables have the same object ID.

---

# Question 2: How do you avoid this?

If you want an independent list, create a **copy** instead of another reference.

## Method 1: `copy()` (Recommended)

```python
a = [1, 2, 3]

b = a.copy()

b.append(4)

print(a)
print(b)
```

Output:

```python
[1, 2, 3]
[1, 2, 3, 4]
```

Memory:

```text
a -----> [1, 2, 3]

b -----> [1, 2, 3, 4]
```

Two separate list objects.

---

## Method 2: List Slicing

```python
b = a[:]
```

---

## Method 3: `list()`

```python
b = list(a)
```

---

# What if the list contains nested lists?

Consider:

```python
a = [[1], [2]]

b = a.copy()

b[0].append(100)

print(a)
```

Output:

```python
[[1, 100], [2]]
```

Why?

`copy()` performs a **shallow copy**.

The outer list is copied, but the inner lists are still shared.

Memory:

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

---

# How to avoid this?

Use a **deep copy**.

```python
import copy

a = [[1], [2]]

b = copy.deepcopy(a)

b[0].append(100)

print(a)
print(b)
```

Output:

```python
[[1], [2]]
[[1, 100], [2]]
```

Now every nested object is copied recursively.

---

# When to use each approach

| Situation                                  | Solution                         |
| ------------------------------------------ | -------------------------------- |
| Just another reference to the same object  | `b = a`                          |
| Independent copy of a flat list            | `a.copy()`, `list(a)`, or `a[:]` |
| Independent copy of nested mutable objects | `copy.deepcopy(a)`               |

---

# Interview Follow-up: Why doesn't this happen with integers?

```python
a = 10
b = a

b += 5

print(a)
print(b)
```

Output:

```python
10
15
```

This is because integers are **immutable**.

`b += 5` creates a **new integer object** instead of modifying the existing one, so `a` continues to reference the original `10`.

---

# Senior AI Engineer Interview Answer (30–45 seconds)

> `b = a` does not copy the list—it copies the **reference**. Both variables point to the same mutable list object, so calling `b.append(4)` modifies that shared object, and the change is visible through `a` as well. To avoid this, create a new list using `a.copy()`, `list(a)`, or `a[:]` for a shallow copy. If the list contains nested mutable objects, use `copy.deepcopy()` to create a completely independent copy of the entire object graph. This distinction is important in production AI systems to avoid unintended side effects when preprocessing data, caching results, or passing mutable objects between components.
