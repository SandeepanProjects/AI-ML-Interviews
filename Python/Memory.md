This is another **very common Senior Python interview question** because it tests your understanding of **Python's object model, memory management, and interning**.

Consider:

```python
a = 10
b = 10

print(id(a))
print(id(b))
print(id(a) == id(b))
```

Output:

```python
True
```

The question is:

> **Why are `a` and `b` pointing to the same object?**

---

# Step 1: Everything in Python is an Object

When you write:

```python
a = 10
```

Python creates (or reuses) an integer object.

```
          Integer Object
        +---------------+
        |      10       |
        +---------------+
               ^
               |
               a
```

Now execute:

```python
b = 10
```

Instead of creating another integer object, Python checks whether an object representing `10` already exists.

Since it does, Python reuses it.

```
          Integer Object
        +---------------+
        |      10       |
        +---------------+
           ^         ^
           |         |
           a         b
```

Therefore:

```python
id(a) == id(b)
```

is

```python
True
```

---

# Step 2: Why does Python do this?

Creating objects is expensive.

Imagine:

```python
for _ in range(10_000_000):
    x = 10
```

If Python created a new integer object every iteration, it would allocate millions of identical objects, wasting both memory and CPU time.

Instead, Python **reuses commonly used immutable objects**.

This optimization is called **integer interning** (or the **small integer cache** in CPython).

---

# Step 3: Small Integer Cache (CPython)

CPython pre-creates integer objects in the range:

```text
-5 to 256
```

These objects are created when the interpreter starts.

Conceptually:

```
Integer Cache

-5
-4
-3
...
0
1
2
...
10
...
100
...
256
```

Whenever you write:

```python
x = 10
```

Python simply returns the cached object.

```
Cache

10
 ^
 |
 +------ a

 ^
 |
 +------ b
```

No new object is created.

---

# Step 4: Demonstration

```python
a = 100
b = 100

print(id(a))
print(id(b))
print(a is b)
```

Output:

```python
True
```

because `100` is cached.

---

Now try:

```python
a = 1000
b = 1000

print(a is b)
```

The result **depends on the Python implementation and context**.

* In CPython, large integer literals appearing in the same code block may still reference the same object due to compiler optimizations.
* If the integers are created dynamically at runtime, they are usually different objects.

For example:

```python
a = int("1000")
b = int("1000")

print(a is b)
```

Typical output:

```python
False
```

because these are two distinct integer objects created at runtime.

**Interview takeaway:** Do **not** rely on `is` for comparing numbers. Use `==`.

---

# Step 5: Why only immutable objects?

Suppose integers were mutable.

```python
a = 10
b = 10
```

If both point to the same object and you could do something like:

```python
a.value = 20
```

then `b` would unexpectedly become `20` as well.

That would make programs unpredictable.

Because integers are **immutable**, sharing them is completely safe.

---

# Step 6: Strings

Python also interns many strings.

```python
a = "hello"
b = "hello"

print(a is b)
```

Often:

```python
True
```

because the same immutable string object is reused.

Again, don't depend on this behavior in your code.

---

# Step 7: Tuples

Tuples are immutable too.

```python
a = (1, 2)
b = (1, 2)

print(a is b)
```

The result may be `True` due to optimizations, but this is **not guaranteed** across implementations or contexts.

---

# Step 8: Lists

Lists are mutable, so they are **never** automatically shared like cached integers.

```python
a = [1, 2]
b = [1, 2]

print(a is b)
```

Output:

```python
False
```

Memory:

```
a ---> [1,2]

b ---> [1,2]
```

Two different list objects are created.

---

# Interview Question

**Why does `id(a) == id(b)` return `True` for:**

```python
a = 10
b = 10
```

**Answer:**

* Python variables store references to objects.
* In CPython, integers from **-5 to 256** are stored in a **small integer cache**.
* When `a = 10` and `b = 10` execute, both variables reference the same cached integer object.
* Since both references point to the same object, `id(a) == id(b)` and `a is b` are `True`.
* This optimization is safe because integers are immutable.

---

## `==` vs `is`

```python
a = int("1000")
b = int("1000")

print(a == b)   # True (same value)
print(a is b)   # False (different objects)
```

* `==` compares **values**.
* `is` compares **object identity** (whether both names refer to the exact same object).

**Best practice:** Use `==` for comparing values. Reserve `is` for identity checks (for example, `x is None`).
