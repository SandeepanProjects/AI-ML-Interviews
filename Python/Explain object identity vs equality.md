This is one of the most frequently asked **Senior Python interview questions** because it tests whether you understand Python's object model, references, and comparison semantics.

The key difference is:

* **Identity** → Are these two variables referring to the **same object** in memory?
* **Equality** → Do these two objects have the **same value**?

---

# Object Identity (`is`)

Identity checks whether two variables point to the exact same object.

Python uses the `is` operator.

```python
a = [1, 2, 3]
b = a

print(a is b)
```

Output

```text
True
```

Why?

Both variables reference the same list object.

Memory:

```text
a --------\
           \
            -----> [1,2,3]
           /
b --------/
```

You can verify this with `id()`:

```python
print(id(a))
print(id(b))
```

Example output:

```text
43822152
43822152
```

The IDs are the same, confirming they are the same object.

---

# Equality (`==`)

Equality compares the values of two objects.

```python
a = [1, 2, 3]
b = [1, 2, 3]

print(a == b)
```

Output

```text
True
```

Although the values are equal, these are different list objects.

```python
print(a is b)
```

Output

```text
False
```

Memory:

```text
a -----> [1,2,3]

b -----> [1,2,3]
```

Two separate objects contain the same values.

---

# Identity vs Equality Together

```python
a = [1, 2, 3]
b = a
c = [1, 2, 3]

print(a == b)
print(a is b)

print(a == c)
print(a is c)
```

Output

```text
True
True
True
False
```

Explanation:

* `a == b` → same values
* `a is b` → same object
* `a == c` → same values
* `a is c` → different objects

---

# Strings

```python
s1 = "Python"
s2 = "Python"

print(s1 == s2)
print(s1 is s2)
```

You may see:

```text
True
True
```

Why?

Python often **interns** identical string literals, so both variables may reference the same object.

However, you should **not** rely on `is` for string comparison.

Always write:

```python
if s1 == s2:
    ...
```

---

# Integers

```python
a = 100
b = 100

print(a is b)
```

Often:

```text
True
```

Python caches small integers (typically from `-5` to `256` in CPython).

For larger integers, implementation details vary.

Again, use `==` to compare numeric values.

---

# Custom Objects

```python
class User:
    def __init__(self, name):
        self.name = name

u1 = User("Alice")
u2 = User("Alice")

print(u1 == u2)
print(u1 is u2)
```

Output:

```text
False
False
```

By default:

* `==` compares object identity because `User` doesn't define equality.
* `is` also compares identity.

---

# Custom Equality with `__eq__`

```python
class User:
    def __init__(self, name):
        self.name = name

    def __eq__(self, other):
        return self.name == other.name

u1 = User("Alice")
u2 = User("Alice")

print(u1 == u2)
print(u1 is u2)
```

Output:

```text
True
False
```

Now equality is based on the `name` attribute rather than object identity.

---

# Why `None` Should Be Compared Using `is`

Correct:

```python
if value is None:
    print("Missing")
```

Avoid:

```python
if value == None:
    ...
```

Why?

`None` is a singleton—there is only one `None` object in a Python process.

Identity comparison is both clearer and immune to custom `__eq__` implementations.

---

# Common Interview Mistake

```python
x = []

if x is []:
    print("Empty")
```

Output:

```text
Nothing
```

Why?

Each `[]` creates a new list object.

Correct:

```python
if x == []:
    print("Empty")
```

Or, more idiomatically:

```python
if not x:
    print("Empty")
```

---

# Real-world AI Examples

### Comparing model names

```python
model = "gpt-4.1"

if model == "gpt-4.1":
    ...
```

Use `==` because you care about the value, not whether it's the same string object.

---

### Checking for an optional parameter

```python
if tokenizer is None:
    tokenizer = load_tokenizer()
```

Use `is None` because you're checking for the singleton `None`.

---

### Comparing configuration objects

```python
config1 == config2
```

Checks whether the configurations contain the same values.

```python
config1 is config2
```

Checks whether both variables reference the exact same configuration object.

---

# Summary

| Feature                   | Identity (`is`)        | Equality (`==`)                 |
| ------------------------- | ---------------------- | ------------------------------- |
| Checks                    | Same object in memory  | Same value/content              |
| Uses                      | Object identity (`id`) | `__eq__()` method               |
| Operator                  | `is`                   | `==`                            |
| Works with custom classes | Yes                    | Yes, if `__eq__` is implemented |
| Use for `None`            | ✅ Yes                  | ❌ Avoid                         |
| Use for strings/numbers   | ❌ No                   | ✅ Yes                           |

---

# Senior AI Engineer Interview Answer (45–60 seconds)

> In Python, **identity** and **equality** are different concepts. Identity, checked with the `is` operator, determines whether two variables reference the exact same object in memory. Equality, checked with `==`, determines whether two objects have the same value by calling the object's `__eq__()` method. For example, two different lists containing `[1, 2, 3]` are equal but not identical. In production code, use `==` when comparing values such as strings, numbers, or lists. Reserve `is` for identity checks, especially when comparing against singletons like `None`, where `if value is None:` is the recommended and idiomatic approach.
