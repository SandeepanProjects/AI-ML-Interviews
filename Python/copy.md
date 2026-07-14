This is one of the **most frequently asked Python interview questions**, especially for **Senior AI Engineer**, **Backend**, and **Python Developer** roles.

Let's understand it internally.

---

# Example

```python
import copy

a = [[1,2],[3,4]]

b = a
c = a.copy()
d = copy.deepcopy(a)
```

Memory looks like this:

```
          +--------------------+
a ------> |  List              |
b ------> |--------------------|
          | --> [1,2] -----------+
          | --> [3,4] -------+   |
          +------------------|---+
                             |    
                +------------+----
                | Nested Lists
                |
                v
              [1,2]
              [3,4]
```

Now let's understand every assignment.

---

# 1. Assignment (No Copy)

```python
b = a
```

Nothing is copied.

Only another variable points to the same object.

```
a --------+
          |
b --------+

      List Object
```

Check identity

```python
print(a is b)
```

Output

```
True
```

Same object.

---

If you modify

```python
b.append([5,6])
```

```
a
↓

[[1,2],[3,4],[5,6]]
```

because both point to same object.

---

# Memory

```
a
 \
  \
   ---> List
  /
 /
b
```

Only one list exists.

---

# 2. Shallow Copy

```python
c = a.copy()
```

or

```python
c = copy.copy(a)
```

Python creates

* new outer list
* but NOT new nested objects

Memory

```
a
 |
 |        +------------------+
 +------> | [ *, * ]         |
          +--|------|--------+
             |      |
             |      |
             v      v
           [1,2]  [3,4]


c
 |
 +------> +------------------+
          | [ *, * ]         |
          +--|------|--------+
             |      |
             |      |
             +------+
             Same nested lists
```

Notice

Outer list is different.

Inner lists are same.

---

Check

```python
print(a is c)
```

```
False
```

Different outer list.

---

But

```python
print(a[0] is c[0])
```

Output

```
True
```

Same nested list.

---

Modify nested list

```python
c[0].append(100)

print(a)
print(c)
```

Output

```
[[1,2,100],[3,4]]
[[1,2,100],[3,4]]
```

Both changed.

Why?

Because nested object is shared.

---

Now modify outer list

```python
c.append([5,6])
```

Output

```
a

[[1,2],[3,4]]

c

[[1,2],[3,4],[5,6]]
```

Outer lists are different.

Only nested objects are shared.

---

# Internal Working

Suppose

```
a

[
    obj1,
    obj2
]
```

Shallow copy does

```
new_list = [
    obj1,
    obj2
]
```

Not

```
new_list = [
    copy(obj1),
    copy(obj2)
]
```

Only references are copied.

---

# 3. Deep Copy

```python
d = copy.deepcopy(a)
```

Python recursively copies everything.

Memory

```
a

+------------+
|  *   *     |
+--|---|-----+
   |   |
   v   v
 [1,2] [3,4]



d

+------------+
|  *   *     |
+--|---|-----+
   |   |
   v   v
 [1,2] [3,4]
```

Everything is new.

Nothing is shared.

---

Check

```python
print(a is d)
```

```
False
```

Outer list different.

---

Check nested

```python
print(a[0] is d[0])
```

Output

```
False
```

Nested list also different.

---

Modify

```python
d[0].append(100)

print(a)
print(d)
```

Output

```
[[1,2],[3,4]]

[[1,2,100],[3,4]]
```

Original unchanged.

---

# Compare IDs

```python
import copy

a = [[1,2],[3,4]]

b = a
c = a.copy()
d = copy.deepcopy(a)

print(id(a), id(b), id(c), id(d))

print(id(a[0]))
print(id(c[0]))
print(id(d[0]))
```

Typical output

```
Outer

a : 4510019200
b : 4510019200
c : 4510031008
d : 4510032000

Nested

a[0] : 4510040000
c[0] : 4510040000
d[0] : 4510050000
```

Notice

```
a == b
a != c
a != d

Nested

a[0] == c[0]

a[0] != d[0]
```

---

# What does `deepcopy()` do internally?

Simplified implementation:

```python
def deepcopy(obj):
    if obj is immutable:
        return obj

    new_obj = create_new_object()

    for item in obj:
        new_obj.append(deepcopy(item))

    return new_obj
```

It recursively traverses the entire object graph and duplicates mutable objects, while safely handling shared references and circular references internally.

---

# Time Complexity

Suppose

```
[
 [1],
 [2],
 [3],
 ...
]
```

## Assignment

```python
b = a
```

Time

```
O(1)
```

No copying.

---

## Shallow Copy

```python
a.copy()
```

Copies only first level.

```
O(n)
```

where `n` is the number of elements in the outer container.

---

## Deep Copy

```python
copy.deepcopy()
```

Copies everything recursively.

```
O(total objects)
```

If there are 1 million nested objects,

all million are copied.

---

# Interview Question

**What is the difference between shallow copy and deep copy?**

| Feature                                           | Assignment (`b = a`) | Shallow Copy (`copy.copy()` / `list.copy()`) | Deep Copy (`copy.deepcopy()`) |
| ------------------------------------------------- | -------------------- | -------------------------------------------- | ----------------------------- |
| New outer object                                  | ❌                    | ✅                                            | ✅                             |
| New nested objects                                | ❌                    | ❌                                            | ✅                             |
| Shares nested references                          | ✅                    | ✅                                            | ❌                             |
| Recursive copy                                    | ❌                    | ❌                                            | ✅                             |
| Time complexity                                   | O(1)                 | O(n) (outer level)                           | O(total nested objects)       |
| Memory usage                                      | Lowest               | Moderate                                     | Highest                       |
| Changes to nested mutable objects affect original | ✅                    | ✅                                            | ❌                             |

### Rule of thumb

* Use **assignment (`=`)** when you intentionally want multiple variables to refer to the same object.
* Use a **shallow copy** when you need a new outer container but the contained objects can be safely shared (or are immutable).
* Use a **deep copy** when you need a completely independent copy of a nested mutable object structure.
