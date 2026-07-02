This is one of the **most common Senior AI Engineer / Backend Engineer / Systems Design coding interview questions**.

Interviewers are not just checking if you know an LRU cache. They are testing whether you understand:

* Data structure selection
* Time complexity
* Why HashMap alone isn't enough
* Why a Doubly Linked List is required
* Memory management
* Production applications (Redis, CPU caches, AI inference caches)
* Clean, extensible design

---

# 1. What is an LRU Cache?

LRU stands for:

```text
Least Recently Used
```

Suppose your cache can hold only **3 items**.

Operations:

```text
put(A)

put(B)

put(C)
```

Cache:

```text
A B C
```

Now access

```text
get(A)
```

A becomes the most recently used.

Cache order:

```text
B C A
```

Now insert

```text
put(D)
```

Cache is full.

Which item should be removed?

```text
B
```

because it is the **Least Recently Used**.

Final cache:

```text
C A D
```

---

# 2. Where is LRU Used?

Almost every large-scale system uses an LRU-like policy:

* CPU caches
* Operating system page replacement
* Browser caches
* CDN edge caches
* Redis (one of several eviction policies)
* Database buffer pools
* Feature stores in ML
* LLM KV caches (or variants optimized for sequence generation)
* Embedding/vector caches
* API response caches

---

# 3. Interview Requirements

A typical interview asks you to implement:

```python
cache.get(key)

cache.put(key, value)
```

with:

```text
Time Complexity

O(1)
```

for both operations.

---

# 4. Why Can't We Use a Dictionary Alone?

Dictionary:

```python
cache = {
    1:10,
    2:20,
    3:30
}
```

Lookup

```python
cache[2]
```

is

```text
O(1)
```

Great.

But when the cache is full,

how do we know

```text
Least Recently Used?
```

Dictionary has no notion of usage order.

---

# 5. Why Can't We Use a List?

Suppose

```text
A B C D
```

Access

```text
B
```

Need

```text
A C D B
```

Removing from the middle of a Python list is:

```text
O(n)
```

Not acceptable.

---

# 6. Data Structures We Need

We combine two structures:

### HashMap

Stores:

```text
Key

↓

Node
```

Lookup:

```text
O(1)
```

### Doubly Linked List

Maintains usage order.

Most recently used:

```text
Head
```

Least recently used:

```text
Tail
```

Operations:

```text
Remove node

O(1)

Insert node

O(1)
```

Together they provide O(1) get and put.

---

# 7. Node Structure

Each node stores:

```python
class Node:
    key
    value
    prev
    next
```

Why store the key?

When we evict the least recently used node, we must also remove its entry from the dictionary.

Without the key in the node, we wouldn't know which dictionary entry to delete.

---

# 8. Overall Design

```text
HashMap
───────────────

1 ─────────► Node1

2 ─────────► Node2

3 ─────────► Node3


Doubly Linked List

HEAD

↓

[3]

↓

[1]

↓

[2]

↓

TAIL
```

Head = Most Recently Used (MRU)

Tail = Least Recently Used (LRU)

---

# 9. Operations

### get(key)

If key exists:

1. Find node from dictionary (O(1))
2. Remove node from current position (O(1))
3. Move node to head (O(1))
4. Return value

---

### put(key,value)

Case 1:

Already exists

```text
Update value

Move to head
```

Case 2:

New key

```text
Insert at head
```

If cache exceeds capacity

```text
Remove tail

Delete from dictionary
```

Everything is O(1).

---

# 10. Complete Implementation

```python
class Node:

    def __init__(self, key=0, value=0):
        self.key = key
        self.value = value

        self.prev = None
        self.next = None


class LRUCache:

    def __init__(self, capacity):

        self.capacity = capacity

        self.cache = {}

        # Dummy nodes
        self.head = Node()
        self.tail = Node()

        self.head.next = self.tail
        self.tail.prev = self.head

    # ------------------

    def _remove(self, node):

        prev = node.prev
        nxt = node.next

        prev.next = nxt
        nxt.prev = prev

    # ------------------

    def _insert_front(self, node):

        node.next = self.head.next
        node.prev = self.head

        self.head.next.prev = node
        self.head.next = node

    # ------------------

    def get(self, key):

        if key not in self.cache:
            return -1

        node = self.cache[key]

        # Move to front
        self._remove(node)
        self._insert_front(node)

        return node.value

    # ------------------

    def put(self, key, value):

        if key in self.cache:

            node = self.cache[key]
            node.value = value

            self._remove(node)
            self._insert_front(node)

            return

        node = Node(key, value)

        self.cache[key] = node

        self._insert_front(node)

        if len(self.cache) > self.capacity:

            # Remove LRU
            lru = self.tail.prev

            self._remove(lru)

            del self.cache[lru.key]
```

---

# 11. Example Execution

```python
cache = LRUCache(2)

cache.put(1,10)

cache.put(2,20)
```

Cache:

```text
Head

↓

2

↓

1

↓

Tail
```

Access

```python
cache.get(1)
```

Cache becomes

```text
1

↓

2
```

Insert

```python
cache.put(3,30)
```

Capacity exceeded.

Evict

```text
2
```

Final:

```text
3

↓

1
```

Dictionary:

```python
{
1:Node,

3:Node
}
```

---

# 12. Time Complexity

| Operation | Complexity |
| --------- | ---------- |
| get       | O(1)       |
| put       | O(1)       |
| remove    | O(1)       |
| insert    | O(1)       |
| eviction  | O(1)       |

Space complexity:

```text
O(capacity)
```

---

# 13. Why Dummy Head and Tail?

Without dummy nodes:

```text
Head could be None

Tail could be None
```

Every insert/remove would need special handling for:

* Empty list
* One node
* Removing the head
* Removing the tail

Dummy nodes eliminate these edge cases, making the code simpler and less error-prone.

---

# 14. Production Considerations

A production-grade LRU cache often adds features beyond this interview implementation:

* **Thread safety:** Use mutexes, read-write locks, or sharded caches to support concurrent access.
* **Time-to-Live (TTL):** Expire entries after a configurable duration.
* **Metrics:** Track hits, misses, evictions, and memory usage for observability.
* **Size-based eviction:** Evict based on memory footprint instead of entry count.
* **Persistence:** Optionally serialize entries to disk or a distributed cache.
* **Distributed caching:** Coordinate across multiple nodes using systems like Redis or Memcached.
* **Admission policies:** Modern systems may combine LRU with policies like TinyLFU to improve hit rates under skewed workloads.

---

# 15. AI/LLM Use Cases

LRU-style caching is particularly useful in AI systems:

* **Embedding cache:** Avoid recomputing embeddings for repeated text.
* **Feature cache:** Reuse expensive feature engineering outputs.
* **Inference cache:** Return cached predictions for identical requests.
* **Tokenizer cache:** Reuse tokenization results for frequently seen prompts.
* **Model artifact cache:** Keep frequently used model weights or adapters in memory.
* **Prompt cache:** Store responses for repeated prompts when appropriate.
* **KV cache management:** While transformer KV caches use specialized policies, LRU-inspired ideas can be applied to managing limited cache resources in long-context or multi-session serving systems.

---

# 16. How to Answer in a Senior AI Engineer Interview

A strong answer would be:

> "An LRU cache provides O(1) `get` and `put` operations by combining a hash map with a doubly linked list. The hash map provides constant-time lookup from key to node, while the doubly linked list maintains items ordered by recency of use. On every access or update, the corresponding node is moved to the front of the list. When the cache reaches capacity, the node at the tail—the least recently used entry—is removed from both the linked list and the hash map. In production, additional concerns include concurrency, TTL-based expiration, memory-aware eviction, metrics, and distributed cache coordination."
