## What is Backtracking?

**Backtracking is a systematic way of trying choices, and when a choice leads to a dead end, undoing that choice and trying another.**

Think of it as:

> **Choose → Explore → Fail? → Undo → Choose another**

It is essentially **DFS (Depth-First Search) + undoing decisions**.

A very common interview pattern is:

```text
                    Start
                  /       \
               Choice A   Choice B
                /   \        /  \
              A1    A2     B1   B2
             ❌     ✅      ❌    ?
```

If we try `A → A1` and discover that it doesn't work, we **backtrack**:

```text
A → A1 → ❌
       ↑
     undo A1
       ↓
A → A2 → ✅
```

---

# 1. The Core Idea

Suppose we want to generate all binary strings of length 3.

We need to make 3 decisions:

```text
position 0 → 0 or 1
position 1 → 0 or 1
position 2 → 0 or 1
```

So the decision tree is:

```text
                         ""
                    /          \
                   0            1
                 /   \        /   \
               00    01     10    11
              / \    / \    / \    / \
            000 001 010 011 100 101 110 111
```

There are:

```text
2 × 2 × 2 = 8
```

possibilities.

Backtracking walks through this tree using DFS.

---

# 2. The Backtracking Template

Almost every backtracking problem can be reduced to this:

```python
def backtrack(state):
    if is_solution(state):
        process(state)
        return

    for choice in choices:
        make_choice(choice)

        backtrack(state)

        undo_choice(choice)
```

The **most important line** is:

```python
undo_choice(choice)
```

That's what makes it backtracking.

---

# 3. Simple Example: Generate Binary Strings

```python
def generate(n):
    result = []
    path = []

    def backtrack():
        if len(path) == n:
            result.append("".join(path))
            return

        for choice in ["0", "1"]:
            path.append(choice)

            backtrack()

            path.pop()

    backtrack()
    return result
```

For:

```python
generate(3)
```

we get:

```text
000
001
010
011
100
101
110
111
```

---

# 4. Let's Understand the Calculations

This is where backtracking becomes much easier to understand.

At every level, we have:

```text
2 choices
```

and:

```text
3 levels
```

Therefore:

[
2^3 = 8
]

leaves.

For a general problem with:

* `n` positions
* `k` choices at each position

the number of possible solutions is:

[
k^n
]

For example:

```text
n = 5
k = 2
```

Then:

[
2^5 = 32
]

possible strings.

If:

```text
n = 10
k = 3
```

then:

[
3^{10} = 59049
]

possibilities.

That's why brute-force backtracking can become expensive very quickly.

---

# 5. What Actually Happens in Memory?

Consider:

```python
path = []
```

Initially:

```text
path = []
```

Choose `0`:

```text
path = [0]
```

Choose `0`:

```text
path = [0, 0]
```

Choose `1`:

```text
path = [0, 0, 1]
```

We reached a solution.

Now:

```python
path.pop()
```

Undo the last choice:

```text
[0, 0, 1]
       ↓
[0, 0]
```

Then try another choice:

```text
[0, 0, 0]
```

Then eventually backtrack again.

The same `path` array is reused.

---

# 6. Why `pop()` Is So Important

Consider:

```python
path.append(choice)

backtrack()

path.pop()
```

The sequence is:

```text
MAKE
  ↓
EXPLORE
  ↓
UNDO
```

For example:

```text
[]
 ↓
[0]
 ↓
[0,1]
 ↓
[0,1,1]
 ↓
solution
 ↓
[0,1]       ← undo
 ↓
[0]         ← undo
 ↓
[0,0]
```

Without `pop()`, your state would contain choices from previous branches.

---

# 7. Backtracking vs Normal DFS

They are closely related.

### Normal DFS

DFS usually says:

```text
Visit node
    ↓
Visit children
    ↓
Continue
```

### Backtracking

Backtracking says:

```text
Make decision
    ↓
Explore
    ↓
Undo decision
    ↓
Try next decision
```

So:

> **Backtracking = DFS over a decision tree + state restoration**

---

# 8. Example: Subsets

Suppose:

```text
nums = [1, 2, 3]
```

Every element has two choices:

```text
take it
don't take it
```

Decision tree:

```text
                         []
                     /        \
                  take 1     skip 1
                  [1]          []
                 /   \        /   \
             take2  skip2  take2  skip2
             [1,2]  [1]     [2]    []
```

Continue until all elements are processed.

The number of subsets is:

[
2^n
]

For 3 elements:

[
2^3 = 8
]

Subsets:

```text
[]
[1]
[2]
[3]
[1,2]
[1,3]
[2,3]
[1,2,3]
```

Code:

```python
def subsets(nums):
    result = []
    path = []

    def backtrack(index):
        if index == len(nums):
            result.append(path.copy())
            return

        # Take nums[index]
        path.append(nums[index])
        backtrack(index + 1)
        path.pop()

        # Don't take nums[index]
        backtrack(index + 1)

    backtrack(0)
    return result
```

---

# 9. The Calculation Behind Time Complexity

This is very important for interviews.

For subsets:

```text
At each element:

take
don't take
```

Therefore:

[
2^n
]

different states/leaves.

But we also have to copy the current path.

A path can contain up to `n` elements.

Therefore:

[
O(n \times 2^n)
]

time.

Space:

### Recursion stack

```text
O(n)
```

### Current path

```text
O(n)
```

### Output

```text
O(n × 2^n)
```

If we include the output:

[
O(n2^n)
]

space.

---

# 10. Permutations — Another Important Calculation

Suppose:

```text
nums = [1,2,3]
```

At the first position:

```text
3 choices
```

Second position:

```text
2 choices
```

Third position:

```text
1 choice
```

Therefore:

[
3 \times 2 \times 1 = 3!
]

So there are:

[
n!
]

permutations.

For:

```text
n = 4
```

we get:

[
4! = 4 \times 3 \times 2 \times 1 = 24
]

For:

```text
n = 10
```

we get:

[
10! = 3,628,800
]

That's why permutation backtracking becomes expensive.

---

# 11. Permutation Code

```python
def permutations(nums):
    result = []
    path = []
    used = [False] * len(nums)

    def backtrack():
        if len(path) == len(nums):
            result.append(path.copy())
            return

        for i in range(len(nums)):
            if used[i]:
                continue

            used[i] = True
            path.append(nums[i])

            backtrack()

            path.pop()
            used[i] = False

    backtrack()
    return result
```

The important part:

```python
used[i] = True
path.append(nums[i])

backtrack()

path.pop()
used[i] = False
```

Notice the symmetry:

```text
DO
 ↓
EXPLORE
 ↓
UNDO
```

---

# 12. Combination Problems

Suppose:

```text
n = 5
k = 2
```

How many ways can we choose 2 elements?

Mathematically:

[
C(n,k)=\frac{n!}{k!(n-k)!}
]

Therefore:

[
C(5,2)
======

\frac{5!}{2!3!}
]

# [

\frac{5×4×3×2×1}{(2×1)(3×2×1)}
]

[
=10
]

So there are 10 combinations.

Backtracking can generate these without generating every permutation.

---

# 13. The Most Important Optimization: Pruning

This is where backtracking becomes powerful.

Suppose you're solving:

> Find combinations whose sum equals 7.

Consider:

```text
[2, 3, 5, 8]
```

If your current path is:

```text
[2, 5]
```

sum:

```text
2 + 5 = 7
```

Solution.

But suppose:

```text
[5, 8]
```

sum:

```text
13
```

If all remaining numbers are positive, there is no point continuing.

So:

```python
if current_sum > target:
    return
```

This is called:

> **Pruning**

Instead of exploring:

```text
        5
       / \
      8   ...
     / \
   ... ...
```

we immediately stop:

```text
5 + 8 = 13 > 7
        ↓
      STOP
```

---

# 14. Why Pruning Matters

Without pruning:

[
O(2^n)
]

potential branches might be explored.

With pruning:

```text
Invalid branch
      ↓
STOP EARLY
```

The actual number of visited nodes can become much smaller.

Important interview point:

> **Pruning does not necessarily change the worst-case Big-O complexity.**

For example, a problem may still have worst-case:

[
O(2^n)
]

but pruning can dramatically improve practical runtime.

---

# 15. Classic Example: N-Queens

This is one of the most important backtracking interview problems.

Problem:

> Put `N` queens on an `N × N` chessboard so that no two queens attack each other.

For:

```text
N = 4
```

We could try:

```text
row 0 → 4 choices
row 1 → 4 choices
row 2 → 4 choices
row 3 → 4 choices
```

Naively:

[
4^4 = 256
]

possibilities.

But we don't need to continue if two queens attack each other.

For example:

```text
Q . . .
. . Q .
```

These queens are on the same diagonal.

Therefore:

```text
INVALID
```

We immediately backtrack.

---

# 16. N-Queens Decision Tree

Conceptually:

```text
Row 0
 ├── Col 0
 │    ├── Row 1 Col 0 ❌
 │    ├── Row 1 Col 1 ❌
 │    ├── Row 1 Col 2 ❌
 │    └── Row 1 Col 3
 │
 ├── Col 1
 │    ├── ...
 │
 ├── Col 2
 │    └── ...
 │
 └── Col 3
      └── ...
```

The algorithm keeps eliminating invalid branches.

That's the essence of backtracking.

---

# 17. The Mathematical Pattern You Should Memorize

When analyzing a backtracking problem, ask:

### Question 1

How many choices exist at each level?

Example:

```text
2 choices
```

### Question 2

How deep is the recursion?

Example:

```text
n
```

### Question 3

Are choices decreasing?

For permutations:

```text
n
n-1
n-2
...
1
```

Therefore:

[
n!
]

### Question 4

Are choices constant?

For subsets:

```text
2
2
2
...
2
```

Therefore:

[
2^n
]

### Question 5

Is there pruning?

If yes, practical runtime can be much lower.

---

# 18. Three Complexity Patterns to Know

### Binary decision

```text
take / don't take
```

Usually:

[
O(2^n)
]

Examples:

* subsets
* subset sum
* partition

---

### K choices at every level

```text
choice 1
choice 2
...
choice k
```

Depth `n`:

[
O(k^n)
]

Examples:

* generating strings
* some path/combination problems

---

### Decreasing choices

```text
n
n-1
n-2
...
1
```

Therefore:

[
O(n!)
]

Examples:

* permutations
* some scheduling/order problems

---

# 19. A Better Mental Model

Don't think:

> "Backtracking is a complicated algorithm."

Think:

> **Backtracking is a loop over possible decisions where recursion explores one decision, and undo restores the previous state.**

The universal pattern is:

```python
def backtrack(state):

    if goal_reached(state):
        save_answer()
        return

    for choice in choices:

        if invalid(choice):
            continue

        # 1. Choose
        apply(choice)

        # 2. Explore
        backtrack(state)

        # 3. Undo
        undo(choice)
```

The three lines to remember are:

```text
CHOOSE
  ↓
EXPLORE
  ↓
UNDO
```

And the optimization is:

```text
CHOOSE
  ↓
CHECK
  ↓
EXPLORE
  ↓
UNDO
```

where `CHECK` is your **pruning condition**.

---

## 20. Backtracking vs Dynamic Programming

This distinction is useful in interviews.

**Backtracking:**

```text
Try choices
   ↓
Explore
   ↓
Undo
```

Usually used when you need to **enumerate/search possible configurations**.

Examples:

* N-Queens
* Sudoku
* permutations
* combinations
* subsets
* word search

**Dynamic Programming:**

```text
Solve subproblem
   ↓
Store result
   ↓
Reuse result
```

Used when the same subproblems occur repeatedly.

So a useful rule is:

> **Backtracking explores possibilities; DP avoids recomputing possibilities.**

---

## 21. Interview Shortcut

When you see a problem containing phrases like:

* "generate all"
* "find all possible"
* "choose"
* "arrange"
* "combination"
* "permutation"
* "place"
* "partition"
* "solve the board"
* "all valid configurations"

immediately ask yourself:

```text
Can I represent this as a decision tree?
                ↓
        What is my choice?
                ↓
        What makes it invalid?
                ↓
        Can I prune?
                ↓
        What do I undo?
```

That thought process will identify most backtracking problems very quickly.
