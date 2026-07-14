These are **core Python concepts** that are asked almost every time in **Senior AI Engineer**, **Backend Engineer**, and **Python Developer** interviews. Understanding not just the syntax but the internal mechanics will help you answer follow-up interview questions confidently.

---

# 1. Positional Arguments

## Definition

A **positional argument** is matched to a function parameter based on its **position** in the function call.

Example:

```python
def greet(name, age):
    print(f"Name: {name}")
    print(f"Age : {age}")

greet("Alice", 30)
```

Output:

```
Name: Alice
Age : 30
```

### How Python Executes This

When Python calls:

```python
greet("Alice", 30)
```

it creates a new **stack frame**.

```
Stack Frame

name → "Alice"
age  → 30
```

Python maps arguments by order:

```
Parameter      Argument

name    ←     "Alice"
age     ←      30
```

If the order changes:

```python
greet(30, "Alice")
```

then

```
name = 30
age = "Alice"
```

No type checking happens automatically.

---

### Common Interview Questions

#### Q1

```python
def add(a, b):
    return a + b

add(10)
```

Output

```
TypeError:
missing 1 required positional argument: 'b'
```

---

#### Q2

```python
add(10,20,30)
```

Output

```
TypeError
```

Too many positional arguments.

---

# 2. Keyword Arguments

Instead of matching by position, Python matches by **parameter name**.

```python
def greet(name, age):
    print(name)
    print(age)

greet(age=30, name="Alice")
```

Output

```
Alice
30
```

Notice order no longer matters.

---

### Internal Mapping

```
Parameter

name  ← keyword "name"

age   ← keyword "age"
```

---

### Mixing Positional and Keyword Arguments

Valid:

```python
greet("Alice", age=30)
```

Python processes:

1. Positional arguments first
2. Keyword arguments second

---

Invalid:

```python
greet(name="Alice",30)
```

Output

```
SyntaxError
```

because positional arguments cannot come after keyword arguments.

---

# Default Arguments

```python
def greet(name, age=25):
    print(name, age)

greet("Alice")
```

Output

```
Alice 25
```

---

# 3. *args

Interviewers almost always ask this.

Suppose you write

```python
def add(a,b):
    return a+b
```

Only two numbers can be passed.

What if the count is unknown?

Use `*args`.

```python
def add(*args):
    print(args)
```

Call

```python
add(1,2,3,4,5)
```

Output

```
(1,2,3,4,5)
```

Notice

`args` is a tuple.

---

### Internal Working

Python converts

```python
add(1,2,3)
```

into

```
args

↓

(1,2,3)
```

Equivalent to

```python
args = (1,2,3)
```

---

### Example

```python
def add(*args):
    total = 0

    for num in args:
        total += num

    return total

print(add(1,2,3,4))
```

Output

```
10
```

---

### Mixing

```python
def func(a,b,*args):
    print(a)
    print(b)
    print(args)

func(1,2,3,4,5)
```

Output

```
1
2
(3,4,5)
```

---

# 4. **kwargs

Used for **variable keyword arguments**.

Example

```python
def person(**kwargs):
    print(kwargs)

person(name="Alice", age=30)
```

Output

```
{
'name':'Alice',
'age':30
}
```

`kwargs` is a dictionary.

---

### Internal Working

Python converts

```python
person(name="Alice", age=30)
```

to

```
kwargs

↓

{
'name':'Alice',
'age':30
}
```

---

Example

```python
def person(**kwargs):

    for k,v in kwargs.items():
        print(k,v)

person(name="Alice",
       age=30,
       city="Delhi")
```

Output

```
name Alice
age 30
city Delhi
```

---

# *args + **kwargs Together

```python
def func(a,b,*args,**kwargs):

    print(a)
    print(b)
    print(args)
    print(kwargs)

func(
    1,
    2,
    3,
    4,
    x=10,
    y=20
)
```

Output

```
1
2
(3,4)
{'x':10,'y':20}
```

---

### Interview Question

Parameter order must be

```python
def func(
    positional,
    *args,
    keyword_only=None,
    **kwargs
):
    pass
```

---

# 5. Lambda Functions

Lambda means **anonymous function**.

Instead of

```python
def square(x):
    return x*x
```

you can write

```python
square = lambda x: x*x

print(square(5))
```

Output

```
25
```

---

### Syntax

```
lambda parameters : expression
```

Only one expression is allowed.

---

### Multiple Parameters

```python
add = lambda a,b:a+b

print(add(3,4))
```

Output

```
7
```

---

### Sorting

Interview favorite.

```python
students = [

("Bob",80),

("Alice",95),

("John",60)

]

students.sort(
    key=lambda x:x[1]
)

print(students)
```

Output

```
[
('John',60),

('Bob',80),

('Alice',95)
]
```

Lambda tells Python

```
Use second element
as sorting key
```

---

# 6. Nested Functions

Functions inside another function.

```python
def outer():

    print("Outer")

    def inner():
        print("Inner")

    inner()

outer()
```

Output

```
Outer
Inner
```

---

### Memory

```
outer()

↓

Stack Frame

↓

inner() created

↓

called

↓

destroyed
```

The inner function exists only while `outer()` is executing unless it is returned.

---

# Why Use Nested Functions?

Hide helper functions.

```python
def calculate():

    def validate():
        print("Checking input")

    validate()

    print("Calculation")
```

No one outside can call

```python
validate()
```

because it is local to `calculate()`.

---

# 7. Closures

This is the most important interview topic among these.

A **closure** is a function that remembers variables from its enclosing scope even after the enclosing function has finished executing.

Example

```python
def outer():

    x = 10

    def inner():
        print(x)

    return inner

f = outer()

f()
```

Output

```
10
```

---

## The Big Question

After

```python
f = outer()
```

`outer()` has already returned.

Normally its local variables should disappear.

So how does `inner()` still know `x`?

---

### Internal Working

Step 1

```
outer()

x = 10
```

Memory

```
Stack

x =10

inner
```

---

Step 2

Return

```
return inner
```

Instead of destroying `x`, Python stores it in a special structure called a **closure cell** because `inner` still references it.

Conceptually:

```
Function Object (inner)

↓

Closure

↓

x =10
```

---

Step 3

Later

```python
f()
```

Python executes

```
print(x)
```

It looks in the closure and finds:

```
x =10
```

---

### Another Example

```python
def multiplier(n):

    def multiply(x):
        return x*n

    return multiply

double = multiplier(2)

triple = multiplier(3)

print(double(10))
print(triple(10))
```

Output

```
20
30
```

Memory

```
double

closure

n=2


triple

closure

n=3
```

Each returned function carries its own remembered state.

---

# Where Closures Are Used

Closures are used extensively in:

* Decorators
* Function factories
* Callback functions
* Event handlers
* Memoization
* Dependency injection patterns
* Frameworks like Flask and FastAPI
* AI libraries that generate configurable functions

---

# Summary

| Concept              | What It Does                                       | Internal Representation                              | Common Use Cases                                        |
| -------------------- | -------------------------------------------------- | ---------------------------------------------------- | ------------------------------------------------------- |
| Positional arguments | Matched by order                                   | Parameters receive arguments in sequence             | Standard function calls                                 |
| Keyword arguments    | Matched by parameter name                          | Mapping by name instead of position                  | Readability, optional ordering                          |
| `*args`              | Collects extra positional arguments                | Tuple                                                | Variable-length APIs                                    |
| `**kwargs`           | Collects extra keyword arguments                   | Dictionary                                           | Configuration, flexible interfaces                      |
| `lambda`             | Anonymous single-expression function               | Function object                                      | `sort()`, `map()`, `filter()`, callbacks                |
| Nested functions     | Function defined inside another                    | Local function object created during outer execution | Encapsulation, helper functions                         |
| Closures             | Returned nested function remembers outer variables | Function object + captured closure cells             | Decorators, factories, callbacks, configurable behavior |

These concepts frequently appear together in interviews because they form the foundation for more advanced Python features like **decorators**, **context managers**, **dependency injection**, and **functional programming patterns** used in production AI systems.

This is one of the **most important Python interview topics**. Senior AI Engineer interviews often go beyond "what is a closure?" and ask:

* **What is a free variable?**
* **Where is it stored?**
* **Why isn't it destroyed when `outer()` returns?**
* **How does Python implement closures internally?**

Let's understand it step by step.

---

# Example

```python
def outer():

    x = 10

    def inner():
        print(x)

    return inner

f = outer()

f()
```

Output

```
10
```

The interesting question is:

> **How is `x` still available after `outer()` has already finished executing?**

Normally, local variables disappear when a function returns.

---

# Step 1: Calling `outer()`

```python
f = outer()
```

Python creates a new stack frame.

```
Stack Frame (outer)

-------------------
x = 10
inner = function
-------------------
```

At this moment:

```
outer()
│
├── x = 10
└── inner()
```

---

# Step 2: Creating `inner`

Python creates the function object.

Conceptually:

```
inner

↓

Function Object
```

Now Python analyzes the code inside `inner`.

```python
def inner():
    print(x)
```

It notices something important:

> `x` is **not defined inside `inner`**.

So Python searches outward.

It finds:

```
outer()

↓

x = 10
```

Therefore,

```
x belongs to outer()

but

is used by inner()
```

This makes `x` a **free variable**.

---

# What is a Free Variable?

A **free variable** is:

> A variable that is **used inside a function but defined in an enclosing scope** (not local to that function).

Example:

```python
def outer():

    x = 10

    def inner():
        print(x)
```

Inside `inner`:

```
Local variables

None
```

Python looks outside:

```
outer

↓

x
```

Therefore

```
x

↓

Free Variable
```

---

# LEGB Rule

Python searches variables in this order:

```
L → Local

E → Enclosing

G → Global

B → Built-in
```

For

```python
print(x)
```

Python checks:

```
Local?

No

↓

Enclosing?

Yes

↓

Found x
```

---

# Step 3: Returning `inner`

Now

```python
return inner
```

Normally,

```
outer()

↓

Stack frame destroyed
```

which would remove

```
x
```

But Python notices:

```
inner still needs x
```

So instead of deleting `x`, Python moves it into a special object called a **closure**.

Conceptually:

```
Function inner

↓

Closure

↓

x = 10
```

---

# What is a Closure?

A **closure** is:

> A function together with the variables it has captured from its enclosing scope.

Think of it as:

```
Function

+

Saved Variables
```

```
inner

+

x=10
```

---

# Memory Diagram

Before `outer()` returns:

```
Stack

outer()

------------------

x =10

inner()

------------------
```

After returning:

```
Stack

(empty)
```

But

```
Function Object

inner

↓

Closure

↓

x =10
```

The stack is gone.

The closure survives.

---

# Step 4: Calling `f()`

Now

```python
f()
```

Python executes

```python
print(x)
```

Where does it find `x`?

Not local.

Not global.

Instead:

```
Function

↓

Closure

↓

x =10
```

Output

```
10
```

---

# Inspecting the Closure

Python lets you inspect it.

```python
def outer():

    x = 10

    def inner():
        print(x)

    return inner

f = outer()

print(f.__closure__)
```

Output

```
(<cell at 0x...: int object at ...>,)
```

The tuple contains **cell objects**.

Each cell stores one captured variable.

---

Access the value:

```python
print(f.__closure__[0].cell_contents)
```

Output

```
10
```

---

# Multiple Free Variables

```python
def outer():

    a = 10
    b = 20

    def inner():
        print(a,b)

    return inner
```

Closure

```
Function

↓

Closure

↓

a=10

b=20
```

Inspect

```python
f = outer()

for cell in f.__closure__:
    print(cell.cell_contents)
```

Output

```
10
20
```

---

# Closure Example: Function Factory

```python
def multiplier(n):

    def multiply(x):
        return x*n

    return multiply
```

```
double = multiplier(2)

triple = multiplier(3)
```

Memory

```
double

↓

multiply()

↓

Closure

↓

n=2
```

```
triple

↓

multiply()

↓

Closure

↓

n=3
```

Now

```python
print(double(5))
```

Python executes

```
5 × 2
```

Output

```
10
```

While

```python
print(triple(5))
```

becomes

```
5 × 3
```

Output

```
15
```

Each returned function has its own closure.

---

# Closure vs Local Variable

```python
def func():

    x = 10

    print(x)
```

Here

```
x
```

is

```
Local Variable
```

because it belongs to `func`.

---

Now

```python
def outer():

    x =10

    def inner():
        print(x)
```

Inside `inner`

```
x
```

is **not local**.

It comes from the enclosing function.

Therefore

```
Free Variable
```

---

# Interview Definition

### What is a Free Variable?

A **free variable** is a variable that is **referenced inside a function but defined in an enclosing (non-global) scope**. It is neither a local variable nor a global variable for that function.

---

### What is a Closure?

A **closure** is a function object that **retains access to the free variables from its enclosing scope even after the enclosing function has finished executing**. Python achieves this by storing the captured variables in **closure cells**, which are attached to the function object.

---

# Summary

| Concept        | Meaning                                              | Example                                     |
| -------------- | ---------------------------------------------------- | ------------------------------------------- |
| Local Variable | Defined and used in the same function                | `x = 10` inside `inner()`                   |
| Free Variable  | Used in a function but defined in an enclosing scope | `x` used in `inner()`, defined in `outer()` |
| Closure        | The function plus its captured free variables        | `inner` + captured `x = 10`                 |
| Closure Cell   | Internal object that stores each captured variable   | Accessible via `f.__closure__`              |

**Key interview takeaway:** A **free variable** is the variable itself (`x` in this example), while a **closure** is the mechanism that allows the returned function (`inner`) to keep using that free variable after `outer()` has returned.

This is one of the **most famous Python interview questions** because it tests your understanding of:

* Default arguments
* Mutable objects
* Function objects
* Memory model
* Object identity

Consider the code:

```python
def add(x=[]):
    x.append(1)
    return x

print(add())
print(add())
```

Output:

```python
[1]
[1, 1]
```

Most beginners expect:

```python
[1]
[1]
```

But that's **not** what happens.

---

# Step 1: When is the default argument created?

Many people think this happens every time:

```python
def add(x=[]):
```

They imagine Python does this:

```
Call 1

x → []

Call 2

x → []
```

This is **incorrect**.

Python evaluates default arguments **once**, when the function is **defined**, not each time it is called.

---

## What Python actually does

When Python executes:

```python
def add(x=[]):
    x.append(1)
    return x
```

it creates the list immediately.

Conceptually:

```
List Object

[]

↑
Stored as function default
```

The function object internally keeps a reference to this list.

You can even inspect it:

```python
print(add.__defaults__)
```

Output:

```python
([],)
```

The tuple contains the default values for the function.

---

# Step 2: First call

```python
add()
```

No argument is supplied.

Python therefore uses the stored default list.

Memory before:

```
Function

↓

Default List

[]
```

Inside the function:

```python
x.append(1)
```

The list becomes

```
[1]
```

Return:

```python
[1]
```

Memory after:

```
Function

↓

Default List

[1]
```

Notice:

The default list was **modified**.

---

# Step 3: Second call

Again:

```python
add()
```

Python **does not create a new list**.

It reuses the existing default list.

Current memory:

```
Function

↓

Default List

[1]
```

Execute:

```python
x.append(1)
```

Now the list becomes

```
[1,1]
```

Return

```python
[1,1]
```

---

# Why?

Because

```python
x
```

points to

the **same list object** every time.

You can prove it.

```python
def add(x=[]):
    print(id(x))
    x.append(1)
    return x

add()
add()
```

Typical output

```python
140435480

140435480
```

Same object.

---

# Visual Memory Diagram

After function definition

```
Function Object

↓

Default Argument

↓

[]
```

First call

```
x
↓

[]

append(1)

↓

[1]
```

Second call

```
x

↓

[1]

append(1)

↓

[1,1]
```

Nothing new is created.

---

# Compare with Immutable Default

```python
def add(x=0):
    x += 1
    return x

print(add())
print(add())
```

Output

```python
1
1
```

Why?

Because integers are immutable.

```
x += 1
```

creates a **new integer object** instead of modifying the existing one.

The stored default (`0`) is never changed.

---

# Correct Way

Instead of

```python
def add(x=[]):
```

use

```python
def add(x=None):

    if x is None:
        x = []

    x.append(1)

    return x
```

Now

```
Call 1

None

↓

New List

↓

[1]
```

Call 2

```
None

↓

New List

↓

[1]
```

Output

```python
[1]
[1]
```

Each call gets a fresh list.

---

# Why use `None`?

Because `None` is immutable and acts as a sentinel value.

Python stores

```
None
```

as the default.

Then inside the function you decide whether to create a new list.

---

# Inspecting Function Defaults

```python
def add(x=[]):
    pass

print(add.__defaults__)
```

Output

```python
([],)
```

Notice that defaults are stored as a tuple on the function object.

If the default list changes:

```python
def add(x=[]):
    x.append(1)

add()

print(add.__defaults__)
```

Output

```python
([1],)
```

The function's stored default itself has been modified.

---

# Interview Questions

### Q1. Why is this dangerous?

```python
def func(data=[]):
```

Because **all calls without an explicit argument share the same mutable object**, leading to unexpected state being carried across calls.

---

### Q2. Which objects cause this problem?

Any **mutable object**, such as:

* `list`
* `dict`
* `set`
* instances of custom mutable classes

Example:

```python
def f(d={}):
    d["count"] = d.get("count", 0) + 1
    return d

print(f())  # {'count': 1}
print(f())  # {'count': 2}
```

---

### Q3. Is this always a bug?

Usually, yes.

However, it can be used intentionally to preserve state across function calls, though using a closure or a class is generally clearer.

---

# Summary

| Code           | Behavior                                                            |
| -------------- | ------------------------------------------------------------------- |
| `def f(x=[]):` | Default list created **once** at function definition time           |
| `f()`          | Reuses the same list on every call                                  |
| `append()`     | Mutates the existing list                                           |
| Result         | State accumulates across calls                                      |
| Best practice  | Use `None` as the default and create a new list inside the function |

## Senior AI Engineer Interview Answer

If asked **"Why does the second call return `[1, 1]`?"**, a strong answer is:

> Default argument expressions are evaluated only once, when the function is defined. The default list is stored in the function object's `__defaults__` tuple. Each call to `add()` without an argument reuses the same list object. Since `list.append()` mutates that shared list in place, the changes persist across calls. The recommended pattern is to use `None` as the default and create a new list inside the function when needed.

This is one of the **most fundamental Python concepts** and is asked very frequently in **Senior AI Engineer**, **Backend**, and **Python** interviews because it is the foundation for **decorators, callbacks, closures, higher-order functions, FastAPI, LangChain, and AI frameworks**.

---

# What is a First-Class Function?

A language supports **first-class functions** if functions are treated like any other object.

In Python, functions are objects.

That means a function can:

1. ✅ Be assigned to a variable
2. ✅ Be passed as an argument
3. ✅ Be returned from another function
4. ✅ Be stored in data structures
5. ✅ Be created at runtime

---

## Proof: Functions are Objects

```python
def greet():
    print("Hello")

print(type(greet))
```

Output

```text
<class 'function'>
```

A function is simply another Python object.

---

# Memory Representation

When Python executes

```python
def greet():
    print("Hello")
```

Python creates a **function object**.

```
Function Object

+------------------+
| greet() code     |
+------------------+

^

greet
```

`greet` is just a variable pointing to a function object.

Exactly like

```python
x = 10
```

where

```
x

↓

Integer Object (10)
```

Similarly,

```
greet

↓

Function Object
```

---

# 1. Can Functions Be Stored?

Yes.

```python
def greet():
    print("Hello")

x = greet

x()
```

Output

```
Hello
```

---

## Internally

```
greet

↓

Function Object

↑

x
```

Both variables point to the same function.

Check it:

```python
print(greet is x)
```

Output

```
True
```

---

# Another Example

```python
def add(a, b):
    return a + b

operation = add

print(operation(5, 6))
```

Output

```
11
```

No copying happened.

Only another reference was created.

---

# 2. Can Functions Be Passed as Arguments?

Yes.

This is called a **Higher-Order Function**.

Example

```python
def greet():
    print("Hello")

def execute(func):
    func()

execute(greet)
```

Output

```
Hello
```

Notice

We wrote

```python
execute(greet)
```

NOT

```python
execute(greet())
```

---

## Why?

Because

```python
greet
```

means

```
Pass the function object.
```

Whereas

```python
greet()
```

means

```
Execute the function immediately
```

---

### Memory

```
execute()

↓

func

↓

Function Object (greet)
```

Calling

```python
func()
```

is equivalent to

```python
greet()
```

---

## Real Example

```python
def square(x):
    return x * x

numbers = [1, 2, 3]

result = list(map(square, numbers))

print(result)
```

Output

```
[1, 4, 9]
```

`map()` receives the function as an argument.

---

## Another Example

```python
def is_even(x):
    return x % 2 == 0

nums = [1,2,3,4]

print(list(filter(is_even, nums)))
```

Output

```
[2,4]
```

Again,

the function is passed.

---

# 3. Can Functions Be Returned?

Absolutely.

Example

```python
def outer():

    def inner():
        print("Hello")

    return inner

f = outer()

f()
```

Output

```
Hello
```

---

## Memory

```
outer()

↓

Creates inner()

↓

Returns Function Object

↓

f

↓

inner()
```

Nothing is executed until

```python
f()
```

---

## Another Example

```python
def power(n):

    def calculate(x):
        return x ** n

    return calculate

square = power(2)

cube = power(3)

print(square(5))
print(cube(5))
```

Output

```
25

125
```

Each returned function remembers its own closure.

---

# 4. Can Functions Be Stored in Data Structures?

Yes.

List

```python
def add(a,b):
    return a+b

def sub(a,b):
    return a-b

operations = [add, sub]

print(operations[0](10,5))
print(operations[1](10,5))
```

Output

```
15

5
```

---

Dictionary

```python
operations = {

    "+": add,

    "-": sub

}

print(operations["+"](10,20))
```

Output

```
30
```

---

This is widely used instead of long `if-elif` chains.

Instead of

```python
if op == "+":
    ...

elif op == "-":
    ...
```

Use

```python
operations[op](a,b)
```

Cleaner and extensible.

---

# 5. Functions Can Be Created Dynamically

```python
def choose(operation):

    if operation == "add":

        def add(a,b):
            return a+b

        return add

    else:

        def multiply(a,b):
            return a*b

        return multiply

f = choose("add")

print(f(3,4))
```

Output

```
7
```

Python created and returned a function based on runtime logic.

---

# Why Is This Important?

Many Python features depend on first-class functions.

## Decorators

```python
@cache
def predict():
    ...
```

A decorator **accepts a function** and **returns a new function**.

---

## Callbacks

```python
button.on_click(handle_click)
```

`handle_click` is passed as a function.

---

## FastAPI

```python
@app.get("/")
def home():
    ...
```

The decorator receives the `home` function object.

---

## LangChain

```python
chain = RunnableLambda(my_function)
```

A function is passed to build a chain.

---

## ThreadPoolExecutor

```python
executor.submit(train_model)
```

The executor stores the function and executes it later.

---

# Interview Questions

## Q1

```python
def hello():
    print("Hello")

x = hello

x()
```

Output

```
Hello
```

because `x` and `hello` reference the same function object.

---

## Q2

```python
def hello():
    print("Hello")

print(hello)
```

Output

```
<function hello at 0x...>
```

because you printed the **function object**, not its return value.

---

## Q3

```python
print(hello())
```

Output

```
Hello
None
```

Why?

1. `hello()` executes and prints `"Hello"`.
2. Since it has no `return`, it implicitly returns `None`.
3. `print()` then prints that `None`.

---

## Q4

```python
def execute(f):
    return f()

execute(print)
```

Output

```
```

`print()` is called with no arguments, so it prints a blank line and returns `None`.

---

# Summary

| Operation              | Example                    | Allowed? |
| ---------------------- | -------------------------- | -------- |
| Assign to a variable   | `f = greet`                | ✅        |
| Pass as an argument    | `execute(greet)`           | ✅        |
| Return from a function | `return greet`             | ✅        |
| Store in a list        | `[add, sub]`               | ✅        |
| Store in a dictionary  | `{"+": add}`               | ✅        |
| Create dynamically     | Nested function + `return` | ✅        |

## Senior AI Engineer Interview Answer

If asked **"What are first-class functions in Python?"**, a strong answer is:

> Python treats functions as first-class objects. A function is an object of type `function`, which means it can be assigned to variables, passed as arguments, returned from other functions, stored in data structures, and created dynamically. This capability enables higher-order functions, decorators, callbacks, closures, and many frameworks such as FastAPI, LangChain, and asyncio.
