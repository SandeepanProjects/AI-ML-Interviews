# Multiprocessing Explained Like a Senior AI Engineer

**Multiprocessing** is one of the most important Python concepts for **Senior AI Engineer**, **ML Infrastructure Engineer**, **LLM Platform Engineer**, and **Backend AI Engineer** interviews.

Interviewers typically ask:

* What is multiprocessing?
* Why do we need it if we already have threads?
* How does it avoid the GIL?
* What happens internally when a process is created?
* How do processes communicate?
* What is shared memory?
* What are `Process`, `Pool`, `Queue`, `Pipe`, and `Manager`?
* How is multiprocessing used in AI systems?

We'll build the concept from the operating system level.

---

# Why Multiprocessing Exists

Suppose you have a CPU-intensive task:

```python
def train():
    for _ in range(1_000_000_000):
        pass
```

You have a machine with:

```text
CPU

Core1
Core2
Core3
Core4
```

You create four Python threads expecting all four cores to work.

```text
Thread1
Thread2
Thread3
Thread4
```

But because of the **Global Interpreter Lock (GIL)** in CPython:

```text
Core1 → Running Thread1

Core2 → Idle

Core3 → Idle

Core4 → Idle
```

Only one thread executes Python bytecode at a time.

This is where **multiprocessing** comes in.

---

# What is Multiprocessing?

Instead of creating multiple **threads**, multiprocessing creates multiple **processes**.

Each process has:

* its own Python interpreter
* its own memory
* its own GIL
* its own address space

```text
+----------------------+
| Process 1            |
| Python Interpreter   |
| GIL                  |
+----------------------+

+----------------------+
| Process 2            |
| Python Interpreter   |
| GIL                  |
+----------------------+

+----------------------+
| Process 3            |
| Python Interpreter   |
| GIL                  |
+----------------------+
```

Now every process can execute Python code independently.

---

# Process vs Thread

## Threads

```text
Python Process

+--------------------------------+

Heap

Globals

Objects

↓

Thread1

Thread2

Thread3

Shared Memory

+--------------------------------+
```

Everything is shared.

---

## Processes

```text
Process1

Heap

Globals

Objects

Memory
```

```text
Process2

Heap

Globals

Objects

Memory
```

Nothing is shared by default.

---

# First Multiprocessing Example

```python
from multiprocessing import Process
import time

def worker():
    print("Worker started")
    time.sleep(2)
    print("Worker finished")

if __name__ == "__main__":
    p = Process(target=worker)

    p.start()

    p.join()

    print("Main finished")
```

Output

```text
Worker started

(wait)

Worker finished

Main finished
```

---

# What Happens Internally?

When you execute:

```python
p.start()
```

the operating system performs something conceptually like:

```text
Python Process

↓

Create New Process

↓

Allocate New Memory

↓

Start New Python Interpreter

↓

Copy Program State

↓

Run worker()

↓

Exit
```

Unlike a thread, this is a completely separate process.

---

# Multiple Processes

```python
from multiprocessing import Process
import os

def worker():

    print(os.getpid())

if __name__ == "__main__":

    processes = []

    for _ in range(4):
        p = Process(target=worker)
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
```

Output

```text
5213
5214
5215
5216
```

Each process has its own PID (Process ID).

---

# Real Parallelism

Suppose:

```python
def compute():
    total = 0
    for i in range(200_000_000):
        total += i
```

Single process

```text
Core1

Compute

↓

Done
```

Four processes

```text
Core1 → Compute

Core2 → Compute

Core3 → Compute

Core4 → Compute
```

All CPU cores are utilized.

---

# Benchmark

```python
from multiprocessing import Process
import time

COUNT = 80_000_000

def work():
    x = 0
    for _ in range(COUNT):
        x += 1

if __name__ == "__main__":

    start = time.time()

    processes = []

    for _ in range(4):
        p = Process(target=work)
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print(time.time() - start)
```

Compared to running the work sequentially four times, multiprocessing can significantly reduce wall-clock time on a machine with multiple CPU cores (actual speedup depends on hardware and process overhead).

---

# Why Doesn't the GIL Matter?

Each process has:

```text
Interpreter

↓

Own GIL
```

So

```text
Process1

GIL
```

does not block

```text
Process2

GIL
```

because they are completely separate interpreters.

---

# Problem: Memory Isn't Shared

Suppose

```python
from multiprocessing import Process

counter = 0

def increment():
    global counter
    counter += 1

if __name__ == "__main__":

    p = Process(target=increment)

    p.start()

    p.join()

    print(counter)
```

Output

```text
0
```

People expect:

```text
1
```

Why?

Because the child process modified **its own copy** of `counter`.

The parent's memory remains unchanged.

---

# Queue

Processes communicate using IPC (Inter-Process Communication).

The simplest mechanism is a `Queue`.

```python
from multiprocessing import Process, Queue

def worker(q):

    q.put("Hello")

if __name__ == "__main__":

    q = Queue()

    p = Process(target=worker, args=(q,))

    p.start()

    print(q.get())

    p.join()
```

Output

```text
Hello
```

Internally

```text
Process1

↓

Queue

↓

Process2
```

The queue safely transfers data between processes.

---

# Pipe

A `Pipe` is a lightweight communication channel.

```python
from multiprocessing import Pipe, Process

def worker(conn):

    conn.send("Done")

if __name__ == "__main__":

    parent, child = Pipe()

    p = Process(target=worker, args=(child,))

    p.start()

    print(parent.recv())

    p.join()
```

Output

```text
Done
```

---

# Pool

Suppose you have:

```text
1000 Images
```

You don't want to manually create 1000 processes.

Use a process pool.

```python
from multiprocessing import Pool

def square(x):
    return x * x

if __name__ == "__main__":

    with Pool(processes=4) as pool:

        result = pool.map(square, range(10))

    print(result)
```

Internally

```text
Pool

↓

Worker1

Worker2

Worker3

Worker4
```

Tasks are distributed automatically.

---

# Shared Memory

Normally

```text
Process1

Array
```

```text
Process2

Different Array
```

To share memory:

```python
from multiprocessing import Value, Process

def worker(counter):

    counter.value += 1

if __name__ == "__main__":

    counter = Value('i', 0)

    p = Process(target=worker, args=(counter,))

    p.start()

    p.join()

    print(counter.value)
```

Output

```text
1
```

`Value` creates shared memory accessible by multiple processes.

---

# Shared Arrays

```python
from multiprocessing import Array, Process

def worker(arr):

    arr[0] = 100

if __name__ == "__main__":

    arr = Array('i', [1, 2, 3])

    p = Process(target=worker, args=(arr,))

    p.start()

    p.join()

    print(arr[:])
```

Output

```text
[100, 2, 3]
```

---

# Manager

A `Manager` provides higher-level shared objects.

```python
from multiprocessing import Manager, Process

def worker(data):

    data.append(10)

if __name__ == "__main__":

    with Manager() as manager:

        data = manager.list()

        p = Process(target=worker, args=(data,))

        p.start()

        p.join()

        print(list(data))
```

Output

```text
[10]
```

---

# Fork vs Spawn

Different operating systems start processes differently.

### Fork (common on Linux)

```text
Parent

↓

Copy Process

↓

Child
```

The child initially inherits the parent's memory using copy-on-write, making startup relatively fast.

---

### Spawn (default on Windows, also available elsewhere)

```text
New Python Interpreter

↓

Import Program

↓

Run Target Function
```

Startup is slower but cleaner because nothing is inherited implicitly.

---

# Why `if __name__ == "__main__"`?

Consider:

```python
p = Process(target=worker)
p.start()
```

On platforms using **spawn**, the child process imports the module again.

Without:

```python
if __name__ == "__main__":
```

the process creation code executes again in the child, potentially creating processes recursively.

Always protect multiprocessing entry points with this guard.

---

# Multiprocessing in AI

Imagine training a model.

Dataset

```text
10 Million Images
```

Sequential

```text
Load Image

↓

Resize

↓

Normalize

↓

Next Image
```

Very slow.

Using multiple worker processes:

```text
Worker1 → Images 1–2500

Worker2 → Images 2501–5000

Worker3 → Images 5001–7500

Worker4 → Images 7501–10000
```

All CPU cores participate in preprocessing.

This is exactly why frameworks like PyTorch allow:

```python
DataLoader(
    dataset,
    batch_size=32,
    num_workers=8
)
```

Each worker is a separate process that loads and preprocesses data in parallel.

---

# Threading vs Multiprocessing

| Threading                | Multiprocessing          |
| ------------------------ | ------------------------ |
| Same process             | Separate processes       |
| Shared memory            | Separate memory          |
| One GIL                  | One GIL per process      |
| Lightweight              | Heavier                  |
| Fast communication       | IPC required             |
| Great for I/O-bound work | Great for CPU-bound work |

---

# Asyncio vs Multiprocessing

| Asyncio                 | Multiprocessing    |
| ----------------------- | ------------------ |
| Single process          | Multiple processes |
| Single thread           | Multiple processes |
| Event loop              | OS scheduler       |
| I/O concurrency         | CPU parallelism    |
| No true CPU parallelism | True parallelism   |

---

# Production AI Use Cases

Multiprocessing is widely used for:

* CPU-heavy data preprocessing
* Image and video processing
* Batch feature engineering
* Parallel document parsing
* Large-scale embedding generation
* Parallel inference on CPU
* Offline training pipelines
* Data loading (for example, PyTorch `DataLoader` workers)

For GPU-based model training or inference, the heavy numerical computation typically happens in optimized native libraries or on the GPU, but multiprocessing is still often used for feeding data efficiently to the accelerator.

---

# Common Interview Questions

### Q1. What is multiprocessing?

Multiprocessing creates multiple independent Python processes, each with its own interpreter and GIL, enabling true parallel execution across CPU cores.

---

### Q2. Why is multiprocessing faster for CPU-bound tasks?

Because each process runs independently on a different CPU core, avoiding the single-interpreter limitation imposed by the GIL.

---

### Q3. Why don't processes share variables?

Each process has its own address space. Changes made in one process are not automatically visible to another unless you explicitly use shared memory or inter-process communication.

---

### Q4. How do processes communicate?

Common mechanisms include:

* `Queue`
* `Pipe`
* `Manager`
* Shared memory (`Value`, `Array`, or the `multiprocessing.shared_memory` module)

---

### Q5. When should you use multiprocessing?

Use multiprocessing for CPU-intensive work such as:

* numerical computation in pure Python
* image processing
* feature engineering
* data preprocessing
* parallel batch inference
* offline machine learning pipelines

Avoid it for workloads that are primarily waiting on network or disk I/O; in those cases, threading or `asyncio` is usually more appropriate.

---

# Senior AI Engineer Takeaway

| Scenario                                | Best Choice                                                   | Reason                                         |
| --------------------------------------- | ------------------------------------------------------------- | ---------------------------------------------- |
| External LLM API calls                  | `asyncio` or threading                                        | Mostly network I/O                             |
| Database and vector database queries    | `asyncio` or threading                                        | Efficient I/O concurrency                      |
| Image preprocessing                     | Multiprocessing                                               | CPU-intensive                                  |
| Feature engineering                     | Multiprocessing                                               | Uses multiple CPU cores                        |
| Large-scale embedding generation on CPU | Multiprocessing                                               | Parallel computation                           |
| Model training data loading             | Multiprocessing                                               | Keeps GPUs fed with data                       |
| GPU model inference                     | Native libraries + GPU (often with multiple worker processes) | Compute happens outside the Python interpreter |

## Key Insight

Think of the three concurrency models like this:

```text
Threading
─────────
One house
Many people
Shared furniture
Easy communication
Need rules (locks)
```

```text
Multiprocessing
───────────────
Many houses
One person per house
Nothing shared
Communication by messages
```

```text
Asyncio
───────
One person
Many tasks
Switches tasks whenever one is waiting
```

The most important interview takeaway is:

* **Threading** is best for **I/O-bound** work.
* **Multiprocessing** is best for **CPU-bound** work.
* **Asyncio** is best for handling **large numbers of concurrent I/O operations** with minimal overhead.
