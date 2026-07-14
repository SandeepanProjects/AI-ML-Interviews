# Multithreading in Python

This is a **very important Senior Python / AI Engineer interview topic**.

Interviewers expect you to know:

* What is a thread?
* Why use threads?
* Python GIL impact
* Race conditions
* Locks
* Semaphores
* Conditions
* Events
* Producer-consumer patterns
* When to use threading vs multiprocessing vs asyncio

---

# 1. What is a Thread?

A **thread** is a lightweight unit of execution inside a process.

A process can have multiple threads.

Example:

```
Process
 |
 |---- Thread 1
 |
 |---- Thread 2
 |
 |---- Thread 3
```

All threads share:

* Same memory
* Same variables
* Same resources

---

# Process vs Thread

| Process                | Thread                 |
| ---------------------- | ---------------------- |
| Independent memory     | Shared memory          |
| Heavyweight            | Lightweight            |
| Higher creation cost   | Lower creation cost    |
| Better CPU parallelism | Better I/O concurrency |
| Separate address space | Same address space     |

---

# Example: Creating Threads

Python uses:

```python
threading.Thread
```

Example:

```python
import threading
import time


def download():

    print("Downloading")

    time.sleep(2)

    print("Completed")


t1 = threading.Thread(target=download)

t1.start()

t1.join()

print("Finished")
```

Output:

```
Downloading
Completed
Finished
```

---

# What does `start()` do?

```python
t1.start()
```

Creates a new OS thread and executes:

```python
download()
```

---

# What does `join()` do?

```python
t1.join()
```

means:

> Wait until this thread completes.

Without:

```python
t1.start()

print("Finished")
```

Output could be:

```
Downloading
Finished
Completed
```

because main thread does not wait.

---

# Why Use Multithreading?

Threads are useful for:

## I/O-bound tasks

Examples:

* API calls
* Database queries
* File operations
* Network requests

Example:

```
Request 1 → API call
Request 2 → API call
Request 3 → API call
```

Instead of:

```
API1 (2 sec)
API2 (2 sec)
API3 (2 sec)

Total = 6 sec
```

Threads:

```
API1
API2
API3

Total ≈ 2 sec
```

---

# Python GIL (Important)

Python has:

## Global Interpreter Lock

The GIL allows only one thread to execute Python bytecode at a time.

Example:

```python
counter += 1
```

internally:

```
READ counter

ADD 1

WRITE counter
```

Two threads can interfere.

---

# Does GIL make threading useless?

No.

Because during I/O:

```
Thread 1
 |
 waiting for API
 |
 releases GIL


Thread 2
 |
 executes
```

Threads are excellent for I/O workloads.

---

# Race Condition

Example:

```python
import threading


counter = 0


def increment():

    global counter

    for _ in range(100000):
        counter += 1


t1 = threading.Thread(target=increment)
t2 = threading.Thread(target=increment)


t1.start()
t2.start()

t1.join()
t2.join()


print(counter)
```

Expected:

```
200000
```

But sometimes:

```
154332
```

Why?

Both threads modify shared data.

---

# Solution: Lock

# 2. Lock

A Lock provides mutual exclusion.

Meaning:

> Only one thread can enter the critical section at a time.

---

Example:

```python
import threading


counter = 0

lock = threading.Lock()


def increment():

    global counter

    for _ in range(100000):

        with lock:

            counter += 1


t1 = threading.Thread(target=increment)
t2 = threading.Thread(target=increment)


t1.start()
t2.start()

t1.join()
t2.join()


print(counter)
```

Output:

```
200000
```

---

# How Lock Works Internally

Without lock:

```
Thread A
READ counter=10

Thread B
READ counter=10

Thread A
WRITE 11

Thread B
WRITE 11
```

Lost update.

---

With lock:

```
Thread A
 |
Acquire Lock
 |
Update counter
 |
Release Lock


Thread B
 |
Acquire Lock
 |
Update counter
```

---

# Lock States

```
LOCKED

or

UNLOCKED
```

Example:

```python
lock.acquire()

try:
    critical_section()

finally:
    lock.release()
```

Better:

```python
with lock:
    critical_section()
```

---

# 3. Semaphore

A Semaphore allows a limited number of threads to access a resource.

Lock:

```
Only 1 thread
```

Semaphore:

```
Allow N threads
```

---

Example:

Suppose:

Only 3 API requests allowed simultaneously.

```python
import threading


semaphore = threading.Semaphore(3)


def call_api():

    with semaphore:

        print("Calling API")

        time.sleep(2)
```

Now:

```
Thread 1 ✓
Thread 2 ✓
Thread 3 ✓

Thread 4 waits
Thread 5 waits
```

---

# Semaphore Internal

It maintains a counter:

```
Semaphore(3)

Available slots:

3

Thread enters

2

Thread enters

1

Thread enters

0

Next thread blocks
```

When a thread exits:

```
0 → 1
```

Another thread can enter.

---

# Lock vs Semaphore

| Lock                | Semaphore           |
| ------------------- | ------------------- |
| Allows one thread   | Allows N threads    |
| Mutual exclusion    | Resource limiting   |
| Binary (0/1)        | Counter-based       |
| Protect shared data | Control concurrency |

---

# 4. Event

An Event is a signaling mechanism.

Think:

> One thread tells another thread something happened.

It has two states:

```
False
True
```

---

Example:

Worker waits for signal:

```python
import threading
import time


event = threading.Event()


def worker():

    print("Waiting")

    event.wait()

    print("Started work")


thread = threading.Thread(
    target=worker
)

thread.start()


time.sleep(3)

event.set()
```

Output:

```
Waiting

(after 3 seconds)

Started work
```

---

# Event Methods

## set()

Set flag:

```python
event.set()
```

State:

```
False → True
```

---

## wait()

Block until set:

```python
event.wait()
```

---

## clear()

Reset:

```python
event.clear()
```

State:

```
True → False
```

---

# Real Example

AI model loading:

Thread 1:

```
Load 20GB model
```

Thread 2:

```
Wait until model ready
```

Using Event:

```
Model Loader

     |
     |
event.set()

     ↓

Inference Thread starts
```

---

# 5. Condition

Condition is used when threads need to wait for a specific state.

It combines:

* Lock
* Notification mechanism

Common use:

Producer-Consumer.

---

Example:

A queue.

Producer:

```
Produces data
```

Consumer:

```
Consumes data
```

Consumer should wait if queue is empty.

---

Implementation:

```python
import threading


queue = []

condition = threading.Condition()


def consumer():

    with condition:

        while len(queue)==0:

            condition.wait()

        item = queue.pop(0)

        print("Consumed",item)



def producer():

    with condition:

        queue.append("data")

        condition.notify()
```

---

# Condition Flow

Consumer:

```
Acquire lock

Queue empty?

YES

wait()
 |
 releases lock
 |
 sleeps
```

Producer:

```
Acquire lock

Add item

notify()

```

Consumer:

```
Wake up

Acquire lock

Consume item
```

---

# Condition vs Event

| Event                | Condition                  |
| -------------------- | -------------------------- |
| Simple signal        | Complex synchronization    |
| Boolean flag         | Wait for state change      |
| One-way notification | Producer-consumer patterns |
| No lock required     | Uses lock                  |

---

# Real AI Production Examples

## 1. Thread Pool for API Calls

Example:

```
1000 user requests

        |
        v

Thread Pool

Thread 1 → OpenAI API
Thread 2 → Database
Thread 3 → Vector DB
```

---

## 2. Lock

Protect:

```
Shared Cache

Redis client

Local model state
```

Example:

```python
with cache_lock:
    update_cache()
```

---

## 3. Semaphore

Limit:

```
Maximum concurrent LLM calls
```

Example:

```
Semaphore(10)

Only 10 requests hit GPU
```

---

## 4. Event

Model initialization:

```
Application starts

       |
Load embedding model

       |
event.set()

       |
Accept requests
```

---

## 5. Condition

Batch processing:

```
Workers generate embeddings

↓

Batch size reaches 100

↓

Notify GPU worker

↓

Process batch
```

---

# Interview Questions

## Q1. What is a thread?

A thread is a lightweight execution unit inside a process. Threads share memory and are useful for concurrent execution, especially I/O-bound workloads.

---

## Q2. Why use Lock?

A lock prevents race conditions by ensuring only one thread modifies shared state at a time.

---

## Q3. Lock vs Semaphore?

Lock:

> One thread at a time.

Semaphore:

> Allows a fixed number of threads simultaneously.

---

## Q4. Event vs Condition?

Event:

> Simple notification mechanism using a boolean flag.

Condition:

> Allows threads to wait until a particular condition/state becomes true.

---

## Q5. Threading vs Multiprocessing?

| Threading      | Multiprocessing |
| -------------- | --------------- |
| Shared memory  | Separate memory |
| Good for I/O   | Good for CPU    |
| Limited by GIL | Bypasses GIL    |
| Lightweight    | Heavyweight     |

---

# Senior AI Engineer Answer

> In production AI systems, threading is primarily used for I/O concurrency such as parallel API calls, database operations, and network requests. Locks protect shared resources like caches and model state. Semaphores control concurrency limits, such as limiting simultaneous LLM requests or GPU jobs. Events coordinate lifecycle signals such as model readiness, while Conditions synchronize complex workflows like producer-consumer pipelines and batch processing.

# Producer–Consumer, Race Condition, Deadlock in Python

These are **very common Senior Python / AI Engineer interview topics** because they test whether you understand **concurrency, synchronization, shared state, and production system reliability**.

They are heavily used in:

* Data pipelines
* ML training pipelines
* LLM inference systems
* Message queues
* Background workers
* Distributed systems

---

# 1. Producer–Consumer Pattern

## Concept

Producer–Consumer is a concurrency design pattern where:

* **Producer** creates data
* **Consumer** processes data

They communicate through a shared queue.

Architecture:

```
              Producer

                 |
                 |
                 v

              Queue

                 |
                 |
                 v

              Consumer
```

Example:

AI pipeline:

```
Documents
    |
    |
 Document Loader (Producer)
    |
    v
 Queue
    |
    v
 Embedding Workers (Consumers)
    |
    v
 Vector Database
```

---

# Why Not Direct Communication?

Bad design:

```python
producer()
consumer()
```

Problem:

* Producer may be faster than consumer
* Consumer may be faster than producer
* Need buffering

Queue solves this.

---

# Python Implementation Using Queue

Python provides:

```python
queue.Queue
```

which is thread-safe.

---

## Producer Consumer Example

```python
import threading
import queue
import time


buffer = queue.Queue(maxsize=5)


def producer():

    for i in range(10):

        print("Producing", i)

        buffer.put(i)

        time.sleep(0.5)


    buffer.put(None)



def consumer():

    while True:

        item = buffer.get()

        if item is None:
            break

        print("Consuming", item)

        time.sleep(1)



producer_thread = threading.Thread(
    target=producer
)


consumer_thread = threading.Thread(
    target=consumer
)


producer_thread.start()

consumer_thread.start()


producer_thread.join()

consumer_thread.join()
```

---

Output:

```
Producing 0
Consuming 0

Producing 1
Producing 2

Consuming 1

Producing 3

Consuming 2
...
```

---

# Internal Flow

## Producer

```python
buffer.put(item)
```

Adds item:

```
Queue:

[1,2,3]
```

---

## Consumer

```python
buffer.get()
```

Removes item:

```
Queue:

[2,3]
```

---

# Why maxsize?

Example:

```python
queue.Queue(maxsize=5)
```

means:

```
Queue capacity = 5
```

If producer creates faster:

```
Producer

1
2
3
4
5
6

BLOCK
```

Producer waits until consumer removes items.

This is called **backpressure**.

---

# Real AI Example

Document ingestion:

```
                 Producer

        PDF Loader Thread

                 |
                 |
                 v

              Queue

                 |
       -----------------
       |       |       |
       v       v       v

   Embed     Embed   Embed
   Worker    Worker  Worker

                 |
                 v

              Vector DB
```

Benefits:

* Faster ingestion
* Controlled memory
* Parallel processing

---

# 2. Race Condition

## Definition

A race condition happens when:

> Multiple threads access shared data at the same time and the final result depends on execution order.

Example:

```python
counter = 0
```

Two threads:

```
Thread A
Thread B
```

Both update:

```
counter += 1
```

---

# Why Is `counter += 1` Dangerous?

It looks atomic:

```python
counter += 1
```

But internally:

```
READ counter

ADD 1

WRITE counter
```

---

Example:

Initial:

```
counter = 10
```

Thread A:

```
READ 10
```

Thread B:

```
READ 10
```

Thread A:

```
WRITE 11
```

Thread B:

```
WRITE 11
```

Final:

```
counter = 11
```

Expected:

```
12
```

One update is lost.

---

# Race Condition Example

```python
import threading


counter = 0


def increment():

    global counter

    for _ in range(100000):

        counter += 1



t1 = threading.Thread(target=increment)

t2 = threading.Thread(target=increment)


t1.start()
t2.start()


t1.join()
t2.join()


print(counter)
```

Expected:

```
200000
```

But may get:

```
145321
```

---

# Fix Race Condition Using Lock

```python
import threading


counter = 0

lock = threading.Lock()


def increment():

    global counter


    for _ in range(100000):

        with lock:

            counter += 1



t1 = threading.Thread(target=increment)

t2 = threading.Thread(target=increment)


t1.start()

t2.start()


t1.join()

t2.join()


print(counter)
```

Output:

```
200000
```

---

# Lock Flow

Without lock:

```
Thread A
   |
   update

Thread B
   |
   update

Collision
```

With lock:

```
Thread A

Acquire Lock

Update

Release


Thread B

Acquire Lock

Update

Release
```

---

# 3. Deadlock

## Definition

Deadlock happens when:

> Two or more threads wait forever for each other to release resources.

Example:

```
Thread A owns Lock 1
waiting for Lock 2


Thread B owns Lock 2
waiting for Lock 1
```

Nobody moves.

---

# Deadlock Example

```python
import threading
import time


lock1 = threading.Lock()

lock2 = threading.Lock()



def task1():

    with lock1:

        print("Task1 got lock1")

        time.sleep(1)

        with lock2:

            print("Task1 got lock2")



def task2():

    with lock2:

        print("Task2 got lock2")

        time.sleep(1)

        with lock1:

            print("Task2 got lock1")



t1 = threading.Thread(target=task1)

t2 = threading.Thread(target=task2)


t1.start()

t2.start()


t1.join()

t2.join()
```

---

Execution:

Thread 1:

```
Acquire lock1
```

Thread 2:

```
Acquire lock2
```

Now:

Thread 1:

```
Waiting for lock2
```

Thread 2:

```
Waiting for lock1
```

Both wait forever.

```
       lock1
         ^
         |
Thread A | Thread B
         |
         v
       lock2
```

---

# How to Prevent Deadlock?

## 1. Always Acquire Locks in Same Order

Bad:

Thread A:

```
lock1
lock2
```

Thread B:

```
lock2
lock1
```

---

Good:

Both:

```
lock1
lock2
```

---

Example:

```python
def task():

    with lock1:

        with lock2:

            work()
```

---

# 2. Use Timeout

Instead of waiting forever:

```python
lock.acquire(timeout=5)
```

Example:

```python
if lock.acquire(timeout=5):

    try:
        process()

    finally:
        lock.release()

else:

    print("Could not acquire lock")
```

---

# 3. Use Higher-Level Abstractions

Prefer:

```python
queue.Queue
```

instead of manually managing:

```python
lock
condition
```

because Queue already handles synchronization.

---

# Producer Consumer with Condition (Interview Version)

Sometimes interviewer asks:

"Implement producer consumer without Queue."

Example:

```python
import threading


buffer = []

condition = threading.Condition()


def producer():

    for i in range(5):

        with condition:

            buffer.append(i)

            print("Produced",i)

            condition.notify()


def consumer():

    for i in range(5):

        with condition:

            while not buffer:

                condition.wait()


            item = buffer.pop(0)

            print("Consumed",item)



t1 = threading.Thread(target=producer)

t2 = threading.Thread(target=consumer)


t1.start()

t2.start()


t1.join()

t2.join()
```

---

# Comparison

| Concept           | Problem Solved                | Tool                   |
| ----------------- | ----------------------------- | ---------------------- |
| Producer Consumer | Data exchange between threads | Queue                  |
| Race Condition    | Incorrect shared data updates | Lock                   |
| Deadlock          | Threads waiting forever       | Lock ordering, timeout |

---

# Senior AI Engineer Production Examples

## 1. LLM Request Queue

```
API Requests

      |
      v

Producer

      |
      v

Queue

      |
      v

GPU Workers
```

Queue controls load.

---

## 2. Embedding Pipeline

```
PDF Loader

Producer

      |
      v

Document Queue

      |
      v

Embedding Workers

      |
      v

Vector DB
```

---

## 3. Shared Model Cache

Multiple threads:

```python
cache["embedding_model"]
```

Need:

```python
with lock:
    update_cache()
```

to avoid race conditions.

---

## 4. GPU Resource Control

Use Semaphore:

```python
gpu_slots = Semaphore(4)
```

Only 4 inference jobs run simultaneously.

---

# Interview Answers

## Q: What is a race condition?

> A race condition occurs when multiple threads access shared mutable state concurrently and the output depends on timing. Locks or other synchronization mechanisms are used to protect critical sections.

---

## Q: What is deadlock?

> Deadlock occurs when threads permanently wait for resources held by each other. It can be prevented using consistent lock ordering, timeouts, avoiding nested locks, and higher-level concurrency primitives.

---

## Q: Why use Queue instead of list in producer-consumer?

> `queue.Queue` is thread-safe and internally handles locking, blocking, and synchronization, making it safer than manually protecting a shared list.

---

## Senior-level summary:

```
Producer Consumer
        |
        v
 Queue + Workers


Race Condition
        |
        v
 Shared State Problem
        |
        v
 Lock


Deadlock
        |
        v
 Bad Lock Ordering
        |
        v
 Timeout / Ordering / Avoid Nested Locks
```

For production AI systems, these three concepts are fundamental for building **high-throughput, reliable inference pipelines and distributed ML workloads**.
