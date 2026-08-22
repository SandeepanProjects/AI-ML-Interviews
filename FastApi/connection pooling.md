These three are important because they explain how **FastAPI + SQLAlchemy + PostgreSQL** behaves in production.

The easiest way to remember them is:

```text
FastAPI Request
      ↓
AsyncSession
      ↓
Transaction
      ↓
Connection Pool
      ↓
PostgreSQL
```

---

# 1. What is connection pooling?

**Connection pooling is the practice of keeping a reusable set of database connections instead of creating a new database connection for every request.**

Creating a database connection has overhead:

```text
Request
  ↓
Create TCP connection
  ↓
Authenticate
  ↓
Establish PostgreSQL connection
  ↓
Execute query
  ↓
Close connection
```

Doing that for every request is expensive.

Instead, SQLAlchemy maintains a **connection pool**:

```text
                    SQLAlchemy Engine
                           │
                    Connection Pool
                           │
             ┌─────────────┼─────────────┐
             ↓             ↓             ↓
        Connection 1   Connection 2   Connection 3
             │             │             │
             └─────────────┼─────────────┘
                           ↓
                       PostgreSQL
```

When a request needs a connection:

```text
Request
   ↓
AsyncSession
   ↓
Get connection from pool
   ↓
Execute query
   ↓
Return connection to pool
```

The connection is generally **returned to the pool**, not destroyed.

---

## SQLAlchemy connection pool example

```python
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

engine = create_async_engine(
    DATABASE_URL,

    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
)

SessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)
```

### What does `pool_size=10` mean?

Roughly:

```text
10 connections maintained in the pool
```

### What does `max_overflow=20` mean?

SQLAlchemy can temporarily create additional connections beyond the pool size when demand increases, subject to the pool configuration and database limits.

So conceptually:

```text
Normal:
10 pooled connections

High traffic:
10 pooled
+
up to 20 overflow
```

Don't blindly choose these values. In production, you tune them based on:

* PostgreSQL connection limits
* number of application workers/pods
* request concurrency
* query latency
* workload

For example, if you deploy:

```text
10 Kubernetes pods
```

and each pod has:

```text
pool_size = 20
```

you could potentially have a much larger aggregate database connection footprint than just 20.

That's an important **senior-level consideration**.

---

# Why is connection pooling important?

### Without pooling

```text
Request 1 → create connection → close
Request 2 → create connection → close
Request 3 → create connection → close
Request 4 → create connection → close
```

### With pooling

```text
             Connection Pool
            /      |       \
           /       |        \
Request 1 → C1     |         |
Request 2 ─────────→ C2      |
Request 3 ─────────────────→ C3
Request 4 → C1
```

Benefits:

* lower connection setup overhead
* better throughput
* controlled database concurrency
* less pressure on PostgreSQL
* better resource utilization

---

# 2. How do transactions work?

A **transaction is a logical unit of database work that should either succeed completely or be rolled back.**

For example, imagine a bank transfer:

```text
Account A: -₹1,000
Account B: +₹1,000
```

You don't want this:

```text
A → -₹1,000
B → ERROR
```

Instead:

```text
BEGIN
   ↓
Debit A
   ↓
Credit B
   ↓
COMMIT
```

If something fails:

```text
BEGIN
   ↓
Debit A
   ↓
Credit B → ERROR
   ↓
ROLLBACK
```

The changes made during the transaction are undone.

---

# ACID

Transactions are generally discussed using **ACID**:

```text
A → Atomicity
C → Consistency
I → Isolation
D → Durability
```

### Atomicity

All operations succeed or none do.

```text
Debit A ✓
Credit B ✓
     ↓
COMMIT
```

or:

```text
Debit A ✓
Credit B ✗
     ↓
ROLLBACK
```

---

### Consistency

The transaction should leave the database in a valid state according to its constraints and rules.

For example:

```text
balance >= 0
```

if your application/database enforces that rule.

---

### Isolation

Concurrent transactions shouldn't incorrectly interfere with each other.

For example:

```text
Transaction A
Transaction B
```

can run concurrently while PostgreSQL's isolation mechanisms control what each transaction can see.

---

### Durability

Once PostgreSQL successfully commits a transaction, the committed data should survive subsequent failures according to the database's durability guarantees.

---

# 3. How do you commit and rollback?

With SQLAlchemy `AsyncSession`:

### Commit

```python
await db.commit()
```

### Rollback

```python
await db.rollback()
```

---

# Basic example

Suppose we create a user:

```python
async def create_user(
    db: AsyncSession,
    email: str,
    name: str,
):

    user = User(
        email=email,
        name=name,
    )

    db.add(user)

    await db.commit()

    await db.refresh(user)

    return user
```

The flow is approximately:

```text
db.add(user)
     ↓
Session tracks object
     ↓
await db.commit()
     ↓
SQLAlchemy flushes pending changes
     ↓
INSERT executed
     ↓
COMMIT
```

---

# What does `db.add()` actually do?

This:

```python
db.add(user)
```

doesn't necessarily immediately execute:

```sql
INSERT INTO users ...
```

Instead, SQLAlchemy puts the object into the session's unit of work.

When you eventually flush/commit:

```python
await db.commit()
```

SQLAlchemy sends the appropriate SQL to PostgreSQL.

Conceptually:

```text
db.add(user)
     ↓
Pending object
     ↓
flush
     ↓
INSERT
     ↓
COMMIT
```

---

# What happens if something fails?

Consider:

```python
async def create_user(
    db: AsyncSession,
):

    try:

        user = User(
            email="test@example.com",
            name="Test",
        )

        db.add(user)

        await db.commit()

        return user

    except Exception:

        await db.rollback()

        raise
```

If the database operation fails:

```text
db.add()
   ↓
commit()
   ↓
Database error
   ↓
except
   ↓
rollback()
   ↓
raise
```

The important line is:

```python
await db.rollback()
```

---

# Why is rollback important?

Suppose a transaction fails:

```text
BEGIN
 ↓
INSERT user
 ↓
UPDATE account
 ↓
ERROR
```

The session may now be in a failed transaction state.

You should rollback:

```python
await db.rollback()
```

Then the session can be used appropriately for subsequent work.

---

# A better transaction pattern

For multiple related operations, use a transaction block:

```python
async with db.begin():

    db.add(user)

    db.add(profile)

    db.add(settings)
```

Conceptually:

```text
async with db.begin()
        ↓
BEGIN
        ↓
Add user
        ↓
Add profile
        ↓
Add settings
        ↓
Everything succeeds?
    /           \
  YES            NO
   ↓              ↓
COMMIT         ROLLBACK
```

This is often cleaner when you want the entire block to be atomic.

---

# Example: creating multiple records

Suppose your application creates:

```text
User
Profile
OrganizationMembership
```

These should all succeed together.

```python
async def create_account(
    db: AsyncSession,
):

    async with db.begin():

        user = User(
            email="user@example.com",
            name="John",
        )

        db.add(user)

        profile = Profile(
            name="John",
        )

        db.add(profile)

        membership = Membership(
            role="member",
        )

        db.add(membership)

    return user
```

If everything succeeds:

```text
User       ✓
Profile    ✓
Membership ✓
     ↓
COMMIT
```

If something fails:

```text
User       ✓
Profile    ✓
Membership ✗
     ↓
ROLLBACK
```

The transaction gives you atomicity.

---

# `commit()` vs `flush()`

This is a common senior-level interview question.

### `flush()`

```python
await db.flush()
```

sends pending changes to the database **within the current transaction**, but does not commit the transaction.

For example:

```python
user = User(
    email="test@example.com"
)

db.add(user)

await db.flush()

print(user.id)
```

You may need `flush()` when you want the database-generated ID before committing.

The transaction is still active.

```text
add()
 ↓
flush()
 ↓
INSERT
 ↓
transaction still open
 ↓
commit()
```

---

### `commit()`

```python
await db.commit()
```

ends the transaction successfully.

```text
BEGIN
 ↓
INSERT
 ↓
UPDATE
 ↓
COMMIT
```

---

# `rollback()` vs `commit()`

Easy way to remember:

```text
commit()
   ↓
"Make my transaction permanent."

rollback()
   ↓
"Undo the current transaction."
```

---

# Example with business logic

Imagine transferring money:

```python
async def transfer_money(
    db: AsyncSession,
    from_account: int,
    to_account: int,
    amount: int,
):

    try:

        source = await get_account(
            db,
            from_account,
        )

        destination = await get_account(
            db,
            to_account,
        )

        if source.balance < amount:
            raise ValueError(
                "Insufficient balance"
            )

        source.balance -= amount

        destination.balance += amount

        await db.commit()

    except Exception:

        await db.rollback()

        raise
```

Flow:

```text
                 BEGIN
                   │
                   ↓
          Read source account
                   │
                   ↓
         Read destination account
                   │
                   ↓
           Debit source
                   │
                   ↓
          Credit destination
                   │
             ┌─────┴─────┐
             ↓           ↓
          Success       Error
             ↓           ↓
          COMMIT      ROLLBACK
```

---

# Important: transaction ≠ connection pool

These are often confused.

## Connection pool

Answers:

> **"How do I efficiently manage database connections?"**

```text
Engine
 ↓
Connection Pool
 ↓
Connections
```

## Transaction

Answers:

> **"Which database operations should succeed or fail together?"**

```text
BEGIN
 ↓
Operation 1
 ↓
Operation 2
 ↓
COMMIT / ROLLBACK
```

They solve different problems.

---

# How they work together

This is the important architecture:

```text
                    FastAPI
                       │
                       ↓
                  AsyncSession
                       │
                       ↓
                  Transaction
                       │
                       ↓
               SQLAlchemy Engine
                       │
                       ↓
                Connection Pool
                       │
                       ↓
                  PostgreSQL
```

More accurately, the session acquires a connection from the engine's pool when it needs to interact with the database.

---

# Production FastAPI example

A clean setup:

```python
# db/session.py

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

engine = create_async_engine(
    DATABASE_URL,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
)

SessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)
```

Dependency:

```python
# db/dependencies.py

async def get_db():
    async with SessionLocal() as db:
        yield db
```

Endpoint:

```python
@router.post("/users")
async def create_user(
    request: UserCreate,
    db: AsyncSession = Depends(get_db),
):

    try:

        user = User(
            email=request.email,
            name=request.name,
        )

        db.add(user)

        await db.commit()

        await db.refresh(user)

        return user

    except Exception:

        await db.rollback()

        raise
```

---

# Even better: transaction boundary in the service layer

In a larger application, I don't want every router to contain complicated transaction logic.

I'd generally structure:

```text
Router
   ↓
Service
   ↓
Repository
   ↓
AsyncSession
```

For example:

```python
class UserService:

    def __init__(
        self,
        db: AsyncSession,
    ):
        self.db = db

    async def create_user(
        self,
        email: str,
        name: str,
    ):

        async with self.db.begin():

            user = User(
                email=email,
                name=name,
            )

            self.db.add(user)

        return user
```

The service owns the business transaction.

---

# What happens when an exception occurs inside `begin()`?

With:

```python
async with db.begin():

    db.add(user)
    db.add(profile)
```

if an exception occurs:

```text
enter transaction
       ↓
add user
       ↓
add profile
       ↓
Exception
       ↓
automatic rollback
```

If everything succeeds:

```text
enter transaction
       ↓
add user
       ↓
add profile
       ↓
automatic commit
```

That's one reason the context-manager pattern is useful.

---

# Senior interview answer

If the interviewer asks:

### 1. "What is connection pooling?"

Say:

> **"Connection pooling maintains a reusable set of database connections. Instead of establishing a new PostgreSQL connection for every request, SQLAlchemy acquires a connection from the pool when needed and returns it to the pool afterward. This reduces connection overhead and prevents uncontrolled database connection creation."**

---

### 2. "How do transactions work?"

Say:

> **"A transaction groups multiple database operations into one logical unit of work. Either all operations are committed or, if something fails, the transaction is rolled back. This provides atomicity and helps maintain database consistency."**

Example:

```python
async with db.begin():
    db.add(user)
    db.add(profile)
```

---

### 3. "How do you commit and rollback?"

Say:

> **"With SQLAlchemy AsyncSession, I use `await db.commit()` to commit a successful transaction and `await db.rollback()` when an operation fails. For structured transaction boundaries, I prefer `async with db.begin()`, which automatically commits on success and rolls back when an exception occurs."**

```python
try:
    async with db.begin():
        db.add(user)
        db.add(profile)

except Exception:
    raise
```

Or explicitly:

```python
try:
    db.add(user)
    await db.commit()

except Exception:
    await db.rollback()
    raise
```

---

## The 3 things to remember

```text
CONNECTION POOL
"Reuse database connections."

TRANSACTION
"Group database operations atomically."

COMMIT / ROLLBACK
"Commit = success.
 Rollback = undo failed transaction."
```

And in a production FastAPI application:

```text
Request
   ↓
Depends(get_db)
   ↓
AsyncSession
   ↓
Transaction
   ↓
Connection from pool
   ↓
PostgreSQL
   ↓
COMMIT / ROLLBACK
   ↓
Connection returned to pool
   ↓
Session closed
```

This distinction—**session lifecycle, connection pooling, and transaction boundaries are separate concepts**—is exactly what makes the answer strong at a senior-level interview.
