## How do you avoid N+1 queries?

This is a **very common SQLAlchemy/FastAPI interview question**.

### First: What is the N+1 problem?

The **N+1 query problem** happens when you execute:

* **1 query** to fetch N parent records
* then **N additional queries** to fetch related data for each parent

Example:

```text
1 query → get 100 users

Then:

100 queries → get orders for each user

Total = 101 queries
```

Instead of:

```text
1 query → users
1 query → all required orders

Total = 2 queries
```

---

# 1. Example of N+1 in SQLAlchemy

Suppose we have:

```python
class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(
        primary_key=True
    )

    name: Mapped[str]


class Order(Base):
    __tablename__ = "orders"

    id: Mapped[int] = mapped_column(
        primary_key=True
    )

    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id")
    )

    amount: Mapped[float]
```

And relationship:

```python
class User(Base):

    ...

    orders: Mapped[list["Order"]] = relationship(
        back_populates="user"
    )
```

---

## Bad approach

You fetch users:

```python
result = await db.execute(
    select(User)
)

users = result.scalars().all()
```

Then:

```python
for user in users:
    print(user.orders)
```

Conceptually this can result in:

```text
SELECT * FROM users;

SELECT * FROM orders WHERE user_id = 1;
SELECT * FROM orders WHERE user_id = 2;
SELECT * FROM orders WHERE user_id = 3;
...
SELECT * FROM orders WHERE user_id = 100;
```

So:

```text
1 + 100 = 101 queries
```

That's the N+1 problem.

---

# 2. Solution: eager loading

SQLAlchemy provides eager-loading strategies.

The most common ones are:

```text
selectinload()
joinedload()
```

---

# 3. `selectinload()`

For one-to-many relationships, `selectinload()` is often a very good choice.

```python
from sqlalchemy import select
from sqlalchemy.orm import selectinload

result = await db.execute(
    select(User)
    .options(
        selectinload(User.orders)
    )
)

users = result.scalars().all()
```

Instead of:

```text
101 queries
```

you typically get:

```text
Query 1:
SELECT users ...

Query 2:
SELECT orders ...
WHERE orders.user_id IN (...)
```

So:

```text
2 queries
```

The database handles the relationship loading in batches.

---

# 4. What does `selectinload()` actually do?

Suppose we have:

```text
Users:

id
--
1
2
3
```

SQLAlchemy can effectively perform:

```sql
SELECT *
FROM users;
```

Then:

```sql
SELECT *
FROM orders
WHERE user_id IN (1, 2, 3);
```

Then SQLAlchemy associates those orders with the appropriate users.

Conceptually:

```text
Users
 ├── User 1 ── Orders
 ├── User 2 ── Orders
 └── User 3 ── Orders
```

without querying orders individually for every user.

---

# 5. `joinedload()`

Another strategy is `joinedload()`:

```python
from sqlalchemy.orm import joinedload

result = await db.execute(
    select(User)
    .options(
        joinedload(User.orders)
    )
)

users = result.unique().scalars().all()
```

This generally uses a SQL JOIN.

Conceptually:

```sql
SELECT ...
FROM users
LEFT OUTER JOIN orders
    ON users.id = orders.user_id;
```

So instead of:

```text
User query
+
N order queries
```

you get a joined result.

---

# 6. `selectinload()` vs `joinedload()`

This is a good senior-level question.

|                   | `selectinload()` | `joinedload()`                    |
| ----------------- | ---------------- | --------------------------------- |
| Strategy          | Separate SELECT  | SQL JOIN                          |
| Number of queries | Usually 2        | Usually 1                         |
| Good for          | Collections      | Many-to-one / small relationships |
| Large collections | Often better     | Can produce large result sets     |
| Row duplication   | Less             | Can be significant                |
| Common use        | One-to-many      | Many-to-one                       |

### Example

For:

```text
User → Orders
```

I'd commonly consider:

```python
selectinload(User.orders)
```

because a user can have many orders.

For:

```text
Order → User
```

where each order has one user:

```python
joinedload(Order.user)
```

can be appropriate.

There isn't a universal rule—you choose based on relationship cardinality and query shape.

---

# 7. Example with `selectinload`

Model:

```python
class User(Base):

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(
        primary_key=True
    )

    orders: Mapped[list["Order"]] = relationship(
        back_populates="user"
    )


class Order(Base):

    __tablename__ = "orders"

    id: Mapped[int] = mapped_column(
        primary_key=True
    )

    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id")
    )

    user: Mapped["User"] = relationship(
        back_populates="orders"
    )
```

Repository:

```python
class UserRepository:

    def __init__(
        self,
        db: AsyncSession,
    ):
        self.db = db

    async def get_users_with_orders(
        self,
    ) -> list[User]:

        result = await self.db.execute(
            select(User)
            .options(
                selectinload(User.orders)
            )
        )

        return list(
            result.scalars().all()
        )
```

Now:

```python
users = await repository.get_users_with_orders()

for user in users:
    for order in user.orders:
        print(order.amount)
```

doesn't need to issue one query per user.

---

# 8. Avoid lazy loading in async applications

This is particularly important with:

```text
FastAPI
+
AsyncSession
```

Don't rely on implicit lazy loading of relationships in async code.

Prefer explicitly loading the relationships you need:

```python
select(User).options(
    selectinload(User.orders)
)
```

This makes database access predictable.

---

# 9. Another solution: explicit JOIN

Sometimes you don't actually need ORM relationship objects.

You can use a JOIN:

```python
result = await db.execute(
    select(User, Order)
    .join(
        Order,
        Order.user_id == User.id
    )
)
```

This can be useful when you need a specific projection.

For example:

```text
User ID
User name
Order ID
Order amount
```

rather than fully materializing ORM graphs.

---

# 10. Don't fetch data you don't need

Avoid:

```python
select(User)
```

if you only need:

```text
id
name
```

You can do:

```python
result = await db.execute(
    select(
        User.id,
        User.name,
    )
)
```

This reduces:

* database work
* network transfer
* memory usage
* serialization overhead

---

# 11. Use pagination

Another way to avoid performance problems is to avoid loading thousands of objects.

Bad:

```python
result = await db.execute(
    select(User)
)

users = result.scalars().all()
```

Imagine:

```text
10 million users
```

Better:

```python
result = await db.execute(
    select(User)
    .offset(offset)
    .limit(limit)
)
```

For large datasets, **keyset/cursor pagination** is often preferable to very large offsets.

Example:

```python
select(User).where(
    User.id > last_seen_id
).order_by(
    User.id
).limit(100)
```

---

# 12. Be careful with nested relationships

Suppose:

```text
User
 └── Orders
      └── Products
```

You can load nested relationships:

```python
result = await db.execute(
    select(User)
    .options(
        selectinload(User.orders)
        .selectinload(Order.products)
    )
)
```

This avoids:

```text
User query
+
N order queries
+
N*M product queries
```

But don't automatically eager-load the entire object graph.

Load what the API actually needs.

---

# 13. N+1 can happen outside ORM relationships too

It isn't only about lazy-loaded relationships.

For example:

```python
users = await get_users()

for user in users:

    orders = await get_orders(
        user.id
    )
```

This is also N+1.

You need to batch it.

Instead of:

```text
get_orders(user1)
get_orders(user2)
get_orders(user3)
...
```

use something conceptually like:

```sql
SELECT *
FROM orders
WHERE user_id IN (...);
```

Then group the results in Python if necessary.

---

# 14. How I would handle it in a production application

For a production FastAPI application, I'd follow these principles:

### 1. Identify query patterns

Use SQL logging/APM/tracing to detect:

```text
1 query
+
100 repeated queries
```

### 2. Explicitly eager-load relationships

```python
selectinload()
```

or:

```python
joinedload()
```

depending on the relationship.

### 3. Fetch only required columns

```python
select(
    User.id,
    User.name,
)
```

### 4. Batch database operations

Instead of:

```python
for user in users:
    await repository.get_orders(user.id)
```

perform one batched query.

### 5. Paginate

Don't return thousands/millions of records unnecessarily.

### 6. Add proper indexes

For:

```sql
WHERE orders.user_id = ?
```

you generally want an index on:

```text
orders.user_id
```

### 7. Measure query count and latency

Don't optimize blindly.

---

# 15. Example: bad vs good

### ❌ Bad

```python
async def get_users(
    db: AsyncSession,
):

    result = await db.execute(
        select(User)
    )

    users = result.scalars().all()

    for user in users:

        # Can cause N additional queries
        print(user.orders)

    return users
```

---

### ✅ Good

```python
async def get_users(
    db: AsyncSession,
):

    result = await db.execute(
        select(User)
        .options(
            selectinload(User.orders)
        )
    )

    users = result.scalars().all()

    return users
```

---

# 16. Interview answer

If the interviewer asks:

> **"How do you avoid N+1 queries in SQLAlchemy?"**

A strong senior-level answer would be:

> **"I avoid N+1 queries by identifying implicit or explicit per-record database calls and replacing them with eager loading or batched queries. In SQLAlchemy, for collection relationships I commonly use `selectinload()`, while `joinedload()` can be useful for suitable many-to-one or small relationships. I also avoid loading unnecessary relationships, use pagination and projections, and monitor query counts and database latency in production."**

Then show:

```python
result = await db.execute(
    select(User)
    .options(
        selectinload(User.orders)
    )
)

users = result.scalars().all()
```

And explain:

```text
❌ N+1

1 user query
+
N order queries


✅ Eager loading

1 user query
+
1 batched order query
```

### The key sentence to remember

> **"The problem isn't simply having multiple queries; the problem is issuing a new query repeatedly inside a loop when the data could have been fetched in a batch."**

That distinction is especially important in a senior AI/backend interview because the same principle applies to **RAG pipelines, PostgreSQL repositories, document retrieval, and agent data access**: **batch I/O rather than performing sequential per-item calls whenever the operation can be safely batched.**
