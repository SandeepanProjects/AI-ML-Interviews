For a **FastAPI + SQLAlchemy + PostgreSQL production application**, **Alembic** is the standard tool you use to manage database schema changes safely.

The important interview distinction is:

```text
SQLAlchemy
    ↓
Defines your database models

Alembic
    ↓
Changes the actual database schema
```

---

# 1. What is Alembic?

**Alembic is a database migration tool for SQLAlchemy.**

Suppose you initially have:

```python
class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(
        primary_key=True
    )

    email: Mapped[str] = mapped_column(
        String(255)
    )
```

Your PostgreSQL table is:

```text
users
----------------
id
email
```

Later, you add:

```python
name: Mapped[str] = mapped_column(
    String(100)
)
```

Now your Python model expects:

```text
users
----------------
id
email
name
```

But PostgreSQL still has:

```text
users
----------------
id
email
```

You need a **schema migration**.

Alembic generates/applies that change:

```text
Old schema
   ↓
Migration
   ↓
New schema
```

---

# 2. Why do we need migrations?

Imagine you have a production database with:

```text
10 million users
```

You modify your SQLAlchemy model.

Simply changing:

```python
class User(Base):
    ...
```

does **not** automatically modify the production database.

You need a controlled schema change.

For example:

```sql
ALTER TABLE users
ADD COLUMN name VARCHAR(100);
```

Alembic allows you to version and execute that change safely.

---

# 3. Think of migrations like Git

This is the easiest way to remember it.

Git manages:

```text
Code versions
```

Alembic manages:

```text
Database schema versions
```

For example:

```text
Git

commit A
   ↓
commit B
   ↓
commit C
```

Alembic:

```text
Migration 001
      ↓
Migration 002
      ↓
Migration 003
```

Your database records which migration revision it is currently at.

---

# 4. Installing Alembic

For an async SQLAlchemy application:

```bash
pip install alembic
```

You might also have:

```bash
pip install sqlalchemy asyncpg
```

---

# 5. Initialize Alembic

From your project root:

```bash
alembic init alembic
```

You get something like:

```text
project/
├── app/
│   ├── models/
│   │   ├── user.py
│   │   └── ...
│   ├── db/
│   │   └── session.py
│   └── main.py
│
├── alembic/
│   ├── versions/
│   ├── env.py
│   ├── script.py.mako
│   └── README
│
└── alembic.ini
```

The important directory is:

```text
alembic/
    versions/
```

This contains your migration files.

---

# 6. Configure the database

You can configure the URL in `alembic.ini`, but in production I prefer loading it from the application's settings/environment.

For example:

```python
# app/config.py

from pydantic_settings import BaseSettings


class Settings(BaseSettings):

    database_url: str

    class Config:
        env_file = ".env"


settings = Settings()
```

`.env`:

```text
DATABASE_URL=postgresql+asyncpg://postgres:password@localhost:5432/mydb
```

---

# 7. Configure Alembic

For async SQLAlchemy, `alembic/env.py` needs to use the async engine setup.

A simplified version:

```python
from sqlalchemy import pool
from sqlalchemy.ext.asyncio import (
    async_engine_from_config,
)

from alembic import context

from app.models.user import Base


config = context.config

target_metadata = Base.metadata
```

Then Alembic uses your metadata to detect model/schema changes.

---

# 8. What is `target_metadata`?

This is an important interview concept.

Your SQLAlchemy models have metadata:

```python
Base.metadata
```

It contains information about:

```text
Tables
Columns
Indexes
Constraints
Relationships
```

Alembic compares:

```text
SQLAlchemy metadata
        vs
Current database schema
```

and can generate migration operations.

So:

```python
target_metadata = Base.metadata
```

tells Alembic:

> "These are the SQLAlchemy models whose schema I want you to track."

---

# 9. Create your first migration

Suppose your model is:

```python
class User(Base):

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(
        primary_key=True
    )

    email: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        unique=True,
    )
```

Run:

```bash
alembic revision --autogenerate -m "create users table"
```

Alembic generates something similar to:

```text
alembic/
└── versions/
    └── 001_create_users_table.py
```

---

# 10. What does the migration file look like?

Something like:

```python
"""create users table"""

from alembic import op
import sqlalchemy as sa


revision = "abc123"

down_revision = None


def upgrade():

    op.create_table(
        "users",

        sa.Column(
            "id",
            sa.Integer(),
            primary_key=True,
        ),

        sa.Column(
            "email",
            sa.String(255),
            nullable=False,
        ),

        sa.UniqueConstraint(
            "email"
        ),
    )


def downgrade():

    op.drop_table("users")
```

There are two important functions:

```python
upgrade()
```

and:

```python
downgrade()
```

---

# 11. What is `upgrade()`?

`upgrade()` moves your database schema forward.

For example:

```python
def upgrade():

    op.add_column(
        "users",
        sa.Column(
            "name",
            sa.String(100),
            nullable=True,
        ),
    )
```

This is conceptually:

```sql
ALTER TABLE users
ADD COLUMN name VARCHAR(100);
```

---

# 12. What is `downgrade()`?

`downgrade()` reverses the migration.

```python
def downgrade():

    op.drop_column(
        "users",
        "name",
    )
```

Conceptually:

```sql
ALTER TABLE users
DROP COLUMN name;
```

So:

```text
upgrade()
    ↓
old schema → new schema

downgrade()
    ↓
new schema → old schema
```

---

# 13. Apply migrations

Run:

```bash
alembic upgrade head
```

`head` means:

> Apply migrations until the latest revision.

For example:

```text
001
 ↓
002
 ↓
003
 ↓
HEAD
```

The database gets migrated through the required revisions.

---

# 14. Check current migration

```bash
alembic current
```

You might see:

```text
abc123 (head)
```

Meaning the database is currently at that revision.

---

# 15. See migration history

```bash
alembic history
```

You might see:

```text
abc123 -> def456 (head), add user status
789xyz -> abc123, add user name
None   -> 789xyz, create users
```

---

# 16. Create a new migration

Suppose you add:

```python
is_active: Mapped[bool] = mapped_column(
    default=True
)
```

Run:

```bash
alembic revision --autogenerate -m "add user active status"
```

Alembic generates another migration:

```text
001_create_users.py
002_add_user_active_status.py
```

Then:

```bash
alembic upgrade head
```

---

# 17. The migration workflow

This is the workflow you should remember for interviews:

```text
1. Change SQLAlchemy model
          ↓
2. Generate migration
          ↓
alembic revision --autogenerate
          ↓
3. Review migration
          ↓
4. Test migration
          ↓
5. Apply migration
          ↓
alembic upgrade head
          ↓
6. Deploy application
```

---

# 18. Very important: Don't blindly trust `--autogenerate`

This is a **senior-level point**.

You should not assume:

```bash
alembic revision --autogenerate
```

always generates the perfect migration.

You should **review the generated migration**.

For example, renaming:

```text
old column:
customer_name

new column:
name
```

Alembic might interpret the change as:

```text
DROP customer_name
CREATE name
```

instead of:

```text
RENAME customer_name → name
```

That could cause data loss.

You might manually write:

```python
op.alter_column(
    "users",
    "customer_name",
    new_column_name="name",
)
```

Always review important production migrations.

---

# 19. Adding a non-null column to an existing table

This is another common production issue.

Suppose you have:

```text
users
---------
10 million existing rows
```

You add:

```python
name = Column(
    String(100),
    nullable=False,
)
```

If you immediately execute:

```sql
ALTER TABLE users
ADD COLUMN name VARCHAR(100) NOT NULL;
```

existing rows don't have values.

The migration may fail.

A safer migration can be:

```text
Step 1
Add column nullable

        ↓

Step 2
Backfill existing rows

        ↓

Step 3
Make column NOT NULL
```

Example:

```python
def upgrade():

    op.add_column(
        "users",
        sa.Column(
            "name",
            sa.String(100),
            nullable=True,
        ),
    )

    op.execute(
        "UPDATE users "
        "SET name = 'Unknown' "
        "WHERE name IS NULL"
    )

    op.alter_column(
        "users",
        "name",
        nullable=False,
    )
```

This is the kind of consideration that demonstrates production experience.

---

# 20. How migrations work in CI/CD

You generally don't want every developer manually modifying production databases.

A deployment pipeline might look like:

```text
Developer
   ↓
Change SQLAlchemy model
   ↓
Create migration
   ↓
Git commit
   ↓
Pull Request
   ↓
Migration review
   ↓
CI tests
   ↓
Build Docker image
   ↓
Deploy
   ↓
Run migrations
   ↓
Start application
```

For example:

```bash
alembic upgrade head
```

can be run as a controlled deployment step.

---

# 21. Don't run migrations from every application pod

This is an important Kubernetes/production consideration.

Suppose you deploy:

```text
10 FastAPI pods
```

If every pod runs:

```bash
alembic upgrade head
```

simultaneously, you're unnecessarily creating competing migration processes.

A better pattern is often:

```text
Kubernetes Deployment
        │
        ├── Migration Job
        │       ↓
        │   alembic upgrade head
        │
        ↓
FastAPI Pods
```

Or execute migrations as a dedicated release/deployment step, depending on your platform.

The key idea:

> **Schema migration should be a controlled deployment operation, not something every application worker independently performs.**

---

# 22. What happens if migration 2 fails?

Suppose:

```text
Migration 001 ✓
Migration 002 ✗
Migration 003
```

The database remains at the appropriate successful revision, and you fix/rework the migration before proceeding.

You can inspect:

```bash
alembic current
```

and:

```bash
alembic history
```

---

# 23. Rolling back

Suppose you're at:

```text
001
 ↓
002
 ↓
003
```

You want to go back one revision.

```bash
alembic downgrade -1
```

Conceptually:

```text
003
 ↓
002
```

Or specify a revision:

```bash
alembic downgrade 002
```

Then:

```text
003 → 002
```

---

# 24. Important production warning about downgrade

Don't assume every production migration is safely reversible.

For example:

```python
def upgrade():
    op.drop_column(
        "users",
        "old_data",
    )
```

The downgrade might technically be:

```python
def downgrade():
    op.add_column(...)
```

but the **original data is already gone**.

So:

```text
Migration reversible at schema level
             ≠
Migration reversible without data loss
```

For destructive migrations, you need careful planning and backups.

---

# 25. FastAPI + SQLAlchemy + Alembic architecture

Your application now looks like:

```text
                         FastAPI
                            │
                            ↓
                       Service Layer
                            │
                            ↓
                     Repository Layer
                            │
                            ↓
                       AsyncSession
                            │
                            ↓
                      SQLAlchemy ORM
                            │
                            ↓
                       PostgreSQL
                            ↑
                            │
                         Alembic
                            │
                   Database migrations
```

Important distinction:

```text
SQLAlchemy
    ↓
Application ↔ Database

Alembic
    ↓
Database schema evolution
```

---

# 26. SQLAlchemy vs Alembic

This is a very common interview question.

| SQLAlchemy                  | Alembic                      |
| --------------------------- | ---------------------------- |
| Python ORM/database toolkit | Migration tool               |
| Defines models              | Defines schema changes       |
| Executes queries            | Executes migrations          |
| `select()`                  | `op.add_column()`            |
| `AsyncSession`              | `alembic upgrade`            |
| Application runtime         | Deployment/schema management |

Example:

### SQLAlchemy

```python
result = await db.execute(
    select(User)
)
```

### Alembic

```python
def upgrade():

    op.add_column(
        "users",
        sa.Column(
            "name",
            sa.String(100),
        ),
    )
```

---

# 27. Interview answer

If the interviewer asks:

### **"What is Alembic?"**

A strong answer:

> **"Alembic is SQLAlchemy's database migration tool. I use it to version and manage database schema changes across development, staging, and production. When I modify SQLAlchemy models, I generate a migration, review the generated SQL operations, test it, and then apply it using `alembic upgrade head`. Each migration has an upgrade and usually a downgrade path."**

---

### **"How do you use migrations?"**

Say:

> **"I first modify the SQLAlchemy model, generate a migration using `alembic revision --autogenerate`, review and possibly edit the migration manually, test it against a representative database, and apply it using `alembic upgrade head`. In production, migrations are run as a controlled CI/CD or Kubernetes migration job rather than independently by every FastAPI pod."**

Commands to remember:

```bash
# Initialize
alembic init alembic

# Generate migration
alembic revision --autogenerate -m "add user name"

# Apply all pending migrations
alembic upgrade head

# Check current revision
alembic current

# View history
alembic history

# Roll back one revision
alembic downgrade -1
```

### The key interview sentence

> **"SQLAlchemy models describe what the schema should look like; Alembic provides the versioned, incremental path for safely changing the actual database schema."**
