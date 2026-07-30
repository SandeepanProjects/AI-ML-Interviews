Role-Based Access Control (RBAC) is one of the **most important security mechanisms** in enterprise AI systems. It ensures that **who the user is** determines **what they are allowed to do**.

In AI agents, RBAC is especially important because LLMs can invoke tools that perform real-world actions.

---

# What is RBAC?

RBAC restricts access based on **roles**, not individual users.

Example:

| Role    | Permissions                     |
| ------- | ------------------------------- |
| Viewer  | Read documents                  |
| Analyst | Read documents, run SQL queries |
| Manager | Approve reports, export data    |
| Admin   | Delete users, manage system     |

Instead of writing:

```python
if user.id == 123:
    allow()
```

You write:

```python
if "delete_user" in user.permissions:
    allow()
```

This scales much better.

---

# Production Architecture

```text
                 User Login
                      │
                      ▼
           Authentication (JWT)
                      │
                      ▼
          User Identity + Role Loaded
                      │
                      ▼
             Authorization Layer
                      │
        ┌─────────────┼─────────────┐
        ▼             ▼             ▼
   SQL Tool      Email Tool    Admin Tool
```

Authentication answers:

> Who are you?

Authorization answers:

> What are you allowed to do?

---

# Step 1: Define Roles

```python
from enum import Enum

class Role(str, Enum):
    VIEWER = "viewer"
    ANALYST = "analyst"
    MANAGER = "manager"
    ADMIN = "admin"
```

---

# Step 2: Define Permissions

Instead of checking roles everywhere, map roles to permissions.

```python
ROLE_PERMISSIONS = {
    Role.VIEWER: {
        "read_documents"
    },

    Role.ANALYST: {
        "read_documents",
        "query_database"
    },

    Role.MANAGER: {
        "read_documents",
        "query_database",
        "approve_report"
    },

    Role.ADMIN: {
        "read_documents",
        "query_database",
        "approve_report",
        "delete_user",
        "manage_roles"
    }
}
```

---

# Step 3: User Model

```python
from dataclasses import dataclass

@dataclass
class User:
    id: str
    name: str
    role: Role

    @property
    def permissions(self):
        return ROLE_PERMISSIONS[self.role]
```

Example:

```python
alice = User(
    id="101",
    name="Alice",
    role=Role.ANALYST
)

print(alice.permissions)
```

Output:

```text
{
    'read_documents',
    'query_database'
}
```

---

# Step 4: Authorization Function

```python
class AuthorizationError(Exception):
    pass


def require_permission(user: User, permission: str):

    if permission not in user.permissions:
        raise AuthorizationError(
            f"{user.role} cannot perform '{permission}'"
        )
```

Usage:

```python
require_permission(
    alice,
    "query_database"
)
```

Works.

```python
require_permission(
    alice,
    "delete_user"
)
```

Raises:

```text
AuthorizationError:
analyst cannot perform 'delete_user'
```

---

# Step 5: Protect Tools

Suppose we expose a SQL tool.

```python
def query_database(user: User, sql: str):

    require_permission(
        user,
        "query_database"
    )

    # Execute parameterized SQL
    return run_sql(sql)
```

Delete tool:

```python
def delete_user(user: User, user_id: str):

    require_permission(
        user,
        "delete_user"
    )

    # delete logic
```

Now even if the LLM tries to call `delete_user`, the authorization layer blocks unauthorized users.

---

# Step 6: Use a Decorator

Instead of repeating authorization checks:

```python
from functools import wraps

def requires(permission):

    def decorator(func):

        @wraps(func)
        def wrapper(user: User, *args, **kwargs):

            require_permission(
                user,
                permission
            )

            return func(
                user,
                *args,
                **kwargs
            )

        return wrapper

    return decorator
```

Usage:

```python
@requires("query_database")
def sql_tool(user, sql):
    return run_sql(sql)


@requires("delete_user")
def delete_user_tool(user, user_id):
    return delete(user_id)
```

This keeps tools clean and consistent.

---

# Step 7: FastAPI Integration

Authenticate with JWT and inject the current user.

```python
from fastapi import Depends, FastAPI

app = FastAPI()

def get_current_user():
    # Normally decode JWT here
    return User(
        id="101",
        name="Alice",
        role=Role.ANALYST
    )


@app.get("/users")
def list_users(
    user: User = Depends(get_current_user)
):

    require_permission(
        user,
        "read_documents"
    )

    return {"status": "allowed"}
```

---

# Step 8: RBAC with LangChain Tools

```python
from langchain_core.tools import tool

@tool
def query_sales(
    user: User,
    month: str
):

    require_permission(
        user,
        "query_database"
    )

    return f"Sales for {month}"
```

Before the tool executes, the permission check runs.

---

# Step 9: RBAC in LangGraph

State carries user information.

```python
from typing import TypedDict

class AgentState(TypedDict):
    user: User
    query: str
```

Node:

```python
def sql_agent(state: AgentState):

    require_permission(
        state["user"],
        "query_database"
    )

    result = run_sql(
        state["query"]
    )

    return {
        "result": result
    }
```

If authorization fails, the graph can route to an error handler.

```python
def router(state):

    if state.get("error"):
        return "access_denied"

    return "writer"
```

---

# Step 10: Human Approval for Sensitive Actions

Even admins may require approval.

```text
User
 │
 ▼
Transfer ₹100,000
 │
 ▼
RBAC Check
 │
 ▼
Human Approval
 │
 ▼
Execute Tool
```

Example:

```python
def transfer_money(
    user,
    amount
):

    require_permission(
        user,
        "approve_report"
    )

    if amount > 50000:
        return {
            "requires_human": True
        }

    return perform_transfer(amount)
```

RBAC and human approval complement each other.

---

# Step 11: Audit Logging

Every authorization decision should be logged.

```python
import logging

logger = logging.getLogger(__name__)

def require_permission(user, permission):

    logger.info(
        "Authorization check",
        extra={
            "user": user.id,
            "role": user.role,
            "permission": permission
        }
    )

    if permission not in user.permissions:
        logger.warning(
            "Authorization denied",
            extra={
                "user": user.id,
                "permission": permission
            }
        )
        raise AuthorizationError
```

Example log:

```text
2026-07-30 18:50
User:101
Role:ANALYST
Permission:delete_user
Result:DENIED
```

---

# Enterprise Database Design

Instead of hardcoding roles, store them in a database.

```text
Users
-------------------------
id
name
role_id

Roles
-------------------------
id
name

Permissions
-------------------------
id
name

RolePermissions
-------------------------
role_id
permission_id
```

Example:

```text
Role: Analyst

↓

Permissions

- read_documents
- query_database
```

This allows administrators to modify permissions without changing code.

---

# RBAC Flow

```text
            User Request
                  │
                  ▼
          JWT Authentication
                  │
                  ▼
            Load User & Role
                  │
                  ▼
       Check Required Permission
                  │
        ┌─────────┴─────────┐
        ▼                   ▼
     Allowed             Denied
        │                   │
        ▼                   ▼
 Execute Tool         Return 403/Error
```

---

# RBAC vs ABAC

| RBAC                              | ABAC                                                              |
| --------------------------------- | ----------------------------------------------------------------- |
| Based on user role                | Based on attributes                                               |
| Easy to implement                 | More flexible                                                     |
| Good for enterprise apps          | Good for fine-grained policies                                    |
| Example: "Admin can delete users" | Example: "Managers can approve reports only for their department" |

Many enterprises combine both:

* RBAC → High-level access
* ABAC → Fine-grained rules

---

# Best Practices

| Practice                                   | Reason                          |
| ------------------------------------------ | ------------------------------- |
| Centralize authorization                   | Avoid duplicated logic          |
| Protect every tool                         | Never trust the LLM             |
| Use permissions, not hardcoded role checks | Easier to maintain              |
| Validate JWTs                              | Prevent identity spoofing       |
| Log every decision                         | Auditing and compliance         |
| Apply least privilege                      | Grant only required permissions |
| Combine RBAC with human approval           | Protect high-risk operations    |
| Keep roles in a database                   | Easier administration           |

---

# Common Interview Questions

### Why check permissions inside the tool instead of only in the prompt?

Because prompts are **not a security boundary**. The LLM can generate unexpected tool calls. Every tool must independently enforce authorization.

---

### Should the LLM decide whether a user is an admin?

No. The application authenticates the user (e.g., via JWT), determines their role from a trusted source, and enforces authorization before any sensitive operation.

---

### How do you support different permissions for different departments?

Use **RBAC + ABAC**. RBAC determines broad permissions (e.g., `query_database`), while ABAC adds contextual rules (e.g., `department == "Finance"`).

---

# Production RBAC Architecture

```text
                    User
                      │
                 JWT Login
                      │
                      ▼
           Authentication Service
                      │
                      ▼
             User + Role Loaded
                      │
                      ▼
          Authorization Middleware
                      │
      ┌───────────────┼────────────────┐
      ▼               ▼                ▼
  SQL Tool       Email Tool      Admin Tool
      │               │                │
      └───────────────┼────────────────┘
                      ▼
               Audit Logging
                      ▼
             Monitoring & Alerts
```

---

# Senior AI Engineer Interview Answer

> **I implement RBAC by separating authentication from authorization. Users authenticate with JWT, and the application loads their role and permissions from a trusted source such as a database. Every protected tool or API declares the permission it requires, and a centralized authorization layer validates that the authenticated user has that permission before execution. I enforce checks inside the tool itself rather than relying on prompts, log every authorization decision for auditing, apply the principle of least privilege, and combine RBAC with human approval or attribute-based checks for high-risk operations. This ensures that even if an LLM attempts an unauthorized tool call, the execution layer prevents it.**
