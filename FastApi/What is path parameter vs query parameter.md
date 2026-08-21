The easiest way to remember it is:

> **Path parameter = identifies the resource.**
> **Query parameter = modifies how you retrieve/filter that resource.**

## 1. Path parameter

A **path parameter** is part of the URL path.

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/users/{user_id}")
async def get_user(user_id: int):
    return {"user_id": user_id}
```

Request:

```http
GET /users/123
```

Here:

```text
/users/{user_id}
        ↑
   path parameter
```

`123` identifies **which user** you want.

### Typical use cases

```text
/users/123
/orders/456
/products/789
/documents/abc123
/conversations/xyz456
```

Think:

> **"Which specific resource?"**

---

# 2. Query parameter

A **query parameter** comes after `?` in the URL.

```python
@app.get("/users")
async def get_users(
    page: int = 1,
    limit: int = 20
):
    return {
        "page": page,
        "limit": limit
    }
```

Request:

```http
GET /users?page=2&limit=10
```

Here:

```text
/users?page=2&limit=10
      ↑
   query parameters
```

The endpoint is still retrieving `/users`, but the query parameters control **how** you retrieve them.

Typical uses:

* pagination
* filtering
* searching
* sorting
* optional configuration

For example:

```http
GET /users?page=2&limit=20
GET /users?role=admin
GET /users?status=active
GET /users?sort=name
```

Think:

> **"How should I retrieve the resources?"**

---

# 3. Path vs Query

Compare:

```http
GET /users/123
```

versus:

```http
GET /users?role=admin
```

The first says:

> Give me **user 123**.

The second says:

> Give me **users**, filtered by `role=admin`.

---

## 4. FastAPI example with both

You can use both together:

```python
@app.get("/users/{user_id}/orders")
async def get_orders(
    user_id: int,
    page: int = 1,
    limit: int = 20
):
    return {
        "user_id": user_id,
        "page": page,
        "limit": limit
    }
```

Request:

```http
GET /users/123/orders?page=2&limit=10
```

Here:

```text
/users/123/orders
       ↑
       │
       └── Path parameter

?page=2&limit=10
 ↑
 └── Query parameters
```

Meaning:

> Get orders belonging to **user 123**, page 2, 10 orders per page.

---

# 5. Important FastAPI distinction

FastAPI determines this based on the function signature and route.

### Path parameter

```python
@app.get("/users/{user_id}")
async def get_user(user_id: int):
    ...
```

Because `user_id` appears in:

```text
/users/{user_id}
```

FastAPI knows it is a path parameter.

### Query parameter

```python
@app.get("/users")
async def get_users(
    page: int = 1
):
    ...
```

`page` isn't part of the route, so FastAPI treats it as a query parameter.

---

# 6. Required vs optional query parameters

You can make a query parameter required:

```python
from fastapi import Query

@app.get("/users")
async def search_users(
    name: str = Query(...)
):
    return {"name": name}
```

Now:

```http
GET /users?name=Sandeep
```

is valid.

But:

```http
GET /users
```

returns a validation error.

You can make it optional:

```python
@app.get("/users")
async def search_users(
    name: str | None = None
):
    return {"name": name}
```

Now both are valid:

```http
GET /users
```

and:

```http
GET /users?name=Sandeep
```

---

# 7. Real-world RAG example

This distinction becomes very useful in your AI/RAG APIs.

### Get a specific document

```http
GET /documents/123
```

`123` is a **path parameter**.

```python
@app.get("/documents/{document_id}")
async def get_document(document_id: int):
    ...
```

You're saying:

> Give me document **123**.

---

### Search documents

```http
GET /documents?query=machine-learning&limit=10
```

`query` and `limit` are **query parameters**.

```python
@app.get("/documents")
async def search_documents(
    query: str,
    limit: int = 10
):
    ...
```

You're saying:

> Search documents for **machine-learning**, returning up to 10 results.

---

# 8. POST example

Path and query parameters aren't limited to GET.

```python
@app.post("/documents/{document_id}/reprocess")
async def reprocess_document(
    document_id: int,
    force: bool = False
):
    return {
        "document_id": document_id,
        "force": force
    }
```

Request:

```http
POST /documents/123/reprocess?force=true
```

Here:

```text
123
 ↓
Path parameter
(document_id)

force=true
 ↓
Query parameter
```

---

# 9. Path vs Query vs Request Body

This is a **very common interview question**.

| Parameter type | Example           | Purpose                  |
| -------------- | ----------------- | ------------------------ |
| Path           | `/users/123`      | Identify resource        |
| Query          | `/users?page=2`   | Filter/control retrieval |
| Body           | `{"name":"John"}` | Send structured data     |

For example:

```http
PUT /users/123?notify=true
```

```json
{
    "name": "John",
    "email": "john@example.com"
}
```

You have:

```text
123
↓
Path parameter
→ Which user?

notify=true
↓
Query parameter
→ How should operation behave?

JSON
↓
Request body
→ What data should be updated?
```

---

## Interview answer

If asked **"What is the difference between path and query parameters?"**, give this:

> **"A path parameter identifies a specific resource and is part of the URL path, such as `/users/{user_id}`. A query parameter provides optional filtering, pagination, sorting or behavioral options, such as `/users?page=2&limit=20`. In FastAPI, path parameters are declared in the route itself, while parameters not present in the route are generally interpreted as query parameters. For example, `/documents/123?limit=10` has `123` as the path parameter and `limit` as the query parameter."**

### Memory trick

```text
PATH  → WHICH resource?
QUERY → HOW to retrieve/modify it?
BODY  → WHAT data to send?
```
