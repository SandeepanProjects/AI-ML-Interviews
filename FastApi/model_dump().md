`model_dump()` is a **Pydantic v2 method that converts a Pydantic `BaseModel` instance into a normal Python dictionary**.

Think:

```text
Pydantic Model
      ↓
model_dump()
      ↓
Python dict
```

---

## 1. Basic example

```python
from pydantic import BaseModel


class User(BaseModel):
    name: str
    age: int
    email: str


user = User(
    name="Sandeep",
    age=37,
    email="sandeep@example.com"
)

data = user.model_dump()

print(data)
```

Output:

```python
{
    "name": "Sandeep",
    "age": 37,
    "email": "sandeep@example.com"
}
```

So:

```python
user
```

is a Pydantic object, while:

```python
user.model_dump()
```

is a normal Python `dict`.

---

# 2. Why is it useful?

Suppose your service receives a Pydantic request:

```python
class UserCreate(BaseModel):
    name: str
    email: str
    age: int
```

Your endpoint:

```python
@app.post("/users")
async def create_user(user: UserCreate):

    data = user.model_dump()

    return data
```

You might use `model_dump()` when passing structured data to:

* a repository
* SQLAlchemy
* another Python function
* logging
* caching
* serialization
* message queues

For example:

```python
await user_repository.create(
    **user.model_dump()
)
```

---

# 3. `model_dump()` vs `dict()`

If you're preparing for interviews, this is important.

### Pydantic v2

Use:

```python
user.model_dump()
```

### Pydantic v1

You may see:

```python
user.dict()
```

So:

```text
Pydantic v1 → .dict()
Pydantic v2 → .model_dump()
```

`model_dump()` is the modern Pydantic v2 API.

---

# 4. Excluding fields

You can control what gets dumped.

```python
data = user.model_dump(
    exclude={"email"}
)
```

Result:

```python
{
    "name": "Sandeep",
    "age": 37
}
```

This is useful when you don't want to pass certain fields to another layer.

---

# 5. Including only specific fields

```python
data = user.model_dump(
    include={"name", "email"}
)
```

Result:

```python
{
    "name": "Sandeep",
    "email": "sandeep@example.com"
}
```

---

# 6. Excluding unset fields

Suppose:

```python
from pydantic import BaseModel


class UserUpdate(BaseModel):
    name: str | None = None
    email: str | None = None
    age: int | None = None
```

Client sends only:

```json
{
    "name": "Sandeep"
}
```

You can do:

```python
data = user.model_dump(
    exclude_unset=True
)
```

Result:

```python
{
    "name": "Sandeep"
}
```

This is **very useful for PATCH/update APIs**.

For example:

```python
@app.patch("/users/{user_id}")
async def update_user(
    user_id: int,
    user: UserUpdate
):
    updates = user.model_dump(
        exclude_unset=True
    )

    await user_service.update(
        user_id,
        updates
    )
```

Now you only update fields that the client actually provided.

---

# 7. Excluding `None`

Suppose:

```python
user = UserUpdate(
    name="Sandeep",
    email=None
)
```

Then:

```python
user.model_dump(
    exclude_none=True
)
```

returns:

```python
{
    "name": "Sandeep"
}
```

This is different from `exclude_unset`.

### `exclude_unset`

Exclude fields that weren't provided.

### `exclude_none`

Exclude fields whose value is `None`.

---

# 8. `model_dump_json()`

Don't confuse:

```python
model_dump()
```

with:

```python
model_dump_json()
```

### `model_dump()`

Returns a Python dictionary:

```python
data = user.model_dump()
```

```python
{
    "name": "Sandeep",
    "age": 37
}
```

### `model_dump_json()`

Returns a JSON string:

```python
json_data = user.model_dump_json()
```

Example:

```json
{"name":"Sandeep","age":37}
```

So:

```text
model_dump()
      ↓
Python dict

model_dump_json()
      ↓
JSON string
```

---

# 9. Nested models

`model_dump()` also handles nested Pydantic models.

```python
from pydantic import BaseModel


class Address(BaseModel):
    city: str
    country: str


class User(BaseModel):
    name: str
    address: Address
```

Create:

```python
user = User(
    name="Sandeep",
    address={
        "city": "Bangalore",
        "country": "India"
    }
)
```

Then:

```python
user.model_dump()
```

gives:

```python
{
    "name": "Sandeep",
    "address": {
        "city": "Bangalore",
        "country": "India"
    }
}
```

---

# 10. Real FastAPI example

Consider:

```python
class ChatRequest(BaseModel):
    question: str
    top_k: int = 5
    temperature: float = 0.2
```

Your endpoint:

```python
@app.post("/chat")
async def chat(request: ChatRequest):

    params = request.model_dump()

    result = await rag_service.generate(
        **params
    )

    return result
```

The Pydantic model:

```text
ChatRequest
    ↓
model_dump()
    ↓
{
    "question": "...",
    "top_k": 5,
    "temperature": 0.2
}
    ↓
RAG Service
```

This is a common pattern.

---

# 11. `model_dump()` and SQLAlchemy

You might see code like:

```python
user_data = user.model_dump()

db_user = User(**user_data)
```

For example:

```python
class UserCreate(BaseModel):
    name: str
    email: str
```

Then:

```python
user_data = user.model_dump()

db_user = User(
    **user_data
)
```

This converts:

```text
Pydantic API model
        ↓
Python dict
        ↓
SQLAlchemy model
```

However, in a larger production application, you may prefer an explicit mapping function rather than blindly passing every field:

```python
db_user = User(
    name=user.name,
    email=user.email
)
```

This gives you tighter control over your persistence layer.

---

# 12. Important security point

Don't blindly do:

```python
db_user = User(
    **request.model_dump()
)
```

if the request contains fields the client shouldn't control.

For example:

```python
class UserCreate(BaseModel):
    name: str
    email: str
    is_admin: bool = False
```

If the client can send:

```json
{
    "name": "Sandeep",
    "email": "x@example.com",
    "is_admin": true
}
```

you may accidentally allow privilege escalation.

Instead, separate schemas:

```python
class UserCreate(BaseModel):
    name: str
    email: str
```

and set privileged fields server-side.

---

# Interview answer

If asked **"What is `model_dump()`?"**, say:

> **"`model_dump()` is a Pydantic v2 method that converts a Pydantic model instance into a Python dictionary. It's commonly used when passing validated request data to service or repository layers, serializing data, or preparing partial updates. It supports options such as `include`, `exclude`, `exclude_unset`, and `exclude_none`. For JSON serialization, Pydantic provides `model_dump_json()`."**

### Most important distinction

```text
user.model_dump()
        ↓
Python dict

user.model_dump_json()
        ↓
JSON string
```

And for FastAPI PATCH APIs:

```python
updates = request.model_dump(exclude_unset=True)
```

is a particularly important pattern to remember.
