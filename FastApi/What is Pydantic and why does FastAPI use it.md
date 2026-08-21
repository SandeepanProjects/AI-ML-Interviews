**Pydantic is a Python library for data validation and serialization using Python type hints.** FastAPI uses it heavily because API applications constantly receive untrusted external data—JSON bodies, query parameters, headers—and need to validate that data before passing it into business logic.

Think of it as:

```text
Client JSON
    ↓
Pydantic
    ↓
Validate + Parse + Convert
    ↓
Python object
    ↓
Business logic
```

---

# 1. What problem does Pydantic solve?

Suppose your API expects:

```json
{
  "name": "Sandeep",
  "age": 37,
  "email": "sandeep@example.com"
}
```

Without Pydantic, you might manually validate:

```python
def create_user(data: dict):

    if "name" not in data:
        raise ValueError("name required")

    if not isinstance(data["age"], int):
        raise ValueError("age must be integer")

    if "email" not in data:
        raise ValueError("email required")

    # business logic
```

This becomes painful as your API grows.

With Pydantic:

```python
from pydantic import BaseModel


class UserCreate(BaseModel):
    name: str
    age: int
    email: str
```

Now Pydantic defines the contract.

---

# 2. Pydantic validates data

If you create:

```python
user = UserCreate(
    name="Sandeep",
    age=37,
    email="sandeep@example.com"
)
```

Pydantic validates the fields.

If required data is missing:

```python
UserCreate(
    name="Sandeep"
)
```

Pydantic raises a validation error because:

```text
age      → required
email    → required
```

---

# 3. Type validation

You specify:

```python
class User(BaseModel):
    name: str
    age: int
```

Pydantic understands:

```text
name → string
age  → integer
```

So:

```python
User(
    name="Sandeep",
    age="abc"
)
```

will fail validation.

Depending on the input and Pydantic configuration, compatible values may also be converted. For example, a numeric string may be parsed into an integer.

---

# 4. Field constraints

You can add validation rules.

```python
from pydantic import BaseModel, Field


class UserCreate(BaseModel):
    name: str = Field(
        min_length=2,
        max_length=100
    )

    age: int = Field(
        ge=18,
        le=100
    )
```

Now you have:

```text
name:
  minimum length = 2
  maximum length = 100

age:
  minimum = 18
  maximum = 100
```

This is much cleaner than writing validation manually.

---

# 5. Pydantic supports nested models

This is very useful for real APIs.

```python
from pydantic import BaseModel


class Address(BaseModel):
    city: str
    country: str


class UserCreate(BaseModel):
    name: str
    email: str
    address: Address
```

The API can receive:

```json
{
  "name": "Sandeep",
  "email": "sandeep@example.com",
  "address": {
    "city": "Bangalore",
    "country": "India"
  }
}
```

Pydantic validates the entire structure.

---

# 6. Pydantic isn't just validation

Pydantic also handles **serialization**.

For example:

```python
user = UserCreate(
    name="Sandeep",
    age=37,
    email="sandeep@example.com"
)
```

You can convert it into a dictionary:

```python
user.model_dump()
```

Result:

```python
{
    "name": "Sandeep",
    "age": 37,
    "email": "sandeep@example.com"
}
```

And you can serialize it to JSON:

```python
user.model_dump_json()
```

So Pydantic is useful for:

```text
Validation
    +
Parsing
    +
Serialization
    +
Schema definition
```

---

# 7. Why does FastAPI use Pydantic?

Because an API receives data from **outside your application**.

For example:

```text
Mobile App
    ↓
HTTP Request
    ↓
FastAPI
    ↓
Pydantic validation
    ↓
Your business logic
```

You don't want invalid data reaching:

```text
Service
   ↓
Database
```

Pydantic creates a boundary:

```text
          UNTRUSTED
              │
              ▼
        ┌────────────┐
        │   Pydantic │
        │ Validation  │
        └──────┬─────┘
               │
               ▼
          TRUSTED DATA
               │
               ▼
            Service
```

---

# 8. FastAPI + Pydantic request body

This is where you will use it most frequently.

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class ChatRequest(BaseModel):
    question: str
    temperature: float = 0.2
    top_k: int = 5


@app.post("/chat")
async def chat(request: ChatRequest):

    return {
        "question": request.question,
        "temperature": request.temperature,
        "top_k": request.top_k
    }
```

Client sends:

```json
{
  "question": "What is RAG?",
  "temperature": 0.2,
  "top_k": 5
}
```

FastAPI automatically:

```text
HTTP JSON
   ↓
Pydantic
   ↓
ChatRequest
   ↓
Endpoint
```

You don't have to manually parse the JSON.

---

# 9. Pydantic also powers Swagger/OpenAPI

This is another major reason FastAPI uses it.

Consider:

```python
class ChatRequest(BaseModel):
    question: str
    temperature: float = 0.2
    top_k: int = 5
```

FastAPI can use that model to generate OpenAPI documentation:

```text
POST /chat

Request body:

question      string    required
temperature   number    default: 0.2
top_k         integer   default: 5
```

So the relationship is:

```text
Pydantic Model
      │
      ├── Validation
      │
      ├── Parsing
      │
      ├── Serialization
      │
      └── JSON Schema
               │
               ▼
            OpenAPI
               │
               ▼
           Swagger UI
```

---

# 10. Pydantic response models

Pydantic isn't only for requests.

You can define responses too.

```python
class UserResponse(BaseModel):
    id: int
    name: str
    email: str


@app.get(
    "/users/{user_id}",
    response_model=UserResponse
)
async def get_user(user_id: int):

    return {
        "id": user_id,
        "name": "Sandeep",
        "email": "sandeep@example.com"
    }
```

Now FastAPI knows exactly what the API is supposed to return.

This is important because you don't want your internal database object to accidentally expose sensitive fields.

---

# 11. Request model vs database model

A senior engineer should distinguish these.

### API schema

```python
class UserCreate(BaseModel):
    name: str
    email: str
    password: str
```

### Database model

```python
class User(Base):
    __tablename__ = "users"

    id = ...
    name = ...
    email = ...
    password_hash = ...
    created_at = ...
```

You generally don't want to expose the database model directly through your API.

Instead:

```text
Client
  ↓
Pydantic Request Model
  ↓
Service
  ↓
SQLAlchemy Model
  ↓
PostgreSQL
```

and on the way back:

```text
PostgreSQL
  ↓
SQLAlchemy
  ↓
Service
  ↓
Pydantic Response Model
  ↓
Client
```

This gives you a clean boundary between your API and persistence layer.

---

# 12. Why not just use dictionaries?

You could write:

```python
@app.post("/chat")
async def chat(request: dict):
    ...
```

But now you lose a lot of structure.

You have to manually access:

```python
request["question"]
request["temperature"]
request["top_k"]
```

and manually validate them.

With Pydantic:

```python
class ChatRequest(BaseModel):
    question: str
    temperature: float
    top_k: int
```

you get:

* type validation
* defaults
* nested models
* constraints
* IDE/type checking
* JSON schema
* OpenAPI documentation
* serialization

So for production APIs, Pydantic models are generally much better than arbitrary dictionaries.

---

# 13. Pydantic in an LLM application

This is particularly important in AI engineering.

You might define:

```python
class ChatRequest(BaseModel):
    question: str
    model: str = "default"
    temperature: float = 0.2
    top_k: int = 5
```

And:

```python
class ChatResponse(BaseModel):
    answer: str
    sources: list[str]
    latency_ms: float
```

Then your API becomes strongly typed:

```text
                FastAPI
                   │
          ┌────────┴────────┐
          ▼                 ▼
   ChatRequest          ChatResponse
     Pydantic              Pydantic
          │                 ▲
          ▼                 │
       Service ────────► RAG/LLM
```

This makes your LLM service much easier to maintain and test.

---

# 14. Pydantic vs Type hints

This is another subtle interview point.

Python type hints alone:

```python
def create_user(age: int):
    ...
```

**do not automatically validate runtime input.**

Pydantic:

```python
class User(BaseModel):
    age: int
```

actually performs runtime validation when the model is instantiated.

So:

```text
Type hint
   → tells tools/developers what type is expected

Pydantic
   → validates/parses data at runtime
```

FastAPI combines the two extremely well.

---

# Interview answer

If the interviewer asks:

**"What is Pydantic and why does FastAPI use it?"**

A strong senior-level answer is:

> **"Pydantic is a Python data-validation and serialization library based on type annotations. FastAPI uses Pydantic to define request and response schemas, validate incoming data, parse JSON into typed Python objects, serialize responses, and generate JSON Schema used by OpenAPI and Swagger. This gives us a strongly typed API contract and keeps validation at the API boundary before data reaches our service or database layers."**

And remember this flow:

```text
Client JSON
    ↓
FastAPI
    ↓
Pydantic
 ┌───────────────┐
 │ Validate      │
 │ Parse         │
 │ Type-check    │
 │ Serialize     │
 │ JSON Schema   │
 └───────┬───────┘
         ↓
     Service Layer
         ↓
    Database / LLM
```

**One-liner for interviews:**

> **"Pydantic is the schema and validation layer of a FastAPI application."**
