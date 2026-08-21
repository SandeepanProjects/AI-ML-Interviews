In FastAPI, you normally define a **request body using a Pydantic `BaseModel`**. This gives you validation, type conversion, clear API contracts, and automatic Swagger/OpenAPI documentation.

## 1. Basic request body

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class UserCreate(BaseModel):
    name: str
    email: str
    age: int


@app.post("/users")
async def create_user(user: UserCreate):
    return {
        "name": user.name,
        "email": user.email,
        "age": user.age
    }
```

The client sends:

```json
{
    "name": "Sandeep",
    "email": "sandeep@example.com",
    "age": 37
}
```

FastAPI:

```text
JSON request
     ↓
Pydantic validation
     ↓
UserCreate object
     ↓
Endpoint
```

So inside the endpoint:

```python
user.name
user.email
user.age
```

are available as typed Python attributes.

---

# 2. Why use Pydantic?

Suppose the API expects:

```json
{
    "name": "Sandeep",
    "email": "sandeep@example.com",
    "age": 37
}
```

but the client sends:

```json
{
    "name": "Sandeep",
    "email": "wrong-email",
    "age": "abc"
}
```

Pydantic validates the input and FastAPI automatically returns a **422 validation error**.

You don't have to manually write:

```python
if "name" not in request:
    ...

if not isinstance(age, int):
    ...

if "@" not in email:
    ...
```

---

# 3. Optional fields

You can define optional fields using defaults or `Optional`.

```python
from pydantic import BaseModel


class UserCreate(BaseModel):
    name: str
    email: str
    age: int | None = None
```

Now this is valid:

```json
{
    "name": "Sandeep",
    "email": "sandeep@example.com"
}
```

because `age` is optional.

---

# 4. Default values

```python
class ChatRequest(BaseModel):
    question: str
    temperature: float = 0.7
    max_tokens: int = 1000
```

The client can send:

```json
{
    "question": "What is RAG?"
}
```

and FastAPI/Pydantic will produce:

```python
temperature = 0.7
max_tokens = 1000
```

This is very common in LLM APIs.

---

# 5. Nested request bodies

Real applications often have nested JSON.

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

Request:

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

Then:

```python
user.address.city
```

returns:

```text
Bangalore
```

---

# 6. Lists in request bodies

For example, creating an order:

```python
from pydantic import BaseModel


class OrderItem(BaseModel):
    product_id: int
    quantity: int


class OrderCreate(BaseModel):
    customer_id: int
    items: list[OrderItem]
```

Request:

```json
{
    "customer_id": 101,
    "items": [
        {
            "product_id": 1,
            "quantity": 2
        },
        {
            "product_id": 5,
            "quantity": 1
        }
    ]
}
```

FastAPI validates the entire nested structure.

---

# 7. Adding validation rules

Pydantic allows you to define constraints.

```python
from pydantic import BaseModel, Field


class UserCreate(BaseModel):
    name: str = Field(min_length=2, max_length=100)
    age: int = Field(ge=18, le=100)
    email: str
```

Now:

```json
{
    "name": "A",
    "age": 12,
    "email": "test@example.com"
}
```

will fail validation.

This is much better than putting validation logic directly inside the endpoint.

---

# 8. Request body + path parameter

You can combine them.

```python
class UserUpdate(BaseModel):
    name: str
    email: str


@app.put("/users/{user_id}")
async def update_user(
    user_id: int,
    user: UserUpdate
):
    return {
        "user_id": user_id,
        "name": user.name,
        "email": user.email
    }
```

Request:

```http
PUT /users/123
```

Body:

```json
{
    "name": "Sandeep",
    "email": "new@example.com"
}
```

Here:

```text
/users/123
       ↑
       Path parameter

JSON body
   ↓
UserUpdate
```

---

# 9. Request body + query parameter

You can also combine all three.

```python
class ChatRequest(BaseModel):
    question: str
    model: str


@app.post("/chat/{conversation_id}")
async def chat(
    conversation_id: int,
    request: ChatRequest,
    stream: bool = False
):
    return {
        "conversation_id": conversation_id,
        "question": request.question,
        "model": request.model,
        "stream": stream
    }
```

Request:

```http
POST /chat/123?stream=true
```

Body:

```json
{
    "question": "Explain RAG",
    "model": "gpt-model"
}
```

So:

```text
123
↓
Path parameter

stream=true
↓
Query parameter

JSON
↓
Request body
```

---

# 10. Production AI/RAG example

For an AI application, you might define:

```python
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(
        min_length=1,
        max_length=5000
    )

    conversation_id: str | None = None

    model: str = "default"

    temperature: float = Field(
        default=0.2,
        ge=0.0,
        le=2.0
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=20
    )
```

Then:

```python
@app.post("/chat")
async def chat(request: ChatRequest):

    result = await chat_service.generate(
        question=request.question,
        conversation_id=request.conversation_id,
        model=request.model,
        temperature=request.temperature,
        top_k=request.top_k
    )

    return result
```

Client:

```json
{
    "question": "What is retrieval augmented generation?",
    "conversation_id": "conv-123",
    "model": "default",
    "temperature": 0.2,
    "top_k": 5
}
```

This is much cleaner than accepting an arbitrary dictionary:

```python
async def chat(request: dict):
    ...
```

because the Pydantic model acts as a **strong API contract**.

---

# 11. Request schema vs database model

A very important production concept:

**Don't necessarily use your SQLAlchemy database model as your API request model.**

Instead:

```text
API
 │
 ▼
Pydantic Schema
 │
 ▼
Service
 │
 ▼
SQLAlchemy Model
 │
 ▼
PostgreSQL
```

For example:

```python
class UserCreate(BaseModel):
    name: str
    email: str
```

is your API schema.

While:

```python
class User(Base):
    __tablename__ = "users"

    id = mapped_column(...)
    name = mapped_column(...)
    email = mapped_column(...)
    password_hash = mapped_column(...)
```

is your database model.

You don't want clients to be able to send:

```json
{
    "id": 999,
    "password_hash": "..."
}
```

just because those fields exist in your database model.

---

# 12. Request body vs response model

You should normally define **separate models**.

```python
class UserCreate(BaseModel):
    name: str
    email: str
    password: str


class UserResponse(BaseModel):
    id: int
    name: str
    email: str
```

Then:

```python
@app.post(
    "/users",
    response_model=UserResponse
)
async def create_user(user: UserCreate):
    ...
```

Notice:

```text
Request:
UserCreate
    ↓
name
email
password

Response:
UserResponse
    ↓
id
name
email
```

You don't return the password to the client.

This separation is important for **security and API design**.

---

## Interview answer

If the interviewer asks:

**"How do you define request bodies in FastAPI?"**

A strong answer is:

> "I define request bodies using Pydantic models that inherit from `BaseModel`. The model acts as the API contract and provides type validation, default values, nested models, and field constraints. FastAPI automatically parses and validates the incoming JSON and returns a validation error if the payload doesn't match the schema. In production, I keep request and response schemas separate from SQLAlchemy database models so that clients can't directly control internal database fields."

A good mental model is:

```text
HTTP Request
     │
     ├── Path parameters → identify resource
     ├── Query parameters → filtering/options
     └── Request body → structured input
                  │
                  ▼
             Pydantic
                  │
                  ▼
              Service
                  │
                  ▼
             Repository
                  │
                  ▼
             Database
```
