In FastAPI, **returning a Python `dict`, list, or Pydantic model is usually enough**. FastAPI automatically serializes it into JSON.

## 1. Simplest approach

```python
from fastapi import FastAPI

app = FastAPI()


@app.get("/users")
async def get_users():
    return {
        "id": 1,
        "name": "Sandeep",
        "role": "AI Engineer"
    }
```

The client receives:

```json
{
  "id": 1,
  "name": "Sandeep",
  "role": "AI Engineer"
}
```

You don't need to manually call `json.dumps()`.

---

## 2. Returning a list

```python
@app.get("/users")
async def get_users():
    return [
        {"id": 1, "name": "Sandeep"},
        {"id": 2, "name": "John"}
    ]
```

Response:

```json
[
  {
    "id": 1,
    "name": "Sandeep"
  },
  {
    "id": 2,
    "name": "John"
  }
]
```

---

# 3. Use a Pydantic response model

For production APIs, I prefer defining the response schema explicitly.

```python
from pydantic import BaseModel


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

FastAPI validates and serializes the returned object according to `UserResponse`.

This gives you a strong API contract:

```text
Endpoint
   ↓
response_model
   ↓
Validation + serialization
   ↓
JSON
```

---

# 4. Returning HTTP status codes

You can specify the status code directly.

For example, when creating a resource:

```python
from fastapi import status


@app.post(
    "/users",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED
)
async def create_user(user: UserCreate):

    return {
        "id": 101,
        "name": user.name,
        "email": user.email
    }
```

Response:

```http
HTTP/1.1 201 Created
```

with:

```json
{
  "id": 101,
  "name": "Sandeep",
  "email": "sandeep@example.com"
}
```

Common status codes:

| Status | Meaning                       |
| ------ | ----------------------------- |
| `200`  | Successful request            |
| `201`  | Resource created              |
| `204`  | Success with no response body |
| `400`  | Bad request                   |
| `401`  | Unauthenticated               |
| `403`  | Forbidden                     |
| `404`  | Resource not found            |
| `422`  | Validation error              |
| `500`  | Server error                  |

---

# 5. Returning errors as JSON

Use `HTTPException`.

```python
from fastapi import HTTPException


@app.get("/users/{user_id}")
async def get_user(user_id: int):

    user = find_user(user_id)

    if user is None:
        raise HTTPException(
            status_code=404,
            detail="User not found"
        )

    return user
```

FastAPI returns JSON similar to:

```json
{
  "detail": "User not found"
}
```

---

# 6. Custom JSON response

Sometimes you need more control over the response.

You can use `JSONResponse`.

```python
from fastapi.responses import JSONResponse


@app.get("/users/{user_id}")
async def get_user(user_id: int):

    return JSONResponse(
        status_code=200,
        content={
            "success": True,
            "data": {
                "id": user_id,
                "name": "Sandeep"
            }
        }
    )
```

Response:

```json
{
  "success": true,
  "data": {
    "id": 123,
    "name": "Sandeep"
  }
}
```

However, **don't use `JSONResponse` everywhere**. For normal APIs, simply returning dictionaries/Pydantic models is cleaner.

---

# 7. Production API response structure

In enterprise applications, you may standardize your response format.

For example:

```python
from typing import Generic, TypeVar
from pydantic import BaseModel

T = TypeVar("T")


class APIResponse(BaseModel, Generic[T]):
    success: bool
    data: T
    message: str | None = None
```

Then:

```python
class UserResponse(BaseModel):
    id: int
    name: str


@app.get("/users/{user_id}")
async def get_user(user_id: int):

    user = UserResponse(
        id=user_id,
        name="Sandeep"
    )

    return APIResponse(
        success=True,
        data=user,
        message="User retrieved successfully"
    )
```

Response:

```json
{
  "success": true,
  "data": {
    "id": 123,
    "name": "Sandeep"
  },
  "message": "User retrieved successfully"
}
```

---

# 8. AI/RAG API example

Suppose you have a RAG endpoint:

```python
class ChatResponse(BaseModel):
    answer: str
    sources: list[str]
    confidence: float
```

Your endpoint:

```python
@app.post(
    "/chat",
    response_model=ChatResponse
)
async def chat(request: ChatRequest):

    result = await rag_service.answer(
        request.question
    )

    return ChatResponse(
        answer=result.answer,
        sources=result.sources,
        confidence=result.confidence
    )
```

The client receives:

```json
{
  "answer": "RAG combines retrieval with generation...",
  "sources": [
    "document_123",
    "document_456"
  ],
  "confidence": 0.91
}
```

This is a good production pattern because the API has an explicit response contract.

---

# 9. What happens internally?

When you write:

```python
return {
    "id": 123,
    "name": "Sandeep"
}
```

the flow is roughly:

```text
Python dict
    ↓
FastAPI
    ↓
Response serialization
    ↓
JSON
    ↓
HTTP response
    ↓
Client
```

So you generally **shouldn't do this**:

```python
import json

return json.dumps({
    "id": 123,
    "name": "Sandeep"
})
```

That returns a JSON **string**, rather than letting FastAPI handle normal JSON serialization.

---

## Interview answer

If asked **"How do you return JSON responses in FastAPI?"**, I'd answer:

> "FastAPI automatically serializes Python dictionaries, lists, and Pydantic models into JSON responses. For production APIs, I usually define a Pydantic `response_model` so the response has a clear contract and is validated and documented automatically. I use appropriate HTTP status codes such as 201 for resource creation and `HTTPException` for errors. I only use `JSONResponse` when I need explicit control over the status code, headers, or response structure."

The key distinction to remember:

```text
dict / Pydantic model
        ↓
FastAPI serialization
        ↓
JSON HTTP response
```

And for a production API:

```text
Request
   ↓
Pydantic Request Model
   ↓
Router
   ↓
Service
   ↓
Repository / LLM / RAG
   ↓
Pydantic Response Model
   ↓
JSON Response
```
