In FastAPI, you create **GET, POST, PUT, and DELETE endpoints** using decorators such as `@app.get()`, `@app.post()`, `@app.put()`, and `@app.delete()`.

For a **Senior AI Engineer interview**, you should understand not just the syntax, but how these endpoints fit into a production architecture.

## 1. Basic example

```python
from fastapi import FastAPI

app = FastAPI()


@app.get("/users")
async def get_users():
    return {"message": "Get users"}


@app.post("/users")
async def create_user():
    return {"message": "Create user"}


@app.put("/users/{user_id}")
async def update_user(user_id: int):
    return {
        "message": "Update user",
        "user_id": user_id
    }


@app.delete("/users/{user_id}")
async def delete_user(user_id: int):
    return {
        "message": "Delete user",
        "user_id": user_id
    }
```

These correspond to:

| HTTP Method | Purpose             | Example            |
| ----------- | ------------------- | ------------------ |
| GET         | Retrieve data       | `GET /users`       |
| POST        | Create data         | `POST /users`      |
| PUT         | Replace/update data | `PUT /users/10`    |
| DELETE      | Delete data         | `DELETE /users/10` |

---

# 2. GET endpoint

A GET endpoint is normally used to retrieve resources.

```python
@app.get("/users")
async def get_users():
    return [
        {"id": 1, "name": "Sandeep"},
        {"id": 2, "name": "John"}
    ]
```

Request:

```http
GET /users
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

### Path parameter

```python
@app.get("/users/{user_id}")
async def get_user(user_id: int):
    return {
        "id": user_id
    }
```

Request:

```http
GET /users/10
```

FastAPI automatically converts and validates:

```text
"10"
 ↓
10
```

because we specified:

```python
user_id: int
```

---

# 3. Query parameters

You can also use query parameters.

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

FastAPI gives:

```python
page = 2
limit = 10
```

This is commonly used for:

* pagination
* filtering
* sorting
* searching

For example:

```http
GET /documents?tenant_id=123&page=2&limit=20
```

---

# 4. POST endpoint

POST is generally used to **create a resource**.

For example, define a Pydantic request model:

```python
from pydantic import BaseModel


class UserCreate(BaseModel):
    name: str
    email: str
    age: int
```

Then:

```python
@app.post("/users")
async def create_user(user: UserCreate):
    return {
        "message": "User created",
        "user": user
    }
```

Client sends:

```json
{
    "name": "Sandeep",
    "email": "sandeep@example.com",
    "age": 37
}
```

FastAPI automatically:

1. Reads JSON
2. Validates it
3. Converts types
4. Creates the Pydantic object
5. Passes it to your function

---

# 5. Response models

You should also define what the API returns.

```python
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

    return UserResponse(
        id=101,
        name=user.name,
        email=user.email
    )
```

This provides a clear API contract.

---

# 6. PUT endpoint

PUT is commonly used to update/replace a resource.

```python
class UserUpdate(BaseModel):
    name: str
    email: str
    age: int
```

Endpoint:

```python
@app.put("/users/{user_id}")
async def update_user(
    user_id: int,
    user: UserUpdate
):
    return {
        "id": user_id,
        "name": user.name,
        "email": user.email,
        "age": user.age
    }
```

Request:

```http
PUT /users/101
```

Body:

```json
{
    "name": "Sandeep Swain",
    "email": "sandeep@example.com",
    "age": 37
}
```

---

# 7. PUT vs PATCH

This is an important interview question.

### PUT

Usually means:

> Replace/update the resource representation.

```http
PUT /users/101
```

```json
{
    "name": "Sandeep",
    "email": "new@example.com",
    "age": 37
}
```

### PATCH

Usually means:

> Partially update the resource.

```http
PATCH /users/101
```

```json
{
    "email": "new@example.com"
}
```

FastAPI supports PATCH as well:

```python
@app.patch("/users/{user_id}")
async def patch_user(
    user_id: int,
    user: UserUpdate
):
    ...
```

In real systems, PATCH is often preferable when only a few fields need changing.

---

# 8. DELETE endpoint

DELETE removes a resource.

```python
@app.delete("/users/{user_id}")
async def delete_user(user_id: int):

    # delete from database

    return {
        "message": "User deleted",
        "user_id": user_id
    }
```

Request:

```http
DELETE /users/101
```

---

# 9. Production FastAPI architecture

In a real project, I wouldn't put database logic directly inside the endpoint.

Instead:

```text
                    Client
                      │
                      ▼
                 FastAPI Router
                      │
                      ▼
                  Service
                      │
                      ▼
                Repository
                      │
                      ▼
                 PostgreSQL
```

For example:

```text
app/
├── main.py
├── routers/
│   └── users.py
├── services/
│   └── user_service.py
├── repositories/
│   └── user_repository.py
├── models/
│   └── user.py
└── schemas/
    └── user.py
```

### Router

```python
from fastapi import APIRouter, Depends

router = APIRouter(
    prefix="/users",
    tags=["Users"]
)


@router.get("/{user_id}")
async def get_user(
    user_id: int,
    service=Depends(get_user_service)
):
    return await service.get_user(user_id)
```

The router should primarily handle:

```text
HTTP concerns
   ↓
validation
   ↓
authentication
   ↓
call service
   ↓
return response
```

---

# 10. Service layer

```python
class UserService:

    def __init__(self, repository):
        self.repository = repository

    async def get_user(self, user_id: int):

        user = await self.repository.get_by_id(user_id)

        if not user:
            raise UserNotFoundException(user_id)

        return user
```

Business logic belongs here.

For example:

```text
Router
  ↓
UserService
  ├── validate business rules
  ├── authorization
  ├── transactions
  └── call repository
       ↓
UserRepository
       ↓
PostgreSQL
```

---

# 11. Production CRUD example

A simplified version would look like:

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()


class UserCreate(BaseModel):
    name: str
    email: str


class UserResponse(BaseModel):
    id: int
    name: str
    email: str


users = {}


@app.get("/users/{user_id}", response_model=UserResponse)
async def get_user(user_id: int):

    user = users.get(user_id)

    if not user:
        raise HTTPException(
            status_code=404,
            detail="User not found"
        )

    return user


@app.post(
    "/users",
    response_model=UserResponse,
    status_code=201
)
async def create_user(user: UserCreate):

    user_id = len(users) + 1

    new_user = {
        "id": user_id,
        "name": user.name,
        "email": user.email
    }

    users[user_id] = new_user

    return new_user


@app.put("/users/{user_id}")
async def update_user(
    user_id: int,
    user: UserCreate
):

    if user_id not in users:
        raise HTTPException(
            status_code=404,
            detail="User not found"
        )

    users[user_id].update({
        "name": user.name,
        "email": user.email
    })

    return users[user_id]


@app.delete("/users/{user_id}")
async def delete_user(user_id: int):

    if user_id not in users:
        raise HTTPException(
            status_code=404,
            detail="User not found"
        )

    del users[user_id]

    return {
        "message": "User deleted"
    }
```

In production, the `users` dictionary would be replaced with PostgreSQL/SQLAlchemy.

---

# 12. How this applies to an AI/RAG API

This is particularly relevant to the type of systems you've been preparing for.

You might have:

```text
POST /chat
POST /documents
GET  /documents/{document_id}
DELETE /documents/{document_id}
POST /search
GET  /conversations/{conversation_id}
```

For example:

### Upload document

```python
@router.post("/documents")
async def upload_document(
    file: UploadFile,
    service=Depends(get_document_service)
):
    return await service.ingest(file)
```

The service might execute:

```text
Upload
  ↓
Parse document
  ↓
Chunk
  ↓
Generate embeddings
  ↓
Store vectors in Qdrant
  ↓
Store metadata in PostgreSQL
```

### Ask question

```python
@router.post("/chat")
async def chat(
    request: ChatRequest,
    service=Depends(get_chat_service)
):
    return await service.chat(request)
```

The service:

```text
Question
   ↓
Query rewrite
   ↓
Hybrid retrieval
   ↓
Qdrant
   ↓
Reranker
   ↓
Context
   ↓
LLM
   ↓
Response
```

---

## Interview answer

If the interviewer asks:

**"How do you create GET, POST, PUT and DELETE endpoints in FastAPI?"**

I'd answer:

> "FastAPI provides decorators such as `@app.get`, `@app.post`, `@app.put`, and `@app.delete`. I use path parameters for resource identifiers, query parameters for filtering and pagination, and Pydantic models for request validation and response schemas. In production, I keep the endpoint thin and use a Router → Service → Repository architecture. The router handles HTTP concerns, the service contains business logic, and the repository handles database access. I also use dependency injection for database sessions, authentication and services, and return appropriate HTTP status codes such as 201 for creation and 404 when a resource isn't found."

That answer demonstrates **API knowledge + validation + architecture + production practices**, rather than just knowing the decorators.
