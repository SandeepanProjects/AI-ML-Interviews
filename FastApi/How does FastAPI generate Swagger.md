FastAPI generates **OpenAPI documentation automatically from your Python code, type hints, Pydantic models, and route definitions**.

This is one of FastAPI's biggest advantages for production APIs.

## 1. The basic idea

When you write:

```python
from fastapi import FastAPI

app = FastAPI()


@app.get("/users/{user_id}")
async def get_user(user_id: int):
    return {
        "id": user_id
    }
```

FastAPI inspects:

```text
@app.get(...)
       ↓
Route information

user_id: int
       ↓
Parameter type

Pydantic models
       ↓
Request/response schemas

status_code=...
       ↓
HTTP response information
```

and builds an **OpenAPI schema** automatically.

The architecture is roughly:

```text
Python Code
     │
     ├── Routes
     ├── Type hints
     ├── Pydantic models
     ├── Parameters
     └── Response models
            │
            ▼
       FastAPI
            │
            ▼
      OpenAPI Schema
            │
       ┌────┴─────┐
       ▼          ▼
   Swagger UI    ReDoc
     /docs       /redoc
```

---

# 2. Swagger UI

When you create:

```python
app = FastAPI()
```

FastAPI automatically exposes Swagger UI at:

```text
/docs
```

For example:

```text
http://localhost:8000/docs
```

You get an interactive API interface where you can:

* see endpoints
* inspect parameters
* see request schemas
* see response schemas
* execute API calls
* inspect responses
* understand validation requirements

You don't have to manually create the Swagger page.

---

# 3. ReDoc

FastAPI also provides ReDoc automatically:

```text
/redoc
```

So:

```text
http://localhost:8000/redoc
```

Both are generated from the same OpenAPI schema.

---

# 4. Where does the OpenAPI schema come from?

FastAPI exposes the generated OpenAPI JSON at:

```text
/openapi.json
```

For example:

```text
http://localhost:8000/openapi.json
```

You will see a JSON document describing your API.

Conceptually:

```json
{
  "openapi": "3.1.0",
  "paths": {
    "/users/{user_id}": {
      "get": {
        "parameters": [...]
      }
    }
  }
}
```

Swagger UI reads this OpenAPI specification and renders the interactive documentation.

So:

```text
FastAPI
   ↓
/openapi.json
   ↓
Swagger UI
/docs
```

---

# 5. Pydantic models automatically become schemas

This is where FastAPI becomes really powerful.

Suppose you define:

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
    return user
```

FastAPI understands:

```text
UserCreate
 ├── name: string
 ├── email: string
 └── age: integer
```

and automatically puts that information into OpenAPI.

Swagger UI can then show the expected request body.

---

# 6. Response models are also documented

Suppose:

```python
class UserResponse(BaseModel):
    id: int
    name: str
    email: str
```

Then:

```python
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

FastAPI documents:

```text
Response
 ├── id: integer
 ├── name: string
 └── email: string
```

This gives frontend and backend teams a clear API contract.

---

# 7. Path parameters are documented automatically

```python
@app.get("/users/{user_id}")
async def get_user(user_id: int):
    ...
```

FastAPI knows:

```text
Parameter:
user_id

Location:
path

Type:
integer

Required:
yes
```

Swagger UI displays this automatically.

---

# 8. Query parameters are documented too

```python
@app.get("/users")
async def get_users(
    page: int = 1,
    limit: int = 20
):
    ...
```

FastAPI generates documentation showing:

```text
page
Type: integer
Default: 1

limit
Type: integer
Default: 20
```

---

# 9. Add descriptions

You can make the documentation much more useful.

```python
@app.get(
    "/users/{user_id}",
    summary="Get a user",
    description="Retrieve a user by their unique ID."
)
async def get_user(user_id: int):
    ...
```

Swagger will show:

```text
Get a user

Retrieve a user by their unique ID.
```

---

# 10. Tags organize your APIs

For a production project, you don't want Swagger showing 100 endpoints in one giant list.

Use tags:

```python
@app.get(
    "/users/{user_id}",
    tags=["Users"]
)
async def get_user(user_id: int):
    ...
```

And:

```python
@app.post(
    "/documents",
    tags=["Documents"]
)
async def upload_document():
    ...
```

Swagger will organize them:

```text
Users
 ├── GET /users/{user_id}
 ├── POST /users
 └── DELETE /users/{user_id}

Documents
 ├── POST /documents
 ├── GET /documents
 └── DELETE /documents/{document_id}
```

---

# 11. Router-level tags

In a real application, you would normally use `APIRouter`.

```python
from fastapi import APIRouter

router = APIRouter(
    prefix="/users",
    tags=["Users"]
)
```

Then:

```python
@router.get("/{user_id}")
async def get_user(user_id: int):
    ...
```

And include the router:

```python
app.include_router(router)
```

FastAPI incorporates those routes into the OpenAPI specification.

This works well with:

```text
app/
├── main.py
├── routers/
│   ├── users.py
│   ├── documents.py
│   └── chat.py
├── schemas/
├── services/
└── repositories/
```

---

# 12. API metadata

You can configure the documentation at application level:

```python
app = FastAPI(
    title="Enterprise AI Platform",
    description="RAG and Agentic AI API",
    version="1.0.0"
)
```

Swagger/OpenAPI will then show:

```text
Enterprise AI Platform
RAG and Agentic AI API
Version: 1.0.0
```

You can also provide contact/license metadata in larger enterprise projects.

---

# 13. AI/RAG example

Imagine your production API has:

```python
@app.post(
    "/chat",
    response_model=ChatResponse,
    tags=["Chat"],
    summary="Ask a question"
)
async def chat(request: ChatRequest):
    ...
```

where:

```python
class ChatRequest(BaseModel):
    question: str
    conversation_id: str | None = None
    top_k: int = 5
```

and:

```python
class ChatResponse(BaseModel):
    answer: str
    sources: list[str]
    confidence: float
```

FastAPI can automatically document:

```text
POST /chat

Request body:
{
    "question": "string",
    "conversation_id": "string",
    "top_k": 5
}

Response:
{
    "answer": "string",
    "sources": ["string"],
    "confidence": 0.0
}
```

This is extremely useful when multiple teams consume your AI service.

---

# 14. Authentication can also be documented

FastAPI's security utilities integrate with OpenAPI.

For example, using OAuth2:

```python
from fastapi.security import OAuth2PasswordBearer

oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="token"
)
```

FastAPI can add the corresponding security scheme to the OpenAPI specification, allowing Swagger UI to provide authentication controls.

In an enterprise application, you might have:

```text
Swagger
   ↓
Authorize
   ↓
JWT token
   ↓
Call protected endpoint
```

This is very useful for testing authenticated APIs.

---

# 15. Can you customize or disable Swagger?

Yes.

You can change the documentation URLs:

```python
app = FastAPI(
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)
```

Or disable Swagger:

```python
app = FastAPI(
    docs_url=None,
    redoc_url=None
)
```

This can be useful when you don't want interactive documentation exposed publicly in a production environment.

---

# 16. Important distinction: Swagger vs OpenAPI

Interviewers sometimes ask this.

**OpenAPI** is the specification.

**Swagger UI** is a tool that renders that specification interactively.

Think:

```text
OpenAPI
   │
   │ specification
   ▼
/openapi.json
   │
   ▼
Swagger UI
/docs
```

So saying:

> "FastAPI generates Swagger"

is common shorthand, but technically:

> **FastAPI generates an OpenAPI schema, and Swagger UI uses that schema to provide interactive documentation.**

---

# 17. Interview answer

If the interviewer asks:

**"How does FastAPI generate Swagger/OpenAPI documentation?"**

A strong senior-level answer is:

> **"FastAPI generates an OpenAPI specification automatically by inspecting the application's route definitions, Python type hints, Pydantic request and response models, parameters, status codes, and security dependencies. The generated OpenAPI schema is available at `/openapi.json`. FastAPI then uses that schema to provide interactive Swagger UI at `/docs` and ReDoc at `/redoc`. In production, I use Pydantic response models, router tags, descriptions, and security schemes to make the API contract explicit and easy for other teams to consume."**

### Remember this flow

```text
FastAPI route
      +
Python type hints
      +
Pydantic models
      +
Security definitions
      ↓
OpenAPI specification
      ↓
/openapi.json
      ↓
┌──────────────┐
│ Swagger UI   │ → /docs
├──────────────┤
│ ReDoc        │ → /redoc
└──────────────┘
```

This is also why **FastAPI is particularly attractive for enterprise microservices and AI/RAG APIs**: your API contract is generated directly from the code rather than maintained separately.
