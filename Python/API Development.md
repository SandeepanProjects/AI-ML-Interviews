# API Development (Senior AI Engineer Interview)

This is one of the **most important interview topics** for Senior AI Engineers.

A production AI system is much more than an ML model. It typically exposes capabilities through APIs that handle authentication, validation, rate limiting, logging, monitoring, and dependency management.

A typical production architecture looks like:

```text
                Client

                  │
                  ▼
           Load Balancer
                  │
                  ▼
             FastAPI Server
                  │
        ┌─────────┴─────────┐
        ▼                   ▼
 Authentication        Middleware
        │                   │
        └─────────┬─────────┘
                  ▼
          Dependency Injection
                  ▼
             Business Logic
                  ▼
        PostgreSQL / Redis / LLM
```

Let's go through each topic in detail.

---

# What is a REST API?

REST (Representational State Transfer) is an architectural style for exposing resources over HTTP.

Example:

```text
/users
/orders
/chat
/documents
```

Instead of calling Python functions directly:

```python
create_user()
```

Clients send HTTP requests:

```http
POST /users
```

Server processes the request and returns JSON.

---

# HTTP Methods

| Method | Purpose        |
| ------ | -------------- |
| GET    | Read data      |
| POST   | Create         |
| PUT    | Replace        |
| PATCH  | Partial update |
| DELETE | Remove         |

Example:

```http
GET /users/10
```

Returns:

```json
{
  "id":10,
  "name":"Alice"
}
```

---

# Building a REST API with FastAPI

Install:

```bash
pip install fastapi uvicorn
```

Simple API:

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Hello"}
```

Run:

```bash
uvicorn main:app --reload
```

Visit:

```
http://localhost:8000/
```

Response:

```json
{
    "message":"Hello"
}
```

---

# Path Parameters

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/users/{user_id}")
def get_user(user_id: int):
    return {"user": user_id}
```

Request:

```
GET /users/100
```

Response:

```json
{
    "user":100
}
```

FastAPI automatically converts:

```text
"100"

↓

100
```

---

# Query Parameters

```python
@app.get("/search")
def search(query: str):
    return {"query": query}
```

Request:

```
/search?query=python
```

Response:

```json
{
  "query":"python"
}
```

---

# Request Body

Use Pydantic models.

```python
from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int
```

API:

```python
@app.post("/users")
def create_user(user: User):
    return user
```

Request:

```json
{
    "name":"Alice",
    "age":30
}
```

FastAPI automatically:

* Parses JSON
* Validates types
* Generates documentation

---

# Why FastAPI?

FastAPI advantages:

* Async support
* Type hints
* Automatic validation
* OpenAPI generation
* Swagger UI
* Excellent performance

---

# Flask vs FastAPI

Flask:

```python
from flask import Flask

app = Flask(__name__)
```

FastAPI:

```python
from fastapi import FastAPI

app = FastAPI()
```

FastAPI automatically provides:

```
/docs
```

Interactive Swagger UI.

Flask requires additional libraries.

---

# FastAPI vs Flask

| Feature       | Flask    | FastAPI     |
| ------------- | -------- | ----------- |
| Async         | Limited  | Native      |
| Validation    | Manual   | Automatic   |
| Documentation | Plugins  | Automatic   |
| Type hints    | Optional | First-class |
| Performance   | Good     | Excellent   |
| OpenAPI       | Plugins  | Built-in    |

For modern AI services, FastAPI is generally preferred.

---

# Authentication

Authentication answers:

> Who are you?

Authorization answers:

> What are you allowed to do?

Example:

```
User

↓

Login

↓

Token

↓

Protected API
```

---

# JWT (JSON Web Token)

Instead of storing sessions on the server:

```
User

↓

Login

↓

JWT

↓

Client stores token

↓

Future requests include token
```

Header:

```
Authorization: Bearer <token>
```

---

# JWT Flow

```
User Login
     │
     ▼
Validate username/password
     │
     ▼
Generate JWT
     │
     ▼
Client stores JWT
     │
     ▼
Future API calls include JWT
```

Server verifies the token before processing the request.

---

# JWT Example

Login:

```python
@app.post("/login")
def login():
    token = create_access_token(
        {"sub": "alice"}
    )

    return {
        "access_token": token
    }
```

Protected endpoint:

```python
@app.get("/profile")
def profile(user=Depends(get_current_user)):
    return user
```

---

# Why JWT?

Advantages:

* Stateless
* Scalable
* No server-side session storage
* Easy for microservices

---

# Middleware

Middleware runs before and/or after every request.

Flow:

```
Request

↓

Middleware

↓

API

↓

Middleware

↓

Response
```

---

## Example

```python
from fastapi import Request

@app.middleware("http")
async def log_requests(request: Request, call_next):

    print(request.url)

    response = await call_next(request)

    return response
```

Every request passes through the middleware.

---

# What Middleware Is Used For

Production middleware commonly handles:

* Logging
* Authentication
* Request IDs
* CORS
* Metrics
* Timing
* Compression
* Security headers

---

# Request Timing Middleware

```python
import time

@app.middleware("http")
async def timing(request, call_next):

    start = time.time()

    response = await call_next(request)

    print(time.time() - start)

    return response
```

Useful for latency monitoring.

---

# Dependency Injection (DI)

One of FastAPI's most powerful features.

Without DI:

```python
def endpoint():

    db = Database()

    ...
```

Every endpoint creates its own dependencies.

With DI:

```python
from fastapi import Depends

def get_db():
    return Database()

@app.get("/users")
def users(db=Depends(get_db)):
    ...
```

FastAPI injects the dependency.

---

# Why Dependency Injection?

Benefits:

* Loose coupling
* Easier testing
* Reusable components
* Better maintainability

---

# AI Example

```
API

↓

Depends(get_llm)

↓

LLM Service

↓

OpenAI
```

Instead of:

```python
llm = OpenAI(...)
```

inside every endpoint.

---

# Dependency Injection for Authentication

```python
def get_current_user():

    ...

@app.get("/me")
def me(
    user=Depends(get_current_user)
):
    return user
```

Authentication logic is shared across endpoints.

---

# Dependency Injection for Database

```python
def get_db():

    db = Session()

    try:
        yield db

    finally:
        db.close()
```

FastAPI automatically closes the connection.

---

# AI Service Example

```
Client

↓

POST /chat

↓

Authentication

↓

Rate Limiter

↓

Middleware

↓

Dependency Injection

↓

LLM Service

↓

Redis Cache

↓

OpenAI

↓

Response
```

---

# Production REST API Best Practices

## Validation

Never trust client input.

Use Pydantic models.

---

## Authentication

Protect every private endpoint.

---

## Logging

Log:

* Request ID
* User ID
* Latency
* Errors

---

## Exception Handling

Use global exception handlers.

Never expose stack traces.

---

## Rate Limiting

Prevent abuse.

Example:

```
100 requests/minute
```

---

## Dependency Injection

Inject:

* Database
* Redis
* LLM
* Configuration
* Authentication

---

## Async

Use async for:

* Database
* HTTP requests
* LLM APIs
* Redis
* S3

Avoid blocking operations in async endpoints.

---

# AI Chat API Example

```
POST /chat

↓

JWT Authentication

↓

Retrieve Documents

↓

Generate Embeddings

↓

Vector Search

↓

LLM

↓

Return Answer
```

---

# Flask vs FastAPI (Interview Answer)

**When would you choose Flask?**

* Small applications
* Legacy projects
* Simple APIs

**When would you choose FastAPI?**

* AI inference services
* High-performance APIs
* Async workloads
* Modern microservices
* Automatic API documentation

---

# Senior AI Engineer Interview Questions

## Why FastAPI instead of Flask?

A good answer:

> FastAPI provides native async support, automatic request validation through Pydantic, OpenAPI documentation generation, dependency injection, and higher performance, making it well suited for production AI services.

---

## What is Dependency Injection?

> Dependency Injection supplies required objects such as database sessions, authentication services, or LLM clients to functions instead of creating them internally. This reduces coupling, improves testability, and promotes code reuse.

---

## Why use Middleware?

> Middleware centralizes cross-cutting concerns such as logging, authentication, metrics, request tracing, CORS handling, and latency measurement without duplicating code across endpoints.

---

## Why JWT?

> JWT enables stateless authentication. The server verifies the signed token on each request instead of storing session state, making it scalable for distributed systems and microservices.

---

## REST API Best Practices

* Use appropriate HTTP methods.
* Return consistent HTTP status codes.
* Validate all input.
* Authenticate protected endpoints.
* Log requests and errors.
* Handle exceptions globally.
* Use dependency injection.
* Keep endpoints thin and move business logic into services.
* Version APIs when introducing breaking changes (e.g., `/api/v1/...`).

---

# Production AI Architecture

```text
                 Client
                    │
                    ▼
             API Gateway
                    │
                    ▼
              FastAPI Service
                    │
      ┌─────────────┼─────────────┐
      ▼             ▼             ▼
 Authentication  Middleware   Rate Limiter
      │
      ▼
 Dependency Injection
      │
      ├──────────► PostgreSQL
      ├──────────► Redis
      ├──────────► Vector Database
      └──────────► LLM Provider
                    │
                    ▼
             JSON Response
```

This architecture is typical for production AI applications such as RAG systems, chatbots, recommendation engines, and model inference APIs, and it demonstrates the concepts interviewers expect from a senior engineer.
