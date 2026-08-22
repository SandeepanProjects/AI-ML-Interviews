# FastAPI Exception Handling — Production-Level Approach

For a production FastAPI application, I recommend thinking about exception handling in layers:

```text
Client
   ↓
FastAPI Router
   ↓
Service
   ↓
Repository
   ↓
Database
```

Exceptions should be translated at the appropriate boundary:

```text
Database error
     ↓
Repository / Service
     ↓
Application exception
     ↓
FastAPI exception handler
     ↓
Consistent HTTP response
```

A good API should **not leak raw database errors, stack traces, SQL statements, or internal implementation details to the client**.

---

# 1. How do you handle exceptions globally?

FastAPI allows you to register global exception handlers using:

```python
@app.exception_handler(...)
```

For example:

```python
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()


@app.exception_handler(Exception)
async def global_exception_handler(
    request: Request,
    exc: Exception,
):
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error"
        },
    )
```

Now an unexpected exception anywhere in the request processing pipeline can be converted into a controlled HTTP response.

For example:

```python
@app.get("/users")
async def get_users():

    raise RuntimeError("Something went wrong")
```

Instead of returning a raw traceback to the client, the API returns:

```json
{
  "detail": "Internal server error"
}
```

### But don't stop here.

In production, you should also **log the exception internally**.

```python
import logging

logger = logging.getLogger(__name__)


@app.exception_handler(Exception)
async def global_exception_handler(
    request: Request,
    exc: Exception,
):
    logger.exception(
        "Unhandled exception",
        extra={
            "path": request.url.path,
            "method": request.method,
        },
    )

    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error"
        },
    )
```

The client gets:

```json
{
  "detail": "Internal server error"
}
```

while your logs contain the actual exception and traceback.

---

# 2. What is `HTTPException`?

`HTTPException` is FastAPI's built-in exception for returning an HTTP error response.

Example:

```python
from fastapi import HTTPException


@app.get("/users/{user_id}")
async def get_user(user_id: int):

    user = await get_user_from_db(user_id)

    if user is None:
        raise HTTPException(
            status_code=404,
            detail="User not found",
        )

    return user
```

The client receives:

```http
HTTP/1.1 404 Not Found
```

```json
{
  "detail": "User not found"
}
```

---

# 3. Why do we `raise HTTPException`?

You **raise** it:

```python
raise HTTPException(...)
```

You don't normally do:

```python
return HTTPException(...)
```

Because `HTTPException` is an exception that FastAPI knows how to handle.

Example:

```python
if not user:
    raise HTTPException(
        status_code=404,
        detail="User not found",
    )
```

FastAPI catches it and converts it into an HTTP response.

---

# 4. `HTTPException` vs normal Python exception

Normal exception:

```python
raise ValueError("Invalid user")
```

FastAPI doesn't automatically know that you want this to become:

```http
400 Bad Request
```

You can define a custom handler for it.

`HTTPException` already carries HTTP semantics:

```python
raise HTTPException(
    status_code=404,
    detail="User not found",
)
```

---

# 5. How do you create custom exception handlers?

Suppose you define your own application exception:

```python
class UserNotFoundError(Exception):
    pass
```

Then register a handler:

```python
@app.exception_handler(UserNotFoundError)
async def user_not_found_handler(
    request: Request,
    exc: UserNotFoundError,
):
    return JSONResponse(
        status_code=404,
        content={
            "detail": "User not found"
        },
    )
```

Now your service can simply do:

```python
class UserService:

    async def get_user(
        self,
        user_id: int,
    ):

        user = await self.repository.get_by_id(
            user_id
        )

        if user is None:
            raise UserNotFoundError()

        return user
```

The service doesn't need to know about HTTP.

That's an important architectural benefit.

---

# 6. Why shouldn't the Service Layer raise `HTTPException`?

Consider:

```python
class UserService:

    async def get_user(self, user_id):

        user = ...

        if user is None:
            raise HTTPException(
                status_code=404,
                detail="User not found",
            )
```

This couples your service to FastAPI.

That's not ideal.

Instead:

```python
class UserNotFoundError(Exception):
    pass
```

Then:

```python
class UserService:

    async def get_user(self, user_id):

        user = ...

        if user is None:
            raise UserNotFoundError()

        return user
```

The FastAPI layer translates it:

```text
UserNotFoundError
        ↓
FastAPI Exception Handler
        ↓
HTTP 404
```

This gives you:

```text
Business layer
    ↓
Framework-independent exceptions

API layer
    ↓
HTTP representation
```

---

# 7. Create an application exception hierarchy

For a larger application, I like having a common base exception:

```python
class AppException(Exception):

    def __init__(
        self,
        message: str,
        code: str,
    ):
        self.message = message
        self.code = code

        super().__init__(message)
```

Then:

```python
class UserNotFoundError(AppException):

    def __init__(self):
        super().__init__(
            message="User not found",
            code="USER_NOT_FOUND",
        )
```

Another:

```python
class UserAlreadyExistsError(AppException):

    def __init__(self):
        super().__init__(
            message="User already exists",
            code="USER_ALREADY_EXISTS",
        )
```

---

# 8. Create a global application exception handler

```python
@app.exception_handler(AppException)
async def app_exception_handler(
    request: Request,
    exc: AppException,
):
    return JSONResponse(
        status_code=400,
        content={
            "error": {
                "code": exc.code,
                "message": exc.message,
            }
        },
    )
```

But in a real application, I'd normally associate an HTTP status with the exception too.

For example:

```python
class AppException(Exception):

    def __init__(
        self,
        message: str,
        code: str,
        status_code: int,
    ):
        self.message = message
        self.code = code
        self.status_code = status_code

        super().__init__(message)
```

Then:

```python
class UserNotFoundError(AppException):

    def __init__(self):

        super().__init__(
            message="User not found",
            code="USER_NOT_FOUND",
            status_code=404,
        )
```

And:

```python
class UserAlreadyExistsError(AppException):

    def __init__(self):

        super().__init__(
            message="User already exists",
            code="USER_ALREADY_EXISTS",
            status_code=409,
        )
```

Handler:

```python
@app.exception_handler(AppException)
async def app_exception_handler(
    request: Request,
    exc: AppException,
):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.code,
                "message": exc.message,
            }
        },
    )
```

---

# 9. How do you return consistent error responses?

This is very important for production APIs.

Don't have one endpoint return:

```json
{
  "error": "User not found"
}
```

another:

```json
{
  "message": "Invalid user"
}
```

and another:

```json
{
  "detail": "Something went wrong"
}
```

Instead, define a consistent structure.

For example:

```json
{
  "error": {
    "code": "USER_NOT_FOUND",
    "message": "User not found",
    "request_id": "abc-123"
  }
}
```

You could define a Pydantic model:

```python
from pydantic import BaseModel


class ErrorResponse(BaseModel):

    code: str
    message: str
    request_id: str | None = None
```

Then return:

```python
return JSONResponse(
    status_code=404,
    content={
        "error": {
            "code": "USER_NOT_FOUND",
            "message": "User not found",
            "request_id": request_id,
        }
    },
)
```

---

# 10. Why include a request ID?

Suppose the user gets:

```json
{
  "error": {
    "code": "INTERNAL_ERROR",
    "message": "Internal server error",
    "request_id": "abc-123"
  }
}
```

They can give support:

> "Request `abc-123` failed."

Your logs can then find:

```text
request_id=abc-123
```

and show:

```text
API request
    ↓
PostgreSQL
    ↓
Redis
    ↓
Qdrant
    ↓
LLM
    ↓
Exception
```

This is especially useful in production AI systems.

---

# 11. How do you handle database exceptions?

This is a critical production concern.

Suppose PostgreSQL raises:

```python
IntegrityError
```

For example, a unique constraint violation:

```text
users.email UNIQUE
```

and you try:

```text
john@example.com
```

when it already exists.

SQLAlchemy might raise:

```python
from sqlalchemy.exc import IntegrityError
```

You should **not return the raw exception**.

Bad:

```python
except IntegrityError as exc:
    raise exc
```

because the client may get internal database information.

---

# 12. Handle `IntegrityError`

For example:

```python
from sqlalchemy.exc import IntegrityError


try:

    await repository.create(user)
    await db.commit()

except IntegrityError:

    await db.rollback()

    raise UserAlreadyExistsError()
```

Then your global handler converts:

```text
IntegrityError
      ↓
UserAlreadyExistsError
      ↓
HTTP 409
```

Client:

```json
{
  "error": {
    "code": "USER_ALREADY_EXISTS",
    "message": "User already exists"
  }
}
```

---

# 13. Always rollback after a failed transaction

This is extremely important with SQLAlchemy.

If:

```python
await db.commit()
```

fails, you generally need:

```python
await db.rollback()
```

before continuing to use that session.

Example:

```python
try:

    await db.commit()

except IntegrityError:

    await db.rollback()

    raise
```

Otherwise the SQLAlchemy session may remain in a failed transaction state.

---

# 14. Database exception categories

Common SQLAlchemy exceptions include:

```text
IntegrityError
OperationalError
DBAPIError
SQLAlchemyError
```

For example:

### `IntegrityError`

Usually:

```text
Unique constraint
Foreign key constraint
NOT NULL constraint
```

Often maps to:

```text
409 Conflict
```

depending on the exact situation.

---

### `OperationalError`

Could indicate:

```text
Database unavailable
Connection problem
Network issue
Database server failure
```

This is generally an internal infrastructure problem.

Typically:

```text
500
```

or sometimes:

```text
503 Service Unavailable
```

depending on your API's semantics.

---

# 15. Don't catch every database exception and return 500 blindly

For example:

```python
except SQLAlchemyError:
    raise HTTPException(
        status_code=500,
        detail="Database error",
    )
```

This hides useful distinctions.

Better:

```text
IntegrityError
     ↓
409 / domain-specific conflict

Connection failure
     ↓
503 or 500

Unexpected DB error
     ↓
500
```

And always log the original exception internally.

---

# 16. 400 vs 401 vs 403 vs 404 vs 409 vs 500

This is one of the most important interview questions.

---

## 400 — Bad Request

Means:

> **The request is invalid or cannot be processed as sent.**

Example:

```http
POST /users
```

with malformed application-level input.

For example:

```json
{
  "age": -100
}
```

Or an invalid request parameter depending on your API semantics.

FastAPI/Pydantic validation errors are often represented as **422 Unprocessable Entity** in FastAPI, rather than 400.

That's an important FastAPI-specific detail.

---

# 17. 401 — Unauthorized

This name is confusing.

`401` generally means:

> **The client has not successfully authenticated.**

Examples:

```text
No access token
Invalid access token
Expired access token
Malformed token
```

Example:

```python
raise HTTPException(
    status_code=401,
    detail="Invalid or expired token",
)
```

Usually include:

```http
WWW-Authenticate: Bearer
```

when using Bearer authentication.

---

# 18. 403 — Forbidden

Means:

> **The user is authenticated, but isn't allowed to perform this operation.**

Example:

```text
User:
role = USER

Endpoint:
POST /admin/users

Required:
role = ADMIN
```

The user is authenticated:

```text
JWT ✓
```

But authorization fails:

```text
ADMIN ✗
```

Return:

```http
403 Forbidden
```

Example:

```python
raise HTTPException(
    status_code=403,
    detail="Insufficient permissions",
)
```

---

# 19. 404 — Not Found

Means:

> **The requested resource does not exist or is intentionally not exposed.**

Example:

```http
GET /users/999
```

where user `999` doesn't exist.

```python
raise HTTPException(
    status_code=404,
    detail="User not found",
)
```

---

# 20. 409 — Conflict

Means:

> **The request conflicts with the current state of the resource.**

Classic example:

```text
Create user
email = john@example.com
```

but that email already exists.

Database:

```text
UNIQUE(email)
```

Result:

```http
409 Conflict
```

Example:

```python
raise HTTPException(
    status_code=409,
    detail="User already exists",
)
```

Other examples:

```text
Duplicate resource
Version conflict
State transition conflict
```

---

# 21. 500 — Internal Server Error

Means:

> **Something unexpected went wrong on the server.**

Examples:

```text
Unexpected Python exception
Bug
Unexpected database error
Unexpected third-party failure
```

Don't expose:

```python
str(exc)
```

to the client.

Bad:

```json
{
  "error": "psycopg2.errors.UniqueViolation: ..."
}
```

Better:

```json
{
  "error": {
    "code": "INTERNAL_ERROR",
    "message": "Internal server error",
    "request_id": "abc-123"
  }
}
```

Log the actual exception internally.

---

# 22. Quick comparison

| Status  | Meaning                 | Example                     |
| ------- | ----------------------- | --------------------------- |
| **400** | Bad request             | Invalid application request |
| **401** | Not authenticated       | Missing/invalid JWT         |
| **403** | Not authorized          | User isn't admin            |
| **404** | Resource doesn't exist  | User ID not found           |
| **409** | State conflict          | Duplicate email             |
| **500** | Unexpected server error | Unhandled exception         |

And for FastAPI specifically:

```text
422
↓
Request validation failed
```

For example:

```python
class UserCreate(BaseModel):
    email: EmailStr
    age: int
```

Request:

```json
{
  "email": "not-an-email",
  "age": "hello"
}
```

FastAPI/Pydantic will normally return a validation error response with status **422**.

---

# 23. A production exception architecture

I would structure it like:

```text
app/
│
├── exceptions/
│   ├── base.py
│   ├── auth.py
│   └── users.py
│
├── handlers/
│   └── exception_handlers.py
│
├── services/
│   └── user_service.py
│
└── repositories/
    └── user_repository.py
```

---

# 24. Base application exception

```python
class AppException(Exception):

    def __init__(
        self,
        message: str,
        code: str,
        status_code: int,
    ):
        self.message = message
        self.code = code
        self.status_code = status_code

        super().__init__(message)
```

---

# 25. Domain exceptions

```python
class UserNotFoundError(AppException):

    def __init__(self):

        super().__init__(
            message="User not found",
            code="USER_NOT_FOUND",
            status_code=404,
        )
```

```python
class UserAlreadyExistsError(AppException):

    def __init__(self):

        super().__init__(
            message="User already exists",
            code="USER_ALREADY_EXISTS",
            status_code=409,
        )
```

```python
class InsufficientPermissionsError(AppException):

    def __init__(self):

        super().__init__(
            message="Insufficient permissions",
            code="INSUFFICIENT_PERMISSIONS",
            status_code=403,
        )
```

---

# 26. Global handler

```python
import logging

from fastapi import Request
from fastapi.responses import JSONResponse


logger = logging.getLogger(__name__)


@app.exception_handler(AppException)
async def app_exception_handler(
    request: Request,
    exc: AppException,
):

    request_id = getattr(
        request.state,
        "request_id",
        None,
    )

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.code,
                "message": exc.message,
                "request_id": request_id,
            }
        },
    )
```

---

# 27. Global unexpected exception handler

```python
@app.exception_handler(Exception)
async def unexpected_exception_handler(
    request: Request,
    exc: Exception,
):

    request_id = getattr(
        request.state,
        "request_id",
        None,
    )

    logger.exception(
        "Unhandled exception",
        extra={
            "request_id": request_id,
            "path": request.url.path,
            "method": request.method,
        },
    )

    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": "INTERNAL_ERROR",
                "message": "Internal server error",
                "request_id": request_id,
            }
        },
    )
```

This gives you:

```text
Expected application exception
        ↓
Known status code
        ↓
Consistent response


Unexpected exception
        ↓
Log full traceback
        ↓
Generic 500 response
```

---

# 28. Database handling in the Service Layer

Example:

```python
from sqlalchemy.exc import IntegrityError


class UserService:

    def __init__(
        self,
        repository,
        db,
    ):
        self.repository = repository
        self.db = db

    async def create_user(
        self,
        email: str,
        name: str,
    ):

        user = User(
            email=email,
            name=name,
        )

        try:

            await self.repository.create(
                user
            )

            await self.db.commit()

            return user

        except IntegrityError:

            await self.db.rollback()

            raise UserAlreadyExistsError()
```

This gives:

```text
PostgreSQL
    ↓
IntegrityError
    ↓
rollback()
    ↓
UserAlreadyExistsError
    ↓
Global handler
    ↓
409 Conflict
```

---

# 29. What should the client see?

For duplicate user:

```json
{
  "error": {
    "code": "USER_ALREADY_EXISTS",
    "message": "User already exists",
    "request_id": "abc-123"
  }
}
```

For authentication failure:

```json
{
  "error": {
    "code": "INVALID_TOKEN",
    "message": "Authentication required",
    "request_id": "abc-123"
  }
}
```

For authorization failure:

```json
{
  "error": {
    "code": "INSUFFICIENT_PERMISSIONS",
    "message": "Insufficient permissions",
    "request_id": "abc-123"
  }
}
```

For unexpected failure:

```json
{
  "error": {
    "code": "INTERNAL_ERROR",
    "message": "Internal server error",
    "request_id": "abc-123"
  }
}
```

Notice that the client gets a **safe, predictable response**, while your logs retain the technical details.

---

# 30. Exception flow in a production FastAPI application

The complete architecture looks like:

```text
                   HTTP Request
                        │
                        ↓
                  FastAPI Router
                        │
                        ↓
                  Service Layer
                        │
               ┌────────┴────────┐
               ↓                 ↓
        Domain exception      Repository
               │                 │
               │                 ↓
               │             PostgreSQL
               │                 │
               │          IntegrityError
               │                 │
               │                 ↓
               │              rollback
               │                 │
               └────────┬────────┘
                        ↓
               Exception Handler
                        │
                        ↓
               Consistent JSON
                        │
                        ↓
                     Client
```

---

# 31. Senior-level interview answer

If asked **"How do you handle exceptions globally in FastAPI?"**, answer:

> **"I define domain/application exceptions separately from HTTP concerns, register global FastAPI exception handlers, and translate known exceptions into consistent HTTP responses. Unexpected exceptions are logged with the correlation ID and returned as a generic 500 response without exposing internal details. For database errors, I catch expected SQLAlchemy exceptions such as `IntegrityError`, rollback the transaction, translate them into a domain exception such as a conflict, and let the global handler generate the response."**

If asked **"How do you distinguish 400, 401, 403, 404, 409 and 500?"**, remember:

```text
400 → Request is invalid
401 → Not authenticated
403 → Authenticated but not allowed
404 → Resource not found
409 → Resource/state conflict
500 → Unexpected server failure
```

And for FastAPI:

```text
422 → Pydantic/request validation failure
```

### The architecture to remember

> **Don't scatter `try/except` and `HTTPException` everywhere. Keep business exceptions in the service/domain layer, translate them centrally at the FastAPI boundary, rollback database transactions when necessary, log the technical details internally, and return a consistent error contract to clients.**
