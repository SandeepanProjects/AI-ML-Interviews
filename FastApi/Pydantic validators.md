You add **custom validators in Pydantic** when built-in constraints such as `min_length`, `ge`, `le`, etc. are not enough.

In FastAPI, the flow is:

```text
HTTP Request
     ↓
FastAPI
     ↓
Pydantic Model
     ↓
Built-in validation
     ↓
Custom validation
     ↓
Endpoint
```

The important thing is that **FastAPI itself doesn't normally contain the validation logic**. You put the validation rules in the Pydantic model.

---

# 1. Simple custom validator

With modern Pydantic v2, use `@field_validator`.

```python
from fastapi import FastAPI
from pydantic import BaseModel, field_validator

app = FastAPI()


class UserCreate(BaseModel):
    username: str
    age: int

    @field_validator("username")
    @classmethod
    def validate_username(cls, value: str) -> str:
        if " " in value:
            raise ValueError("Username cannot contain spaces")

        return value
```

Then:

```python
@app.post("/users")
async def create_user(user: UserCreate):
    return user
```

If the client sends:

```json
{
    "username": "sandeep swain",
    "age": 37
}
```

Pydantic raises a validation error because the username contains a space.

---

# 2. Why `@field_validator`?

Suppose you have:

```python
class UserCreate(BaseModel):
    username: str
```

You can use built-in validation:

```python
username: str = Field(min_length=3)
```

But suppose your business rule is:

> Username must contain only letters, numbers and underscores.

That's custom validation.

```python
from pydantic import BaseModel, field_validator
import re


class UserCreate(BaseModel):
    username: str

    @field_validator("username")
    @classmethod
    def validate_username(cls, value: str) -> str:

        if not re.match(r"^[a-zA-Z0-9_]+$", value):
            raise ValueError(
                "Username can contain only letters, numbers and underscores"
            )

        return value
```

---

# 3. Validator can also normalize data

A validator doesn't have to only reject data.

It can also transform it.

For example, normalize an email:

```python
from pydantic import BaseModel, field_validator


class UserCreate(BaseModel):
    email: str

    @field_validator("email")
    @classmethod
    def normalize_email(cls, value: str) -> str:
        return value.strip().lower()
```

Input:

```json
{
    "email": "  SANDEEP@EXAMPLE.COM "
}
```

After Pydantic validation:

```python
user.email
```

becomes:

```text
sandeep@example.com
```

This is useful for normalization.

---

# 4. `mode="before"` vs `mode="after"`

This is an important interview topic.

By default:

```python
@field_validator("email")
```

runs **after Pydantic has performed its normal parsing/validation**.

You can explicitly say:

```python
@field_validator("email", mode="after")
```

Example:

```python
class User(BaseModel):
    age: int

    @field_validator("age", mode="after")
    @classmethod
    def validate_age(cls, value: int) -> int:
        if value < 18:
            raise ValueError("User must be at least 18")

        return value
```

### `mode="before"`

Use this when you want to process the raw input before Pydantic's normal parsing.

```python
class User(BaseModel):
    age: int

    @field_validator("age", mode="before")
    @classmethod
    def normalize_age(cls, value):
        if isinstance(value, str):
            value = value.strip()

        return value
```

Flow:

```text
Input
 ↓
before validator
 ↓
Pydantic parsing/type validation
 ↓
after validator
 ↓
Validated model
```

---

# 5. Validating multiple fields together

Sometimes the validation rule depends on **multiple fields**.

For example:

> `password` and `confirm_password` must match.

For this, Pydantic v2 provides `@model_validator`.

```python
from pydantic import BaseModel, model_validator


class UserCreate(BaseModel):
    password: str
    confirm_password: str

    @model_validator(mode="after")
    def validate_passwords(self):
        if self.password != self.confirm_password:
            raise ValueError("Passwords do not match")

        return self
```

Request:

```json
{
    "password": "secret123",
    "confirm_password": "secret456"
}
```

Validation fails.

This is different from:

```python
@field_validator("password")
```

because the rule involves **two fields**.

---

# 6. Another multi-field example

Imagine an LLM API:

```python
class ChatRequest(BaseModel):
    temperature: float
    top_p: float

    @model_validator(mode="after")
    def validate_generation_parameters(self):

        if self.temperature == 0 and self.top_p != 1:
            raise ValueError(
                "When temperature is 0, top_p must be 1"
            )

        return self
```

Now your API can enforce an application-specific rule before the request reaches your LLM service.

---

# 7. Custom validator with FastAPI

Complete example:

```python
from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator

app = FastAPI()


class UserCreate(BaseModel):
    username: str = Field(min_length=3)
    email: str
    age: int = Field(ge=18)

    @field_validator("username")
    @classmethod
    def validate_username(cls, value: str) -> str:

        value = value.strip()

        if " " in value:
            raise ValueError(
                "Username cannot contain spaces"
            )

        return value

    @field_validator("email")
    @classmethod
    def validate_email(cls, value: str) -> str:

        value = value.strip().lower()

        if not value.endswith("@example.com"):
            raise ValueError(
                "Only example.com emails are allowed"
            )

        return value


@app.post("/users")
async def create_user(user: UserCreate):
    return {
        "username": user.username,
        "email": user.email,
        "age": user.age
    }
```

The client sends:

```json
{
    "username": "  sandeep123 ",
    "email": " SANDEEP@EXAMPLE.COM ",
    "age": 37
}
```

Pydantic normalizes it to something like:

```json
{
    "username": "sandeep123",
    "email": "sandeep@example.com",
    "age": 37
}
```

---

# 8. What happens when validation fails?

Suppose:

```json
{
    "username": "ab",
    "email": "test@gmail.com",
    "age": 15
}
```

You have:

```python
username: str = Field(min_length=3)
age: int = Field(ge=18)
```

FastAPI automatically returns a validation error response, typically:

```text
422 Unprocessable Entity
```

with details about the invalid fields.

You don't normally need to write:

```python
try:
    ...
except ValidationError:
    ...
```

inside every endpoint.

That's one of the major benefits of putting validation in Pydantic.

---

# 9. Custom reusable validation

Suppose multiple models need the same rule.

You can create reusable validators or custom types instead of duplicating logic.

For example, using `Annotated` and a reusable validator:

```python
from typing import Annotated
from pydantic import AfterValidator, BaseModel


def normalize_username(value: str) -> str:
    value = value.strip().lower()

    if not value.isalnum():
        raise ValueError("Username must be alphanumeric")

    return value


Username = Annotated[
    str,
    AfterValidator(normalize_username)
]


class UserCreate(BaseModel):
    username: Username


class AdminCreate(BaseModel):
    username: Username
```

Now both models share the same validation rule.

This is useful in large applications.

---

# 10. Don't put business logic into validators

This is a **very important senior-level distinction**.

Good validator:

```python
@field_validator("email")
@classmethod
def validate_email(cls, value):
    if "@" not in value:
        raise ValueError("Invalid email")

    return value
```

Bad idea:

```python
@field_validator("email")
@classmethod
def validate_email(cls, value):

    # Don't query PostgreSQL here
    user = database.find_user_by_email(value)

    ...
```

Validators should generally handle:

```text
Input shape
Normalization
Field constraints
Cross-field consistency
```

They should not handle:

```text
Database queries
LLM calls
External APIs
Complex business workflows
Authorization
Transactions
```

Those belong in the service/domain layer.

---

# 11. Good production architecture

For a FastAPI application:

```text
                 HTTP Request
                      ↓
                   FastAPI
                      ↓
              Pydantic Model
                      ↓
             ┌────────────────┐
             │ Validation     │
             │ Normalization  │
             │ Cross-fields   │
             └───────┬────────┘
                     ↓
                  Router
                     ↓
                  Service
                     ↓
              Repository / APIs
```

For example:

```python
@app.post("/chat")
async def chat(request: ChatRequest):
    return await chat_service.generate(request)
```

The endpoint remains thin.

---

# 12. Custom validation in an AI/RAG API

This is where you can give a strong interview example.

```python
from pydantic import BaseModel, Field, model_validator


class RAGRequest(BaseModel):
    query: str = Field(min_length=1, max_length=5000)
    top_k: int = Field(default=5, ge=1, le=20)
    score_threshold: float | None = Field(
        default=None,
        ge=0,
        le=1
    )

    @model_validator(mode="after")
    def validate_search_config(self):

        if self.score_threshold is not None and self.top_k > 10:
            raise ValueError(
                "score_threshold cannot be combined with top_k > 10"
            )

        return self
```

Then:

```python
@app.post("/search")
async def search(request: RAGRequest):

    results = await rag_service.search(
        query=request.query,
        top_k=request.top_k,
        score_threshold=request.score_threshold
    )

    return results
```

This ensures invalid RAG configurations never reach Qdrant or your retrieval service.

---

# 13. Pydantic v1 vs v2

This can come up in interviews.

### Pydantic v2

Use:

```python
@field_validator(...)
```

and:

```python
@model_validator(...)
```

### Older Pydantic v1

You may see:

```python
@validator(...)
```

and:

```python
@root_validator(...)
```

For modern FastAPI projects, **Pydantic v2 syntax is what you should know**.

---

# Interview answer

If asked:

**"How do you add custom validators with Pydantic and FastAPI?"**

A strong answer is:

> **"I use Pydantic's `@field_validator` for validation or normalization of individual fields and `@model_validator` when a rule depends on multiple fields. I use `mode='before'` when I need to transform raw input before normal Pydantic parsing and `mode='after'` when I want to validate the parsed value. FastAPI automatically executes these validators when it creates the request model and returns a validation error for invalid input. I keep validators focused on input validation and normalization and keep database calls and business logic in the service layer."**

### Remember this:

```text
One field
   ↓
@field_validator

Multiple fields
   ↓
@model_validator

Raw input
   ↓
mode="before"

Parsed value
   ↓
mode="after"
```

And the senior-level rule:

> **Pydantic validators validate data; service classes execute business logic.**
