# Pydantic Explained Properly (Production Guide with Code)

One of the most common interview questions for **Senior Python**, **FastAPI**, **AI Engineer**, and **LLM Engineer** roles is:

> **What is Pydantic? Why do we use it? How is it different from TypedDict?**

Many people answer:

> "Pydantic is used for validation."

That is only a small part of the story.

Pydantic is the **data validation and serialization layer** for many modern Python applications, especially **FastAPI**, **LangChain**, **LangGraph**, and AI platforms.

---

# Why Do We Need Pydantic?

Imagine you have an API:

```text
POST /users
```

Client sends:

```json
{
    "name":"Alice",
    "age":"25",
    "email":"alice@gmail.com"
}
```

Notice:

```text
age

↓

String
```

But your application expects:

```text
age

↓

Integer
```

Without validation:

```python
user = request.json()

print(user["age"] + 10)
```

Produces:

```python
"25" + 10
```

Error.

We need validation before using the data.

---

# Pydantic Solution

```python
from pydantic import BaseModel

class User(BaseModel):

    name: str
    age: int
    email: str
```

Now:

```python
payload = {

    "name":"Alice",

    "age":"25",

    "email":"alice@gmail.com"

}

user = User(**payload)

print(user)
```

Output:

```python
User(
    name='Alice',
    age=25,
    email='alice@gmail.com'
)
```

Notice:

Pydantic converted:

```text
"25"

↓

25
```

automatically.

---

# Invalid Data

```python
payload = {

    "name":"Alice",

    "age":"abc",

    "email":"alice@gmail.com"
}

User(**payload)
```

Output:

```text
ValidationError

age

Input should be a valid integer
```

Instead of failing later, Pydantic fails immediately.

---

# How Pydantic Works

```text
Incoming JSON

↓

Pydantic Model

↓

Validation

↓

Type Conversion

↓

Python Object

↓

Business Logic
```

---

# Production Example (FastAPI)

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class UserRequest(BaseModel):

    name: str

    age: int

@app.post("/users")

def create_user(user: UserRequest):

    return {

        "message":"User Created",

        "age": user.age
    }
```

Request:

```json
{
    "name":"Alice",
    "age":"25"
}
```

FastAPI automatically:

* Parses JSON
* Creates Pydantic object
* Validates
* Converts types
* Returns 422 for invalid input

---

# Nested Models

Production APIs often contain nested data.

```json
{
    "name":"Alice",
    "address":{
        "city":"Bangalore",
        "zip":"560001"
    }
}
```

Model:

```python
from pydantic import BaseModel

class Address(BaseModel):

    city: str

    zip: str

class User(BaseModel):

    name: str

    address: Address
```

Usage:

```python
user = User(**payload)

print(user.address.city)
```

Output:

```text
Bangalore
```

---

# Lists

```python
from pydantic import BaseModel

class Order(BaseModel):

    items: list[str]
```

Input:

```python
Order(

    items=[
        "Laptop",
        "Mouse"
    ]
)
```

---

# Optional Fields

```python
from typing import Optional

class User(BaseModel):

    name: str

    phone: Optional[str] = None
```

Now:

```python
User(name="Alice")
```

works.

---

# Default Values

```python
class User(BaseModel):

    country: str = "India"
```

Missing field:

```python
User(name="Alice")
```

Output:

```text
country="India"
```

---

# Field Constraints

```python
from pydantic import BaseModel, Field

class User(BaseModel):

    age: int = Field(

        ge=18,

        le=60
    )
```

Valid:

```python
age = 25
```

Invalid:

```python
age = 12
```

Raises:

```text
ValidationError
```

---

# Email Validation

```python
from pydantic import BaseModel
from pydantic import EmailStr

class User(BaseModel):

    email: EmailStr
```

Valid:

```text
alice@gmail.com
```

Invalid:

```text
abc
```

Validation fails automatically.

---

# Model Serialization

Convert to dictionary.

Pydantic v2:

```python
user.model_dump()
```

Output:

```python
{

    "name":"Alice",

    "age":25
}
```

Convert to JSON.

```python
user.model_dump_json()
```

---

# Aliases

Suppose API sends:

```json
{

    "firstName":"Alice"
}
```

Python prefers:

```python
first_name
```

Model:

```python
from pydantic import BaseModel, Field

class User(BaseModel):

    first_name: str = Field(
        alias="firstName"
    )
```

Now both styles are handled cleanly.

---

# Custom Validation (Pydantic v2)

```python
from pydantic import BaseModel
from pydantic import field_validator

class User(BaseModel):

    age: int

    @field_validator("age")

    @classmethod
    def validate_age(cls, value):

        if value < 18:

            raise ValueError(
                "Must be adult"
            )

        return value
```

---

# Computed Fields

```python
from pydantic import BaseModel
from pydantic import computed_field

class User(BaseModel):

    first: str

    last: str

    @computed_field
    @property
    def full_name(self):

        return f"{self.first} {self.last}"
```

---

# Environment Configuration

Pydantic Settings is commonly used for configuration.

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):

    database_url: str

    openai_api_key: str

settings = Settings()
```

Reads from environment variables automatically.

Very common in AI applications.

---

# Pydantic in LangChain

Structured output:

```python
from pydantic import BaseModel

class Answer(BaseModel):

    summary: str

    confidence: float
```

The LLM is instructed to return data matching this schema.

Example:

```python
answer.summary
```

instead of parsing raw JSON manually.

---

# Pydantic in LangGraph

Graph state:

```python
from pydantic import BaseModel

class GraphState(BaseModel):

    question: str

    documents: list[str]

    answer: str
```

Each node receives validated state.

---

# Production AI Architecture

```text
HTTP Request

↓

FastAPI

↓

Pydantic Validation

↓

Business Logic

↓

Database

↓

LLM

↓

Pydantic Response

↓

JSON
```

---

# What is TypedDict?

Now compare with TypedDict.

TypedDict is part of Python typing.

It only provides **type hints**.

No runtime validation.

Example:

```python
from typing import TypedDict

class User(TypedDict):

    name: str

    age: int
```

Usage:

```python
user: User = {

    "name":"Alice",

    "age":"25"
}
```

Python executes this without complaint.

At runtime:

```python
print(type(user))
```

Output:

```python
<class 'dict'>
```

TypedDict does **not** convert `"25"` to `25` and does **not** raise validation errors.

Static type checkers such as `mypy` or `pyright` may flag the mismatch during development, but the running program will not.

---

# Runtime Comparison

Pydantic:

```python
User(

    age="abc"
)
```

Output:

```text
ValidationError
```

TypedDict:

```python
user = {

    "age":"abc"
}
```

Output:

```text
No Exception
```

The error appears only later if your code assumes `age` is an integer.

---

# Type Conversion

Pydantic:

```python
age="25"

↓

25
```

TypedDict:

```python
age="25"

↓

"25"
```

No conversion occurs.

---

# Validation

Pydantic:

```text
Incoming Data

↓

Validate

↓

Convert

↓

Return Object
```

TypedDict:

```text
Incoming Data

↓

Store

↓

Done
```

---

# Nested Structures

TypedDict supports nested typing:

```python
from typing import TypedDict

class Address(TypedDict):

    city: str

class User(TypedDict):

    name: str

    address: Address
```

Again:

No validation.

---

# Performance

TypedDict is faster because:

It is simply a dictionary.

Pydantic performs validation.

So:

```text
TypedDict

↓

Fast

↓

No Validation
```

```text
Pydantic

↓

Slightly Slower

↓

Validation
```

The overhead is usually worth it for API boundaries and external inputs.

---

# When Should You Use Pydantic?

Use Pydantic for:

* FastAPI request/response models
* AI agent state
* LangChain structured outputs
* Configuration
* External APIs
* User input
* Database DTOs
* Data exchanged between services

---

# When Should You Use TypedDict?

Use TypedDict for:

* Internal dictionaries
* Static type checking
* Performance-sensitive internal code
* Interoperating with libraries that already use dictionaries

---

# Pydantic vs TypedDict

| Feature                    | Pydantic                        | TypedDict                      |
| -------------------------- | ------------------------------- | ------------------------------ |
| Runtime validation         | ✅                               | ❌                              |
| Type conversion            | ✅                               | ❌                              |
| JSON serialization helpers | ✅                               | ❌                              |
| Nested validation          | ✅                               | ❌                              |
| Custom validators          | ✅                               | ❌                              |
| Default values             | ✅                               | Limited (normal dict behavior) |
| Performance                | Slightly slower                 | Faster                         |
| Runtime object             | `BaseModel` instance            | Plain `dict`                   |
| Best for                   | APIs, external data, AI systems | Internal typed dictionaries    |

---

# Production AI Example

Suppose you're building an AI customer support platform.

**Incoming API request (Pydantic):**

```python
class ChatRequest(BaseModel):
    user_id: str
    question: str
```

**Internal state passed between functions (TypedDict):**

```python
from typing import TypedDict

class RetrievalContext(TypedDict):
    query_embedding: list[float]
    top_k: int
    tenant_id: str
```

The API validates external input with Pydantic, while internal processing uses lightweight typed dictionaries where runtime validation is unnecessary.

---

# Senior AI Engineer Interview Answer

> **Pydantic is a runtime data validation and serialization library widely used with FastAPI, LangChain, LangGraph, and AI platforms. It validates external data, performs type conversion, supports nested models, constraints, custom validators, JSON serialization, and configuration management. It is ideal at application boundaries where data comes from users, APIs, databases, or LLMs. TypedDict, on the other hand, is purely a static typing construct. It provides dictionary type hints for tools like mypy but performs no runtime validation or type conversion. I use Pydantic for validating and serializing external data, and TypedDict for lightweight internal dictionaries where static type checking is sufficient and runtime validation would add unnecessary overhead.
