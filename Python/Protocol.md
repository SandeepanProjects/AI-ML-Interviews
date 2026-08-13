In Python, a **Protocol** is a way to define an **interface based on behavior rather than inheritance**.

The easiest way to think about it is:

> **"If an object has these methods/attributes, I can use it."**

This is especially useful in **large production applications**, dependency injection, repository patterns, service layers, testing, and clean architecture.

---

# 1. Why do we need Protocol?

Suppose you have a notification service:

```python
class EmailNotifier:
    def send(self, message: str) -> None:
        print(f"Email: {message}")


class SMSNotifier:
    def send(self, message: str) -> None:
        print(f"SMS: {message}")
```

Both classes have:

```python
send(message)
```

We want a service that can work with either:

```python
def notify(notifier, message):
    notifier.send(message)
```

Python allows this naturally because of **duck typing**.

> "If it walks like a duck and quacks like a duck, treat it like a duck."

But there is a problem with large projects:

* IDE/type checker doesn't know exactly what `notifier` should provide.
* You don't have an explicit contract.
* Developers can accidentally pass an incompatible object.
* Dependency injection becomes harder to document.
* Static type checking becomes weaker.

This is where `Protocol` helps.

---

# 2. Basic Protocol

```python
from typing import Protocol


class Notifier(Protocol):

    def send(self, message: str) -> None:
        ...
```

Now create implementations:

```python
class EmailNotifier:

    def send(self, message: str) -> None:
        print(f"Email: {message}")


class SMSNotifier:

    def send(self, message: str) -> None:
        print(f"SMS: {message}")
```

Notice something important:

**We did not inherit from `Notifier`.**

```python
class EmailNotifier(Notifier):  # NOT required
```

is unnecessary.

Instead:

```python
class EmailNotifier:
    def send(self, message: str) -> None:
        ...
```

already satisfies the protocol structurally.

---

# 3. Using the Protocol

```python
def notify_user(
    notifier: Notifier,
    message: str
) -> None:

    notifier.send(message)
```

Now:

```python
email = EmailNotifier()
sms = SMSNotifier()

notify_user(email, "Your order has shipped")
notify_user(sms, "Your order has shipped")
```

Output:

```text
Email: Your order has shipped
SMS: Your order has shipped
```

The function doesn't care about the concrete implementation.

It only cares:

> Does this object have `send(str) -> None`?

---

# 4. Protocol vs Abstract Base Class

This is one of the most important interview questions.

## Abstract Base Class

With an ABC:

```python
from abc import ABC, abstractmethod


class Notifier(ABC):

    @abstractmethod
    def send(self, message: str) -> None:
        pass
```

Then implementations normally inherit:

```python
class EmailNotifier(Notifier):

    def send(self, message: str) -> None:
        print(f"Email: {message}")
```

The relationship is explicit:

```text
EmailNotifier
      ↓
  inherits
      ↓
   Notifier
```

---

## Protocol

With Protocol:

```python
from typing import Protocol


class Notifier(Protocol):

    def send(self, message: str) -> None:
        ...
```

Implementation:

```python
class EmailNotifier:

    def send(self, message: str) -> None:
        print(f"Email: {message}")
```

No inheritance.

Relationship:

```text
EmailNotifier
      ↓
has required behavior
      ↓
   Notifier
```

This is called **structural typing**.

---

# 5. Nominal vs Structural Typing

This distinction is extremely important.

### Traditional inheritance = nominal typing

Python says:

> "This class is a `Notifier` because it explicitly inherits from `Notifier`."

```python
class EmailNotifier(Notifier):
    ...
```

### Protocol = structural typing

Python/type checker says:

> "This class behaves like a `Notifier`, because it implements the required interface."

```python
class EmailNotifier:
    def send(self, message: str) -> None:
        ...
```

So:

```text
Nominal typing
    ↓
"What are you?"

Structural typing
    ↓
"What can you do?"
```

---

# 6. `@runtime_checkable`

By default, Protocol is primarily for **static type checking**.

For example:

```python
from typing import Protocol


class Notifier(Protocol):

    def send(self, message: str) -> None:
        ...
```

You can't reliably do:

```python
isinstance(email, Notifier)
```

unless you explicitly make the protocol runtime-checkable.

Use:

```python
from typing import Protocol, runtime_checkable


@runtime_checkable
class Notifier(Protocol):

    def send(self, message: str) -> None:
        ...
```

Now:

```python
email = EmailNotifier()

print(isinstance(email, Notifier))
```

Output:

```text
True
```

This is useful when you actually need a runtime check.

But remember:

**Protocol's main purpose is static typing, not runtime validation.**

---

# 7. Protocol with attributes

Protocols aren't limited to methods.

Suppose we need a repository that has:

```python
save()
find_by_id()
```

and a property:

```python
name
```

We can define:

```python
from typing import Protocol


class Repository(Protocol):

    name: str

    def save(self, data: dict) -> None:
        ...

    def find_by_id(self, id: int) -> dict | None:
        ...
```

Implementation:

```python
class PostgresRepository:

    name = "postgres"

    def save(self, data: dict) -> None:
        print("Saving to PostgreSQL")

    def find_by_id(self, id: int) -> dict | None:
        return {"id": id}
```

This satisfies the protocol.

---

# 8. Protocol in Dependency Injection

This is where Protocol becomes **very powerful in production applications**.

Imagine your application has:

```text
API
 ↓
Service
 ↓
Repository
 ↓
PostgreSQL
```

You don't want your service tightly coupled to PostgreSQL.

Bad design:

```python
class UserService:

    def __init__(self):
        self.repository = PostgresUserRepository()
```

Now:

```text
UserService
     ↓
PostgresUserRepository
     ↓
PostgreSQL
```

The service knows too much.

Instead, define a protocol.

```python
from typing import Protocol


class UserRepository(Protocol):

    async def get_user(self, user_id: int) -> dict | None:
        ...

    async def create_user(self, user: dict) -> dict:
        ...
```

Now the service depends on the abstraction:

```python
class UserService:

    def __init__(self, repository: UserRepository):
        self.repository = repository

    async def get_user(self, user_id: int):

        return await self.repository.get_user(user_id)
```

The service doesn't care whether the implementation is:

```text
PostgreSQL
MongoDB
DynamoDB
Mock
In-memory database
```

As long as it satisfies:

```python
UserRepository
```

---

# 9. Real FastAPI example

This pattern fits very well with the FastAPI architecture you've been learning.

### Protocol

```python
from typing import Protocol


class UserRepository(Protocol):

    async def get_by_id(
        self,
        user_id: int
    ) -> dict | None:
        ...

    async def create(
        self,
        user: dict
    ) -> dict:
        ...
```

---

### PostgreSQL implementation

```python
class PostgresUserRepository:

    async def get_by_id(
        self,
        user_id: int
    ) -> dict | None:

        # SQLAlchemy query here
        return {
            "id": user_id,
            "name": "Sandeepan"
        }

    async def create(
        self,
        user: dict
    ) -> dict:

        # INSERT into PostgreSQL
        return user
```

---

### Service

```python
class UserService:

    def __init__(
        self,
        repository: UserRepository
    ):
        self.repository = repository

    async def get_user(
        self,
        user_id: int
    ):

        user = await self.repository.get_by_id(user_id)

        if user is None:
            raise ValueError("User not found")

        return user
```

---

### Dependency Injection

```python
def get_user_repository() -> UserRepository:

    return PostgresUserRepository()
```

Then:

```python
from fastapi import Depends


def get_user_service(
    repository: UserRepository = Depends(get_user_repository)
):

    return UserService(repository)
```

Router:

```python
from fastapi import APIRouter, Depends

router = APIRouter()


@router.get("/users/{user_id}")
async def get_user(
    user_id: int,
    service: UserService = Depends(get_user_service)
):

    return await service.get_user(user_id)
```

Now the architecture becomes:

```text
             FastAPI Router
                   │
                   ▼
             UserService
                   │
                   ▼
          UserRepository
             Protocol
                   │
                   ▼
      PostgresUserRepository
                   │
                   ▼
              PostgreSQL
```

This is a clean architecture approach.

---

# 10. The real power: swapping implementations

Suppose tomorrow you decide to use MongoDB.

You don't need to modify:

```python
UserService
```

Create:

```python
class MongoUserRepository:

    async def get_by_id(
        self,
        user_id: int
    ) -> dict | None:

        # MongoDB implementation
        return {"id": user_id}

    async def create(
        self,
        user: dict
    ) -> dict:

        return user
```

It satisfies the same protocol:

```python
class UserRepository(Protocol):

    async def get_by_id(
        self,
        user_id: int
    ) -> dict | None:
        ...

    async def create(
        self,
        user: dict
    ) -> dict:
        ...
```

Therefore:

```python
service = UserService(
    MongoUserRepository()
)
```

works.

No changes to the service.

---

# 11. Protocol is extremely useful for testing

This is another major production use case.

Suppose your service normally uses:

```python
PostgresUserRepository
```

But in unit tests, you don't want a real PostgreSQL database.

Create:

```python
class MockUserRepository:

    async def get_by_id(
        self,
        user_id: int
    ) -> dict | None:

        return {
            "id": user_id,
            "name": "Test User"
        }

    async def create(
        self,
        user: dict
    ) -> dict:

        return user
```

Then:

```python
repository = MockUserRepository()

service = UserService(repository)

user = await service.get_user(1)

assert user["id"] == 1
```

Architecture:

```text
Production

UserService
     ↓
UserRepository Protocol
     ↓
PostgresUserRepository
     ↓
PostgreSQL


Testing

UserService
     ↓
UserRepository Protocol
     ↓
MockUserRepository
```

This gives you **loose coupling + testability**.

---

# 12. Protocol for external APIs

Protocols are also very useful when integrating external services.

Suppose your AI application uses an LLM.

You could define:

```python
from typing import Protocol


class LLMClient(Protocol):

    async def generate(
        self,
        prompt: str
    ) -> str:
        ...
```

OpenAI implementation:

```python
class OpenAIClient:

    async def generate(
        self,
        prompt: str
    ) -> str:

        # call OpenAI
        return "Generated response"
```

Anthropic implementation:

```python
class AnthropicClient:

    async def generate(
        self,
        prompt: str
    ) -> str:

        # call Anthropic
        return "Generated response"
```

Your service:

```python
class AIService:

    def __init__(self, llm: LLMClient):
        self.llm = llm

    async def answer(self, question: str):

        return await self.llm.generate(question)
```

Now:

```python
AIService(OpenAIClient())
```

or:

```python
AIService(AnthropicClient())
```

works without modifying `AIService`.

This is particularly useful for **multi-model architectures and fallback models**.

---

# 13. Protocol for vector databases

For a RAG system, you could define:

```python
class VectorStore(Protocol):

    async def add_documents(
        self,
        documents: list[dict]
    ) -> None:
        ...

    async def similarity_search(
        self,
        query: str,
        k: int
    ) -> list[dict]:
        ...
```

Then:

```python
class QdrantVectorStore:

    async def add_documents(
        self,
        documents: list[dict]
    ) -> None:
        ...

    async def similarity_search(
        self,
        query: str,
        k: int
    ) -> list[dict]:
        ...
```

You could later implement:

```python
class PineconeVectorStore:
    ...
```

or:

```python
class FAISSVectorStore:
    ...
```

Your RAG service only depends on:

```python
VectorStore
```

not Qdrant/Pinecone/FAISS.

---

# 14. Protocol with properties

You can also define properties.

```python
from typing import Protocol


class Model(Protocol):

    @property
    def model_name(self) -> str:
        ...

    def generate(self, prompt: str) -> str:
        ...
```

Implementation:

```python
class GPTModel:

    @property
    def model_name(self) -> str:
        return "gpt-model"

    def generate(self, prompt: str) -> str:
        return "response"
```

---

# 15. Generic Protocol

Protocols can also use generics.

For example, a generic repository:

```python
from typing import Protocol, TypeVar, Generic

T = TypeVar("T")


class Repository(Protocol[T]):

    async def get_by_id(
        self,
        id: int
    ) -> T | None:
        ...

    async def save(
        self,
        entity: T
    ) -> T:
        ...
```

Then:

```python
class User:
    def __init__(self, id: int, name: str):
        self.id = id
        self.name = name
```

A user repository can conceptually implement:

```python
Repository[User]
```

This becomes useful in larger codebases where you have many entity types.

---

# 16. Protocol vs duck typing

You might ask:

> "Python already supports duck typing. Why do I need Protocol?"

Good question.

Without Protocol:

```python
def process(payment_gateway):

    payment_gateway.charge(100)
```

Python will allow it.

But static tools don't have a formal contract.

With Protocol:

```python
class PaymentGateway(Protocol):

    def charge(self, amount: float) -> bool:
        ...
```

Then:

```python
def process(
    gateway: PaymentGateway
):

    return gateway.charge(100)
```

Now developers, IDEs and tools have a clear contract.

So:

```text
Duck typing
    ↓
Runtime flexibility


Protocol
    ↓
Duck typing
+
explicit interface
+
static type checking
+
better IDE support
+
dependency inversion
+
testability
```

---

# 17. Protocol vs inheritance

A good rule:

### Use inheritance when:

There is a genuine **"is-a" relationship**.

```text
Dog
 ↓
Animal
```

```python
class Dog(Animal):
    ...
```

### Use Protocol when:

You care about **capabilities/behavior**.

```text
Anything that can:
    send()
```

```python
class Notifier(Protocol):
    def send(self, message: str) -> None:
        ...
```

For example:

```text
EmailNotifier
SMSNotifier
PushNotifier
SlackNotifier
```

They don't necessarily need to belong to the same inheritance hierarchy.

---

# 18. Protocol vs ABC — interview answer

If an interviewer asks:

> **What is the difference between Protocol and ABC in Python?**

A strong answer is:

> `ABC` provides nominal abstraction through inheritance, whereas `Protocol` provides structural typing. With an ABC, a class generally needs to explicitly inherit from the base class. With a Protocol, a class only needs to implement the required methods and attributes. Protocols are therefore particularly useful for dependency inversion, dependency injection, testing, and defining interfaces between loosely coupled components.

---

# 19. Production architecture

For the type of **FastAPI + PostgreSQL + Redis + Qdrant + LLM** applications you've been working on, you can use Protocols at boundaries:

```text
                    API
                     │
                     ▼
              Service Layer
                     │
          ┌──────────┼───────────┐
          ▼          ▼           ▼
       Protocol    Protocol    Protocol
          │          │           │
          ▼          ▼           ▼
     Repository   Cache       LLMClient
          │          │           │
          ▼          ▼           ▼
     PostgreSQL    Redis     OpenAI/Claude
                               /Llama
```

For RAG:

```text
                 RAGService
                     │
       ┌─────────────┼─────────────┐
       ▼             ▼             ▼
   Embedder       Retriever      Reranker
   Protocol       Protocol       Protocol
       │             │             │
       ▼             ▼             ▼
 OpenAI Embed      Qdrant       Cohere
                    /FAISS
```

This gives you **dependency inversion**.

The business logic doesn't depend directly on infrastructure.

---

# 20. One important misconception

A Protocol does **not** automatically enforce implementation at runtime.

For example:

```python
class UserRepository(Protocol):

    def get_user(self, id: int):
        ...
```

This doesn't mean Python will automatically prevent:

```python
class BadRepository:
    pass
```

from existing.

Protocol is primarily a **type-checking contract**.

Tools such as:

```text
mypy
pyright
Pylance
```

can detect violations.

For example:

```python
class BadRepository:
    pass


repo: UserRepository = BadRepository()
```

A static type checker can report that `BadRepository` doesn't implement `get_user()`.

---

# 21. The mental model to remember

Think of Protocol as:

```text
                PROTOCOL
                   │
           "I don't care who
            you are."
                   │
                   ▼
        "I care what you can do."
                   │
          ┌────────┼────────┐
          ▼        ▼        ▼
       Class A   Class B   Class C
          │        │        │
       send()   send()   send()
```

That's the core idea.

And in a production application:

```text
Business Logic
      │
      ▼
   Protocol
      │
      ├───────────────┐
      ▼               ▼
Production Impl    Test Impl
      │               │
 PostgreSQL          Mock
 Qdrant              Fake
 OpenAI              Stub
 Redis               In-memory
```

**This is why Protocol is so valuable in senior-level Python architecture:** it lets your service layer depend on **interfaces/behavior instead of concrete infrastructure implementations**, while still giving you strong static typing.
