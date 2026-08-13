# mypy in Python — Properly Explained

**mypy** is a **static type checker for Python**.

Python itself is dynamically typed:

```python
def add(a, b):
    return a + b
```

Python doesn't check what `a` and `b` are until the program actually runs.

With mypy, you add type hints:

```python
def add(a: int, b: int) -> int:
    return a + b
```

Then mypy analyzes your code **before you run it** and can catch incorrect usage.

---

# 1. Why do we need mypy?

Consider:

```python
def calculate_total(price: float, quantity: int) -> float:
    return price * quantity
```

Now somewhere else:

```python
total = calculate_total("100", 5)
```

Python won't necessarily complain when you define the function.

But this is clearly wrong:

```text
"100" → expected float
```

mypy can detect it before runtime.

You can think of the workflow as:

```text
                 Python code
                     │
                     ▼
                  mypy
                     │
             ┌───────┴────────┐
             │                │
          Valid            Invalid
             │                │
             ▼                ▼
          Run code       Fix type error
```

---

# 2. Installing mypy

Install it with:

```bash
pip install mypy
```

Then:

```bash
mypy app.py
```

For example:

```python
# app.py

def add(a: int, b: int) -> int:
    return a + b


result = add(10, "20")
```

Run:

```bash
mypy app.py
```

You may get something similar to:

```text
Argument 2 to "add" has incompatible type "str"; expected "int"
```

That's the main purpose of mypy.

---

# 3. Runtime vs static checking

This distinction is extremely important.

Python:

```python
def greet(name: str) -> str:
    return f"Hello {name}"
```

The annotation:

```python
name: str
```

doesn't automatically make Python reject:

```python
greet(123)
```

Python is still dynamically typed.

mypy checks it separately.

```text
Python runtime
     │
     └── Executes code

mypy
     │
     └── Analyzes types before execution
```

So:

> **Type hints don't enforce types at runtime. mypy checks those type hints statically.**

---

# 4. Basic type annotations

### Variables

```python
name: str = "Sandeepan"

age: int = 37

salary: float = 100000.0

is_active: bool = True
```

mypy can detect:

```python
age: int = "37"
```

because:

```text
Expected: int
Got: str
```

---

# 5. Function parameters and return types

A very common pattern:

```python
def calculate_salary(
    base_salary: float,
    bonus: float
) -> float:

    return base_salary + bonus
```

Now:

```python
salary = calculate_salary(
    100000,
    20000
)
```

mypy knows:

```text
base_salary → float
bonus       → float
return      → float
```

If you accidentally write:

```python
def calculate_salary(
    base_salary: float,
    bonus: float
) -> float:

    return "100000"
```

mypy catches the return type mismatch.

---

# 6. `list`

Instead of:

```python
users = []
```

you can specify:

```python
users: list[str] = []
```

Now:

```python
users.append("Sandeepan")
```

is valid.

But:

```python
users.append(123)
```

will be flagged by mypy.

For objects:

```python
user_ids: list[int] = [1, 2, 3]
```

---

# 7. `dict`

You can specify both key and value types:

```python
users: dict[int, str] = {
    1: "John",
    2: "Alice"
}
```

This means:

```text
key   → int
value → str
```

So:

```python
users[3] = "Bob"
```

is valid.

But:

```python
users["4"] = "Bob"
```

is invalid.

---

# 8. `Optional`

Suppose a database lookup may not find a user.

```python
def get_user(user_id: int) -> dict | None:
    ...
```

This means:

```text
return value = dict OR None
```

For older Python versions, you'll often see:

```python
from typing import Optional

def get_user(user_id: int) -> Optional[dict]:
    ...
```

Modern Python generally uses:

```python
dict | None
```

---

# 9. Why Optional matters

Consider:

```python
user = get_user(10)

print(user["name"])
```

mypy may complain because:

```text
user could be None
```

You need to handle that:

```python
user = get_user(10)

if user is None:
    print("User not found")
else:
    print(user["name"])
```

This is called **type narrowing**.

---

# 10. Type narrowing

mypy can understand conditions that narrow types.

Example:

```python
def print_name(name: str | None) -> None:

    if name is None:
        return

    print(name.upper())
```

Before the `if`:

```text
name: str | None
```

After:

```python
if name is None:
    return
```

mypy knows:

```text
name: str
```

So:

```python
name.upper()
```

is safe.

---

# 11. `Union`

Older Python code often uses:

```python
from typing import Union

def process(value: Union[int, str]) -> str:
    return str(value)
```

Modern Python:

```python
def process(value: int | str) -> str:
    return str(value)
```

Both mean:

```text
value can be int OR str
```

---

# 12. `Any`

This is very important.

```python
from typing import Any

data: Any = get_data()
```

`Any` basically tells the type checker:

> "Don't worry about this value's type."

For example:

```python
data: Any = "hello"

data.foo.bar.baz()
```

mypy generally won't complain because you've explicitly opted out of type checking for that value.

That's why overusing `Any` is dangerous.

Bad:

```python
def process(data: Any):
    ...
```

Better:

```python
def process(data: UserRequest):
    ...
```

or:

```python
def process(data: dict[str, str]):
    ...
```

---

# 13. `TypedDict`

This is particularly useful for API responses and JSON-like structures.

```python
from typing import TypedDict


class User(TypedDict):
    id: int
    name: str
    email: str
```

Now:

```python
user: User = {
    "id": 1,
    "name": "John",
    "email": "john@example.com"
}
```

mypy understands the structure.

This:

```python
print(user["name"])
```

is valid.

But:

```python
print(user["age"])
```

will be flagged because `age` isn't defined in `User`.

This is one reason `TypedDict` is useful for structured JSON.

---

# 14. `Protocol` + mypy

This connects directly to your previous question.

Suppose:

```python
from typing import Protocol


class PaymentGateway(Protocol):

    def charge(self, amount: float) -> bool:
        ...
```

Implementation:

```python
class StripeGateway:

    def charge(self, amount: float) -> bool:
        print(f"Charging {amount}")
        return True
```

This works structurally.

Now:

```python
def process_payment(
    gateway: PaymentGateway,
    amount: float
) -> bool:

    return gateway.charge(amount)
```

mypy understands:

```text
process_payment()
       │
       ▼
PaymentGateway Protocol
       │
       ▼
StripeGateway
```

If you create:

```python
class BrokenGateway:

    def refund(self, amount: float) -> bool:
        return True
```

and:

```python
gateway: PaymentGateway = BrokenGateway()
```

mypy can report that `BrokenGateway` doesn't implement the required `charge()` method.

This is where **Protocol + mypy** becomes powerful.

---

# 15. Classes

mypy also checks classes.

```python
class User:

    def __init__(
        self,
        user_id: int,
        name: str
    ) -> None:

        self.user_id = user_id
        self.name = name

    def get_name(self) -> str:
        return self.name
```

Now:

```python
user = User(1, "John")

name: str = user.get_name()
```

Everything is type-safe.

But:

```python
user = User("1", "John")
```

will be flagged.

---

# 16. Inheritance

mypy checks overridden methods too.

```python
class Animal:

    def speak(self) -> str:
        return "sound"


class Dog(Animal):

    def speak(self) -> str:
        return "woof"
```

Correct.

But:

```python
class Dog(Animal):

    def speak(self) -> int:
        return 10
```

is invalid because the subclass violates the parent's method contract.

---

# 17. Generics

Generics are important for production code.

Example:

```python
from typing import TypeVar

T = TypeVar("T")


def first_item(items: list[T]) -> T:
    return items[0]
```

Now:

```python
numbers = first_item([1, 2, 3])
```

mypy infers:

```text
numbers → int
```

And:

```python
names = first_item(["John", "Alice"])
```

becomes:

```text
names → str
```

So the function preserves the type.

---

# 18. Generic Repository

This becomes useful in your backend architecture.

```python
from typing import Generic, TypeVar

T = TypeVar("T")


class Repository(Generic[T]):

    def get(self, id: int) -> T:
        ...
```

Then:

```python
class User:
    ...


class UserRepository(Repository[User]):
    ...
```

Now mypy understands that:

```python
user = repository.get(1)
```

returns:

```text
User
```

rather than just `Any`.

---

# 19. Async code

For the FastAPI/SQLAlchemy applications you've been studying, this is important.

```python
async def get_user(
    user_id: int
) -> User | None:

    ...
```

Notice:

```python
async def
```

and:

```python
-> User | None
```

This means:

```text
await get_user(1)
        ↓
User | None
```

mypy understands async return types.

---

# 20. Async Repository Protocol

A production-style example:

```python
from typing import Protocol


class UserRepository(Protocol):

    async def get_by_id(
        self,
        user_id: int
    ) -> User | None:
        ...

    async def create(
        self,
        user: User
    ) -> User:
        ...
```

Implementation:

```python
class PostgresUserRepository:

    async def get_by_id(
        self,
        user_id: int
    ) -> User | None:

        # SQLAlchemy query
        return None

    async def create(
        self,
        user: User
    ) -> User:

        return user
```

Service:

```python
class UserService:

    def __init__(
        self,
        repository: UserRepository
    ) -> None:

        self.repository = repository

    async def get_user(
        self,
        user_id: int
    ) -> User:

        user = await self.repository.get_by_id(user_id)

        if user is None:
            raise ValueError("User not found")

        return user
```

mypy can verify the relationships between all these components.

---

# 21. Mypy configuration

In a real project, you usually don't just run:

```bash
mypy app.py
```

You create:

```text
mypy.ini
```

For example:

```ini
[mypy]
python_version = 3.12
strict = true
warn_return_any = true
warn_unused_ignores = true
disallow_untyped_defs = true
check_untyped_defs = true
no_implicit_optional = true
```

Then:

```bash
mypy app/
```

---

# 22. What does `strict = true` mean?

This is important in production projects.

Without strict checking, mypy can be relatively permissive.

With:

```ini
strict = true
```

mypy becomes much more aggressive about finding type problems.

For example, this:

```python
def calculate(a, b):
    return a + b
```

has no type annotations.

Strict mode can flag it.

You should generally aim for **strong typing in important production code**, rather than using `Any` everywhere.

---

# 23. `reveal_type()`

One of my favorite mypy debugging tools is:

```python
reveal_type(value)
```

Example:

```python
user: dict[str, str] = {
    "name": "John"
}

reveal_type(user)
```

mypy reports something similar to:

```text
Revealed type is "builtins.dict[builtins.str, builtins.str]"
```

This is extremely useful when you're unsure what mypy thinks a variable's type is.

---

# 24. Common mypy mistake

Suppose:

```python
def get_user() -> dict:
    return {
        "id": 1,
        "name": "John"
    }
```

`dict` is too vague.

Better:

```python
from typing import TypedDict


class User(TypedDict):
    id: int
    name: str


def get_user() -> User:
    return {
        "id": 1,
        "name": "John"
    }
```

Now mypy understands the structure.

---

# 25. Mypy in a FastAPI project

A production architecture could look like:

```text
app/
│
├── api/
│   └── users.py
│
├── services/
│   └── user_service.py
│
├── repositories/
│   ├── protocol.py
│   └── postgres.py
│
├── models/
│   └── user.py
│
└── main.py
```

`protocol.py`:

```python
from typing import Protocol


class UserRepository(Protocol):

    async def get_by_id(
        self,
        user_id: int
    ) -> User | None:
        ...
```

`postgres.py`:

```python
class PostgresUserRepository:

    async def get_by_id(
        self,
        user_id: int
    ) -> User | None:

        ...
```

`user_service.py`:

```python
class UserService:

    def __init__(
        self,
        repository: UserRepository
    ) -> None:

        self.repository = repository
```

Then mypy verifies that the concrete repository satisfies the expected interface.

This gives you:

```text
                mypy
                 │
                 ▼
       ┌──────────────────┐
       │ Type correctness │
       └────────┬─────────┘
                │
     ┌──────────┼───────────┐
     ▼          ▼           ▼
  FastAPI    Service     Repository
     │          │           │
     └──────────┼───────────┘
                ▼
             Protocol
```

---

# 26. Mypy vs Pydantic

This distinction is **very important for FastAPI**.

### mypy

Primarily checks types **statically**:

```python
def create_user(name: str) -> User:
    ...
```

It helps developers catch errors before execution.

### Pydantic

Validates data **at runtime**:

```python
from pydantic import BaseModel


class CreateUser(BaseModel):
    name: str
    age: int
```

If an API receives:

```json
{
    "name": "John",
    "age": "abc"
}
```

Pydantic can reject the data at runtime.

So:

```text
                 Type Safety
                     │
          ┌──────────┴──────────┐
          │                     │
       mypy                  Pydantic
          │                     │
      Static                  Runtime
      checking               validation
          │                     │
      Developer              Incoming
       code                   data
```

They complement each other.

---

# 27. Mypy vs Pyright

You may also encounter **Pyright**.

Both are static type checkers.

```text
mypy
  │
  └── popular Python type checker

Pyright
  │
  └── another powerful Python type checker
```

Pyright is also heavily used through VS Code/Pylance.

For most production Python projects, either can work well.

---

# 28. Where mypy fits in CI/CD

This is where it becomes especially valuable.

Your Git workflow could be:

```text
Developer
    │
    ▼
Write code
    │
    ▼
pytest
    │
    ▼
mypy
    │
    ▼
ruff
    │
    ▼
Docker build
    │
    ▼
Deploy
```

For example:

```bash
pytest
mypy app/
ruff check .
```

If mypy fails:

```text
❌ CI failed
   Type error detected
```

The code doesn't get deployed.

---

# 29. Real production example

Imagine someone changes:

```python
class UserService:

    async def get_user(self, user_id: int) -> User:

        user = await self.repository.get_by_id(user_id)

        if user is None:
            raise UserNotFound()

        return user
```

Later somebody changes the repository:

```python
async def get_by_id(
    self,
    user_id: int
) -> dict | None:
    ...
```

but the service expects:

```text
User
```

Mypy can identify mismatches at the boundaries.

This is particularly valuable in large teams because you don't have to manually inspect every caller after a refactor.

---

# 30. The senior-engineer mental model

Think about mypy as a **compiler-like safety layer for Python**.

Python:

```text
Very flexible
      ↓
Easy to write
      ↓
But type errors may appear at runtime
```

mypy:

```text
Type annotations
      ↓
Static analysis
      ↓
Find incorrect assumptions
      ↓
Before production
```

And in the architecture you've been learning:

```text
                 FastAPI
                    │
                    ▼
              Service Layer
                    │
             ┌──────┴──────┐
             │             │
          Protocol      Protocol
             │             │
             ▼             ▼
       Repository       LLMClient
             │             │
             ▼             ▼
        PostgreSQL      OpenAI/etc.
             │
             ▼
           mypy
             │
             ▼
       Static safety
```

### The key relationship to remember

**Type hints** → describe what your code expects.

**mypy** → checks whether your code follows those type hints.

**Protocol** → defines behavioral interfaces.

**Pydantic** → validates data at runtime.

**pytest** → verifies behavior.

Together, they give a Python production codebase much stronger guarantees than relying on Python's dynamic typing alone.
