# Python Testing (Senior AI Engineer Interview)

Testing is one of the **most frequently asked topics** for Senior Python, Backend, and AI Engineer interviews.

Common interview questions:

* What is `pytest`?
* Why use `pytest` instead of `unittest`?
* What are fixtures?
* What is mocking?
* What is `monkeypatch`?
* How do you test external APIs?
* How do you measure test coverage?

In production AI systems, testing ensures:

* Model pipelines work correctly
* APIs remain stable
* Refactoring doesn't break functionality
* External dependencies don't make tests flaky

---

# Types of Testing

```text
                 Testing

                    │
     ┌──────────────┼──────────────┐
     ▼              ▼              ▼

 Unit Test     Integration     End-to-End

 Test one      Test multiple   Test entire
 function       components      application
```

---

# 1. pytest

## What is pytest?

`pytest` is the most popular Python testing framework.

Install:

```bash
pip install pytest
```

---

## Simple Example

Suppose:

```python
# calculator.py

def add(a, b):
    return a + b
```

Test:

```python
# test_calculator.py

from calculator import add

def test_add():
    assert add(2, 3) == 5
```

Run:

```bash
pytest
```

Output:

```text
1 passed
```

---

## Why pytest?

Compared to `unittest`:

```python
import unittest

class TestCalculator(unittest.TestCase):

    def test_add(self):
        self.assertEqual(add(2,3),5)
```

Pytest:

```python
def test_add():
    assert add(2,3)==5
```

Advantages:

* Less boilerplate
* Better error messages
* Fixtures
* Parameterization
* Plugins
* Easier mocking

---

# Assertions

```python
assert 10 == 10

assert "AI".upper() == "AI"

assert len([1,2,3]) == 3
```

If assertion fails:

```python
assert add(2,3)==10
```

Pytest shows:

```text
Expected:10
Actual:5
```

Very readable.

---

# Parameterized Tests

Instead of:

```python
def test_add1():
    ...

def test_add2():
    ...

def test_add3():
    ...
```

Use:

```python
import pytest

@pytest.mark.parametrize(
    "a,b,result",
    [
        (1,2,3),
        (5,10,15),
        (-1,1,0)
    ]
)
def test_add(a,b,result):

    assert add(a,b)==result
```

Runs three tests automatically.

---

# 2. Fixtures

## What is a Fixture?

A fixture prepares resources before a test and cleans them up afterward.

Think of it as **test setup**.

Example:

```python
import pytest

@pytest.fixture
def sample_user():

    return {
        "name":"John",
        "age":30
    }
```

Use:

```python
def test_name(sample_user):

    assert sample_user["name"]=="John"
```

Pytest automatically injects the fixture.

---

## Why Fixtures?

Without fixtures:

```python
def test1():

    user=create_user()

def test2():

    user=create_user()

def test3():

    user=create_user()
```

Repeated setup.

With fixtures:

```python
@pytest.fixture
def user():

    return create_user()
```

Reusable across many tests.

---

## Fixture Scope

```python
@pytest.fixture(scope="function")
```

Runs before every test.

Other scopes:

* `function`
* `class`
* `module`
* `session`

Example:

Database connection:

```python
@pytest.fixture(scope="session")
def database():

    connect()

    yield db

    disconnect()
```

Created once for the whole test session.

---

# 3. Mock

## Why Mock?

Suppose your function calls an external API.

```python
def get_weather():

    requests.get(...)
```

Problems:

* Internet required
* Slow
* API may fail
* Costs money

Instead:

Replace the real dependency with a fake object.

---

Example

```python
from unittest.mock import Mock

api = Mock()

api.get.return_value = {
    "temperature":25
}

print(api.get())
```

Output:

```python
{
    "temperature":25
}
```

No real API call happened.

---

# Mocking a Function

Suppose:

```python
def fetch_user():

    return requests.get(...).json()
```

Test:

```python
from unittest.mock import patch

@patch("requests.get")
def test_fetch(mock_get):

    mock_get.return_value.json.return_value = {
        "name":"John"
    }

    result = fetch_user()

    assert result["name"]=="John"
```

Real HTTP request is never executed.

---

# AI Example

LLM:

```python
def ask_llm(prompt):

    return openai.chat(...)
```

Don't call OpenAI in tests.

Instead:

```python
mock_llm.return_value = "Hello"
```

Fast and deterministic.

---

# Mock vs Stub vs Fake

| Type | Purpose                    |
| ---- | -------------------------- |
| Mock | Verify interactions        |
| Stub | Return fixed values        |
| Fake | Lightweight implementation |

Example:

Mock:

```python
mock.send.assert_called_once()
```

Checks whether a method was called.

---

# 4. monkeypatch

`monkeypatch` temporarily changes objects during a test.

Very common in pytest.

---

Example:

Environment variable:

```python
import os

os.getenv("API_KEY")
```

Test:

```python
def test_api(monkeypatch):

    monkeypatch.setenv(
        "API_KEY",
        "test-key"
    )

    assert os.getenv("API_KEY")=="test-key"
```

After the test:

Environment restored automatically.

---

## Replace Functions

Suppose:

```python
def get_time():

    return datetime.now()
```

Test:

```python
def fake_time():

    return "10:00"

monkeypatch.setattr(
    module,
    "get_time",
    fake_time
)
```

Now:

```python
get_time()
```

returns:

```text
10:00
```

---

# AI Example

Suppose:

```python
embedding_model.encode()
```

Instead of loading a huge model:

```python
def fake_encode(text):

    return [0.1,0.2]
```

Monkeypatch replaces the real encoder.

Tests become very fast.

---

# 5. Coverage

Coverage measures:

> How much of your code is exercised by tests?

Install:

```bash
pip install pytest-cov
```

Run:

```bash
pytest --cov=app
```

Output:

```text
Name          Coverage

api.py          100%

llm.py           95%

vector.py        90%
```

Overall:

```text
94%
```

---

# Missing Coverage

Example:

```python
def login(user):

    if user=="admin":
        return True

    return False
```

Suppose test:

```python
assert login("admin")
```

Coverage:

```text
True branch

✓ Covered

False branch

✗ Not covered
```

Need another test:

```python
assert login("guest")==False
```

---

# Production AI Example

Suppose:

```text
FastAPI

↓

Retriever

↓

Embedding

↓

Vector DB

↓

LLM
```

Unit tests:

```
Retriever
```

Mock:

```
Vector DB
```

Mock:

```
LLM
```

Only test retrieval logic.

---

# Example Project Structure

```text
project/

    app/

        api.py
        llm.py
        vector.py

    tests/

        test_api.py
        test_llm.py
        test_vector.py
```

Run:

```bash
pytest
```

---

# What Should Be Mocked?

Mock:

* LLM APIs
* Database
* Redis
* AWS
* Kafka
* Vector DB
* Email
* File system (sometimes)
* Time

Don't mock:

* Pure business logic
* Utility functions
* Data transformations

---

# AI Example

Embedding service:

```python
def embed(text):

    return model.encode(text)
```

Test:

```python
mock_encode.return_value = [
    0.1,
    0.2
]

assert embed("AI")==[
    0.1,
    0.2
]
```

No GPU required.

---

# CI/CD

Typical pipeline:

```text
Git Push

↓

GitHub Actions

↓

pytest

↓

coverage

↓

lint

↓

build Docker

↓

deploy
```

If tests fail:

Deployment stops.

---

# Best Practices

* Write isolated unit tests.
* Mock external services.
* Use fixtures for reusable setup.
* Use monkeypatch for environment variables and temporary replacements.
* Aim for meaningful coverage rather than chasing 100%.
* Keep tests deterministic and fast.

---

# Interview Questions

### Why use pytest instead of unittest?

> `pytest` has a simpler syntax, powerful fixtures, parameterized testing, an extensive plugin ecosystem, and better failure reporting.

---

### What is a fixture?

> A fixture provides reusable setup and teardown logic for tests, improving readability and reducing duplication.

---

### What is mocking?

> Mocking replaces real dependencies with controllable fake objects so tests are fast, isolated, and deterministic.

---

### What is monkeypatch?

> `monkeypatch` temporarily modifies attributes, functions, or environment variables during a test and automatically restores the original state afterward.

---

### Why use coverage?

> Coverage shows which lines and branches of code are exercised by tests, helping identify untested code. However, high coverage alone does not guarantee high-quality tests.

---

# Senior AI Engineer Summary

| Concept       | Purpose                   | AI Example                                             |
| ------------- | ------------------------- | ------------------------------------------------------ |
| `pytest`      | Testing framework         | Test RAG pipeline                                      |
| `fixtures`    | Reusable setup            | Shared test database or sample documents               |
| `mock`        | Replace dependencies      | Mock OpenAI, Qdrant, Redis                             |
| `monkeypatch` | Temporary runtime changes | Override API keys, fake embedding model                |
| `coverage`    | Measure tested code       | Ensure retrieval, ranking, and API paths are exercised |

In production AI systems, a common strategy is:

* **Unit tests** for business logic.
* **Mocks** for external services (LLMs, vector databases, cloud APIs).
* **Fixtures** for reusable test data.
* **Integration tests** to verify components work together.
* **Coverage reports** in CI/CD to detect untested changes before deployment.
