# Logging in Python (Senior AI Engineer Interview)

Logging is one of the **most important production topics** for Senior Python and AI Engineer interviews.

Interviewers commonly ask:

* Why not use `print()`?
* How do you configure production logging?
* What are log levels?
* What is log rotation?
* What is structured logging?
* Why use JSON logging?
* How do AI systems use logging?

---

# Why not use `print()`?

Many beginners debug like this:

```python
print("Request received")
print("Calling LLM...")
print("Response:", response)
```

Problems:

* No timestamps
* No log levels
* No filtering
* No file output
* Difficult to search
* Difficult to aggregate
* Not suitable for distributed systems

---

# Python Logging Module

```python
import logging

logging.basicConfig(level=logging.INFO)

logging.info("Application started")
logging.warning("High memory usage")
logging.error("Database connection failed")
```

Output:

```text
INFO:root:Application started
WARNING:root:High memory usage
ERROR:root:Database connection failed
```

---

# Log Levels

| Level    | When to Use                                    |
| -------- | ---------------------------------------------- |
| DEBUG    | Detailed debugging information                 |
| INFO     | Normal application flow                        |
| WARNING  | Something unexpected but application continues |
| ERROR    | Operation failed                               |
| CRITICAL | Application cannot continue                    |

Example:

```python
logging.debug("Embedding vector generated")
logging.info("User logged in")
logging.warning("Cache miss")
logging.error("Database timeout")
logging.critical("GPU unavailable")
```

---

# Creating a Logger

Instead of using the root logger:

```python
import logging

logger = logging.getLogger(__name__)

logger.info("Started")
```

Why?

If your project has:

```text
app/
    api.py
    llm.py
    vector.py
```

Each module gets its own logger:

```python
logger = logging.getLogger(__name__)
```

Logs become:

```text
app.api
app.llm
app.vector
```

This makes debugging much easier.

---

# Production Logging Configuration

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
```

Example output:

```text
2026-07-14 15:20:10 INFO app.llm Request received
```

Each field has a purpose:

```text
Timestamp
Log level
Module
Message
```

---

# Logging to a File

```python
import logging

logging.basicConfig(
    filename="app.log",
    level=logging.INFO
)

logging.info("Application started")
```

Output:

```text
app.log
```

Contains:

```text
INFO Application started
```

---

# Production Logging to Console + File

```python
import logging

logger = logging.getLogger(__name__)

logger.setLevel(logging.INFO)

formatter = logging.Formatter(
    "%(asctime)s %(levelname)s %(message)s"
)

console = logging.StreamHandler()
console.setFormatter(formatter)

file = logging.FileHandler("app.log")
file.setFormatter(formatter)

logger.addHandler(console)
logger.addHandler(file)
```

Now logs go to:

* Console
* Log file

---

# Rotating Logs

## Problem

Imagine your AI service runs for 6 months.

```
app.log

↓

20 GB

↓

100 GB

↓

500 GB
```

This becomes difficult to manage.

---

# Solution

Use rotating logs.

```python
from logging.handlers import RotatingFileHandler
import logging

logger = logging.getLogger(__name__)

handler = RotatingFileHandler(
    "app.log",
    maxBytes=1024 * 1024,
    backupCount=5
)

logger.addHandler(handler)
```

Meaning:

```
1 MB reached

↓

Rename

app.log

↓

app.log.1

↓

New app.log
```

Keeps:

```
app.log
app.log.1
app.log.2
...
```

Only the latest 5 backups.

---

# Timed Rotating Logs

Rotate every day.

```python
from logging.handlers import TimedRotatingFileHandler

handler = TimedRotatingFileHandler(
    "app.log",
    when="midnight",
    interval=1,
    backupCount=30
)
```

Useful for production servers.

---

# Structured Logging

Traditional logging:

```text
User logged in
```

Hard to search.

Structured logging:

```text
user_id=123
endpoint=/chat
latency=250ms
tokens=350
```

Each field is searchable.

---

Example

Instead of:

```python
logger.info("User logged in")
```

Use:

```python
logger.info(
    "User logged in",
    extra={
        "user_id":123,
        "ip":"10.0.0.1"
    }
)
```

Now log aggregation tools can index those fields.

---

# JSON Logging

Modern systems don't read plain text logs. They ingest structured formats like JSON.

Example log:

```json
{
  "timestamp":"2026-07-14T15:30:00",
  "level":"INFO",
  "service":"chat-api",
  "user_id":123,
  "latency_ms":210,
  "tokens":350,
  "model":"gpt-4.1"
}
```

Benefits:

* Machine readable
* Searchable
* Easy to aggregate
* Compatible with Elasticsearch, Loki, Datadog, Splunk, CloudWatch

---

## Example using `python-json-logger`

Install:

```bash
pip install python-json-logger
```

Configure:

```python
import logging
from pythonjsonlogger import jsonlogger

handler = logging.StreamHandler()

formatter = jsonlogger.JsonFormatter()

handler.setFormatter(formatter)

logger = logging.getLogger()

logger.addHandler(handler)

logger.setLevel(logging.INFO)

logger.info(
    "Inference complete",
    extra={
        "latency_ms": 180,
        "tokens": 420
    }
)
```

Output:

```json
{
  "message":"Inference complete",
  "latency_ms":180,
  "tokens":420
}
```

---

# AI Production Logging

Suppose a RAG application.

Flow:

```text
User

↓

API

↓

Retriever

↓

Vector DB

↓

LLM

↓

Response
```

Each stage should log:

```text
Request ID

User ID

Latency

Model

Embedding Time

Vector Search Time

LLM Time

Token Count

Cost

Errors
```

---

Example

```python
logger.info(
    "LLM completed",
    extra={
        "request_id":"abc123",
        "model":"gpt-4.1",
        "latency_ms":520,
        "prompt_tokens":850,
        "completion_tokens":220
    }
)
```

---

# Correlation IDs

Production systems often have many microservices.

```
Gateway

↓

Chat API

↓

Retriever

↓

LLM Service

↓

Redis
```

How do you connect logs?

Use a request ID.

Example:

```text
request_id=8fa92d...
```

Every service logs the same ID.

```
Gateway

request_id=8fa92d

↓

Retriever

request_id=8fa92d

↓

LLM

request_id=8fa92d
```

Now all logs for one request can be traced.

---

# What Should Never Be Logged?

Never log:

* Passwords
* API keys
* JWT tokens
* Credit card numbers
* Personal identifiers unless required and protected
* Full prompts containing sensitive user data (unless properly redacted)

Instead:

```text
User authenticated

user_id=123
```

Not:

```text
Password=MySecretPassword
```

---

# Logging Architecture in AI Systems

```text
            User Request
                  │
                  ▼
            FastAPI Service
                  │
                  ▼
          Structured Logger
                  │
        ┌─────────┴─────────┐
        ▼                   ▼
    Console            JSON Log File
        │                   │
        └─────────┬─────────┘
                  ▼
        Log Collection Agent
                  │
                  ▼
     Elasticsearch / Loki / CloudWatch
                  │
                  ▼
            Grafana / Kibana
```

---

# Production Best Practices

* Use `logging.getLogger(__name__)` instead of the root logger.
* Log in JSON or another structured format.
* Include timestamps, request IDs, service names, and log levels.
* Rotate logs by size or time.
* Use appropriate log levels (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`).
* Never log secrets or sensitive information.
* Send logs to a centralized logging platform instead of relying on local files.

---

# Interview Questions

### Why use logging instead of print()?

> Logging provides configurable log levels, timestamps, persistence, formatting, filtering, and integration with monitoring tools. `print()` is only suitable for simple debugging.

---

### What is structured logging?

> Structured logging stores log information as key-value pairs instead of free-form text, making logs searchable and easier for monitoring systems to analyze.

---

### Why use JSON logging?

> JSON is machine-readable and integrates well with centralized logging systems like Elasticsearch, Loki, Splunk, Datadog, and CloudWatch. It allows efficient searching, filtering, and dashboard creation.

---

### Why rotate logs?

> Without rotation, log files grow indefinitely, consuming disk space and becoming difficult to manage. Rotation creates new log files based on size or time while retaining a limited history.

---

### What do you log in an LLM application?

A strong senior-level answer is:

* Request ID / Correlation ID
* User or tenant ID (if appropriate)
* Model name
* Prompt and completion token counts
* Latency for retrieval, inference, and total request
* API cost
* Retry attempts
* Cache hits/misses
* Errors and stack traces
* Rate limiting events
* Resource utilization (when relevant)

These logging practices are standard in production AI platforms because they support debugging, performance tuning, cost analysis, and operational monitoring.
