# How do you implement CORS in FastAPI?

**CORS (Cross-Origin Resource Sharing)** is a browser security mechanism that controls whether a frontend running on one **origin** can make requests to a backend running on another origin.

For example:

```text
Frontend
https://app.example.com
        │
        │ API request
        ↓
Backend
https://api.example.com
```

These are different origins, so the browser applies CORS rules.

---

# 1. What is an Origin?

An origin consists of:

```text
scheme + host + port
```

For example:

```text
https://example.com:443
```

is an origin.

These are different origins:

```text
http://localhost:3000
http://localhost:8000
```

because the ports differ.

Also:

```text
https://app.example.com
https://api.example.com
```

are different because the hosts differ.

---

# 2. Why do we need CORS?

Suppose your React frontend runs on:

```text
http://localhost:3000
```

and FastAPI runs on:

```text
http://localhost:8000
```

Your frontend does:

```javascript
fetch("http://localhost:8000/users")
```

The browser sees:

```text
Frontend origin
http://localhost:3000

Backend origin
http://localhost:8000
```

Different origins.

The browser therefore checks whether the backend allows the frontend origin.

---

# 3. FastAPI CORS implementation

FastAPI uses Starlette's `CORSMiddleware`.

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=[
        "GET",
        "POST",
        "PUT",
        "DELETE",
    ],
    allow_headers=[
        "Authorization",
        "Content-Type",
    ],
)
```

Now the frontend at:

```text
http://localhost:3000
```

is allowed to call the API.

---

# 4. What do these options mean?

## `allow_origins`

Specifies which frontend origins are allowed.

```python
allow_origins=[
    "https://app.example.com",
]
```

You can specify multiple:

```python
allow_origins=[
    "https://app.example.com",
    "https://admin.example.com",
]
```

For local development:

```python
allow_origins=[
    "http://localhost:3000",
    "http://localhost:5173",
]
```

---

# 5. Don't blindly use `"*"` in production

You may see:

```python
allow_origins=["*"]
```

This means essentially:

> Allow requests from any origin.

It's convenient for development but generally too permissive for production.

Prefer:

```python
allow_origins=[
    "https://app.example.com",
]
```

This follows the principle:

> **Allow only the origins that actually need access.**

---

# 6. `allow_methods`

Controls which HTTP methods browsers may use.

```python
allow_methods=[
    "GET",
    "POST",
    "PUT",
    "DELETE",
]
```

You could also use:

```python
allow_methods=["*"]
```

but explicit methods are preferable when you know what your API requires.

---

# 7. `allow_headers`

Controls which request headers are permitted.

For JWT authentication, you commonly need:

```python
allow_headers=[
    "Authorization",
    "Content-Type",
]
```

Because the frontend might send:

```http
Authorization: Bearer eyJ...
```

---

# 8. `allow_credentials`

This is relevant when the browser needs to send credentials such as:

* Cookies
* HTTP authentication information
* Certain browser-managed credentials

Example:

```python
allow_credentials=True
```

If you're using cookie-based authentication, this setting is particularly important.

---

# 9. Important: CORS and JWT are different

A common interview mistake is saying:

> "CORS authenticates the user."

It doesn't.

CORS is primarily a **browser cross-origin access control mechanism**.

JWT authentication answers:

> **Who is this user?**

CORS answers:

> **Is this browser origin allowed to make this cross-origin request?**

For example:

```text
CORS
 ↓
Is https://app.example.com allowed?

JWT
 ↓
Who is the user?
```

They solve different problems.

---

# 10. Preflight requests

One of the most important CORS concepts is the **preflight request**.

Suppose your frontend sends:

```http
POST /users
Authorization: Bearer ...
Content-Type: application/json
```

The browser may first send:

```http
OPTIONS /users
```

with headers such as:

```http
Origin: https://app.example.com
Access-Control-Request-Method: POST
Access-Control-Request-Headers: authorization, content-type
```

This asks the server:

> "Is this origin allowed to make this POST request with these headers?"

FastAPI's `CORSMiddleware` handles the CORS response.

---

# 11. What does the server return?

The server can respond with headers such as:

```http
Access-Control-Allow-Origin: https://app.example.com
Access-Control-Allow-Methods: GET, POST, PUT, DELETE
Access-Control-Allow-Headers: Authorization, Content-Type
```

The browser checks these headers.

If the policy allows the request:

```text
Browser
   ↓
Actual POST request
```

Otherwise, the browser blocks access to the response.

---

# 12. Production configuration

Don't hardcode origins directly into the application if you're deploying across environments.

Use configuration.

For example:

```text
.env

CORS_ORIGINS=https://app.example.com,https://admin.example.com
```

Then:

```python
from pydantic_settings import BaseSettings


class Settings(BaseSettings):

    cors_origins: str

    @property
    def cors_origin_list(self) -> list[str]:
        return [
            origin.strip()
            for origin in self.cors_origins.split(",")
            if origin.strip()
        ]

    class Config:
        env_file = ".env"
```

Then:

```python
settings = Settings()

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=[
        "GET",
        "POST",
        "PUT",
        "DELETE",
    ],
    allow_headers=[
        "Authorization",
        "Content-Type",
    ],
)
```

---

# 13. Environment-specific CORS

In a real application, I might have:

### Development

```text
http://localhost:3000
http://localhost:5173
```

### Staging

```text
https://staging-app.example.com
```

### Production

```text
https://app.example.com
```

Configuration:

```python
if settings.environment == "production":

    cors_origins = [
        "https://app.example.com",
    ]

elif settings.environment == "staging":

    cors_origins = [
        "https://staging-app.example.com",
    ]

else:

    cors_origins = [
        "http://localhost:3000",
        "http://localhost:5173",
    ]
```

Then:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=[
        "GET",
        "POST",
        "PUT",
        "DELETE",
    ],
    allow_headers=[
        "Authorization",
        "Content-Type",
    ],
)
```

---

# 14. CORS with cookies

Suppose you're using:

```text
Frontend
https://app.example.com

Backend
https://api.example.com
```

and authentication uses an HTTP-only cookie.

You might configure:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://app.example.com",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)
```

The frontend needs to make credentialed requests, for example:

```javascript
fetch(
    "https://api.example.com/profile",
    {
        credentials: "include"
    }
)
```

The important point is that **credentialed CORS requests require an explicit allowed origin**; don't use a wildcard origin for this configuration.

---

# 15. CORS does not protect your API from non-browser clients

This is another excellent interview point.

Suppose your API allows:

```text
https://app.example.com
```

through CORS.

That does **not** mean someone cannot call your API using:

```text
curl
Postman
Python requests
another backend
```

CORS is enforced by **browsers**.

It is not an API authentication mechanism.

Therefore you still need:

```text
Authentication
Authorization
Rate limiting
Input validation
TLS
Security controls
```

---

# 16. CORS in a production AI API

For an AI application:

```text
React / Next.js
https://app.example.com
          │
          │ HTTPS
          ↓
      FastAPI
          │
    ┌─────┼─────┐
    ↓     ↓     ↓
Postgres Redis Qdrant
          │
          ↓
         LLM
```

You might configure:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://app.example.com",
    ],
    allow_credentials=True,
    allow_methods=[
        "GET",
        "POST",
    ],
    allow_headers=[
        "Authorization",
        "Content-Type",
        "X-Request-ID",
    ],
)
```

Notice `X-Request-ID` because your frontend may propagate a correlation ID.

---

# 17. CORS vs CSRF

These are also frequently confused.

### CORS

Controls:

```text
Which browser origins can make/read cross-origin requests?
```

### CSRF

Protects against:

```text
A malicious site causing a user's browser
to perform an authenticated action.
```

Especially when authentication uses cookies, you need to think about **CSRF protection separately**.

CORS does not replace CSRF protection.

---

# 18. Interview answer

If the interviewer asks:

> **"How do you implement CORS in FastAPI?"**

A strong answer is:

> **"I use FastAPI's `CORSMiddleware`. I explicitly configure the trusted frontend origins, allowed HTTP methods, required headers, and whether credentials are allowed. In production I don't use a wildcard origin; I load the allowed origins from environment-specific configuration. CORS is a browser security mechanism, not authentication, so I still use JWT/session authentication and authorization separately."**

Code:

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://app.example.com",
    ],
    allow_credentials=True,
    allow_methods=[
        "GET",
        "POST",
        "PUT",
        "DELETE",
    ],
    allow_headers=[
        "Authorization",
        "Content-Type",
        "X-Request-ID",
    ],
)
```

### Remember this:

```text
CORS
 ↓
Browser cross-origin policy

Authentication
 ↓
Who are you?

Authorization
 ↓
What are you allowed to do?

CSRF
 ↓
Can a malicious site trick the browser
into performing an authenticated action?
```

That distinction is what makes the answer **senior-level rather than just "add `CORSMiddleware`."**
