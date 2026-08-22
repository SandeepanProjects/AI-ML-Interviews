Absolutely. These are **core senior-level FastAPI authentication and security interview questions**. The best way to learn them is to build one consistent authentication architecture and then explain each concept from that architecture.

I'll use:

* **FastAPI**
* **JWT**
* **OAuth2 Bearer authentication**
* **Argon2 password hashing**
* **PostgreSQL + SQLAlchemy**
* **Access + refresh tokens**
* **RBAC**
* **Token revocation**
* **Dependency Injection**

> For production, use a proper identity provider such as an enterprise OAuth/OIDC provider when appropriate. The code below demonstrates how the architecture works and is suitable for interview discussion.

---

# 1. How do you implement JWT authentication?

A typical architecture is:

```text
                    Client
                      │
                POST /auth/login
                      │
                      ↓
             Verify email/password
                      │
                      ↓
               Create JWT tokens
                      │
             ┌────────┴────────┐
             ↓                 ↓
       Access Token       Refresh Token
             │                 │
             ↓                 ↓
        API requests      /auth/refresh
             │
             ↓
      Validate JWT
             │
             ↓
       Current User
             │
             ↓
      Authorization/RBAC
             │
             ↓
          Endpoint
```

---

# 2. Project structure

A clean FastAPI authentication implementation could look like:

```text
app/
│
├── main.py
│
├── auth/
│   ├── router.py
│   ├── security.py
│   ├── jwt.py
│   └── dependencies.py
│
├── users/
│   ├── models.py
│   ├── repository.py
│   └── router.py
│
├── db/
│   ├── session.py
│   └── dependencies.py
│
├── core/
│   └── config.py
│
└── schemas/
    └── auth.py
```

---

# 3. User model

Let's start with the database.

```python
# app/users/models.py

from sqlalchemy import String, Boolean
from sqlalchemy.orm import Mapped, mapped_column

from app.db.session import Base


class User(Base):

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(
        primary_key=True
    )

    email: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        index=True,
        nullable=False,
    )

    password_hash: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    role: Mapped[str] = mapped_column(
        String(50),
        default="user",
        nullable=False,
    )

    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
    )
```

---

# 4. Password hashing

Never store passwords directly.

### ❌ Never do this

```python
user.password = password
```

Instead:

```text
Password
   ↓
Argon2
   ↓
Password hash
   ↓
Database
```

Use Argon2:

```bash
pip install pwdlib[argon2]
```

Then:

```python
# app/auth/security.py

from pwdlib import PasswordHash


password_hash = PasswordHash.recommended()


def hash_password(password: str) -> str:
    return password_hash.hash(password)


def verify_password(
    password: str,
    hashed_password: str,
) -> bool:
    return password_hash.verify(
        password,
        hashed_password,
    )
```

---

# 5. JWT configuration

```python
# app/auth/jwt.py

from datetime import datetime, timedelta, timezone

from jose import jwt, JWTError


SECRET_KEY = "CHANGE_ME_IN_PRODUCTION"

ALGORITHM = "HS256"

ACCESS_TOKEN_EXPIRE_MINUTES = 15

REFRESH_TOKEN_EXPIRE_DAYS = 7
```

In production:

```text
❌ Hardcode SECRET_KEY

SECRET_KEY = "abc123"
```

Instead:

```text
Environment / Secret Manager
             ↓
        SECRET_KEY
```

For example:

```python
from pydantic_settings import BaseSettings


class Settings(BaseSettings):

    jwt_secret_key: str

    jwt_algorithm: str = "HS256"

    access_token_expire_minutes: int = 15

    refresh_token_expire_days: int = 7

    class Config:
        env_file = ".env"
```

---

# 6. Creating JWT tokens

JWT contains three components:

```text
HEADER.PAYLOAD.SIGNATURE
```

Example payload:

```json
{
    "sub": "123",
    "type": "access",
    "role": "admin",
    "exp": 1780000000
}
```

Create a helper:

```python
# app/auth/jwt.py

def create_token(
    user_id: int,
    token_type: str,
    expires_delta: timedelta,
    role: str | None = None,
) -> str:

    now = datetime.now(timezone.utc)

    payload = {
        "sub": str(user_id),
        "type": token_type,
        "iat": now,
        "exp": now + expires_delta,
    }

    if role is not None:
        payload["role"] = role

    return jwt.encode(
        payload,
        SECRET_KEY,
        algorithm=ALGORITHM,
    )
```

Access token:

```python
def create_access_token(
    user_id: int,
    role: str,
) -> str:

    return create_token(
        user_id=user_id,
        token_type="access",
        expires_delta=timedelta(
            minutes=ACCESS_TOKEN_EXPIRE_MINUTES
        ),
        role=role,
    )
```

Refresh token:

```python
def create_refresh_token(
    user_id: int,
) -> str:

    return create_token(
        user_id=user_id,
        token_type="refresh",
        expires_delta=timedelta(
            days=REFRESH_TOKEN_EXPIRE_DAYS
        ),
    )
```

---

# 7. Login endpoint

Request schema:

```python
# app/schemas/auth.py

from pydantic import BaseModel, EmailStr


class LoginRequest(BaseModel):

    email: EmailStr

    password: str


class TokenResponse(BaseModel):

    access_token: str

    refresh_token: str

    token_type: str = "bearer"
```

Login:

```python
# app/auth/router.py

from fastapi import APIRouter, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.jwt import (
    create_access_token,
    create_refresh_token,
)
from app.auth.security import verify_password
from app.schemas.auth import (
    LoginRequest,
    TokenResponse,
)
from app.users.models import User

router = APIRouter(
    prefix="/auth",
    tags=["Authentication"],
)


@router.post(
    "/login",
    response_model=TokenResponse,
)
async def login(
    request: LoginRequest,
    db: AsyncSession,
):

    result = await db.execute(
        select(User).where(
            User.email == request.email
        )
    )

    user = result.scalar_one_or_none()

    if user is None:
        raise HTTPException(
            status_code=401,
            detail="Invalid credentials",
        )

    if not verify_password(
        request.password,
        user.password_hash,
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid credentials",
        )

    if not user.is_active:
        raise HTTPException(
            status_code=403,
            detail="User is inactive",
        )

    access_token = create_access_token(
        user.id,
        user.role,
    )

    refresh_token = create_refresh_token(
        user.id
    )

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
    )
```

In real code, `db` would normally be injected using:

```python
db: AsyncSession = Depends(get_db)
```

---

# 8. What is OAuth2?

OAuth 2.0 is an **authorization framework**.

It defines mechanisms for allowing a client/application to obtain and use access tokens to access protected resources.

A simplified flow:

```text
User
 │
 ↓
Authorization Server
 │
 ↓
Access Token
 │
 ↓
Client
 │
 ↓
Resource/API Server
```

Important distinction:

> **OAuth2 itself is not a JWT format.**

OAuth2 can use JWT access tokens, but it doesn't require JWT.

---

# 9. OAuth2 vs JWT

This is a very common interview trap.

They are not competing technologies.

```text
OAuth2
   ↓
Authorization framework

JWT
   ↓
Token format
```

You can have:

```text
OAuth2 + JWT
```

For example:

```text
OAuth2 authorization flow
        ↓
JWT access token
        ↓
FastAPI API
```

---

# 10. What is OpenID Connect?

For authentication, you'll often encounter **OpenID Connect (OIDC)**.

Think:

```text
OAuth2
   ↓
Authorization

OIDC
   ↓
Authentication / identity
```

OIDC is built on top of OAuth 2.0.

Enterprise identity providers commonly use OAuth2/OIDC.

---

# 11. JWT vs Session Authentication

This is another very common interview question.

## Session-based authentication

Flow:

```text
Login
 ↓
Server creates session
 ↓
Session stored server-side
 ↓
Client receives session cookie
 ↓
Client sends cookie
 ↓
Server looks up session
```

Example:

```text
Cookie:
session_id=abc123
```

Server:

```text
abc123 → User 123
```

---

## JWT authentication

Flow:

```text
Login
 ↓
Server creates JWT
 ↓
Client receives token
 ↓
Client sends JWT
 ↓
Server validates token
 ↓
User identity obtained
```

JWT can be stateless:

```text
JWT
 ↓
Signature verification
 ↓
User ID
```

---

# 12. JWT vs Session comparison

| Feature            | JWT                       | Session                         |
| ------------------ | ------------------------- | ------------------------------- |
| State              | Can be stateless          | Server-side state               |
| Storage            | Client-side token         | Server-side session             |
| Revocation         | More complicated          | Usually easier                  |
| Horizontal scaling | Easy if stateless         | Requires shared session store   |
| Token size         | Larger                    | Small session ID                |
| Immediate logout   | More complex              | Easy                            |
| Distributed APIs   | Convenient                | Requires shared session storage |
| Microservices      | Often useful              | More infrastructure             |
| Security           | Depends on implementation | Depends on implementation       |

Important:

> JWT isn't automatically more secure than sessions.

The security depends on how you implement them.

---

# 13. Access token vs Refresh token

A good production design separates them.

### Access token

Used for API requests:

```http
Authorization: Bearer <access-token>
```

Short lifetime:

```text
5–15 minutes
```

### Refresh token

Used to obtain a new access token.

Longer lifetime:

```text
Days/weeks
```

Flow:

```text
Login
 │
 ├── Access Token ───→ API
 │
 └── Refresh Token
          │
          ↓
     /auth/refresh
          │
          ↓
    New Access Token
```

Why short-lived access tokens?

If stolen:

```text
Attacker
   ↓
Access token
   ↓
Limited lifetime
```

The damage window is smaller.

---

# 14. Where should tokens be stored?

This is nuanced and important.

## Browser application

For sensitive browser applications, a common recommendation is:

### Access token

Keep it in memory where practical.

Avoid:

```javascript
localStorage.setItem(
    "access_token",
    token
)
```

because an XSS vulnerability can expose localStorage contents.

### Refresh token

A common approach is:

```http
Set-Cookie:
refresh_token=...
HttpOnly;
Secure;
SameSite=Lax;
```

This means JavaScript cannot directly read the cookie.

Conceptually:

```text
Browser
│
├── Access token → memory
│
└── Refresh token → HttpOnly Secure cookie
```

For cookie-based authentication, you also need to think carefully about **CSRF protections**, especially depending on your `SameSite` configuration and application architecture.

---

## Mobile applications

For iOS/Android, use platform secure storage:

```text
iOS
 ↓
Keychain

Android
 ↓
Keystore / encrypted secure storage
```

Don't put long-lived credentials in plain local storage.

---

# 15. How do you validate JWT tokens?

JWT validation should include more than just decoding.

You should verify:

```text
1. Signature
2. Algorithm
3. Expiration
4. Token type
5. Required claims
6. Issuer
7. Audience
8. User status
9. Revocation state, if applicable
```

Example:

```python
from jose import jwt, JWTError

EXPECTED_ISSUER = "my-auth-service"
EXPECTED_AUDIENCE = "my-api"


def decode_access_token(token: str) -> dict:

    try:

        payload = jwt.decode(
            token,
            SECRET_KEY,
            algorithms=[ALGORITHM],
            issuer=EXPECTED_ISSUER,
            audience=EXPECTED_AUDIENCE,
        )

        if payload.get("type") != "access":
            raise ValueError(
                "Not an access token"
            )

        user_id = payload.get("sub")

        if not user_id:
            raise ValueError(
                "Missing subject"
            )

        return payload

    except JWTError as exc:
        raise ValueError(
            "Invalid token"
        ) from exc
```

Never do this:

```python
jwt.decode(
    token,
    SECRET_KEY,
    options={
        "verify_signature": False
    }
)
```

for authentication.

---

# 16. How do you protect an endpoint?

Create:

```python
get_current_user()
```

Then:

```python
@router.get("/profile")
async def profile(
    current_user: User = Depends(
        get_current_user
    ),
):
    return {
        "id": current_user.id,
        "email": current_user.email,
    }
```

Now an unauthenticated request is rejected.

Architecture:

```text
GET /profile
      │
      ↓
get_current_user()
      │
      ├── Extract token
      ├── Verify signature
      ├── Check expiry
      ├── Validate claims
      └── Load user
             │
             ↓
          Endpoint
```

---

# 17. Create `get_current_user`

Here's a clean implementation:

```python
# app/auth/dependencies.py

from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="/auth/login"
)


async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db),
) -> User:

    try:
        payload = decode_access_token(token)

    except ValueError:

        raise HTTPException(
            status_code=401,
            detail="Invalid authentication credentials",
            headers={
                "WWW-Authenticate": "Bearer"
            },
        )

    user_id = int(payload["sub"])

    result = await db.execute(
        select(User).where(
            User.id == user_id
        )
    )

    user = result.scalar_one_or_none()

    if user is None:
        raise HTTPException(
            status_code=401,
            detail="User not found",
        )

    if not user.is_active:
        raise HTTPException(
            status_code=403,
            detail="Inactive user",
        )

    return user
```

---

# 18. Authentication vs Authorization

This is extremely important.

## Authentication

> **Who are you?**

Example:

```text
JWT
 ↓
user_id = 123
 ↓
User 123 authenticated
```

## Authorization

> **What are you allowed to do?**

Example:

```text
User 123
role = user

DELETE /users/456
        ↓
403 Forbidden
```

because they don't have permission.

---

# 19. How do you implement RBAC?

RBAC = **Role-Based Access Control**.

For example:

```text
admin
manager
user
viewer
```

Database:

```text
users
--------------------------------
id | email | role
--------------------------------
1  | a@x.com | admin
2  | b@x.com | user
3  | c@x.com | viewer
```

Create a dependency:

```python
from fastapi import Depends, HTTPException


def require_role(required_role: str):

    async def role_checker(
        current_user: User = Depends(
            get_current_user
        ),
    ) -> User:

        if current_user.role != required_role:

            raise HTTPException(
                status_code=403,
                detail="Insufficient permissions",
            )

        return current_user

    return role_checker
```

Then:

```python
@router.delete("/users/{user_id}")
async def delete_user(
    user_id: int,
    admin: User = Depends(
        require_role("admin")
    ),
):
    return {
        "message": "User deleted"
    }
```

Now:

```text
Request
  ↓
JWT
  ↓
get_current_user()
  ↓
require_role("admin")
  ↓
Endpoint
```

---

# 20. Better RBAC: multiple roles

You often don't want:

```python
role == "admin"
```

only.

Create:

```python
def require_roles(
    *allowed_roles: str
):

    async def checker(
        current_user: User = Depends(
            get_current_user
        ),
    ) -> User:

        if current_user.role not in allowed_roles:

            raise HTTPException(
                status_code=403,
                detail="Insufficient permissions",
            )

        return current_user

    return checker
```

Then:

```python
@router.post("/documents")
async def create_document(
    user: User = Depends(
        require_roles(
            "admin",
            "editor",
        )
    ),
):
    ...
```

Now:

```text
admin  → allowed
editor → allowed
user   → forbidden
viewer → forbidden
```

---

# 21. How do you handle expired tokens?

JWT contains:

```json
{
    "sub": "123",
    "exp": 1780000000
}
```

When the token expires, JWT validation fails.

Return:

```http
401 Unauthorized
```

For example:

```python
try:

    payload = jwt.decode(
        token,
        SECRET_KEY,
        algorithms=[ALGORITHM],
    )

except JWTError:

    raise HTTPException(
        status_code=401,
        detail="Token expired or invalid",
    )
```

A better implementation can distinguish expiration from other validation failures if desired.

The client then uses its refresh token:

```text
Access token expired
        ↓
POST /auth/refresh
        ↓
Validate refresh token
        ↓
Issue new access token
        ↓
Retry API request
```

---

# 22. How do you implement token refresh?

Don't simply accept any JWT and issue a new token.

Use a separate refresh token.

Request:

```json
{
    "refresh_token": "..."
}
```

For a browser architecture, the refresh token is often sent via a secure HttpOnly cookie instead.

Example:

```python
class RefreshRequest(BaseModel):
    refresh_token: str
```

Endpoint:

```python
@router.post(
    "/refresh",
    response_model=TokenResponse,
)
async def refresh_token(
    request: RefreshRequest,
    db: AsyncSession = Depends(get_db),
):

    try:
        payload = decode_refresh_token(
            request.refresh_token
        )

    except ValueError:

        raise HTTPException(
            status_code=401,
            detail="Invalid refresh token",
        )

    user_id = int(payload["sub"])

    # Verify refresh token exists
    # and has not been revoked.

    user = await get_user(
        db,
        user_id,
    )

    if user is None or not user.is_active:
        raise HTTPException(
            status_code=401,
            detail="Invalid user",
        )

    access_token = create_access_token(
        user_id=user.id,
        role=user.role,
    )

    return TokenResponse(
        access_token=access_token,
        refresh_token=request.refresh_token,
    )
```

---

# 23. Production refresh-token design

For production systems, I recommend **refresh-token rotation**.

Instead of:

```text
Refresh Token A
      ↓
Access Token B

Refresh Token A
      ↓
Access Token C

Refresh Token A
      ↓
Access Token D
```

use:

```text
Refresh Token A
      ↓
Access Token B
Refresh Token B
      ↓
Revoke A
```

Then:

```text
Refresh A
   ↓
Revoke A
   ↓
Issue B
```

If somebody tries to reuse A:

```text
Refresh A
   ↓
Already revoked
   ↓
Reject
   ↓
Potential token-family compromise
```

This is called **refresh token rotation**.

---

# 24. How do you revoke tokens?

This is where JWT has a major tradeoff.

A self-contained JWT is normally valid until its expiration.

Suppose:

```text
Access token
expires in 15 minutes
```

The user logs out after 1 minute.

The token itself hasn't expired.

So how do you revoke it?

There are several strategies.

---

# 25. Strategy 1 — Short-lived access tokens

Use:

```text
Access token: 5–15 minutes
Refresh token: longer lifetime
```

Then stolen access tokens have a limited lifetime.

This is often the simplest approach.

---

# 26. Strategy 2 — Refresh-token revocation

Store refresh-token records in PostgreSQL or Redis.

For example:

```text
refresh_tokens
------------------------------------------------
id
user_id
token_hash
expires_at
revoked_at
created_at
```

Important: **store a hash of the refresh token**, not necessarily the raw token.

Logout:

```python
async def revoke_refresh_token(
    token_hash: str,
):
    ...
```

Then:

```text
Refresh request
      ↓
Hash presented token
      ↓
Find DB record
      ↓
revoked_at?
      │
 ┌────┴────┐
 No        Yes
 ↓          ↓
Allow     Reject
```

---

# 27. Strategy 3 — JWT denylist / blacklist

For high-security systems, you can include a JWT ID:

```json
{
    "sub": "123",
    "jti": "unique-token-id",
    "exp": 1780000000
}
```

Store revoked IDs:

```text
Redis

revoked:jti:abc123
TTL = remaining token lifetime
```

During authentication:

```python
if await redis.exists(
    f"revoked:jti:{jti}"
):
    raise HTTPException(
        status_code=401,
        detail="Token revoked",
    )
```

This gives you immediate revocation, but introduces state and Redis dependency.

---

# 28. JWT logout tradeoff

This is an important interview point.

> **JWT is often described as stateless, but immediate revocation requires some server-side state such as a denylist, token version, session store, or refresh-token database.**

So:

```text
Pure JWT
   ↓
Stateless
   ↓
Easy horizontal scaling
   ↓
Hard immediate revocation
```

Whereas:

```text
Session
   ↓
Stateful
   ↓
Easy revocation
```

---

# 29. How do you secure API endpoints?

Authentication is only one layer.

A production API should use **defense in depth**.

Think:

```text
                    API
                     │
              ┌──────┴──────┐
              ↓             ↓
         Authentication   Rate Limit
              │
              ↓
         Authorization
              │
              ↓
         Input Validation
              │
              ↓
          Business Logic
              │
              ↓
        Database Security
              │
              ↓
        Audit / Monitoring
```

---

# 30. Use HTTPS

Never send credentials/tokens over plain HTTP in production.

```text
Client
   │
 HTTPS
   ↓
Load Balancer
   ↓
FastAPI
```

Use TLS everywhere, especially between services when appropriate.

---

# 31. Validate all input

FastAPI + Pydantic helps here.

```python
class TransferRequest(BaseModel):

    account_id: int

    amount: Decimal

    currency: Literal[
        "USD",
        "EUR",
        "INR",
    ]

    @field_validator("amount")
    @classmethod
    def validate_amount(cls, value):
        if value <= 0:
            raise ValueError(
                "Amount must be positive"
            )

        return value
```

Then:

```python
@router.post("/transfer")
async def transfer(
    request: TransferRequest,
    user: User = Depends(
        get_current_user
    ),
):
    ...
```

Never blindly trust request data.

---

# 32. Use RBAC/permissions

Don't just authenticate:

```python
user = Depends(get_current_user)
```

Also authorize:

```python
user = Depends(
    require_roles(
        "admin",
        "manager",
    )
)
```

Authentication:

```text
Who are you?
```

Authorization:

```text
Are you allowed?
```

---

# 33. Rate limiting

For login:

```text
POST /auth/login
```

you should protect against brute force.

For example:

```text
5 attempts/minute/IP
```

For expensive AI endpoints:

```text
POST /chat
```

you might additionally apply:

```text
requests/user/minute
tokens/user/day
cost/user/month
```

In distributed deployments, Redis is commonly used for centralized rate-limit state.

---

# 34. Protect sensitive endpoints

For example:

```python
@router.delete("/users/{user_id}")
async def delete_user(
    user_id: int,
    admin: User = Depends(
        require_roles("admin")
    ),
):
    ...
```

This gives:

```text
JWT authentication
        ↓
User identification
        ↓
Admin authorization
        ↓
Delete
```

---

# 35. Prevent IDOR

This is a very important API security concept.

Suppose:

```http
GET /documents/123
```

User A changes:

```http
GET /documents/124
```

If document 124 belongs to User B, your API must not return it merely because the JWT is valid.

Bad:

```python
document = await repo.get(document_id)
return document
```

Better:

```python
document = await repo.get_for_user(
    document_id=document_id,
    user_id=current_user.id,
)
```

Database query:

```python
select(Document).where(
    Document.id == document_id,
    Document.owner_id == current_user.id,
)
```

This is especially important in **multi-tenant AI/RAG systems**.

---

# 36. Multi-tenant security

If your application has:

```text
Tenant A
 ├── Document 1
 └── Document 2

Tenant B
 ├── Document 3
 └── Document 4
```

A request from Tenant A must never retrieve Tenant B's data.

Your query should include tenant isolation:

```python
select(Document).where(
    Document.id == document_id,
    Document.tenant_id == current_user.tenant_id,
)
```

And your vector search should also filter:

```python
qdrant_filter = {
    "must": [
        {
            "key": "tenant_id",
            "match": {
                "value": current_user.tenant_id
            },
        }
    ]
}
```

The security boundary must exist at the **database/vector retrieval layer**, not just in the frontend.

---

# 37. Secure JWT configuration

Avoid weak settings:

```python
SECRET_KEY = "password123"
```

Use:

```text
Secret Manager
     ↓
Environment
     ↓
FastAPI configuration
```

And use a strong randomly generated secret.

For asymmetric architectures, you can use:

```text
RS256 / ES256
```

where:

```text
Private key
   ↓
Identity/Auth service signs

Public key
   ↓
API services verify
```

This is particularly useful when many services need to verify tokens without possessing the signing private key.

---

# 38. Don't put sensitive data inside JWT

JWT payloads are generally **encoded, not encrypted**.

Don't put:

```json
{
    "password": "...",
    "credit_card": "...",
    "secret": "..."
}
```

A JWT payload should contain only what is needed.

For example:

```json
{
    "sub": "123",
    "role": "admin",
    "iss": "auth-service",
    "aud": "my-api",
    "exp": 1780000000
}
```

Remember:

```text
JWT ≠ encrypted data
```

---

# 39. Complete authentication dependency

Putting the pieces together:

```python
# auth/dependencies.py

from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession


oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="/auth/login"
)


async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db),
) -> User:

    credentials_exception = HTTPException(
        status_code=401,
        detail="Invalid authentication credentials",
        headers={
            "WWW-Authenticate": "Bearer"
        },
    )

    try:

        payload = jwt.decode(
            token,
            settings.jwt_secret_key,
            algorithms=[
                settings.jwt_algorithm
            ],
            issuer=settings.jwt_issuer,
            audience=settings.jwt_audience,
        )

        if payload.get("type") != "access":
            raise credentials_exception

        user_id = payload.get("sub")

        if not user_id:
            raise credentials_exception

    except JWTError:

        raise credentials_exception

    result = await db.execute(
        select(User).where(
            User.id == int(user_id)
        )
    )

    user = result.scalar_one_or_none()

    if user is None:
        raise credentials_exception

    if not user.is_active:
        raise HTTPException(
            status_code=403,
            detail="User is inactive",
        )

    return user
```

---

# 40. Complete protected endpoint

```python
@router.get("/profile")
async def get_profile(
    current_user: User = Depends(
        get_current_user
    ),
):
    return {
        "id": current_user.id,
        "email": current_user.email,
        "role": current_user.role,
    }
```

Request:

```http
GET /profile
Authorization: Bearer <access_token>
```

Flow:

```text
Request
   ↓
Extract Bearer token
   ↓
Decode JWT
   ↓
Verify signature
   ↓
Check exp
   ↓
Check issuer/audience/type
   ↓
Extract sub
   ↓
Load user
   ↓
Check active
   ↓
current_user
   ↓
Endpoint
```

---

# 41. Complete RBAC implementation

```python
def require_roles(
    *allowed_roles: str,
):

    async def checker(
        current_user: User = Depends(
            get_current_user
        ),
    ) -> User:

        if current_user.role not in allowed_roles:

            raise HTTPException(
                status_code=403,
                detail="Insufficient permissions",
            )

        return current_user

    return checker
```

Admin endpoint:

```python
@router.delete("/users/{user_id}")
async def delete_user(
    user_id: int,
    current_user: User = Depends(
        require_roles("admin")
    ),
):
    ...
```

Admin + manager:

```python
@router.post("/documents")
async def create_document(
    current_user: User = Depends(
        require_roles(
            "admin",
            "manager",
        )
    ),
):
    ...
```

---

# 42. What happens when access token expires?

Let's say:

```text
Access token
expires in 15 minutes
```

After 15 minutes:

```text
Client
  │
  │ expired access token
  ↓
FastAPI
  │
  ↓
JWT validation
  │
  ↓
401 Unauthorized
```

Client:

```text
401
 ↓
Use refresh token
 ↓
POST /auth/refresh
 ↓
New access token
 ↓
Retry original request
```

---

# 43. Complete token lifecycle

This is the diagram I'd remember for interviews:

```text
                  LOGIN
                    │
                    ↓
            Verify credentials
                    │
                    ↓
          ┌─────────┴─────────┐
          ↓                   ↓
     Access Token        Refresh Token
     5–15 minutes          days/weeks
          │                   │
          ↓                   │
     API requests             │
          │                   │
          ↓                   │
       expires                │
          │                   ↓
          └────────────→ /auth/refresh
                              │
                              ↓
                       Validate refresh
                              │
                              ↓
                       Rotate/revoke
                              │
                              ↓
                      New access token
```

---

# 44. Senior interview answers

### 1. How do you implement JWT authentication?

> **"I authenticate the user's credentials, issue a short-lived signed access token and a longer-lived refresh token, and protect API endpoints with a FastAPI `get_current_user` dependency. The dependency extracts the bearer token, validates the signature and claims such as `exp`, `iss`, `aud`, and token type, then loads the user and injects it into the endpoint."**

---

### 2. What is OAuth2?

> **"OAuth 2.0 is an authorization framework for obtaining access tokens and accessing protected resources. JWT is a token format, so OAuth2 and JWT are not alternatives. An OAuth2 system can use JWTs as access tokens."**

---

### 3. JWT vs session authentication?

> **"A session stores authentication state server-side and gives the client a session identifier, usually in a cookie. JWT can carry the authentication claims in a signed token, allowing stateless verification. JWT makes distributed APIs easier to scale but immediate revocation is more difficult. Sessions make revocation straightforward but require shared server-side session state when horizontally scaling."**

---

### 4. Access token vs refresh token?

> **"The access token is short-lived and is sent to protected APIs. The refresh token is longer-lived and is used only to obtain a new access token. I keep access tokens short-lived and use refresh-token rotation to reduce the impact of token theft."**

---

### 5. Where should tokens be stored?

> **"For browser applications, I generally avoid localStorage for sensitive long-lived tokens because XSS can expose it. A common architecture is to keep the access token in memory and put the refresh token in a Secure, HttpOnly, appropriately configured SameSite cookie, with CSRF protections where needed. Native mobile apps should use platform secure storage such as iOS Keychain."**

---

### 6. How do you validate JWT tokens?

> **"I verify the cryptographic signature using the expected algorithm and key, validate expiration and required claims such as subject, issuer, audience and token type, and then verify that the user is active and authorized. For systems requiring immediate revocation, I additionally check token/session state."**

---

### 7. How do you implement RBAC?

> **"I authenticate the user first and then use a reusable FastAPI dependency such as `require_roles('admin')` to check the user's role or permissions before allowing the endpoint to execute."**

---

### 8. Authentication vs authorization?

> **"Authentication determines who the user is. Authorization determines what that authenticated user is allowed to do."**

```text
Authentication
      ↓
Who are you?

Authorization
      ↓
What can you do?
```

---

### 9. How do you protect an endpoint?

```python
@router.get("/profile")
async def profile(
    user: User = Depends(
        get_current_user
    ),
):
    ...
```

> **"I protect the endpoint with a `get_current_user` dependency that validates the access token. For privileged endpoints I chain an authorization dependency such as `require_roles`."**

---

### 10. How do you handle expired tokens?

> **"JWT validation rejects the expired access token with 401. The client then uses its valid refresh token to request a new access token. I don't extend an expired access token indefinitely."**

---

### 11. How do you implement token refresh?

> **"I issue a short-lived access token and longer-lived refresh token at login. When the access token expires, the client sends the refresh token to a dedicated refresh endpoint. The server validates the refresh token, checks its server-side state if applicable, rotates it, and issues a new access token."**

---

### 12. How do you revoke tokens?

> **"Because a self-contained JWT normally remains valid until expiration, immediate revocation requires state. I typically use short-lived access tokens plus server-side refresh-token revocation and rotation. For higher-security scenarios I can maintain a JWT `jti` denylist in Redis with a TTL matching the token lifetime."**

---

### 13. How do you secure API endpoints?

> **"I use HTTPS, strong authentication, authorization/RBAC, strict request validation, short-lived access tokens, secure refresh-token handling, rate limiting, secure secret management, audit logging, dependency and package security, database authorization, tenant isolation, and monitoring. For a multi-tenant RAG system, I also enforce `tenant_id` filtering at both PostgreSQL and vector-search layers so authentication cannot accidentally expose another tenant's data."**

---

# The architecture you should remember

For a **senior AI/FastAPI interview**, this is the strongest mental model:

```text
                         CLIENT
                           │
                           │ HTTPS
                           ↓
                    FastAPI API
                           │
                           ↓
                OAuth2 Bearer Token
                           │
                           ↓
                 get_current_user()
                           │
              ┌────────────┴────────────┐
              ↓                         ↓
        Verify JWT                 Check User
        ├─ signature               ├─ active
        ├─ exp                     └─ tenant
        ├─ iss
        ├─ aud
        └─ token type
              │
              ↓
         Current User
              │
              ↓
       Authorization/RBAC
              │
       ┌──────┴──────┐
       ↓             ↓
    allowed        denied
       │             │
       ↓             ↓
   Business        403
     Logic
       │
       ↓
 PostgreSQL / Redis / Qdrant / LLM
```

And the **production security model** is:

```text
Authentication
      ↓
Authorization
      ↓
Tenant Isolation
      ↓
Input Validation
      ↓
Business Rules
      ↓
Database/Vector Access
      ↓
Audit + Monitoring
```

If you can explain that architecture clearly in an interview, rather than just saying "`Depends(get_current_user)`", you're demonstrating **senior-level FastAPI security knowledge**.
