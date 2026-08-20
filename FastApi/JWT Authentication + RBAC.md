# JWT Authentication + RBAC in FastAPI

This is a **very common senior FastAPI interview question**. A strong answer should cover not only how to create a JWT, but also **password hashing, token validation, dependency injection, roles/permissions, expiration, refresh tokens, and where authorization belongs**.

The architecture I would use is:

```text
Client
  │
  │ Authorization: Bearer <JWT>
  ▼
FastAPI Router
  │
  ▼
JWT Authentication Dependency
  │
  ├── Decode token
  ├── Verify signature
  ├── Check expiration
  └── Extract user_id / tenant_id
  │
  ▼
Current User
  │
  ▼
RBAC Dependency
  │
  ├── Check role
  └── Check permission
  │
  ▼
Service Layer
  │
  ▼
Repository
  │
  ▼
Database
```

---

# 1. Authentication vs Authorization

First, distinguish these in an interview.

### Authentication

Answers:

> **Who are you?**

Example:

```text
JWT
  ↓
user_id = 123
  ↓
User = Sandeep
```

### Authorization

Answers:

> **What are you allowed to do?**

Example:

```text
User = Sandeep
Role = ADMIN

Can:
  ✓ create user
  ✓ delete user
  ✓ view reports
```

So:

```text
Authentication
     ↓
Who is the user?

Authorization / RBAC
     ↓
What can that user do?
```

---

# 2. What is JWT?

JWT stands for **JSON Web Token**.

A JWT generally contains:

```text
HEADER.PAYLOAD.SIGNATURE
```

For example:

```text
eyJhbGciOiJIUzI1NiIs...
.
eyJzdWIiOiIxMjMiLCJyb2xlIjoiYWRtaW4i...
.
signature
```

Conceptually:

```text
Header
{
    "alg": "HS256",
    "typ": "JWT"
}

Payload
{
    "sub": "123",
    "role": "admin",
    "exp": 1234567890
}

Signature
HMAC(...)
```

The signature allows the server to detect whether the token has been modified.

---

# 3. Important: JWT is signed, not encrypted

This is a common interview trap.

JWT payloads are generally **Base64URL encoded**, not encrypted.

So don't put:

```text
❌ password
❌ credit card number
❌ secrets
❌ sensitive personal information
```

inside the JWT payload.

Instead:

```json
{
    "sub": "user-123",
    "role": "admin",
    "exp": 1780000000
}
```

---

# 4. Install dependencies

A typical implementation can use:

```bash
pip install fastapi pyjwt pwdlib[argon2]
```

You could also use other well-maintained password/JWT libraries; the important architectural concepts are the same.

---

# 5. Password hashing

**Never store passwords directly.**

Bad:

```python
user.password = password
```

Instead:

```text
Password
   ↓
Argon2id
   ↓
Password hash
   ↓
Database
```

For example:

```python
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

If the user enters:

```text
MyPassword123
```

the database stores something like:

```text
$argon2id$v=19$...
```

not the original password.

---

# 6. User model

A simple SQLAlchemy model could look like:

```python
class User(Base):
    __tablename__ = "users"

    id = mapped_column(
        UUID,
        primary_key=True,
    )

    email = mapped_column(
        String,
        unique=True,
        nullable=False,
    )

    password_hash = mapped_column(
        String,
        nullable=False,
    )

    role = mapped_column(
        String,
        nullable=False,
        default="user",
    )

    is_active = mapped_column(
        Boolean,
        default=True,
    )
```

For a more mature RBAC system, I wouldn't necessarily keep a single `role` column. We'll get to that later.

---

# 7. JWT configuration

Keep secrets out of source code.

For example:

```python
from pydantic_settings import BaseSettings


class Settings(BaseSettings):

    jwt_secret_key: str
    jwt_algorithm: str = "HS256"

    access_token_expire_minutes: int = 15
```

Environment:

```text
JWT_SECRET_KEY=very-long-random-secret
```

In production, retrieve secrets from a proper secret-management system rather than committing them to Git.

---

# 8. Create an access token

Using PyJWT:

```python
from datetime import datetime, timedelta, timezone

import jwt


def create_access_token(
    user_id: str,
    role: str,
) -> str:

    now = datetime.now(timezone.utc)

    payload = {
        "sub": user_id,
        "role": role,
        "iat": now,
        "exp": now + timedelta(
            minutes=15
        ),
    }

    return jwt.encode(
        payload,
        settings.jwt_secret_key,
        algorithm=settings.jwt_algorithm,
    )
```

The important claims are:

### `sub`

Subject:

```text
Who does this token belong to?
```

### `exp`

Expiration:

```text
When does the token stop being valid?
```

### `iat`

Issued-at time:

```text
When was this token created?
```

You may also use:

```text
jti
aud
iss
```

depending on your security architecture.

---

# 9. Login flow

The login endpoint might be:

```python
@router.post("/login")
async def login(
    form: OAuth2PasswordRequestForm = Depends(),
    service: AuthService = Depends(
        get_auth_service
    ),
):
    return await service.login(
        email=form.username,
        password=form.password,
    )
```

The service:

```python
async def login(
    self,
    email: str,
    password: str,
):

    user = await self.user_repository.get_by_email(
        email
    )

    if not user:
        raise InvalidCredentialsError()

    if not verify_password(
        password,
        user.password_hash,
    ):
        raise InvalidCredentialsError()

    access_token = create_access_token(
        user_id=str(user.id),
        role=user.role,
    )

    return {
        "access_token": access_token,
        "token_type": "bearer",
    }
```

The client receives:

```json
{
    "access_token": "eyJhbGciOi...",
    "token_type": "bearer"
}
```

---

# 10. Client sends JWT

For subsequent requests:

```http
GET /users/me
Authorization: Bearer eyJhbGciOi...
```

FastAPI can extract the bearer token.

```python
from fastapi.security import OAuth2PasswordBearer

oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="/auth/login"
)
```

Then:

```python
async def get_token(
    token: str = Depends(oauth2_scheme),
):
    return token
```

---

# 11. Decode and validate JWT

Now create:

```python
from fastapi import HTTPException, status
import jwt


async def get_current_user(
    token: str = Depends(oauth2_scheme),
):

    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
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
        )

        user_id = payload.get("sub")

        if not user_id:
            raise credentials_exception

    except jwt.PyJWTError:
        raise credentials_exception

    return user_id
```

Now:

```python
@router.get("/me")
async def get_me(
    user_id = Depends(get_current_user),
):
    return {
        "user_id": user_id
    }
```

---

# 12. But don't trust the role blindly

A common implementation is:

```python
role = payload.get("role")
```

and then:

```python
if role != "admin":
    raise HTTPException(403)
```

This can be acceptable in some architectures, but there's an important design consideration:

> **JWT claims are only trustworthy if the token is properly signed and you accept that the authorization state is valid for the token's lifetime.**

For sensitive enterprise RBAC, you may want to load the current user and authorization state from your database/cache.

For example:

```python
async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db),
):

    payload = decode_token(token)

    user_id = payload["sub"]

    user = await user_repository.get_by_id(
        user_id
    )

    if not user or not user.is_active:
        raise HTTPException(401)

    return user
```

Then:

```text
JWT
 ↓
user_id
 ↓
Database / cache
 ↓
Current user
 ↓
Current role/permissions
```

This is particularly useful when administrators can revoke or change permissions.

---

# 13. Implement RBAC

RBAC means:

> **Role-Based Access Control**

For example:

```text
ADMIN
MANAGER
USER
VIEWER
```

Permissions:

```text
users:read
users:create
users:update
users:delete

reports:read

payments:create
payments:refund
```

A role maps to permissions.

```text
ADMIN
 ├── users:read
 ├── users:create
 ├── users:update
 ├── users:delete
 ├── reports:read
 └── payments:refund

MANAGER
 ├── users:read
 ├── users:update
 └── reports:read

USER
 ├── users:read
 └── reports:read
```

---

# 14. Simple RBAC implementation

For a small application:

```python
ROLE_PERMISSIONS = {
    "admin": {
        "users:read",
        "users:create",
        "users:update",
        "users:delete",
    },

    "manager": {
        "users:read",
        "users:update",
    },

    "user": {
        "users:read",
    },
}
```

Then:

```python
def require_permission(
    permission: str,
):

    async def dependency(
        user = Depends(get_current_user),
    ):

        permissions = ROLE_PERMISSIONS.get(
            user.role,
            set(),
        )

        if permission not in permissions:
            raise HTTPException(
                status_code=403,
                detail="Insufficient permissions",
            )

        return user

    return dependency
```

---

# 15. Use it in an endpoint

For example:

```python
@router.delete("/users/{user_id}")
async def delete_user(
    user_id: UUID,
    user = Depends(
        require_permission("users:delete")
    ),
):
    ...
```

The flow:

```text
Request
   ↓
JWT authentication
   ↓
Current user
   ↓
Get role
   ↓
Get permissions
   ↓
Check users:delete
   ↓
Allowed?
  / \
Yes  No
 ↓    ↓
200  403
```

---

# 16. Role-based dependency

Sometimes you specifically want:

> Only admins can access this endpoint.

You can create:

```python
def require_role(
    required_role: str,
):

    async def dependency(
        user = Depends(get_current_user),
    ):

        if user.role != required_role:
            raise HTTPException(
                status_code=403,
                detail="Insufficient permissions",
            )

        return user

    return dependency
```

Then:

```python
@router.delete("/users/{user_id}")
async def delete_user(
    user_id: UUID,
    admin = Depends(
        require_role("admin")
    ),
):
    ...
```

---

# 17. Role vs Permission

This is an important senior-level distinction.

Role-based:

```python
require_role("admin")
```

Permission-based:

```python
require_permission("users:delete")
```

I generally prefer **permission-based authorization** for larger systems.

Why?

Imagine:

```text
Today:

ADMIN → users:delete
```

Tomorrow:

```text
ADMIN → users:delete
SUPER_ADMIN → users:delete
USER_MANAGER → users:delete
```

If your endpoint checks:

```python
role == "admin"
```

you need to change application code.

If it checks:

```python
permission == "users:delete"
```

the role-to-permission mapping can change independently.

---

# 18. Production RBAC database design

For enterprise applications, I'd typically use:

```text
users
roles
permissions
user_roles
role_permissions
```

Relationship:

```text
User
 │
 │ many-to-many
 ▼
Role
 │
 │ many-to-many
 ▼
Permission
```

Example:

```text
users
----------------
id
email
password_hash
tenant_id


roles
----------------
id
name


permissions
----------------
id
name


user_roles
----------------
user_id
role_id


role_permissions
----------------
role_id
permission_id
```

---

# 19. Why this is better

Suppose:

```text
Alice → ADMIN
Bob   → MANAGER
John  → VIEWER
```

And:

```text
ADMIN
  ↓
users:create
users:delete
reports:read

MANAGER
  ↓
users:update
reports:read

VIEWER
  ↓
reports:read
```

You can modify permissions without modifying endpoint code.

---

# 20. Multi-tenant RBAC

For enterprise SaaS, this gets more interesting.

You might have:

```text
User
 │
 ├── tenant_id
 │
 └── roles
```

For example:

```text
Tenant A

Alice → Admin
Bob   → Analyst


Tenant B

John  → Admin
Mike  → Viewer
```

The same user may potentially have different permissions in different tenant contexts.

So the authorization model might become:

```text
User
  +
Tenant
  +
Role
  ↓
Permissions
```

Your dependency could therefore establish:

```python
async def get_current_context(
    user = Depends(get_current_user),
    tenant = Depends(get_current_tenant),
):
    ...
```

Then:

```text
JWT
 ↓
User
 ↓
Tenant
 ↓
Role
 ↓
Permission
```

---

# 21. JWT claims in multi-tenant systems

You might have:

```json
{
    "sub": "user-123",
    "tenant_id": "tenant-456",
    "iss": "auth-service",
    "aud": "api",
    "iat": 1780000000,
    "exp": 1780000900
}
```

But don't blindly trust authorization information merely because it's in the token.

For high-security systems, validate:

```text
signature
issuer
audience
expiration
user status
tenant membership
current permissions
```

---

# 22. Access token vs Refresh token

A production authentication system usually doesn't want a long-lived access token.

Instead:

```text
Login
  │
  ├── Access Token → short-lived
  │
  └── Refresh Token → longer-lived
```

For example:

```text
Access Token
15 minutes

Refresh Token
days/weeks
```

Then:

```text
Access token expires
       ↓
Client sends refresh token
       ↓
Auth service validates it
       ↓
New access token
```

---

# 23. Why short-lived access tokens?

Suppose an attacker steals an access token.

If:

```text
Access token = 30 days
```

the attacker potentially has access for a long time.

If:

```text
Access token = 15 minutes
```

the exposure window is much smaller.

You still need a secure refresh-token strategy.

---

# 24. Refresh token rotation

For higher-security applications:

```text
Refresh Token A
       ↓
Refresh endpoint
       ↓
Access Token B
+
Refresh Token B
       ↓
Invalidate A
```

This is called **refresh-token rotation**.

You can maintain refresh-token records:

```text
refresh_tokens
----------------
id
user_id
token_hash
expires_at
revoked_at
device_id
```

Don't casually store raw refresh tokens if your security model allows hashing them instead.

---

# 25. Logout with JWT

A pure stateless JWT access token has an interesting property:

If the server simply issued:

```text
JWT expires in 15 minutes
```

then "logout" doesn't magically invalidate the already-issued token.

For stronger revocation semantics, you can use:

```text
Short-lived access token
+
Revocable refresh token
```

or maintain a denylist/revocation mechanism when necessary.

---

# 26. 401 vs 403

This is another common interview question.

### `401 Unauthorized`

Means:

> Authentication is missing or invalid.

Examples:

```text
No token
Invalid token
Expired token
Bad signature
```

### `403 Forbidden`

Means:

> Authentication succeeded, but the user doesn't have permission.

Example:

```text
Valid JWT
User = viewer
Endpoint requires admin
```

Therefore:

```text
No/invalid identity → 401
Identity exists but insufficient permission → 403
```

---

# 27. Where should JWT logic live?

I would separate it.

For example:

```text
app/
├── auth/
│   ├── jwt.py
│   ├── password.py
│   └── dependencies.py
│
├── services/
│   └── auth_service.py
│
└── api/
    └── routes/
        └── auth.py
```

### `jwt.py`

Responsible for:

```text
create token
decode token
validate claims
```

### `password.py`

Responsible for:

```text
hash password
verify password
```

### `dependencies.py`

Responsible for:

```text
get_current_user
require_role
require_permission
```

### `auth_service.py`

Responsible for:

```text
login
refresh
logout/revocation
authentication workflow
```

---

# 28. Complete dependency chain

Your endpoint might look like:

```python
@router.delete("/users/{user_id}")
async def delete_user(
    user_id: UUID,
    current_user = Depends(
        require_permission("users:delete")
    ),
    service: UserService = Depends(
        get_user_service
    ),
):

    return await service.delete_user(
        user_id
    )
```

The dependency graph:

```text
                     Request
                        │
                        ▼
                 Authorization Header
                        │
                        ▼
                 OAuth2 Bearer
                        │
                        ▼
                     JWT
                        │
                 ┌──────┴──────┐
                 │             │
              Verify         Decode
                 │             │
                 └──────┬──────┘
                        ▼
                  Current User
                        │
                        ▼
                  RBAC Check
                        │
                 ┌──────┴──────┐
                 ▼             ▼
              Allowed         Denied
                 │              │
                 ▼              ▼
              Service          403
                 │
                 ▼
             Repository
                 │
                 ▼
             PostgreSQL
```

---

# 29. Security considerations interviewers expect

If you're interviewing for a senior role, don't stop at:

> "I generate a JWT and decode it."

Mention these:

### Password security

Use:

```text
Argon2id
```

or another strong password hashing scheme.

Never:

```text
MD5
SHA256(password)
plain text
```

---

### JWT secret management

Don't:

```python
SECRET_KEY = "mysecret123"
```

in source code.

Use:

```text
environment variables
secret manager
Kubernetes secrets
cloud secret-management service
```

---

### Short-lived access tokens

Use relatively short expiry.

---

### Refresh-token rotation

Useful for stronger security.

---

### Validate claims

Check:

```text
signature
exp
iat
iss
aud
sub
```

as appropriate.

---

### HTTPS

Always transmit authentication credentials over TLS.

---

### Don't put secrets in JWT

JWT payload isn't encryption.

---

### Rate-limit login

Protect:

```text
POST /login
POST /refresh
```

against brute-force and abuse.

---

### Audit authorization

For enterprise systems, log important events:

```text
user_id
tenant_id
action
resource
result
timestamp
request_id
```

Be careful not to log passwords or raw tokens.

---

# 30. JWT + RBAC in an AI platform

For the type of enterprise AI platform you've been discussing, I'd structure it approximately like this:

```text
                         Client
                           │
                           ▼
                    FastAPI Gateway
                           │
                           ▼
                    JWT Middleware/
                    Dependencies
                           │
                  ┌────────┴────────┐
                  ▼                 ▼
                User              Tenant
                  │                 │
                  └────────┬────────┘
                           ▼
                       RBAC/ABAC
                           │
                  ┌────────┴─────────┐
                  ▼                  ▼
               Permission         Resource
                  │                  │
                  └────────┬─────────┘
                           ▼
                       ChatService
                           │
             ┌─────────────┼─────────────┐
             ▼             ▼             ▼
          Postgres       Qdrant         LLM
             │             │             │
             └─────────────┼─────────────┘
                           ▼
                         Redis
```

For example:

```text
POST /tenants/{tenant_id}/documents/search
```

Before retrieval:

```text
JWT valid?
   ↓
User active?
   ↓
User belongs to tenant?
   ↓
Has documents:search permission?
   ↓
Apply tenant filter
   ↓
Qdrant search
```

That last step is extremely important:

> **RBAC should not be the only tenant-isolation mechanism.**

Even if a user has permission to search documents, your database/vector-store queries should also enforce the tenant boundary.

For example:

```python
results = await qdrant.search(
    query_vector,
    filter={
        "tenant_id": tenant_id
    }
)
```

So you have **defense in depth**:

```text
Authentication
     +
Authorization
     +
Tenant isolation
     +
Resource-level access control
```

---

# 31. RBAC vs ABAC

A senior interviewer may follow up:

> "Would you always use RBAC?"

Not necessarily.

### RBAC

```text
User → Role → Permission
```

Good for:

```text
Admin
Manager
Analyst
Viewer
```

### ABAC

Attribute-Based Access Control:

```text
User attributes
+
Resource attributes
+
Environment/context
        ↓
Authorization decision
```

For example:

```text
User:
department = finance

Document:
department = finance
classification = confidential

Context:
location = corporate_network
```

Policy:

```text
Allow if:
user.department == document.department
AND
user.clearance >= document.classification
```

For complex enterprise AI systems, you may eventually need:

```text
RBAC + ABAC
```

rather than role-only authorization.

---

# 32. Best interview answer

If the interviewer asks:

> **"How would you implement JWT authentication and RBAC in FastAPI?"**

A strong senior-level answer would be:

> "I'd separate authentication from authorization. During login, I'd verify the user's password using a strong password hashing algorithm such as Argon2id and issue a short-lived signed access JWT containing minimal claims such as `sub`, `iat`, and `exp`. For subsequent requests, FastAPI's OAuth2 bearer dependency extracts the token, and an authentication dependency validates the signature and claims and resolves the current user.
>
> For authorization, I'd implement RBAC using roles and permissions. Instead of hardcoding `role == admin` throughout the application, I'd preferably expose reusable dependencies such as `require_permission('users:delete')`. In a larger system I'd store users, roles, permissions, and their relationships in the database.
>
> I'd use short-lived access tokens and a secure refresh-token mechanism, potentially with refresh-token rotation and revocation. I'd return 401 for invalid authentication and 403 for authenticated users who lack permission.
>
> For a multi-tenant system, I'd additionally validate tenant membership and enforce tenant filters at the database and vector-store layers, because authorization checks alone aren't sufficient for data isolation. I'd also keep secrets in a secret manager, use HTTPS, rate-limit authentication endpoints, and audit sensitive authorization events."

---

## The architecture you should memorize

```text
              LOGIN
                │
                ▼
        Email + Password
                │
                ▼
        Verify Argon2 hash
                │
                ▼
          Create JWT
                │
                ▼
       Short-lived Access Token
                │
                ▼
     Authorization: Bearer JWT
                │
                ▼
        ┌───────────────┐
        │ JWT Validation│
        └───────┬───────┘
                ▼
          Current User
                │
                ▼
       Tenant / Context
                │
                ▼
        RBAC Permission
                │
          ┌─────┴─────┐
          ▼           ▼
       Allowed      Denied
          │           │
          ▼           ▼
       Service       403
          │
          ▼
     Repository
          │
          ▼
      Database
```

**One sentence to remember for the interview:**

> **JWT answers "who are you?", RBAC answers "what can you do?", and tenant/resource-level checks answer "what data are you allowed to access?"**
