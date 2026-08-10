Yes. **Authentication** is one of the most important patterns in a real FastAPI project because almost every enterprise API needs to answer two questions:

1. **Authentication — Who are you?**
2. **Authorization — What are you allowed to do?**

A production architecture usually looks like:

```text
Client
  |
  |  Login credentials
  v
Authentication API
  |
  v
Identity verification
  |
  v
Access Token
  |
  v
Client
  |
  | Authorization: Bearer <token>
  v
FastAPI
  |
  v
Authentication Dependency
  |
  v
Current User
  |
  v
Authorization / RBAC
  |
  v
Service
```

---

# 1. Authentication vs Authorization

This distinction is extremely important.

### Authentication

> "Are you really Sandeep?"

For example:

```text
email + password
       ↓
verify password
       ↓
user identity
       ↓
issue access token
```

### Authorization

> "Sandeep is authenticated. Is he allowed to delete this document?"

```text
User
 ↓
Role
 ↓
Permission
 ↓
Allow / Deny
```

So:

```text
Authentication → WHO are you?

Authorization → WHAT can you do?
```

---

# 2. Common authentication patterns

In real-world applications, you'll commonly encounter:

```text
1. Session-based authentication
2. JWT authentication
3. OAuth 2.0
4. OpenID Connect
5. API keys
6. Service-to-service authentication
7. mTLS
```

For modern FastAPI enterprise applications, a common architecture is:

```text
User
 ↓
OAuth2 / OIDC Identity Provider
 ↓
Access Token
 ↓
FastAPI
 ↓
JWT validation
 ↓
Authorization
```

Examples of identity providers include:

* Microsoft Entra ID
* Auth0
* Okta
* Keycloak
* Amazon Cognito

---

# 3. JWT authentication

JWT is extremely common in API architectures.

A JWT looks conceptually like:

```text
xxxxx.yyyyy.zzzzz
```

It has:

```text
Header
Payload
Signature
```

For example, the payload could contain:

```json
{
  "sub": "12345",
  "tenant_id": "tenant-001",
  "role": "admin",
  "exp": 1786350000
}
```

Important fields:

```text
sub       → user identifier
tenant_id → tenant
role      → role
exp       → expiration
iss       → issuer
aud       → audience
```

**Do not blindly trust these claims.** The token must be cryptographically validated and claims such as `iss`, `aud`, and expiration should be checked according to your identity provider's configuration.

---

# 4. JWT flow

genui{"data_networks_databases_learning_block":{"type_id":"NETWORK_FAULT_TOLERANCE"}}

The typical flow:

```text
                 LOGIN
                   |
                   v
              Identity
               Provider
                   |
                   v
             Access Token
                   |
                   v
                Client
                   |
                   | Authorization:
                   | Bearer <token>
                   v
                FastAPI
                   |
                   v
             Validate JWT
                   |
                   v
             Current User
                   |
                   v
             Authorization
                   |
                   v
                Service
```

---

# 5. FastAPI authentication dependency

FastAPI provides security utilities.

For example:

```python
from fastapi.security import (
    OAuth2PasswordBearer,
)

oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="/api/v1/auth/login"
)
```

Then:

```python
from fastapi import Depends

async def get_current_user(
    token: str = Depends(oauth2_scheme),
):
    ...
```

FastAPI extracts:

```http
Authorization: Bearer eyJ...
```

and gives the token to:

```python
get_current_user()
```

---

# 6. Validate the token

A simplified example using PyJWT:

```python
import jwt

from fastapi import HTTPException, status


SECRET_KEY = "..."

ALGORITHM = "HS256"


def decode_access_token(token: str):

    try:

        payload = jwt.decode(
            token,
            SECRET_KEY,
            algorithms=[ALGORITHM],
            options={
                "require": [
                    "exp",
                    "sub",
                ]
            },
        )

        return payload

    except jwt.ExpiredSignatureError:

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token expired",
        )

    except jwt.InvalidTokenError:

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
        )
```

In production, however, you should usually use an external identity provider and validate its JWTs using its published signing keys rather than maintaining your own shared secret.

---

# 7. Get the current user

Now:

```python
async def get_current_user(
    token: str = Depends(oauth2_scheme),
):

    payload = decode_access_token(token)

    user_id = payload.get("sub")

    if not user_id:
        raise HTTPException(
            status_code=401,
            detail="Invalid token",
        )

    return {
        "id": user_id,
        "tenant_id": payload.get("tenant_id"),
        "role": payload.get("role"),
    }
```

Then your API can simply do:

```python
@router.get("/profile")
async def profile(
    user=Depends(get_current_user),
):

    return user
```

The router doesn't need to parse JWTs.

---

# 8. Authentication belongs in dependencies

This is a very important FastAPI design pattern.

Don't do this:

```python
@router.get("/documents")
async def documents(
    authorization: str = Header(...)
):

    # manually parse token
    # validate token
    # find user
    # check permissions
    # ...
```

Instead:

```python
@router.get("/documents")
async def documents(
    user=Depends(get_current_user),
):
    ...
```

The authentication concern is centralized.

---

# 9. Database-backed authentication

In many applications, the JWT identifies the user, but you still retrieve the user from PostgreSQL.

For example:

```python
async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db),
):

    payload = decode_access_token(token)

    user_id = payload["sub"]

    user = await db.get(
        User,
        user_id,
    )

    if not user:
        raise HTTPException(
            status_code=401,
            detail="User not found",
        )

    return user
```

Now the dependency chain is:

```text
Request
   |
   v
Bearer Token
   |
   v
JWT Validation
   |
   v
User ID
   |
   v
PostgreSQL
   |
   v
User
```

This is particularly useful when user status/roles can change after a token was issued.

---

# 10. Authentication + Service Layer

Now connect this with what we just discussed about the service layer.

Router:

```python
@router.get("/documents")
async def get_documents(
    current_user: User = Depends(
        get_current_user
    ),
    service: DocumentService = Depends(
        get_document_service
    ),
):

    return await service.get_documents(
        current_user
    )
```

The router doesn't contain authentication logic.

The service gets the authenticated identity.

---

# 11. Authorization

Authentication tells us:

```text
User = 123
```

Authorization determines:

```text
Can User 123 delete this document?
```

For example:

```python
def require_admin(
    user: User = Depends(get_current_user),
):

    if user.role != "admin":

        raise HTTPException(
            status_code=403,
            detail="Admin access required",
        )

    return user
```

Now:

```python
@router.delete("/{document_id}")
async def delete_document(
    document_id: UUID,
    user: User = Depends(require_admin),
):

    ...
```

Flow:

```text
JWT
 ↓
get_current_user()
 ↓
User
 ↓
require_admin()
 ↓
role == admin?
 ↓
YES → endpoint
NO  → 403
```

---

# 12. Authentication vs authorization HTTP status codes

Typically:

### `401 Unauthorized`

Means:

> "You have not successfully authenticated."

Examples:

```text
Missing token
Invalid token
Expired token
Invalid credentials
```

### `403 Forbidden`

Means:

> "You are authenticated, but you aren't allowed to perform this operation."

Example:

```text
User is authenticated
User role = viewer
Endpoint requires admin
```

So:

```text
401 → Who are you?
403 → I know who you are, but you can't do this.
```

---

# 13. RBAC

A common authorization pattern is **Role-Based Access Control**.

Example:

```text
ADMIN
  ├── users:read
  ├── users:write
  ├── documents:read
  ├── documents:write
  └── documents:delete

EDITOR
  ├── documents:read
  └── documents:write

VIEWER
  └── documents:read
```

Instead of writing:

```python
if user.role == "admin":
```

everywhere, create permission dependencies.

```python
def require_permission(
    permission: str,
):
    async def checker(
        user: User = Depends(get_current_user),
    ):

        if permission not in user.permissions:

            raise HTTPException(
                status_code=403,
                detail="Permission denied",
            )

        return user

    return checker
```

Then:

```python
@router.delete("/{document_id}")
async def delete_document(
    document_id: UUID,
    user: User = Depends(
        require_permission(
            "documents:delete"
        )
    ),
):

    ...
```

This scales much better.

---

# 14. Multi-tenant authentication

For enterprise SaaS, you often have:

```text
User
 ↓
Tenant
 ↓
Role
 ↓
Permissions
```

For example:

```text
Tenant A
 ├── Admin
 ├── Developer
 └── Viewer

Tenant B
 ├── Admin
 └── Viewer
```

The JWT might contain:

```json
{
  "sub": "user-123",
  "tenant_id": "tenant-abc",
  "roles": ["admin"]
}
```

But don't rely solely on a client-provided tenant identifier.

Your authorization layer should establish the tenant context from a trusted identity/claims mapping and enforce tenant isolation at the data-access layer.

---

# 15. Tenant-aware repository

For example:

```python
class DocumentRepository:

    def __init__(
        self,
        db: AsyncSession,
        tenant_id: UUID,
    ):
        self.db = db
        self.tenant_id = tenant_id

    async def get_documents(self):

        result = await self.db.execute(
            select(Document)
            .where(
                Document.tenant_id
                == self.tenant_id
            )
        )

        return result.scalars().all()
```

Now every query automatically scopes itself to the tenant.

This is extremely important for enterprise systems.

You don't want:

```text
Tenant A
   |
   | query
   v
ALL documents
```

You want:

```text
Tenant A
   |
   v
tenant_id = A
   |
   v
Only Tenant A documents
```

---

# 16. Authentication architecture for your enterprise RAG platform

For the type of platform you've been working on, I would structure it like:

```text
                         CLIENT
                           |
                           v
                     API Gateway
                           |
                           v
                        FastAPI
                           |
                           v
                 Authentication Layer
                           |
              +------------+------------+
              |                         |
              v                         v
          JWT/OIDC                  User lookup
              |                         |
              +------------+------------+
                           |
                           v
                       CurrentUser
                           |
                           v
                     Authorization
                           |
                  +--------+--------+
                  |                 |
                  v                 v
             Tenant Check      Permission Check
                  |                 |
                  +--------+--------+
                           |
                           v
                       Service
                           |
          +----------------+----------------+
          |                |                |
          v                v                v
      PostgreSQL         Qdrant             LLM
```

---

# 17. Authentication dependency hierarchy

You can create reusable dependencies:

```python
get_current_user()
```

↓

```python
get_current_tenant()
```

↓

```python
require_permission()
```

↓

```python
get_document_service()
```

↓

```python
router
```

For example:

```python
async def get_current_tenant(
    user: User = Depends(get_current_user),
):

    return user.tenant
```

Then:

```python
def get_document_service(
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
    tenant = Depends(get_current_tenant),
):

    repository = DocumentRepository(
        db=db,
        tenant_id=tenant.id,
    )

    return DocumentService(
        repository=repository,
        user=user,
    )
```

---

# 18. Password authentication

If you're implementing your own username/password authentication rather than using an IdP, **never store passwords directly**.

Bad:

```text
password = "mypassword123"
```

Instead:

```text
password
   ↓
Argon2id/bcrypt
   ↓
password hash
   ↓
PostgreSQL
```

At login:

```text
password supplied
       ↓
verify against stored hash
       ↓
correct?
   /       \
 yes       no
  |         |
token      401
```

Argon2id is a strong modern choice for password hashing.

---

# 19. Access token + refresh token

A common architecture is:

```text
Access Token
   ↓
Short lifetime

Refresh Token
   ↓
Longer lifetime
```

For example:

```text
Access token → 10-15 minutes
Refresh token → days/weeks
```

The exact values depend on your security requirements.

Flow:

```text
Login
 ↓
Access Token + Refresh Token
 ↓
Client
 ↓
Access token expires
 ↓
Refresh token
 ↓
New access token
```

Don't make access tokens unnecessarily long-lived.

---

# 20. Refresh token rotation

For higher-security systems, use refresh token rotation.

Conceptually:

```text
Refresh Token A
      |
      v
Refresh
      |
      +---- invalidate A
      |
      +---- issue B
```

Next:

```text
Refresh Token B
      |
      v
Refresh
      |
      +---- invalidate B
      |
      +---- issue C
```

If a previously invalidated refresh token is reused, you can treat that as a potential token theft event and revoke the token family/session.

---

# 21. Where should tokens be stored?

This depends heavily on the client.

For browser applications, avoid putting sensitive long-lived tokens in `localStorage` because XSS can expose them.

A common browser architecture is:

```text
Access token
   +
HttpOnly Secure cookie
```

with appropriate:

```text
SameSite
CSRF protection
HTTPS
```

For mobile/native clients, secure OS-provided storage such as iOS Keychain or Android Keystore-backed mechanisms is preferred.

---

# 22. API keys

Not every client needs a human user identity.

For machine-to-machine APIs, you may use:

```http
X-API-Key: abc123...
```

For example:

```text
Internal Service A
       |
       | API key
       v
FastAPI Service B
```

But API keys should be:

* generated securely
* stored hashed where appropriate
* scoped
* revocable
* rotated
* rate-limited
* audited

Don't treat an API key as equivalent to a user's identity unless your design explicitly makes that mapping.

---

# 23. Service-to-service authentication

In microservices:

```text
API
 |
 v
Agent Service
 |
 v
RAG Service
 |
 v
Embedding Service
```

You don't necessarily want each service to use a user's password.

Common approaches include:

```text
OAuth2 client credentials
mTLS
short-lived service tokens
workload identity
```

For example:

```text
Agent Service
      |
      | client credentials
      v
Identity Provider
      |
      v
Service Access Token
      |
      v
RAG Service
```

---

# 24. Authentication middleware vs dependency

This is another important interview topic.

### Middleware

Runs broadly around requests.

Good for:

```text
request IDs
logging
CORS
timing
tracing
global processing
```

### Dependency

Excellent for route-specific authentication and authorization.

For example:

```python
user = Depends(get_current_user)
```

You can also use dependencies at the router level when an entire group of endpoints requires authentication.

---

# 25. Authentication in API routers

You can protect an entire router:

```python
router = APIRouter(
    prefix="/documents",
    dependencies=[
        Depends(get_current_user)
    ],
)
```

Now all endpoints require authentication.

```text
/documents
/documents/{id}
/documents/search
/documents/upload
```

all require authentication.

Then individual routes can add stricter permissions:

```python
@router.delete(
    "/{document_id}",
    dependencies=[
        Depends(
            require_permission(
                "documents:delete"
            )
        )
    ],
)
async def delete_document(...):
    ...
```

This is very clean for enterprise APIs.

---

# 26. Security architecture

A production system should think about:

```text
Authentication
     ↓
Authorization
     ↓
Tenant isolation
     ↓
Input validation
     ↓
Rate limiting
     ↓
Audit logging
     ↓
Monitoring
```

For an AI application, additionally:

```text
Prompt injection protection
PII protection
Tool authorization
Document-level access control
LLM output validation
```

Authentication is only the first security layer.

---

# 27. Example complete FastAPI structure

A production-oriented project might have:

```text
app/
│
├── main.py
│
├── api/
│   └── v1/
│       ├── auth.py
│       ├── users.py
│       ├── documents.py
│       └── chat.py
│
├── auth/
│   ├── dependencies.py
│   ├── jwt.py
│   ├── permissions.py
│   └── password.py
│
├── services/
│   ├── auth_service.py
│   ├── user_service.py
│   ├── document_service.py
│   └── chat_service.py
│
├── repositories/
│   ├── user_repository.py
│   ├── document_repository.py
│   └── audit_repository.py
│
├── db/
│   └── session.py
│
├── models/
│   ├── user.py
│   ├── role.py
│   └── permission.py
│
└── schemas/
    ├── auth.py
    ├── user.py
    └── document.py
```

The flow:

```text
                         FastAPI
                            |
                            v
                         Router
                            |
                            v
                   Authentication
                            |
                            v
                    Current User
                            |
                            v
                    Authorization
                            |
                            v
                         Service
                            |
                            v
                       Repository
                            |
                            v
                       PostgreSQL
```

---

# 28. The most important design principle

Don't make your service responsible for figuring out:

> "How do I extract and decode the Authorization header?"

That's an infrastructure/security dependency.

Instead:

```python
async def get_current_user(...):
    ...
```

produces:

```python
User
```

Then the service can simply operate on:

```python
user.id
user.tenant_id
user.roles
user.permissions
```

This keeps your business logic clean.

---

# 29. Authentication + everything you've learned

You've now got a fairly complete architecture:

```text
                         HTTP
                          |
                          v
                       Router
                          |
             +------------+------------+
             |                         |
             v                         v
       Authentication            Request Schema
             |
             v
       Current User
             |
             v
       Authorization
             |
             v
        Tenant Context
             |
             v
          Service
             |
       +-----+------+
       |            |
       v            v
 Repository      External APIs
       |
       v
 SQLAlchemy
       |
       v
 PostgreSQL
```

And for your enterprise RAG system:

```text
Client
  |
  v
FastAPI
  |
  v
JWT/OIDC Authentication
  |
  v
Current User
  |
  v
Tenant + RBAC
  |
  v
ChatService
  |
  +------> ConversationRepository ---> PostgreSQL
  |
  +------> RAGService
  |            |
  |            +----> Qdrant
  |            |
  |            +----> Reranker
  |
  +------> LLMService
               |
               +----> Azure OpenAI / OpenAI / other LLM
```

That is the pattern I would use as the foundation for a **production FastAPI enterprise AI platform**.

### Interview summary

If asked **"Explain authentication architecture in FastAPI"**, a strong answer is:

> "I normally separate authentication and authorization. Authentication is implemented as reusable FastAPI dependencies that validate an OAuth2/OIDC or JWT access token and resolve the current user. Authorization is then handled through RBAC or permission-based dependencies, with tenant context enforced for multi-tenant applications. The router remains thin and passes the authenticated user into the service layer. Services contain business logic, repositories handle persistence, and database access is injected separately. For production systems I prefer an external identity provider such as Entra ID, Auth0, Okta, or Keycloak rather than building the identity system myself. I also use short-lived access tokens, secure refresh-token handling where applicable, TLS, audit logging, rate limiting, and strict tenant/resource-level authorization."
