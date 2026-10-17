# Protecting Secrets and API Keys in an Enterprise AI Project

At staff AI engineer level, secrets management is not just about putting an API key in a `.env` file. It is about designing a system in which credentials are never unnecessarily exposed, access is restricted to the correct workload, compromised credentials can be revoked quickly, and secret usage can be audited.

Consider your enterprise AI platform built with FastAPI, LangGraph, PostgreSQL/pgvector, Redis, Celery, and external LLM providers.

It may need:

- OpenAI or another LLM provider API key
- PostgreSQL credentials
- Redis authentication credentials
- JWT signing keys
- Cloud credentials for AWS
- Vector database credentials
- Third-party search API keys
- TLS certificates and private keys
- Credentials used by CI/CD pipelines

Each secret has a different purpose, access scope, lifetime and rotation strategy. Treating all of them as one shared `API_KEY` is a security design flaw.

## 1. Understand the complete secrets-management architecture

Secret provisioning

Create credentials and assign ownership, scope and rotation policy

Secrets manager

AWS Secrets Manager, Azure Key Vault, Google Secret Manager or Vault

Workload identity

IAM role, workload identity or short-lived identity token

Authorized AI service

FastAPI, Celery worker or a dedicated model gateway

External dependency

LLM API, database, Redis, storage or search service

Cross-cutting controls

Least privilege, encryption, redaction, rotation, audit logs, secret scanning and incident response.

A secrets manager stores and controls access to secrets. A workload identity allows your service to authenticate to that manager without embedding another long-lived cloud access key in the application. These controls work together.

For AWS, the official [Secrets Manager best practices](https://docs.aws.amazon.com/secretsmanager/latest/userguide/best-practices.html?utm_source=chatgpt.com) cover least-privilege access, encryption, rotation, auditing and secure secret retrieval. OWASP also recommends centralized secret lifecycle management and fine-grained access control.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://docs.aws.amazon.com\&sz=32)

AWS Secrets Manager

+1



## 2. Classify the secrets before writing code

Not every credential should be handled identically.

| Secret                      | Recommended protection                                           |
| --------------------------- | ---------------------------------------------------------------- |
| LLM provider key            | Secrets manager or an approved model gateway                     |
| PostgreSQL password         | Secrets manager; consider short-lived database credentials       |
| Redis credential            | Secrets manager or workload-specific authentication              |
| JWT signing key             | KMS/HSM-backed signing where suitable, or a protected key store  |
| AWS credentials             | IAM roles or workload identity instead of static access keys     |
| CI/CD deployment credential | Short-lived federated identity with scoped permissions           |
| TLS private key             | Protected certificate/key-management service                     |
| Local developer API key     | Untracked local configuration or approved developer secret store |

One important distinction: JWT signing keys are not the same as JWT access tokens. Signing keys are long-lived or rotated cryptographic material that must be protected carefully. Access tokens are credentials issued to clients and should have appropriate expiration, audience, issuer and scope validation.

For an enterprise AI system, I would also separate secrets by environment:

```
financial-ai/
├── dev/
│   ├── llm-provider-key
│   └── database-credentials
├── staging/
│   ├── llm-provider-key
│   └── database-credentials
└── prod/
    ├── llm-provider-key
    └── database-credentials
```

These are logical secret names, not files to commit to Git. Production and development should use separate credentials, access policies and ideally separate security boundaries.

A developer who can run the local application should not automatically be able to retrieve the production database password.

## 3. Never hardcode API keys in source code

### Incorrect implementation

```
from openai import OpenAIclient = OpenAI(    api_key="sk-your-real-api-key")
```

This is dangerous even in a private repository. A key may escape through:

- Git history, branches, pull requests or forks
- Logs, debugging output and exception reports
- Docker image layers or build artifacts
- Notebook outputs and screenshots
- CI/CD logs
- Copies of the source code or local backups

Removing the key from the latest source file does not invalidate the exposed credential.

### Better for local development: an untracked `.env` file

Create a `.env` file locally:

```
OPENAI_API_KEY=your-local-development-key
DATABASE_URL=postgresql://app_user:local_password@localhost:5432/financial_ai
REDIS_URL=redis://localhost:6379/0
```

Add it to `.gitignore`:

```
.env
.env.*
!.env.example
```

Be careful: this pattern ignores `.env.example` but does not automatically ignore every possible secret file, certificate or private key. Review your actual repository's ignore rules.

Commit only a placeholder template:

```
OPENAI_API_KEY=
DATABASE_URL=
REDIS_URL=
```

This is appropriate for local development, not a complete production secrets-management strategy. Environment variables and `.env` files can be exposed through process inspection, diagnostic tools, crash dumps or careless logging. OWASP advises minimizing exposure and using a dedicated secrets-management mechanism where practical.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://cheatsheetseries.owasp.org\&sz=32)

OWASP Cheat Sheet Series

+1



### Load local configuration safely with Python

Install:

```
pip install pydantic-settings python-dotenv openai
```

Create `app/config/settings.py`:

```
from functools import lru_cachefrom pydantic import SecretStrfrom pydantic_settings import BaseSettings, SettingsConfigDictclass Settings(BaseSettings):    openai_api_key: SecretStr    database_url: SecretStr    redis_url: SecretStr    model_config = SettingsConfigDict(        env_file=".env",  # Local development convenience        env_file_encoding="utf-8",        extra="ignore",    )@lru_cachedef get_settings() -> Settings:    return Settings()
```

Use the configuration:

```
from openai import AsyncOpenAIfrom app.config.settings import get_settingsdef create_llm_client() -> AsyncOpenAI:    settings = get_settings()    return AsyncOpenAI(        api_key=settings.openai_api_key.get_secret_value(),        timeout=30.0,        max_retries=2,    )
```

`SecretStr` helps prevent accidental exposure when settings are represented or printed. It does not encrypt the secret, prevent a malicious process from reading it, or make explicitly calling `get_secret_value()` safe in every context.

Staff-level rule: never log the full settings object or dump all environment variables during debugging. Log configuration status, such as whether a required value exists, without revealing its value.

## 4. Production: retrieve secrets from AWS Secrets Manager

For an AWS deployment, prefer workload identity—such as an ECS task role or an EKS workload identity configuration—over long-lived AWS access keys stored in your application.

The flow should be:

1. Your FastAPI service starts under an authorized workload identity.
2. The AWS SDK obtains temporary AWS credentials through the configured identity mechanism.
3. The service requests only the specific secret it needs.
4. Secrets Manager authorizes the request and returns the secret over TLS.
5. The application keeps the value in memory only as long as operationally necessary.
6. Access and rotation events are monitored.

AWS documents encryption, access control, caching, rotation and auditing in its [official best-practices guide](https://docs.aws.amazon.com/secretsmanager/latest/userguide/best-practices.html?utm_source=chatgpt.com).&#x20;

[image](https://www.google.com/s2/favicons?domain=https://docs.aws.amazon.com\&sz=32)

AWS Secrets Manager

+1



### Step 1: Install the AWS SDK

```
pip install boto3
```

### Step 2: Store the secret in AWS Secrets Manager

For example, create a secret named:

```
prod/financial-ai/llm
```

Its JSON value might contain:

```
{
  "OPENAI_API_KEY": "replace-with-real-secret"
}
```

Do not commit that JSON value to Git or paste it into deployment configuration. Provision the secret securely through your organization's approved process.

### Step 3: Create a least-privilege IAM policy

The application role should have permission to read only the required secret, rather than all secrets in the account.

Illustrative policy structure:

```
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": "secretsmanager:GetSecretValue",
      "Resource": "arn:aws:secretsmanager:REGION:ACCOUNT_ID:secret:prod/financial-ai/llm-EXAMPLE"
    }
  ]
}
```

Replace the placeholders with the actual region, account and secret ARN. The exact ARN may include a generated suffix.

If you use a customer-managed KMS key, the role and key policy must also permit the necessary decryption path. Do not add broad permissions such as `secretsmanager:*` or `kms:*` merely to make an access error disappear. AWS documents the corresponding IAM and KMS requirements in its [access-control documentation](https://docs.aws.amazon.com/secretsmanager/latest/userguide/auth-and-access.html?utm_source=chatgpt.com).&#x20;

[image](https://www.google.com/s2/favicons?domain=https://docs.aws.amazon.com\&sz=32)

AWS Secrets Manager



### Step 4: Write a secret provider

Create `app/security/secrets.py`:

```
import jsonfrom functools import lru_cacheimport boto3from botocore.config import Configdef _create_secrets_manager_client():    # Uses the AWS SDK credential chain.    # In production, configure an IAM workload identity.    return boto3.client(        "secretsmanager",        region_name="ap-south-1",        config=Config(            connect_timeout=3,            read_timeout=5,            retries={"max_attempts": 3, "mode": "standard"},        ),    )@lru_cache(maxsize=1)def get_llm_secret() -> dict[str, str]:    client = _create_secrets_manager_client()    response = client.get_secret_value(        SecretId="prod/financial-ai/llm"    )    secret_string = response.get("SecretString")    if not secret_string:        raise RuntimeError("Expected a JSON string secret")    data = json.loads(secret_string)    api_key = data.get("OPENAI_API_KEY")
```

This demonstrates a simple cached retrieval pattern. It is not a complete secret-rotation implementation.

Important production refinements:

- Make the region and secret identifier non-secret deployment configuration.
- Use a supported caching strategy with a defined refresh and expiry policy rather than an indefinite process-local cache.
- Handle transient retrieval failures without logging the secret or silently falling back to an old credential forever.
- Ensure secret retrieval happens in a controlled startup or client-initialization path, not repeatedly for every model call.
- Use an approved SDK caching component where suitable. AWS provides guidance and caching components in its [Secrets Manager documentation](https://docs.aws.amazon.com/secretsmanager/latest/userguide/best-practices.html?utm_source=chatgpt.com).&#x20;

  [image](https://www.google.com/s2/favicons?domain=https://docs.aws.amazon.com\&sz=32)

  AWS Secrets Manager



For a multi-worker FastAPI application, remember that each worker process has its own memory and cache. A secret refresh in one process does not automatically update the others.

### Step 5: Construct the LLM client

```
from openai import AsyncOpenAIfrom app.security.secrets import get_llm_secretdef create_llm_client() -> AsyncOpenAI:    secret = get_llm_secret()    return AsyncOpenAI(        api_key=secret["OPENAI_API_KEY"],        timeout=30.0,        max_retries=2,    )
```

In an actual application, construct and manage the client using FastAPI lifespan or your application's dependency-injection container. Reuse the client rather than constructing a new one for every request.

For an enterprise system, I would normally use separate providers for secrets retrieval and LLM client construction. That makes the code testable and makes it possible to swap AWS Secrets Manager for another approved secret store without changing agent logic.

## 5. Use workload identity instead of static cloud credentials

This is one of the most important staff-level security decisions.

### The weaker approach

You store an AWS access key and secret access key in a production `.env` file, then use them to retrieve your LLM API key.

Now your application has two credential risks:

1. The long-lived AWS credentials that grant access to AWS.
2. The LLM API key retrieved using those credentials.

### The stronger approach

```
FastAPI / Celery workload
        |
        | Workload identity
        v
AWS temporary credentials
        |
        | IAM authorization
        v
Secrets Manager
        |
        v
LLM provider API key
```

For example:

- On ECS, use an appropriately scoped task role.
- On EKS, configure the supported workload identity mechanism for your cluster.
- In CI/CD, use an identity federation mechanism such as GitHub Actions OIDC, where supported, instead of storing long-lived cloud access keys.
- On developer machines, use your organization's approved developer authentication flow.

Do not give a workload access to every secret simply because it runs inside AWS. A compromised service should not automatically expose unrelated database, deployment or administrative credentials.

## 6. Protect every secret in your AI architecture

Your project has multiple trust boundaries. Protecting only the LLM key is insufficient.

| Component           | Recommended approach                                                                                              |
| ------------------- | ----------------------------------------------------------------------------------------------------------------- |
| FastAPI API         | Read only the secrets required by that API                                                                        |
| Celery worker       | Separate workload identity and secret permissions                                                                 |
| LangGraph agents    | Receive authorized clients or narrow tool interfaces, not raw secret-store access                                 |
| PostgreSQL/pgvector | Restricted database role; separate migration and runtime privileges                                               |
| Redis               | Authentication, TLS where appropriate, network isolation and limited commands/access                              |
| Qdrant              | Authentication and TLS where supported, network restrictions and collection-level access controls where available |
| LLM provider        | Scoped provider keys, usage limits and provider-side restrictions where available                                 |
| CI/CD pipeline      | Short-lived identity, protected branches and restricted secret access                                             |
| Kubernetes          | External secret integration or supported secret injection; RBAC and encryption at rest                            |
| Logs and traces     | Redaction, restricted access and retention controls                                                               |

### A particularly important agent-security rule

Do not expose the secrets manager as an unrestricted tool to an LLM agent.

Bad design:

```
Agent
  └── read_any_secret(secret_name)
```

An agent that can choose arbitrary secret names may attempt to retrieve database passwords, signing keys or other credentials.

Prefer:

```
Research agent
  └── search_market_data(query)

Database tool
  └── execute_authorized_query(parameters)

LLM client
  └── uses its configured provider credential internally
```

The agent should receive the capability it needs, not unrestricted access to credentials. The secret is injected into the trusted client or service layer and is never placed in the agent's prompt or tool output.

## 7. Secret rotation: what happens when a key changes?

A secret should have a documented lifecycle:

```
Create → Store → Grant access → Consume → Rotate → Revoke → Audit
```

Rotation is not simply changing a string in a dashboard. Your application must begin using the new credential, and the old credential must eventually become unusable.

### Example: rotating an LLM provider key

1. Create a new provider key with the minimum required scope.
2. Store it in the approved secrets manager.
3. Update the application's secret reference or active secret version.
4. Refresh the application client safely.
5. Run a health check against the provider.
6. Confirm that production requests are using the new credential.
7. Revoke the old key.
8. Audit usage and watch for authentication failures.

For provider keys, automatic rotation may not be available or may require custom orchestration. Do not assume AWS Secrets Manager can rotate every third-party API key automatically. Rotation capabilities depend on the upstream provider and its API.

### Database credential rotation requires more care

For database credentials, use a strategy that avoids taking down the service:

- Prefer short-lived or dynamically generated credentials if the database and infrastructure support them.
- Consider separate application identities or alternating credentials.
- Allow a controlled overlap period when the database supports it.
- Refresh connection pools after credentials change.
- Validate new connections before revoking the previous credential.

Existing PostgreSQL connections may remain authenticated even after a password changes. Updating the secret alone does not necessarily replace those connections or invalidate every already-established session.

### Avoid an indefinitely cached secret

The previous `@lru_cache` example illustrates basic caching but is not enough for rotation. For production, use a refreshable provider with a bounded cache lifetime or a supported managed integration.

A minimal interface might be:

```
from typing import Protocolclass SecretProvider(Protocol):    async def get_secret(self, name: str) -> str:        """Return a secret value for an authorized caller."""        ...    async def refresh_secret(self, name: str) -> None:        """Refresh the cached version after a rotation event."""        ...
```

An implementation can use an async AWS SDK, a dedicated secret-store client, or a managed secret-injection mechanism. The application can trigger refresh on a rotation event or on cache expiry.

Be careful with refresh concurrency: if hundreds of workers discover an expired cache simultaneously, they should not all fetch the same secret at once. Use single-flight refresh, bounded retries and jitter where appropriate.

Critical distinction: rotation, revocation and expiration are different. Changing the value in the secret store does not necessarily revoke the old credential at the upstream provider. The old credential must be invalidated through the system that issued it. OWASP discusses secret lifecycle, rotation and revocation in its [Secrets Management Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Secrets_Management_Cheat_Sheet.html?utm_source=chatgpt.com).&#x20;

[image](https://www.google.com/s2/favicons?domain=https://cheatsheetseries.owasp.org\&sz=32)

OWASP Cheat Sheet Series



## 8. Prevent secrets from leaking through Git, Docker and CI/CD

A secret can be stored securely at runtime and still leak during the build or deployment process.

### Git secret scanning

Use multiple layers:

- A pre-commit secret scanner to catch accidental commits.
- Server-side repository scanning to inspect commits and branches.
- Pull-request scanning before merging.
- Periodic scans for historical exposure.
- Dependency and infrastructure configuration scanning.

Tools commonly used for this include Gitleaks and TruffleHog. Treat detection results carefully: a false positive should be reviewed, but a real credential should be revoked immediately.

Example with Gitleaks:

```
gitleaks git .
```

Check the installed version's CLI documentation for the precise supported options. Run scans in CI and prevent merges when verified secrets are found.

### Docker: never bake secrets into images

Incorrect:

```
ARG OPENAI_API_KEY
ENV OPENAI_API_KEY=$OPENAI_API_KEY
```

Also avoid copying `.env` files into the image.

A safer pattern is to build an image containing application code and dependencies only, then supply secrets through the deployment platform's approved secret-injection mechanism at runtime.

Example `.dockerignore`:

```
.env
.env.*
!.env.example
.git
**/__pycache__
**/.venv
```

This is a baseline, not a complete policy. Add any local credential files, certificates, private keys, notebooks or artifacts that should never enter the build context.

Docker build arguments and image environment declarations are not secret stores. A value used during a build can leak through image metadata, layers, build logs or cache. If a build genuinely needs a secret, use the build system's supported secret-mount mechanism and do not persist the secret into an image layer.

### CI/CD

A secure deployment pipeline should:

1. Authenticate through short-lived workload identity where possible.
2. Access only the secrets required for that job.
3. Avoid printing environment variables or command lines containing secret values.
4. Restrict untrusted pull-request workflows from accessing production credentials.
5. Prevent secret exposure through artifacts and caches.
6. Audit privileged workflow changes and deployment approvals.
7. Scan build output and revoke any credential that leaks.

Treat the CI/CD system as a production security boundary. A compromised pipeline can often access more powerful credentials than an ordinary application instance. OWASP's [secrets-management guidance](https://cheatsheetseries.owasp.org/cheatsheets/Secrets_Management_Cheat_Sheet.html?utm_source=chatgpt.com) discusses CI/CD hardening and access controls.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://cheatsheetseries.owasp.org\&sz=32)

OWASP Cheat Sheet Series



## 9. Protect secrets in logs, prompts, traces and exceptions

In AI applications, the leak may happen long after the credential is stored securely.

### Never do this

```
logger.info("LLM configuration: %s", settings)logger.debug("Request headers: %s", request.headers)logger.exception("LLM call failed: %s", exc)
```

Depending on the object and exception, these statements can expose API keys, authorization headers, connection strings, request content or other sensitive data.

### Use safe, structured logs

```
import logginglogger = logging.getLogger(__name__)def log_llm_failure(    *,    provider: str,    model: str,    request_id: str,    error_type: str,) -> None:    logger.warning(        "llm_call_failed",        extra={            "provider": provider,            "model": model,            "request_id": request_id,            "error_type": error_type,        },    )
```

The caller should supply a sanitized error category, not an arbitrary exception message that may contain headers or payloads.

For traces, follow the same rule. Do not attach API keys to span attributes, do not log full request headers, and do not capture raw prompts or completions without a deliberate privacy and access-control design.

### Guardrails are not secrets management

An LLM guardrail may detect a secret-like string in a prompt or response, but it cannot replace secure credential storage, identity-based authorization or key revocation.

Similarly, a prompt instruction such as “never reveal the API key” does not protect a key that has already been placed in the model's context. The safest design is to ensure that the model never receives the credential in the first place.

## 10. Kubernetes and distributed AI workloads

If you deploy FastAPI, LangGraph workers and Celery on Kubernetes, think about how secrets reach each pod.

Options include:

- An external secrets integration that fetches from a managed secret store.
- Kubernetes Secrets, with encryption at rest configured and strict RBAC.
- A supported CSI driver or secret-injection mechanism.
- Workload identity for authenticating to the cloud secret store.

Kubernetes Secrets are not automatically a complete secrets-management solution. Their security depends on cluster configuration, RBAC, encryption, node access and the mechanism used to inject them.

Apply these principles:

- Give the API and Celery workers separate service accounts where their access requirements differ.
- Restrict which pods can read each secret.
- Do not expose secrets through container arguments, image layers or debugging endpoints.
- Keep secret values out of Helm values files committed to Git.
- Restrict `kubectl exec`, node administration and privileged workloads.
- Understand how a rotated secret reaches existing pods and clients.
- Use network policies and TLS to reduce access to internal services.

For a financial AI system, a research worker generally should not have permission to retrieve the production JWT signing key merely because it shares a Kubernetes namespace with the API.

## 11. What to do if an API key is exposed

Assume a developer accidentally commits an LLM key to a repository.

Use an incident-response sequence:

1\. Contain immediately

Revoke or disable the exposed credential at the issuing provider. Do not wait until the Git history is cleaned up.

2\. Replace the credential

Create a new key with the minimum required permissions, update the secret store and verify that the application works with it.

3\. Investigate the exposure

Search repository history, CI logs, build artifacts, traces and relevant access records. Review provider usage and unexpected activity.

4\. Remove the source of exposure

Remove the secret from current files and clean history where appropriate. Coordinate history rewriting with repository owners, but do not confuse cleanup with revocation.

5\. Close the control gap

Add or repair secret scanning, review access policies, rotate related credentials if necessary and document the incident.

If a database credential, JWT signing key or cloud credential is exposed, the incident may require additional steps such as terminating sessions, rotating dependent credentials, reviewing issued tokens or checking unauthorized access. The response depends on the credential's privileges and the systems it protects.

## 12. Production readiness checklist

## Security review checklist

0/16 checked

Storage and access

No real secrets in source code, Git history or Docker images

Production secrets stored in an approved secret manager

Least-privilege workload identities and per-environment access policies

Local, staging and production credentials are separated

Runtime protection

Secrets are not exposed through logs, prompts, traces or error responses

Secret refresh and rotation behavior are tested

LLM clients and database pools handle credential updates safely

TLS and appropriate network restrictions protect service connections

CI/CD and detection

Secret scanning runs in pre-commit and CI

CI/CD uses scoped, preferably short-lived credentials

Secret access and rotation events are audited

Incident-response steps for exposed credentials are documented

AI-specific controls

Agents cannot read arbitrary secrets or environment variables

Tools expose narrow capabilities rather than raw credentials

Tenant boundaries are enforced in retrieval, tools and telemetry

Provider usage and unexpected credential activity are monitored

Copy checklist

## 13. How I would explain this in a staff-level interview

> I treat secrets management as a cross-cutting security capability, not as a configuration-file concern. I classify secrets by sensitivity and purpose, store production credentials in an approved secrets manager, and use workload identity so services do not need long-lived cloud access keys.
>
> Each service receives least-privilege access to only the secrets it needs. I separate credentials across environments, avoid exposing secrets to LLM prompts or agent tools, and apply redaction to logs and traces. I implement rotation with the upstream provider, validate that new credentials work, refresh clients safely and revoke the old credentials.
>
> I also secure the CI/CD pipeline, scan for leaked secrets, prevent credentials from entering Docker images, audit secret access and document incident-response procedures. For distributed AI workflows, I ensure each worker has its own appropriate permissions and that secret refresh works across processes.
>
> My objective is to minimize credential exposure, reduce the blast radius of a compromised service, detect misuse and ensure a leaked credential can be revoked without creating unnecessary downtime.

The key principle is: store secrets centrally, authenticate workloads securely, grant minimum access, keep secrets out of model context and telemetry, rotate and revoke them correctly, and continuously test the entire lifecycle.

## Further reading

- [OWASP Secrets Management Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Secrets_Management_Cheat_Sheet.html?utm_source=chatgpt.com) — lifecycle, rotation, least privilege and CI/CD security.
- [AWS Secrets Manager best practices](https://docs.aws.amazon.com/secretsmanager/latest/userguide/best-practices.html?utm_source=chatgpt.com) — secret retrieval, caching, auditing and rotation.
- [AWS Secrets Manager access control](https://docs.aws.amazon.com/secretsmanager/latest/userguide/auth-and-access.html?utm_source=chatgpt.com) — IAM and resource permissions.