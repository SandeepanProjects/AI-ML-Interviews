# How would you perform rollback for a fine-tuned LLM?

Rollback means:

> **Quickly switching production traffic from a bad model version back to a previously validated, stable version.**

For an LLM, rollback must consider not only application errors but also **model quality regressions**.

For example:

```text
Model v1.0  → Good
Model v1.1  → Deployed
Model v1.1  → Higher hallucinations ❌
                    │
                    ▼
              ROLLBACK
                    │
                    ▼
Model v1.0  → Production again ✅
```

---

# 1. Why would you roll back a model?

A new fine-tuned model may cause:

```text
┌─────────────────────────────────┐
│ Production Problems             │
├─────────────────────────────────┤
│ Higher error rate               │
│ Higher latency                  │
│ GPU OOM                         │
│ Higher hallucination            │
│ Worse instruction following     │
│ Safety regression               │
│ Higher cost                     │
│ Model loading failure           │
│ Bad customer feedback           │
└─────────────────────────────────┘
```

Example:

```text
                    Model v1       Model v2

Latency               800ms          2.5s ❌
Error Rate             0.2%          5%   ❌
Hallucination          2%            12%  ❌
User Rating            4.6           3.2  ❌
```

A model can be technically healthy but still need rollback because its **answers are worse**.

---

# 2. The most important rule: never overwrite the old model

Bad approach:

```text
models/
    customer-support/
        model.safetensors
```

Then replacing it:

```text
v1 → overwritten by v2
```

If v2 fails:

```text
Where is v1? ❌
```

Instead:

```text
models/
│
├── customer-support-v1.0.0/
│
├── customer-support-v1.1.0/
│
└── customer-support-v1.2.0/
```

Each model must be immutable.

---

# 3. Version your model artifacts

A model version should include more than weights.

```text
customer-support-v1.2.0/

├── model/
│   ├── model.safetensors
│   └── config.json
│
├── tokenizer/
│
├── training_config.json
│
├── evaluation.json
│
├── dataset_version.txt
│
├── git_commit.txt
│
└── metadata.json
```

Example metadata:

```json
{
    "model_name": "customer-support",
    "version": "1.2.0",
    "base_model": "llama-3.1-8b",
    "dataset_version": "support-data-v4",
    "training_commit": "abc123",
    "evaluation_score": 0.91
}
```

This gives you reproducibility:

```text
Production model
      │
      ▼
Which version?
      │
      ▼
v1.2.0
      │
      ├── Training data?
      ├── Base model?
      ├── Hyperparameters?
      └── Evaluation results?
```

---

# 4. Basic application-level rollback

Suppose your deployment uses an environment variable:

```text
MODEL_VERSION=v1.2.0
```

Your application loads:

```python
import os

MODEL_VERSION = os.getenv(
    "MODEL_VERSION",
    "v1.0.0"
)

MODEL_PATH = (
    f"/models/customer-support-"
    f"{MODEL_VERSION}"
)
```

Current deployment:

```text
MODEL_VERSION=v1.2.0
```

Rollback:

```text
MODEL_VERSION=v1.1.0
```

Then restart the deployment.

Conceptually:

```text
Before:

Production
    │
    ▼
v1.2.0


Rollback:

Production
    │
    ▼
v1.1.0
```

This is simple, but production rollback should be more controlled.

---

# 5. Store model versions in a model registry

The deployment system should not need to know internal storage paths.

Instead:

```text
Model Registry

customer-support
│
├── v1.0.0
│      └── Archived
│
├── v1.1.0
│      └── Production
│
└── v1.2.0
       └── Candidate
```

Production deployment can resolve an alias:

```text
customer-support@production
```

Initially:

```text
production → v1.1.0
```

If v1.2.0 is promoted:

```text
production → v1.2.0
```

Rollback:

```text
production → v1.1.0
```

This is safer than changing random file paths.

---

# 6. Model alias rollback

Conceptually:

```python
class ModelRegistry:

    def __init__(self):
        self.aliases = {}

    def set_alias(
        self,
        model_name,
        alias,
        version
    ):
        self.aliases[
            f"{model_name}:{alias}"
        ] = version

    def get_version(
        self,
        model_name,
        alias
    ):
        return self.aliases.get(
            f"{model_name}:{alias}"
        )
```

Promote:

```python
registry.set_alias(
    model_name="customer-support",
    alias="production",
    version="v1.2.0"
)
```

Rollback:

```python
registry.set_alias(
    model_name="customer-support",
    alias="production",
    version="v1.1.0"
)
```

The important idea:

```text
Application
    │
    ▼
"production"
    │
    ▼
Model Registry
    │
    ▼
Current model version
```

You change the alias, not the model files.

---

# 7. Kubernetes rollback

Suppose you deployed:

```text
fine-tuned-model:v1.1
```

Then deploy:

```text
fine-tuned-model:v1.2
```

Kubernetes keeps deployment history.

Deployment:

```yaml
apiVersion: apps/v1
kind: Deployment

metadata:
  name: llm-inference

spec:
  replicas: 3

  template:

    spec:

      containers:

        - name: llm

          image: registry.example.com/fine-tuned-model:v1.2
```

Check history:

```bash
kubectl rollout history \
  deployment/llm-inference
```

Example:

```text
REVISION  IMAGE

1         model:v1.0
2         model:v1.1
3         model:v1.2
```

Rollback:

```bash
kubectl rollout undo \
  deployment/llm-inference
```

Rollback to a specific version:

```bash
kubectl rollout undo \
  deployment/llm-inference \
  --to-revision=2
```

Now:

```text
Production

v1.2 ❌

     │
     ▼

v1.1 ✅
```

---

# 8. Rollback should be automated when possible

You should define rollback conditions.

For example:

```text
Error rate > 5%
```

or:

```text
P95 latency > 3 seconds
```

or:

```text
Hallucination rate > 10%
```

Example:

```python
def should_rollback(metrics):

    if metrics["error_rate"] > 0.05:
        return True

    if metrics["p95_latency_ms"] > 3000:
        return True

    if metrics["hallucination_rate"] > 0.10:
        return True

    return False
```

Usage:

```python
metrics = {
    "error_rate": 0.07,
    "p95_latency_ms": 1200,
    "hallucination_rate": 0.05
}

if should_rollback(metrics):

    print(
        "ROLLBACK REQUIRED"
    )
```

Output:

```text
ROLLBACK REQUIRED
```

---

# 9. Canary deployment with automatic rollback

This is one of the safest strategies.

Initial:

```text
                    Traffic

                ┌───────┴───────┐
                ▼               ▼

             95%               5%

             v1.1              v1.2
             Stable            New
```

Monitor v1.2.

```text
Error Rate

v1.1 = 0.2%
v1.2 = 6.5% ❌
```

Automatically:

```text
v1.2

5% traffic
   │
   ▼

Problem detected
   │
   ▼

0% traffic
   │
   ▼

100% → v1.1
```

Traffic routing:

```python
import random


def choose_model():

    value = random.random()

    if value < 0.05:
        return "v1.2"

    return "v1.1"
```

After rollback:

```python
def choose_model():
    return "v1.1"
```

In production, you normally let Kubernetes, a service mesh, or your load balancer perform this routing rather than using random routing inside application code.

---

# 10. Blue-Green deployment

Another powerful rollback strategy.

You run two complete environments.

```text
BLUE

Model v1.1
Stable
```

```text
GREEN

Model v1.2
New
```

Initially:

```text
Users
  │
  ▼
BLUE
v1.1
```

Deploy and test:

```text
Users
  │
  ▼
BLUE
v1.1


GREEN
v1.2
Testing
```

Switch:

```text
Users
  │
  ▼
GREEN
v1.2
```

If something goes wrong:

```text
Users
  │
  ▼
BLUE
v1.1
```

Rollback is nearly immediate because the old environment is still running.

---

# 11. Shadow deployment

This is especially useful for LLMs.

The production model answers the user:

```text
User
  │
  ▼
v1.1
  │
  ▼
Response → User
```

At the same time:

```text
                    ┌──► v1.1 → User
User Request ────────┤
                    └──► v1.2 → Evaluation only
```

The new model receives production-like traffic, but its response is not shown to users.

Compare:

```text
v1.1:
"The password can be reset from Settings."

v1.2:
"Contact the system administrator."
```

You can evaluate:

```text
Quality
Latency
Hallucinations
Safety
Cost
```

before sending v1.2 real traffic.

This reduces the chance of needing a rollback.

---

# 12. LLM-specific rollback is more complicated

Traditional software:

```text
New version
    │
    ▼
API crashes
    │
    ▼
Rollback
```

LLM:

```text
New model
    │
    ▼
API works perfectly
    │
    ▼
But answers are worse ❌
```

Therefore, monitor both:

```text
Technical Metrics
```

and:

```text
Model Quality Metrics
```

### Technical

```text
Latency
Error rate
GPU memory
Throughput
Timeouts
```

### Model quality

```text
Human feedback
LLM-as-a-judge
Hallucination rate
Instruction following
Safety violations
Task success rate
```

Example:

```python
def evaluate_deployment(metrics):

    technical_failure = (
        metrics["error_rate"] > 0.05
    )

    quality_failure = (
        metrics["quality_score"] < 0.80
    )

    safety_failure = (
        metrics["safety_violation_rate"]
        > 0.01
    )

    return (
        technical_failure
        or quality_failure
        or safety_failure
    )
```

---

# 13. Rollback API

You can build an internal deployment service.

```python
from fastapi import FastAPI

app = FastAPI()


deployment = {
    "current_version": "v1.2.0",
    "previous_version": "v1.1.0"
}
```

Rollback endpoint:

```python
@app.post("/internal/rollback")
async def rollback():

    current = (
        deployment["current_version"]
    )

    previous = (
        deployment["previous_version"]
    )

    deployment[
        "current_version"
    ] = previous

    deployment[
        "previous_version"
    ] = current

    return {
        "status": "rolled_back",
        "active_version": previous
    }
```

Result:

```json
{
    "status": "rolled_back",
    "active_version": "v1.1.0"
}
```

In production, this endpoint should be strongly protected and should trigger an audited deployment workflow rather than simply changing an in-memory dictionary.

---

# 14. Database-driven deployment state

For production, store deployment information.

Example table:

```sql
CREATE TABLE model_deployments (

    id UUID PRIMARY KEY,

    model_name VARCHAR(255),

    model_version VARCHAR(50),

    environment VARCHAR(50),

    status VARCHAR(50),

    deployed_at TIMESTAMP,

    rolled_back_at TIMESTAMP
);
```

Example:

```text
model_name          version     environment

support-model       v1.1.0      production
support-model       v1.2.0      production
```

Deployment state:

```text
v1.1.0 → PREVIOUS
v1.2.0 → ACTIVE
```

Rollback:

```text
v1.1.0 → ACTIVE
v1.2.0 → ROLLED_BACK
```

This provides an audit trail.

---

# 15. A production rollback workflow

```text
                Deploy v1.2
                    │
                    ▼
              Canary 5%
                    │
                    ▼
              Monitor Metrics
                    │
          ┌─────────┴─────────┐
          │                   │
        Healthy             Failure
          │                   │
          ▼                   ▼
     Increase Traffic     Stop Traffic
          │                   │
          ▼                   ▼
       100% v1.2         Rollback v1.1
                              │
                              ▼
                         Investigate
                              │
                              ▼
                       Fix / Retrain
```

---

# 16. Recommended production implementation

I would combine:

```text
Model Registry
      +
Immutable Model Versions
      +
Canary Deployment
      +
Automated Monitoring
      +
Kubernetes Rollback
      +
Quality Evaluation
```

Architecture:

```text
                    Model Registry

                         │

            ┌────────────┴────────────┐

            ▼                         ▼

         v1.1                       v1.2

       Stable                     Candidate

            │                         │

            └───────────┬─────────────┘
                        │
                        ▼
                  Traffic Router

                   95%      5%

                    │        │

                    ▼        ▼

                  v1.1     v1.2

                        │
                        ▼

                    Monitoring

                        │

              ┌─────────┴─────────┐

              ▼                   ▼

           Healthy              Failure

              │                   │

              ▼                   ▼

          Promote              Rollback
```

---

# 17. Interview-ready answer

> **I design rollback before deploying the model. Every fine-tuned model is versioned as an immutable artifact and stored in a model registry along with its tokenizer, training configuration, dataset version, evaluation results, and metadata. I never overwrite a production model.**
>
> **I deploy new versions using a canary or blue-green strategy. Initially, only a small percentage of traffic goes to the new model while I compare latency, error rate, GPU utilization, hallucination rate, safety metrics, task success, and user feedback against the stable model.**
>
> **If predefined thresholds are violated, I immediately stop traffic to the new model and route traffic back to the last known-good version. In Kubernetes, this can be done with a rollout undo or by changing the model version deployed through the model registry.**
>
> **For LLMs, rollback decisions must consider both infrastructure failures and quality regressions because a model can be technically healthy while generating worse answers. I therefore combine technical monitoring with automated and human model evaluation.**

## The key principle

```text
A good rollback system does not ask:

"How do I go back?"

It is designed so that:

"Going back is faster and safer than deploying forward."
```
