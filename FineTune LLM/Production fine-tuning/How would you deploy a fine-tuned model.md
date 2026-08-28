# How would you deploy a fine-tuned LLM?

Deploying a fine-tuned model means taking the trained model (or LoRA adapters), loading it into an inference server, exposing it through an API, and operating it reliably.

A typical production architecture is:

```text
                 CI/CD Pipeline
                       │
                       ▼
              Model Registry
                       │
                       ▼
              Model Artifact
                       │
              ┌────────┴────────┐
              ▼                 ▼
         Staging           Production
              │                 │
              ▼                 ▼
        Load Model      vLLM/TGI Server
                              │
                              ▼
                         FastAPI/API
                              │
                     ┌────────┼────────┐
                     ▼        ▼        ▼
                   Auth    Redis   Monitoring
                              │
                              ▼
                           Users
```

---

# 1. Deployment options

There are several ways to deploy a fine-tuned model.

| Approach                | Best for                        |
| ----------------------- | ------------------------------- |
| Hugging Face `pipeline` | Development/testing             |
| FastAPI + Transformers  | Small/simple deployment         |
| vLLM                    | High-throughput production      |
| TGI                     | Production Hugging Face serving |
| Kubernetes              | Large-scale deployment          |
| Managed endpoint        | Faster cloud deployment         |

For a production LLM system, I would usually use:

```text
Fine-tuned LLM
     ↓
vLLM
     ↓
FastAPI Gateway
     ↓
Kubernetes
```

---

# 2. First decision: full model or LoRA adapter?

Suppose you fine-tuned using LoRA.

You may have:

```text
base_model/
    ├── model.safetensors
    └── config.json

lora_adapter/
    ├── adapter_model.safetensors
    └── adapter_config.json
```

There are two options.

## Option A: Load base model + LoRA adapter

```text
Base LLM
   +
LoRA Adapter
   ↓
Fine-tuned model
```

This saves storage.

## Option B: Merge LoRA into the base model

```text
Base LLM
   +
LoRA Adapter
   ↓
Merged Model
   ↓
Deploy
```

This can simplify serving, although serving adapters directly is often useful when multiple fine-tuned variants share one base model.

---

# 3. Load a LoRA model for inference

Install:

```bash
pip install torch transformers peft accelerate
```

Code:

```python
import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)

from peft import (
    PeftModel
)


BASE_MODEL = "meta-llama/Llama-3.1-8B"

ADAPTER_PATH = "./customer-support-adapter"


tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL
)


base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,

    torch_dtype=torch.bfloat16,

    device_map="auto"
)


model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_PATH
)

model.eval()
```

Now:

```text
User Prompt
     ↓
Tokenizer
     ↓
Base Model + LoRA
     ↓
Generated Response
```

---

# 4. Run inference

```python
def generate_response(
    prompt: str
):

    inputs = tokenizer(
        prompt,

        return_tensors="pt"
    )

    inputs = {
        key: value.to(
            model.device
        )
        for key, value
        in inputs.items()
    }

    with torch.no_grad():

        output = model.generate(

            **inputs,

            max_new_tokens=256,

            temperature=0.7,

            do_sample=True,

            top_p=0.9
        )

    response = tokenizer.decode(

        output[0],

        skip_special_tokens=True
    )

    return response
```

Usage:

```python
response = generate_response(
    "How do I reset my password?"
)

print(response)
```

---

# 5. Merge LoRA before deployment

Sometimes you want one standalone model.

```python
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)

from peft import PeftModel

import torch


BASE_MODEL = "meta-llama/Llama-3.1-8B"

ADAPTER_PATH = "./adapter"

OUTPUT_PATH = "./merged-model"


base_model = (
    AutoModelForCausalLM
    .from_pretrained(

        BASE_MODEL,

        torch_dtype=torch.float16,

        device_map="auto"
    )
)


model = PeftModel.from_pretrained(

    base_model,

    ADAPTER_PATH
)


merged_model = (
    model.merge_and_unload()
)


merged_model.save_pretrained(
    OUTPUT_PATH
)


tokenizer = (
    AutoTokenizer.from_pretrained(
        BASE_MODEL
    )
)

tokenizer.save_pretrained(
    OUTPUT_PATH
)
```

Now:

```text
merged-model/

├── config.json
├── model.safetensors
└── tokenizer files
```

You can deploy it without PEFT.

---

# 6. Simple FastAPI deployment

Project:

```text
llm-service/

├── app/
│   ├── main.py
│   ├── model.py
│   └── schemas.py
│
├── requirements.txt
└── Dockerfile
```

---

## `app/schemas.py`

```python
from pydantic import (
    BaseModel,
    Field
)


class GenerateRequest(
    BaseModel
):

    prompt: str = Field(
        min_length=1,
        max_length=10000
    )

    max_new_tokens: int = Field(
        default=256,
        ge=1,
        le=2048
    )

    temperature: float = Field(
        default=0.7,
        ge=0,
        le=2
    )


class GenerateResponse(
    BaseModel
):

    response: str
```

---

## `app/model.py`

Load the model once during startup.

```python
import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)


class ModelService:

    def __init__(
        self
    ):

        self.model = None

        self.tokenizer = None


    def load_model(
        self,
        model_path: str
    ):

        self.tokenizer = (
            AutoTokenizer
            .from_pretrained(
                model_path
            )
        )

        self.model = (
            AutoModelForCausalLM
            .from_pretrained(

                model_path,

                torch_dtype=
                    torch.bfloat16,

                device_map=
                    "auto"
            )
        )

        self.model.eval()


    def generate(

        self,

        prompt: str,

        max_new_tokens: int,

        temperature: float

    ):

        inputs = (
            self.tokenizer(

                prompt,

                return_tensors="pt"
            )
        )

        inputs = {

            key:
            value.to(
                self.model.device
            )

            for key, value

            in inputs.items()
        }


        with torch.inference_mode():

            output = (
                self.model.generate(

                    **inputs,

                    max_new_tokens=
                        max_new_tokens,

                    temperature=
                        temperature,

                    do_sample=
                        temperature > 0
                )
            )


        generated_tokens = (
            output[0][
                inputs[
                    "input_ids"
                ].shape[1]:
            ]
        )


        return (
            self.tokenizer.decode(

                generated_tokens,

                skip_special_tokens=True
            )
        )
```

Notice this important part:

```python
generated_tokens = output[0][
    inputs["input_ids"].shape[1]:
]
```

Without it, your response can include the original prompt.

---

# 7. FastAPI application

## `app/main.py`

```python
from contextlib import (
    asynccontextmanager
)

from fastapi import (
    FastAPI,
    HTTPException
)

from app.model import (
    ModelService
)

from app.schemas import (
    GenerateRequest,
    GenerateResponse
)


model_service = (
    ModelService()
)


@asynccontextmanager
async def lifespan(
    app: FastAPI
):

    model_service.load_model(
        "./merged-model"
    )

    yield

    # Cleanup
    model_service.model = None


app = FastAPI(
    title="Fine-Tuned LLM API",
    lifespan=lifespan
)


@app.get(
    "/health"
)
async def health():

    if model_service.model is None:

        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )

    return {
        "status": "healthy"
    }


@app.post(
    "/generate",
    response_model=GenerateResponse
)
async def generate(
    request: GenerateRequest
):

    try:

        response = (
            model_service.generate(

                prompt=request.prompt,

                max_new_tokens=
                    request.max_new_tokens,

                temperature=
                    request.temperature
            )
        )

        return {
            "response": response
        }

    except RuntimeError as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )
```

Run:

```bash
uvicorn app.main:app \
  --host 0.0.0.0 \
  --port 8000
```

Test:

```bash
curl -X POST \
  http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "How do I reset my password?",
    "max_new_tokens": 100
  }'
```

---

# 8. Why FastAPI alone is not ideal for large LLM production

This architecture:

```text
FastAPI
   │
   ▼
Transformers Model
```

works for:

```text
Development
Small traffic
Internal tools
```

But at scale:

```text
User 1 ─┐
User 2 ─┤
User 3 ─┼──► FastAPI ──► GPU
User N ─┘
```

Problems:

```text
❌ Requests processed inefficiently
❌ GPU batching is limited
❌ Low throughput
❌ Memory management is harder
❌ Multiple model copies possible
```

For high traffic, use an inference engine.

---

# 9. Deploy using vLLM

[vLLM](https://docs.vllm.ai/?utm_source=chatgpt.com) is designed for efficient LLM inference.

Architecture:

```text
Users
  │
  ▼
FastAPI Gateway
  │
  ▼
vLLM
  │
  ▼
GPU
```

Install:

```bash
pip install vllm
```

Run a merged model:

```bash
vllm serve \
    ./merged-model \
    --host 0.0.0.0 \
    --port 8000
```

vLLM provides an OpenAI-compatible API.

Example:

```python
import requests


response = requests.post(

    "http://localhost:8000/v1/chat/completions",

    json={

        "model":
            "customer-support-model",

        "messages": [

            {
                "role": "user",

                "content":
                    "How do I reset my password?"
            }

        ],

        "max_tokens": 256
    }
)

print(
    response.json()
)
```

---

# 10. Production API gateway

Instead of exposing vLLM directly:

```text
                 Client
                   │
                   ▼
              Load Balancer
                   │
                   ▼
             FastAPI Gateway
              /          \
             ▼            ▼
           Auth       Rate Limit
             │            │
             └─────┬──────┘
                   ▼
                 vLLM
                   │
                   ▼
                  GPU
```

Example gateway:

```python
import httpx

from fastapi import (
    FastAPI,
    HTTPException
)

app = FastAPI()


VLLM_URL = (
    "http://vllm:8000"
)


@app.post(
    "/chat"
)
async def chat(
    request: dict
):

    try:

        async with (
            httpx.AsyncClient(
                timeout=60
            )
        ) as client:

            response = (
                await client.post(

                    f"{VLLM_URL}"
                    "/v1/chat/completions",

                    json=request
                )
            )

        response.raise_for_status()

        return response.json()


    except httpx.TimeoutException:

        raise HTTPException(

            status_code=504,

            detail="LLM timeout"
        )
```

In a real system you would also add:

```text
JWT authentication
RBAC
Rate limiting
Request validation
Logging
Tracing
Metrics
Cost tracking
```

---

# 11. Dockerize the model API

## `requirements.txt`

```text
fastapi
uvicorn
torch
transformers
peft
accelerate
```

## Dockerfile

```dockerfile
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install \
    --no-cache-dir \
    -r requirements.txt

COPY app ./app

COPY merged-model \
    ./merged-model

EXPOSE 8000

CMD [
    "uvicorn",
    "app.main:app",
    "--host",
    "0.0.0.0",
    "--port",
    "8000"
]
```

For GPU production, you would generally use a CUDA-compatible base image rather than this CPU-oriented example.

Build:

```bash
docker build \
  -t fine-tuned-llm:v1 .
```

Run:

```bash
docker run \
  --gpus all \
  -p 8000:8000 \
  fine-tuned-llm:v1
```

---

# 12. Kubernetes deployment

For GPU deployment:

```yaml
apiVersion: apps/v1
kind: Deployment

metadata:
  name: llm-inference

spec:

  replicas: 2

  selector:

    matchLabels:

      app: llm-inference


  template:

    metadata:

      labels:

        app: llm-inference


    spec:

      containers:

        - name: llm

          image: registry.example.com/llm:v1

          ports:

            - containerPort: 8000


          resources:

            limits:

              nvidia.com/gpu: 1

            requests:

              cpu: "4"

              memory: "16Gi"
```

Service:

```yaml
apiVersion: v1
kind: Service

metadata:
  name: llm-service

spec:

  selector:

    app: llm-inference

  ports:

    - port: 80

      targetPort: 8000
```

---

# 13. Add health checks

Kubernetes should know whether the model is ready.

```yaml
livenessProbe:

  httpGet:

    path: /health

    port: 8000

  initialDelaySeconds: 60


readinessProbe:

  httpGet:

    path: /health

    port: 8000

  initialDelaySeconds: 60
```

A better design separates:

```text
/live
```

from:

```text
/ready
```

Example:

```python
@app.get("/live")
async def live():
    return {"status": "alive"}


@app.get("/ready")
async def ready():

    if model_service.model is None:

        raise HTTPException(
            status_code=503,
            detail="Model not ready"
        )

    return {
        "status": "ready"
    }
```

Why?

```text
Liveness:
"Is the process alive?"

Readiness:
"Can the process serve requests?"
```

---

# 14. Model versioning during deployment

Never deploy:

```text
latest
```

Instead:

```text
customer-support-model:v1.0.0
```

Then:

```text
customer-support-model:v1.1.0
```

Track:

```python
MODEL_VERSION = (
    "1.1.0"
)

MODEL_NAME = (
    "customer-support-llm"
)
```

Return the model version:

```python
@app.get(
    "/model-info"
)
async def model_info():

    return {

        "model":
            MODEL_NAME,

        "version":
            MODEL_VERSION
    }
```

This helps debugging:

```text
Bad response
     ↓
Which model generated it?
     ↓
customer-support-llm:v1.1.0
```

---

# 15. Canary deployment

Never immediately send 100% traffic to a new fine-tuned model.

```text
              100 Users
                 │
        ┌────────┴────────┐
        ▼                 ▼
      95%                5%
        │                 │
        ▼                 ▼
    Model v1          Model v2
```

Monitor:

```text
Latency
Error rate
Hallucination rate
User feedback
Safety metrics
```

If v2 performs well:

```text
5%
 ↓
25%
 ↓
50%
 ↓
100%
```

---

# 16. Rollback

Keep the previous model:

```text
Production

Current:
Model v1.2

Previous:
Model v1.1
```

If:

```text
v1.2
```

causes:

```text
High latency
Bad answers
Increased errors
```

rollback:

```text
v1.2
   ↓
v1.1
```

With Kubernetes:

```bash
kubectl rollout undo \
    deployment/llm-inference
```

---

# 17. Production deployment pipeline

A good CI/CD flow:

```text
Developer
   │
   ▼
Git Push
   │
   ▼
Run Tests
   │
   ├── Unit Tests
   ├── API Tests
   └── Model Tests
   │
   ▼
Build Image
   │
   ▼
Security Scan
   │
   ▼
Deploy Staging
   │
   ▼
Model Evaluation
   │
   ├── Accuracy
   ├── Hallucination
   ├── Safety
   └── Latency
   │
   ▼
Approval
   │
   ▼
Canary Deployment
   │
   ▼
Production
```

---

# 18. A realistic production architecture

```text
                    ┌──────────────────┐
                    │    Client Apps   │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ API Gateway      │
                    │ Auth / RBAC      │
                    │ Rate Limiting    │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ FastAPI Service  │
                    │ Validation       │
                    │ Logging          │
                    └────────┬─────────┘
                             │
                   ┌─────────┴─────────┐
                   ▼                   ▼
            ┌────────────┐       ┌────────────┐
            │ Redis      │       │ PostgreSQL │
            │ Cache      │       │ Audit Data │
            └────────────┘       └────────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ vLLM Inference   │
                    │ Dynamic Batching │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Fine-Tuned Model │
                    │ Base + LoRA      │
                    └──────────────────┘
```

---

# 19. What happens from training to deployment?

```text
1. Train model
       │
       ▼
2. Evaluate model
       │
       ▼
3. Save artifact
       │
       ▼
4. Register model
       │
       ▼
5. Deploy staging
       │
       ▼
6. Run evaluation tests
       │
       ▼
7. Canary deployment
       │
       ▼
8. Monitor
       │
       ▼
9. Promote to production
```

---

# Interview-ready answer

> **I would deploy a fine-tuned model by first evaluating and versioning the model artifact, then registering it in a model registry. For LoRA fine-tuning, I can either serve the base model with the LoRA adapter or merge the adapter into the base model depending on serving requirements.**
>
> **For development or low traffic, I can use Transformers with FastAPI. For production, I would typically use an optimized inference engine such as vLLM behind an API gateway. The API layer handles authentication, validation, rate limiting, logging, and request tracing, while vLLM handles GPU inference and dynamic batching.**
>
> **I would containerize the service, deploy it to Kubernetes with GPU resource requests, readiness and liveness probes, autoscaling, monitoring, and checkpointed model versions. I would deploy new model versions using canary releases, monitor latency, error rate, quality and hallucination metrics, and maintain rollback capability to the previous stable model.**

## Key takeaway

```text
Fine-tuning
    ↓
Evaluation
    ↓
Model Registry
    ↓
Container
    ↓
Staging
    ↓
Canary
    ↓
Production
    ↓
Monitoring
    ↓
Rollback / Retraining
```

This is the production approach I would describe in a **Senior AI Engineer interview**.
