# How would you serve a LoRA model?

Serving a LoRA model means serving:

```text
Base Model
    +
LoRA Adapter
    =
Fine-tuned Behavior
```

Unlike full fine-tuning, you usually **do not create a completely separate copy of the base model for every fine-tuned model**.

This is one of LoRA's biggest deployment advantages.

---

# 1. What does a LoRA deployment look like?

Suppose you have:

```text
Base Model:
Llama-3.1-8B
```

and three LoRA adapters:

```text
customer-support-adapter
sql-adapter
finance-adapter
```

Conceptually:

```text
                 ┌──────────────────────┐
                 │   Base LLM (8B)      │
                 └──────────┬───────────┘
                            │
             ┌──────────────┼──────────────┐
             ▼              ▼              ▼
        Support LoRA      SQL LoRA     Finance LoRA
             │              │              │
             ▼              ▼              ▼
        Support Agent      SQL Agent    Finance Agent
```

Instead of storing:

```text
8 GB model × 3
```

you store approximately:

```text
Base Model
+
Small LoRA adapters
```

This is particularly useful when you have many fine-tuned variants.

---

# 2. How does LoRA work during inference?

During fine-tuning, LoRA learns small low-rank matrices.

The original weight is:

```text
W
```

LoRA learns:

```text
ΔW = B × A
```

During inference:

```text
Effective Weight = W + ΔW
```

Conceptually:

```text
Input
   │
   ▼
┌─────────────────────┐
│ Base Model Weight W │
└──────────┬──────────┘
           +
┌──────────▼──────────┐
│ LoRA Update ΔW      │
│      B × A          │
└──────────┬──────────┘
           │
           ▼
     Model Output
```

The base weights normally remain frozen.

---

# 3. Option 1: Serve base model + LoRA using Hugging Face PEFT

This is the easiest approach for development or low traffic.

Install:

```bash
pip install torch transformers peft accelerate
```

Load the base model:

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

ADAPTER_PATH = (
    "./lora-adapters/customer-support"
)
```

```python
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL
)

base_model = (
    AutoModelForCausalLM
    .from_pretrained(

        BASE_MODEL,

        torch_dtype=torch.bfloat16,

        device_map="auto"
    )
)
```

Load the adapter:

```python
model = PeftModel.from_pretrained(

    base_model,

    ADAPTER_PATH
)

model.eval()
```

Architecture:

```text
Prompt
   │
   ▼
Tokenizer
   │
   ▼
Base Model
   +
LoRA Adapter
   │
   ▼
Generated Response
```

Generate:

```python
def generate(
    prompt: str
):

    inputs = tokenizer(

        prompt,

        return_tensors="pt"
    ).to(model.device)


    with torch.inference_mode():

        output = model.generate(

            **inputs,

            max_new_tokens=256,

            temperature=0.7,

            do_sample=True
        )


    generated_tokens = output[0][
        inputs["input_ids"].shape[1]:
    ]


    return tokenizer.decode(

        generated_tokens,

        skip_special_tokens=True
    )
```

---

# 4. Expose the LoRA model using FastAPI

A basic project structure:

```text
lora-serving/

├── app/
│   ├── main.py
│   ├── model_service.py
│   └── schemas.py
│
├── adapters/
│   └── customer-support/
│
└── requirements.txt
```

## `schemas.py`

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
        le=1024
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

## `model_service.py`

Load the base model and adapter only once.

```python
import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)

from peft import PeftModel


class LoRAModelService:

    def __init__(self):

        self.model = None

        self.tokenizer = None


    def load_model(
        self,
        base_model_name: str,
        adapter_path: str
    ):

        self.tokenizer = (
            AutoTokenizer
            .from_pretrained(
                base_model_name
            )
        )

        if (
            self.tokenizer.pad_token
            is None
        ):

            self.tokenizer.pad_token = (
                self.tokenizer.eos_token
            )


        base_model = (
            AutoModelForCausalLM
            .from_pretrained(

                base_model_name,

                torch_dtype=
                    torch.bfloat16,

                device_map="auto"
            )
        )


        self.model = (
            PeftModel.from_pretrained(

                base_model,

                adapter_path
            )
        )

        self.model.eval()


    @torch.inference_mode()
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


        generation_kwargs = {

            **inputs,

            "max_new_tokens":
                max_new_tokens,

            "pad_token_id":
                self.tokenizer.eos_token_id
        }


        if temperature > 0:

            generation_kwargs.update({

                "do_sample": True,

                "temperature":
                    temperature,

                "top_p": 0.9
            })

        else:

            generation_kwargs.update({

                "do_sample": False
            })


        output = (
            self.model.generate(
                **generation_kwargs
            )
        )


        input_length = (
            inputs["input_ids"]
            .shape[1]
        )


        generated_tokens = output[
            0,
            input_length:
        ]


        return (
            self.tokenizer.decode(

                generated_tokens,

                skip_special_tokens=True
            )
        )
```

---

## `main.py`

```python
from contextlib import (
    asynccontextmanager
)

from fastapi import (
    FastAPI,
    HTTPException
)

from app.model_service import (
    LoRAModelService
)

from app.schemas import (
    GenerateRequest,
    GenerateResponse
)


model_service = LoRAModelService()


@asynccontextmanager
async def lifespan(
    app: FastAPI
):

    model_service.load_model(

        base_model_name=
            "meta-llama/Llama-3.1-8B",

        adapter_path=
            "./adapters/customer-support"
    )

    yield

    model_service.model = None


app = FastAPI(
    title="LoRA Inference Service",
    lifespan=lifespan
)


@app.get("/health")
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

    except RuntimeError:

        raise HTTPException(
            status_code=500,
            detail="Inference failed"
        )
```

Run:

```bash
uvicorn app.main:app \
  --host 0.0.0.0 \
  --port 8000
```

---

# 5. Important production issue: don't load the model per request

Bad:

```python
@app.post("/generate")
async def generate(
    request
):

    model = load_model()

    return model.generate(
        request.prompt
    )
```

Why is this bad?

```text
Request
   │
   ▼
Load 8B model
   │
   ▼
Load LoRA
   │
   ▼
Generate
   │
   ▼
Destroy model
```

Problems:

```text
❌ Very high latency
❌ GPU memory allocation overhead
❌ Low throughput
❌ Possible GPU OOM
```

Correct:

```text
Application Starts
       │
       ▼
Load Base Model Once
       │
       ▼
Load LoRA Once
       │
       ▼
Keep in GPU Memory
       │
       ▼
Serve Many Requests
```

---

# 6. Serving multiple LoRA adapters

This is one of the strongest LoRA deployment patterns.

Suppose:

```text
Base Model
    │
    ├── Support Adapter
    │
    ├── Finance Adapter
    │
    └── SQL Adapter
```

You can load multiple adapters into the same model.

```python
from peft import (
    PeftModel
)


model = PeftModel.from_pretrained(

    base_model,

    "./adapters/support",

    adapter_name="support"
)
```

Load another adapter:

```python
model.load_adapter(

    "./adapters/finance",

    adapter_name="finance"
)
```

Load SQL adapter:

```python
model.load_adapter(

    "./adapters/sql",

    adapter_name="sql"
)
```

Now:

```text
One Base Model
       │
       ├── support
       ├── finance
       └── sql
```

Select the adapter:

```python
model.set_adapter(
    "support"
)
```

Or:

```python
model.set_adapter(
    "finance"
)
```

---

# 7. Dynamic adapter routing

Imagine this API:

```text
POST /generate
```

Request:

```json
{
    "adapter": "finance",
    "prompt": "Explain the customer's portfolio"
}
```

Schema:

```python
from pydantic import (
    BaseModel
)


class GenerateRequest(
    BaseModel
):

    adapter: str

    prompt: str
```

Routing:

```python
ALLOWED_ADAPTERS = {
    "support",
    "finance",
    "sql"
}


def generate(
    adapter_name: str,
    prompt: str
):

    if (
        adapter_name
        not in ALLOWED_ADAPTERS
    ):

        raise ValueError(
            "Invalid adapter"
        )


    model.set_adapter(
        adapter_name
    )


    # Generate response

    return response
```

Architecture:

```text
                 User Request
                      │
                      ▼
                Adapter Router
                      │
         ┌────────────┼────────────┐
         ▼            ▼            ▼
      Support      Finance        SQL
      Adapter      Adapter        Adapter
         │            │            │
         └────────────┼────────────┘
                      │
                      ▼
                 Base Model
                      │
                      ▼
                     GPU
```

---

# 8. Concurrency problem with `set_adapter()`

The previous implementation has a serious problem.

Suppose:

```text
Request A
adapter = finance
```

At the same time:

```text
Request B
adapter = support
```

This code:

```python
model.set_adapter(
    adapter_name
)
```

changes shared model state.

Possible race condition:

```text
Request A:
set_adapter(finance)

Request B:
set_adapter(support)

Request A:
generate()
```

Request A might accidentally use:

```text
support
```

instead of:

```text
finance
```

That is a major problem in concurrent production systems.

---

# 9. Simple solution: lock adapter switching

```python
import asyncio


adapter_lock = (
    asyncio.Lock()
)
```

Then:

```python
async def generate(
    adapter_name,
    prompt
):

    async with adapter_lock:

        model.set_adapter(
            adapter_name
        )

        output = model.generate(
            prompt
        )

    return output
```

This prevents race conditions.

However:

```text
One request
    │
    ▼
Acquire lock
    │
    ▼
Inference
    │
    ▼
Release lock
```

The problem is throughput.

All requests may become effectively serialized.

For high traffic, this is not the best solution.

---

# 10. Production approach: use an inference server with LoRA support

For production, I would usually use an optimized inference server such as [vLLM](https://docs.vllm.ai/?utm_source=chatgpt.com) rather than manually switching adapters inside a shared Hugging Face model.

Conceptually:

```text
                    API Gateway
                         │
                         ▼
                    Adapter Router
                         │
                         ▼
                Inference Server
                         │
             ┌───────────┴───────────┐
             ▼                       ▼
       Base Model               LoRA Adapters
             │                       │
             └───────────┬───────────┘
                         │
                         ▼
                        GPU
```

The inference server can manage requests more efficiently.

---

# 11. Example vLLM LoRA serving

Conceptually, start the server with LoRA support:

```bash
vllm serve \
  meta-llama/Llama-3.1-8B \
  --enable-lora \
  --max-loras 8
```

The important idea is:

```text
Base Model
    ↓
Loaded once
    ↓
Multiple LoRA adapters
    ↓
Serve requests efficiently
```

You can register adapters for specific names.

Conceptually:

```text
support-lora
finance-lora
sql-lora
```

The request selects the appropriate adapter/model identity.

Example request:

```python
import requests


response = requests.post(

    "http://localhost:8000/v1/chat/completions",

    json={

        "model":
            "support-lora",

        "messages": [

            {
                "role": "user",

                "content":
                    "How do I reset my password?"
            }
        ]
    }
)

print(
    response.json()
)
```

This is better than doing:

```python
model.set_adapter()
```

inside every FastAPI request.

---

# 12. Dynamic LoRA loading

Imagine you have hundreds of adapters.

You cannot necessarily load all of them into GPU memory:

```text
Base Model

+ Adapter 1
+ Adapter 2
+ Adapter 3
...
+ Adapter 500
```

Instead:

```text
Adapter Registry
       │
       ▼
Frequently Used
Adapters in Memory
       │
       ▼
Less Frequently Used
Adapters loaded on demand
```

A conceptual cache:

```python
from collections import OrderedDict


class AdapterCache:

    def __init__(
        self,
        max_adapters=10
    ):

        self.max_adapters = (
            max_adapters
        )

        self.adapters = (
            OrderedDict()
        )


    def get(
        self,
        adapter_name
    ):

        if (
            adapter_name
            in self.adapters
        ):

            self.adapters.move_to_end(
                adapter_name
            )

            return self.adapters[
                adapter_name
            ]

        return None


    def add(
        self,
        adapter_name,
        adapter
    ):

        if (
            len(self.adapters)
            >=
            self.max_adapters
        ):

            self.adapters.popitem(
                last=False
            )

        self.adapters[
            adapter_name
        ] = adapter
```

This is an LRU-style concept.

In production, adapter lifecycle management should generally be handled by the inference system rather than hand-written application-level GPU memory management.

---

# 13. Merge LoRA before serving

Another approach:

```text
Base Model
     +
LoRA
     │
     ▼
Merged Model
     │
     ▼
Serve
```

Code:

```python
from transformers import (
    AutoModelForCausalLM
)

from peft import (
    PeftModel
)


base_model = (
    AutoModelForCausalLM
    .from_pretrained(
        BASE_MODEL
    )
)


model = (
    PeftModel
    .from_pretrained(

        base_model,

        ADAPTER_PATH
    )
)
```

Merge:

```python
merged_model = (
    model.merge_and_unload()
)
```

Save:

```python
merged_model.save_pretrained(
    "./merged-model"
)
```

Then:

```text
merged-model/
    │
    ├── model.safetensors
    ├── config.json
    └── tokenizer files
```

Serve like a normal model.

---

# 14. Base + LoRA vs merged model

| Feature               | Base + LoRA                | Merged model           |
| --------------------- | -------------------------- | ---------------------- |
| Storage               | Efficient                  | Larger                 |
| Multiple adapters     | Excellent                  | Poor                   |
| Switch behavior       | Possible                   | No                     |
| Deployment simplicity | Medium                     | Easy                   |
| Adapter sharing       | Yes                        | No                     |
| Best for              | Multi-tenant/domain models | Single dedicated model |

Example.

### SaaS application

```text
Tenant A → Adapter A
Tenant B → Adapter B
Tenant C → Adapter C
```

Use:

```text
Base + LoRA
```

### One customer support model

```text
One model
One behavior
One deployment
```

You can consider:

```text
Merged Model
```

---

# 15. Multi-tenant LoRA serving

A very interesting production architecture is:

```text
                    API Gateway
                         │
                         ▼
                     JWT Auth
                         │
                         ▼
                  Identify Tenant
                         │
                         ▼
                   Tenant Router
                         │
              ┌──────────┼──────────┐
              ▼          ▼          ▼
           Tenant A   Tenant B   Tenant C
              │          │          │
              ▼          ▼          ▼
          LoRA A      LoRA B      LoRA C
              │          │          │
              └──────────┼──────────┘
                         ▼
                     Base Model
                         │
                         ▼
                        GPU
```

Example:

```python
def get_adapter_for_tenant(
    tenant_id: str
):

    adapter_mapping = {

        "tenant_a":
            "support_adapter",

        "tenant_b":
            "finance_adapter",

        "tenant_c":
            "sql_adapter"
    }

    return adapter_mapping.get(
        tenant_id
    )
```

But you should never allow users to directly choose arbitrary adapter paths:

```text
/adapters/../../secret
```

Instead:

```text
User identity
    ↓
Server-side mapping
    ↓
Authorized adapter
```

---

# 16. Secure LoRA serving

LoRA adapters can represent customer-specific behavior or proprietary training data.

Important controls:

```text
JWT Authentication
        │
        ▼
Tenant Identification
        │
        ▼
RBAC
        │
        ▼
Adapter Authorization
```

Example:

```python
def authorize_adapter(
    user,
    adapter_name
):

    allowed_adapters = (
        get_allowed_adapters(
            user.tenant_id
        )
    )

    if (
        adapter_name
        not in allowed_adapters
    ):

        raise PermissionError(
            "Adapter access denied"
        )
```

Never trust:

```text
adapter_name
```

directly from the client.

---

# 17. Production architecture I would use

```text
                          Client
                            │
                            ▼
                      Load Balancer
                            │
                            ▼
                       API Gateway
                    Auth / Rate Limit
                            │
                            ▼
                     FastAPI Service
                            │
                            ▼
                     Adapter Router
                            │
                 ┌──────────┴──────────┐
                 │                     │
                 ▼                     ▼
           Redis Adapter Cache      Registry
                 │                     │
                 └──────────┬──────────┘
                            ▼
                    vLLM Inference
                            │
                     Base Model
                            │
               ┌────────────┼────────────┐
               ▼            ▼            ▼
            LoRA A       LoRA B       LoRA C
                            │
                            ▼
                           GPU
```

---

# 18. What should you monitor?

For LoRA serving, monitor normal LLM metrics:

```text
Request latency
P50 latency
P95 latency
Error rate
Tokens/sec
GPU utilization
GPU memory
Queue length
```

And LoRA-specific metrics:

```text
Adapter selected
Adapter load latency
Adapter cache hit rate
Adapter memory usage
Adapter load failures
Requests per adapter
```

Example metrics:

```python
metrics = {

    "adapter_name":
        "finance",

    "adapter_load_ms":
        120,

    "adapter_cache_hit":
        True,

    "inference_latency_ms":
        850
}
```

This helps identify:

```text
Why is finance adapter slow?

→ Adapter is being loaded frequently
→ Low cache hit rate
→ Increase adapter cache
```

---

# 19. Recommended deployment strategy

### Development

```text
Transformers
+
PEFT
+
FastAPI
```

Good for:

```text
Testing
Internal applications
Low traffic
```

### Production, one LoRA adapter

```text
Base + LoRA
or
Merged Model
```

served through an optimized inference server.

### Production, multiple adapters

```text
Base Model
+
Multiple LoRA adapters
+
Adapter-aware inference server
+
API Gateway
```

### Large multi-tenant system

```text
Model Registry
+
Adapter Registry
+
Dynamic Adapter Loading
+
Caching
+
Authorization
+
Autoscaling
```

---

# Interview-ready answer

> **To serve a LoRA fine-tuned model, I first load the base model and attach the LoRA adapter using PEFT. For development or low traffic, I can expose that model through FastAPI, making sure the model is loaded once during application startup rather than per request.**
>
> **For production, I usually use an optimized inference engine with LoRA support. This allows the base model to remain loaded once while multiple adapters can provide different domain-specific or tenant-specific behavior. I avoid manually calling `set_adapter()` in concurrent request handlers because changing shared adapter state can create race conditions.**
>
> **For a single dedicated adapter, I can either serve the adapter directly or merge it into the base model for simpler deployment. For multi-adapter or multi-tenant systems, I keep adapters separate, route requests to authorized adapters, manage adapter caching and loading, and monitor adapter-specific latency, memory usage, and failure rates.**

## One-line summary

```text
Load the base model once → attach/select the LoRA adapter →
use an optimized inference server for concurrency →
route and authorize adapters → monitor and autoscale.
```
