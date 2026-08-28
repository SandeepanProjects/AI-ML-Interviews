# How would you reduce LLM inference cost?

Reducing inference cost means reducing the total resources required to serve useful model responses.

A simplified cost model is:

```text
Inference Cost
=
Model Compute Cost
+
GPU Cost
+
Input Token Cost
+
Output Token Cost
+
Infrastructure Cost
```

For a production LLM application, the biggest levers are usually:

1. Use the right model size
2. Reduce input/output tokens
3. Cache responses
4. Batch requests
5. Use efficient inference engines
6. Quantize the model
7. Route requests to cheaper models
8. Autoscale infrastructure
9. Optimize prompts and RAG
10. Monitor cost per request

---

# 1. Use the smallest model that meets your quality requirement

A common mistake is:

```text
Every request
    ↓
70B model
```

Instead:

```text
Simple request
    ↓
Small model

Medium complexity
    ↓
Medium model

Complex request
    ↓
Large model
```

Example architecture:

```text
                    User Request
                         │
                         ▼
                   Request Router
                    /     |      \
                   /      |       \
                  ▼       ▼        ▼
              Small    Medium    Large
              Model    Model     Model
               $$$      $$$$      $$$$$$$
```

## Simple router

```python
class ModelRouter:

    def choose_model(
        self,
        prompt: str
    ) -> str:

        prompt_length = len(
            prompt.split()
        )

        if prompt_length < 20:
            return "small-model"

        if prompt_length < 100:
            return "medium-model"

        return "large-model"
```

Usage:

```python
router = ModelRouter()

model = router.choose_model(
    "What is Python?"
)

print(model)
```

Output:

```text
small-model
```

### Better production approach

Use:

```text
Request complexity
Task type
Historical quality
Latency requirements
Cost budget
```

Example:

```python
def route_request(
    task_type: str,
    complexity: float
):

    if task_type == "classification":
        return "small-model"

    if complexity < 0.4:
        return "small-model"

    if complexity < 0.7:
        return "medium-model"

    return "large-model"
```

---

# 2. Reduce input tokens

Every unnecessary token costs money.

Bad prompt:

```text
You are an extremely intelligent AI assistant.

You must always be helpful.

You must always be accurate.

You must always answer professionally.

You must provide detailed answers.

...
```

Thousands of unnecessary tokens:

```text
Request
   +
Huge system prompt
   +
Conversation history
   +
Large RAG context
   =
High cost
```

Instead:

```python
SYSTEM_PROMPT = """
Answer accurately.
Use the provided context.
If the answer is unknown, say so.
"""
```

Shorter prompts:

```text
Fewer tokens
     ↓
Lower cost
     ↓
Lower latency
```

---

# 3. Limit conversation history

Do not send the entire conversation forever.

Bad:

```text
Message 1
Message 2
Message 3
...
Message 500
```

Instead:

```text
Recent Messages
+
Conversation Summary
```

Example:

```python
def build_context(
    messages,
    summary
):

    recent_messages = messages[-10:]

    return [
        {
            "role": "system",
            "content": summary
        },
        *recent_messages
    ]
```

Architecture:

```text
Old conversation
       │
       ▼
   Summarization
       │
       ▼
Conversation summary
       +
Recent messages
       │
       ▼
LLM
```

This can dramatically reduce repeated input tokens.

---

# 4. Optimize RAG context

One of the biggest cost problems in enterprise RAG is:

```text
Question
   ↓
Retrieve 50 documents
   ↓
Send everything to LLM ❌
```

Instead:

```text
Question
   ↓
Retrieve Top 20
   ↓
Rerank
   ↓
Select Top 3-5
   ↓
LLM
```

Example:

```python
def retrieve_context(
    query,
    vector_store,
    reranker
):

    documents = vector_store.search(
        query=query,
        limit=20
    )

    ranked_documents = reranker.rerank(
        query=query,
        documents=documents
    )

    return ranked_documents[:5]
```

Then enforce a token budget:

```python
MAX_CONTEXT_TOKENS = 6000


def select_documents(
    documents,
    tokenizer
):

    selected = []

    total_tokens = 0

    for document in documents:

        token_count = len(
            tokenizer.encode(
                document
            )
        )

        if (
            total_tokens
            +
            token_count
            >
            MAX_CONTEXT_TOKENS
        ):
            break

        selected.append(
            document
        )

        total_tokens += token_count

    return selected
```

This is extremely important.

More context is **not always better**.

---

# 5. Semantic caching

Many users ask similar questions.

Example:

```text
"What is your refund policy?"

"Tell me about refunds"

"Can I get my money back?"
```

These are semantically similar.

Instead of calling the LLM every time:

```text
User Request
      │
      ▼
Semantic Cache
      │
 ┌────┴─────┐
 │          │
Hit         Miss
 │          │
 ▼          ▼
Return      LLM
Cached      Response
Answer
```

## Redis example

```python
import hashlib
import json

import redis


redis_client = redis.Redis(
    host="localhost",
    port=6379,
    decode_responses=True
)


def generate_cache_key(
    prompt: str
):

    return hashlib.sha256(
        prompt.encode()
    ).hexdigest()
```

Check cache:

```python
def get_cached_response(
    prompt: str
):

    key = (
        generate_cache_key(
            prompt
        )
    )

    response = (
        redis_client.get(key)
    )

    if response:

        return json.loads(
            response
        )

    return None
```

Store:

```python
def cache_response(
    prompt: str,
    response: dict
):

    key = (
        generate_cache_key(
            prompt
        )
    )

    redis_client.setex(

        key,

        3600,

        json.dumps(
            response
        )
    )
```

Full flow:

```python
def answer_question(
    prompt
):

    cached = (
        get_cached_response(
            prompt
        )
    )

    if cached:

        return cached

    response = call_llm(
        prompt
    )

    cache_response(
        prompt,
        response
    )

    return response
```

If cache hit:

```text
LLM Cost = $0
```

for that request.

### Important

Exact cache:

```text
Same text → Same response
```

Semantic cache:

```text
Similar meaning → Reuse response
```

Semantic caching generally uses embeddings.

---

# 6. Batch requests

Without batching:

```text
Request 1 → GPU
Request 2 → GPU
Request 3 → GPU
```

The GPU may be inefficient.

With batching:

```text
Request 1 ─┐
Request 2 ─┼──► Batch ───► GPU
Request 3 ─┘
```

Example:

```python
prompts = [
    "Question 1",
    "Question 2",
    "Question 3"
]

outputs = model.generate(
    **tokenizer(
        prompts,
        return_tensors="pt",
        padding=True
    )
)
```

Production engines such as vLLM dynamically batch requests.

```text
Request 1 ───┐
Request 2 ───┤
Request 3 ───┼──► Dynamic Batch
Request N ───┘
                     │
                     ▼
                    GPU
```

Higher GPU utilization generally means better cost efficiency.

---

# 7. Use an optimized inference engine

Naive:

```text
FastAPI
    ↓
Transformers
    ↓
GPU
```

Better for high throughput:

```text
API Gateway
    ↓
Inference Engine
    ↓
GPU
```

For example, [vLLM](https://docs.vllm.ai/?utm_source=chatgpt.com) is commonly used for efficient LLM serving.

Example command:

```bash
vllm serve ./fine-tuned-model \
    --tensor-parallel-size 2
```

Benefits typically include:

```text
Continuous batching
Efficient KV-cache management
Higher throughput
Better GPU utilization
```

Higher throughput means:

```text
Same GPU
   ↓
More requests
   ↓
Lower cost per request
```

---

# 8. Quantization

LLMs often use:

```text
FP32
FP16
BF16
INT8
INT4
```

Memory:

```text
FP32 → 4 bytes/parameter
FP16 → 2 bytes/parameter
INT8 → 1 byte/parameter
INT4 → 0.5 byte/parameter
```

Example:

```text
7B model

FP16:

7B × 2 bytes
≈ 14 GB
```

INT4 approximately:

```text
7B × 0.5 bytes
≈ 3.5 GB
```

Real memory usage is higher because inference also needs KV cache and other overhead.

Load a quantized model:

```python
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)

import torch


model_name = "your-model"


quantization_config = (
    BitsAndBytesConfig(

        load_in_4bit=True,

        bnb_4bit_quant_type="nf4",

        bnb_4bit_compute_dtype=
            torch.bfloat16
    )
)


model = (
    AutoModelForCausalLM
    .from_pretrained(

        model_name,

        quantization_config=
            quantization_config,

        device_map="auto"
    )
)
```

Benefits:

```text
Less GPU memory
     ↓
Smaller GPU
     ↓
Lower cost
```

But always test quality and latency.

---

# 9. Limit output tokens

Output tokens can be expensive.

Bad:

```python
response = model.generate(
    max_new_tokens=4096
)
```

For a simple question:

```text
"What is Python?"
```

you probably don't need 4,096 output tokens.

Better:

```python
response = model.generate(
    max_new_tokens=300
)
```

Dynamic limits:

```python
def get_max_tokens(
    task_type
):

    limits = {

        "classification": 20,

        "extraction": 100,

        "qa": 300,

        "summarization": 1000,

        "coding": 2000
    }

    return limits.get(
        task_type,
        500
    )
```

This is one of the easiest ways to control cost.

---

# 10. Stop generation early

Use stop sequences.

For example:

```python
STOP_WORDS = [
    "\nUser:",
    "\nHuman:"
]
```

Conceptually:

```text
Model generates answer
        │
        ▼
Stop sequence detected
        │
        ▼
Stop generation
```

For structured output:

```text
{
  "answer": "..."
}
```

stop after the valid JSON is complete instead of allowing the model to continue generating unnecessary text.

---

# 11. Prompt caching

Some applications repeatedly send the same large prefix.

Example:

```text
System Prompt
+
Company Policy
+
Tool Definitions
+
User Question
```

The first three sections may be almost identical across requests.

Conceptually:

```text
Static Prompt Prefix
       │
       ▼
Cached
       │
       +
User-specific content
       │
       ▼
Inference
```

This reduces repeated computation where supported by your inference platform.

---

# 12. Optimize KV cache

During generation, LLMs store:

```text
Key states
Value states
```

This is called the KV cache.

Longer prompts:

```text
More tokens
     ↓
Larger KV cache
     ↓
More GPU memory
```

Optimize by:

```text
Limit context length
Summarize old history
Retrieve fewer RAG chunks
Use smaller context windows when appropriate
```

Example:

```python
MAX_HISTORY_TOKENS = 4000

MAX_RAG_TOKENS = 6000

MAX_INPUT_TOKENS = (
    MAX_HISTORY_TOKENS
    +
    MAX_RAG_TOKENS
)
```

Token budgeting is important.

---

# 13. Use RAG instead of putting everything in the prompt

Bad:

```text
Send entire company documentation
to the LLM
for every request ❌
```

Better:

```text
Question
   │
   ▼
Embedding
   │
   ▼
Vector Search
   │
   ▼
Top Relevant Documents
   │
   ▼
LLM
```

Example:

```python
def answer_with_rag(
    question
):

    documents = (
        vector_db.search(
            query=question,
            limit=5
        )
    )

    context = "\n".join(
        doc.text
        for doc in documents
    )

    prompt = f"""
Context:
{context}

Question:
{question}
"""

    return call_llm(
        prompt
    )
```

This reduces:

```text
Input tokens
GPU computation
Inference cost
```

---

# 14. Use a cheap model for preprocessing

Not every LLM task needs your best model.

Example pipeline:

```text
User Request
      │
      ▼
Small Model
      │
      ├── Classification
      ├── Intent Detection
      └── Routing
              │
              ▼
       Only complex requests
              │
              ▼
          Large Model
```

Example:

```python
def process_request(
    request
):

    intent = (
        small_model.classify(
            request
        )
    )

    if intent == "simple_faq":

        return (
            small_model.generate(
                request
            )
        )

    return (
        large_model.generate(
            request
        )
    )
```

---

# 15. Use deterministic decoding when possible

For:

```text
Classification
Extraction
JSON
Routing
```

you often don't need highly random generation.

```python
generation_config = {

    "do_sample": False,

    "temperature": 0
}
```

This can improve consistency and avoid unnecessary retries. The main cost saving usually comes indirectly: fewer invalid outputs and fewer repeated requests.

---

# 16. Reduce retries

Retries can multiply costs.

Example:

```text
Request
   ↓
LLM call → Invalid JSON
   ↓
Retry
   ↓
Another LLM call
```

Better:

```text
Structured generation
      +
Schema validation
      +
Constrained output
```

Example:

```python
from pydantic import (
    BaseModel
)


class Answer(
    BaseModel):

    answer: str

    confidence: float
```

Validate:

```python
def validate_response(
    response
):

    return Answer.model_validate_json(
        response
    )
```

The goal:

```text
First request succeeds
```

instead of:

```text
Request
   ↓
Retry
   ↓
Retry
   ↓
Success
```

---

# 17. Autoscale GPUs

Bad:

```text
10 GPUs
Running 24/7
```

Even at night:

```text
Traffic = 5%
Cost = 100% ❌
```

Better:

```text
Traffic
   │
   ▼
Autoscaler
   │
   ├── Low traffic → fewer replicas
   │
   └── High traffic → more replicas
```

Example Kubernetes autoscaling concept:

```yaml
apiVersion: autoscaling/v2

kind: HorizontalPodAutoscaler

metadata:
  name: llm-inference

spec:

  minReplicas: 1

  maxReplicas: 10

  metrics:

    - type: Resource

      resource:

        name: cpu

        target:

          type: Utilization

          averageUtilization: 70
```

For GPU LLM serving, you would often scale based on more relevant custom metrics too:

```text
Request queue length
KV-cache utilization
Active requests
Tokens/sec
GPU utilization
Latency
```

---

# 18. Scale to zero for non-real-time workloads

For workloads like:

```text
Nightly summarization
Document processing
Batch extraction
Report generation
```

you may not need GPUs running continuously.

```text
Job arrives
    │
    ▼
Start GPU worker
    │
    ▼
Process job
    │
    ▼
Stop worker
```

This can significantly reduce idle infrastructure cost.

---

# 19. Track cost per request

You cannot optimize what you do not measure.

A useful record:

```python
from dataclasses import (
    dataclass
)


@dataclass
class InferenceMetrics:

    model: str

    input_tokens: int

    output_tokens: int

    latency_ms: float

    gpu_cost: float
```

Calculate estimated cost:

```python
def calculate_cost(
    input_tokens,
    output_tokens,
    input_cost_per_million,
    output_cost_per_million
):

    input_cost = (

        input_tokens
        /
        1_000_000

        *
        input_cost_per_million
    )

    output_cost = (

        output_tokens
        /
        1_000_000

        *
        output_cost_per_million
    )

    return (
        input_cost
        +
        output_cost
    )
```

Example:

```python
cost = calculate_cost(

    input_tokens=2000,

    output_tokens=500,

    input_cost_per_million=2.0,

    output_cost_per_million=8.0
)

print(cost)
```

This lets you compare:

```text
Model A:

Cost/request = $0.002
Quality      = 0.90


Model B:

Cost/request = $0.02
Quality      = 0.91
```

Perhaps Model B is not worth 10× the cost.

---

# 20. Build a cost-aware router

A more realistic approach:

```text
Request
   │
   ▼
Complexity Detection
   │
   ▼
Cost/Quality Router
   │
   ├── Small Model
   ├── Medium Model
   └── Large Model
```

Example:

```python
MODELS = {

    "small": {
        "max_complexity": 0.4,
        "cost": 1
    },

    "medium": {
        "max_complexity": 0.7,
        "cost": 3
    },

    "large": {
        "max_complexity": 1.0,
        "cost": 10
    }
}


def choose_cheapest_model(
    complexity
):

    for model_name, config in MODELS.items():

        if (
            complexity
            <=
            config["max_complexity"]
        ):

            return model_name

    return "large"
```

For:

```text
Complexity = 0.3
```

Output:

```text
small
```

For:

```text
Complexity = 0.9
```

Output:

```text
large
```

A production router should also consider quality thresholds and fallbacks, not just estimated complexity.

---

# 21. Implement fallback instead of always using the expensive model

```text
Small Model
    │
    ▼
High confidence?
   / \
 Yes  No
 │     │
 ▼     ▼
Return Large Model
```

Example:

```python
def answer(
    question
):

    response = (
        small_model.generate(
            question
        )
    )

    if (
        response.confidence
        >=
        0.85
    ):

        return response

    return (
        large_model.generate(
            question
        )
    )
```

Most easy requests stay cheap.

Only difficult requests escalate.

---

# 22. Full production architecture

```text
                     User
                      │
                      ▼
               API Gateway
                      │
          ┌───────────┴────────────┐
          │                        │
          ▼                        ▼
      Rate Limit                 Cache
                                   │
                            ┌──────┴──────┐
                            │             │
                           Hit           Miss
                            │             │
                            ▼             ▼
                         Return      Request Router
                                          │
                              ┌───────────┼───────────┐
                              ▼           ▼           ▼
                            Small       Medium       Large
                            Model       Model        Model
                              │           │           │
                              └───────────┼───────────┘
                                          │
                                          ▼
                                   Optimized Server
                                      (vLLM)
                                          │
                                          ▼
                                         GPU
                                          │
                                          ▼
                                    Metrics/Cost
```

---

# 23. What I would prioritize in a real project

If I had an expensive production LLM system, I would optimize in this order:

### Step 1 — Measure

```text
Cost/request
Input tokens
Output tokens
Latency
Cache hit rate
GPU utilization
```

### Step 2 — Reduce wasted tokens

```text
Prompt optimization
History summarization
RAG reranking
Context token budgets
Output limits
```

### Step 3 — Add caching

```text
Exact cache
Semantic cache
Prompt prefix cache
```

### Step 4 — Improve serving

```text
Dynamic batching
Efficient KV cache
Quantization
Optimized inference engine
```

### Step 5 — Model routing

```text
Small → default
Large → difficult requests
```

### Step 6 — Infrastructure

```text
Autoscaling
Scale-to-zero
Right-size GPUs
```

---

# Interview-ready answer

> **To reduce inference cost, I first measure cost per request, input and output tokens, latency, cache hit rate, throughput, and GPU utilization. The largest savings usually come from reducing unnecessary tokens and using the smallest model that meets the quality requirement.**
>
> **I optimize prompts, summarize long conversation history, use RAG with reranking and strict context budgets, and limit output tokens. I add exact and semantic caching so repeated or similar requests do not trigger another model call.**
>
> **For serving, I use efficient inference engines with continuous batching and efficient KV-cache management, and I evaluate quantization to reduce GPU memory requirements. I also use model routing and fallback strategies so simple requests go to smaller models and only complex requests use expensive models.**
>
> **Finally, I autoscale infrastructure based on traffic and inference metrics, track cost per request and per model, and continuously compare cost against answer quality. The goal is not simply to minimize cost—it is to minimize cost while maintaining an acceptable quality and latency SLA.**

## One-line answer

```text
Reduce tokens → cache results → batch requests → use efficient serving →
quantize → route to smaller models → autoscale → continuously measure
cost vs quality.
```
