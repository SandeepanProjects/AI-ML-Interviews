# How to Serve an LLM — Senior/Staff AI Engineer Guide

Serving an LLM means making a trained language model available to an application so that users or other services can send requests and receive generated responses reliably.

At a senior AI engineer level, you need to understand more than calling `model.generate()`. You should know how model weights are loaded, how inference works, how an API exposes the model, how concurrent requests are handled, how GPUs and memory are managed, and how to scale, secure, monitor, and deploy the service in production.

For example, in your financial compliance AI platform, a FastAPI application might receive a document-review request, retrieve relevant policies from PostgreSQL/pgvector, and send the prompt and evidence to an LLM inference service. The inference service generates a structured result, which the application validates before returning it or requesting human approval.

## 1. Understand the LLM serving architecture

Client / Frontend

Web application, mobile application, or another service

API Gateway / Load Balancer

Authentication, rate limits, routing, TLS

Application API — FastAPI

Validation, business rules, prompt construction, RAG, authorization

LLM Inference Server

vLLM, Hugging Face TGI, or a managed model endpoint

Model Weights + GPU / CPU

Tokenization, model execution, sampling, generated tokens

The important distinction is that FastAPI and the LLM inference server have different responsibilities.

- FastAPI handles your application's business logic, authentication, input validation, RAG, and workflow orchestration.
- The inference server loads and executes the model, schedules inference requests, manages GPU memory, and generates tokens.
- The API gateway or load balancer controls external access and distributes traffic.
- Observability infrastructure records latency, errors, throughput, resource use, and model-specific metrics.

You can combine some of these components for a small application, but separating them is often better for a production system because the API and GPU inference layer have different scaling requirements.

## 2. The main ways to serve an LLM

There are four common approaches.

[HP ZBook Ultra G1a 14” Mobile Workstation PC – AI Laptop  | HP® United Kingdom](https://images.openai.com/static-rsc-4/gwY_ZHelZgIhk_JPqZi5eIMfC40eKIkPFTqX3PxPvu5Gk59tfE-ccLkZOL7VuVJZZvSk0Pzo4SvdvT-ure5pXGufvVgI7a4XtCvvKbFHdiiwtobeJBZHXNR2JRI8RdGQOigFuNNKkyX5hXwnSKmRyOH5tLwePuWk6pb1lXIgDkA?purpose=inline)

Approach 1 — Run inference directly in Python

Load the model using Hugging Face Transformers and generate text inside your application process.

Best for learning and prototypes

Simple to implement, but you must handle concurrency, memory, and scaling yourself.

[Life of an inference request (vLLM V1): How LLMs are served efficiently at scale](https://images.openai.com/static-rsc-4/k7PDP3P9bQLsS1e60Xw6u1Y0aB_JZoqefrttyzmcw9YKukoc8DTFd8QeO7H1oDrcTfa02qA8MCD6fRmrH1Lv3ockAz-m1xV__SfHIO37WsyaIaoAk7RO8Ipx3JTFNOOPP42YYrJt1-pO5GUBBcLyhDTud9pdUcN4_69erQ8kRNw?purpose=inline)

Approach 2 — Self-host an inference server

Use vLLM or another inference engine to serve an open-weight model through an HTTP API.

Common production pattern

Better control over batching, GPU utilization, concurrency, and deployment.

[Networking capabilities optimize traffic for generative AI apps | Google Cloud Blog](https://images.openai.com/static-rsc-4/pDt1UwsFQaE8N6ZY-XU1LY6kPBwYIq9Mgt7e6srS9oOK8KGd3E2c9-eSY6WMlfkKkWjAtclpQj8l64whqCtl9il0qU9ybDbwXkCaBHbNryA3c3JpU-yY3tYijI3DSm6ukJtQ-mJJNw9uhjXyS_u5IKhS6XwBDOeYcJZ34UPwmXU?purpose=inline)

Approach 3 — Use a hosted model API

Call a provider's model endpoint instead of managing model weights and GPU servers.

Fastest to launch

Less infrastructure management, but you depend on provider availability, pricing, limits, and data-handling terms.

[Running Self-Hosted LLMs on Kubernetes (The Hard Way, But the Right Way) | by Braham Garg | DevOps | Golang | Python | AWS | K8s | Feb, 2026 | Medium](https://images.openai.com/static-rsc-4/au8QMRMHw3kgdDDrcfa_hbgDzq73ht1D_U2eiK7H8tECQNXYg1I6ibDkBX6HbxdkpNEAk52Hllm3bmTEjA4Pzp0KHv5wDlCQH7dAeZs24QmDACHMXJN0KTEWIGHCSDZElEbmipS_DoLmYqYQT2eXsuZmt41y-QBPOD7bYhq5Ywg?purpose=inline)

Approach 4 — Serve at scale with Kubernetes

Run inference replicas on GPU nodes, expose them through internal services, and manage rollout, scaling, health checks, and recovery.

Suitable when you need production-grade availability and predictable operations. Kubernetes does not itself make inference faster; the inference engine and deployment configuration still matter.

These approaches are not mutually exclusive. For example, you might use FastAPI plus vLLM on Kubernetes, or FastAPI plus a managed model API.

## 3. Approach 1: Serve a model directly using Hugging Face Transformers

Let's start with the simplest implementation so that the underlying process is clear.

### Install dependencies

```
pip install torch transformers accelerate fastapi uvicorn
```

You also need enough RAM or GPU memory for the selected model. Some models require additional dependencies, model-access permissions, or a particular hardware configuration.

### Load the model and generate text

Create `model_service.py`:

```
import torchfrom transformers import AutoTokenizer, AutoModelForCausalLMMODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"device = "cuda" if torch.cuda.is_available() else "cpu"tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)model = AutoModelForCausalLM.from_pretrained(    MODEL_ID,    torch_dtype=(        torch.float16 if device == "cuda" else torch.float32    ),)model.to(device)model.eval()def generate_answer(question: str) -> str:    messages = [        {            "role": "system",            "content": "You are a helpful assistant.",        },        {            "role": "user",            "content": question,        },    ]    prompt = tokenizer.apply_chat_template(        messages,        tokenize=False,        add_generation_prompt=True,
```

This is a small instruct model suitable for demonstrating the mechanics. It is not a recommendation to use that model for high-stakes production tasks.

### What happens internally?

1. Load the tokenizer. The tokenizer converts text into token IDs.
2. Load the model weights. The model parameters are loaded into CPU or GPU memory.
3. Format the chat prompt. The model receives a sequence that distinguishes system instructions from user input.
4. Tokenize the prompt. Text becomes tensors that the model can process.
5. Run inference. `model.generate()` repeatedly predicts the next token.
6. Decode the output. The generated token IDs are converted back into readable text.

`model.eval()` disables training-specific behavior such as dropout where applicable. `torch.inference_mode()` avoids autograd bookkeeping and reduces inference overhead.

`max_new_tokens` limits the generated continuation, which is important for controlling latency and resource consumption.

`do_sample=False` uses deterministic greedy decoding for this example. It does not guarantee the answer is factually correct.

### Important limitation

This code loads the model at module import time. That can be acceptable for a local experiment, but in a web service you should use application startup or lifespan management, load the model once per worker, and avoid loading a separate copy for every request.

Also, directly calling `model.generate()` in a synchronous API handler can block a request worker. Concurrent GPU requests may compete for memory and compute, and naive batching can be inefficient. For substantial traffic, an inference server such as vLLM is generally a better starting point.

## 4. Expose the model through a FastAPI endpoint

Now we will turn the local model into an HTTP service.

Create `api.py`:

```
from contextlib import asynccontextmanagerimport anyiofrom fastapi import FastAPI, HTTPExceptionfrom pydantic import BaseModel, Fieldfrom model_service import generate_answerclass GenerateRequest(BaseModel):    question: str = Field(min_length=1, max_length=4000)class GenerateResponse(BaseModel):    answer: str@asynccontextmanagerasync def lifespan(app: FastAPI):    # The model_service module loads the model once when imported.    # This is a simplified lifecycle for a local demonstration.    yieldapp = FastAPI(    title="LLM Inference API",    lifespan=lifespan,)@app.post("/generate", response_model=GenerateResponse)async def generate(request: GenerateRequest):    try:        # Avoid blocking the async event loop with synchronous generation.        answer = await anyio.to_thread.run_sync(            generate_answer,
```

Install the additional dependency and start the API:

```
pip install anyio
uvicorn api:app --host 127.0.0.1 --port 8000
```

Send a request:

```
curl -X POST http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"question":"Explain retrieval-augmented generation."}'
```

Expected response shape:

```
{
  "answer": "Retrieval-augmented generation combines..."
}
```

The exact answer will depend on the model.

### What does this architecture achieve?

- FastAPI validates incoming requests.
- Pydantic checks that `question` is present and within the specified length.
- The model generates a response.
- FastAPI serializes the result as JSON.

However, this is a teaching implementation, not a production-ready inference server.

`anyio.to_thread.run_sync()` prevents synchronous generation from directly blocking the event loop, but it does not provide efficient GPU scheduling or guarantee that concurrent requests will be safe. A semaphore, bounded queue, or dedicated inference service may be necessary. A production system should also implement authentication, rate limiting, request timeouts, structured logging, graceful shutdown, and safe error handling.

For a model with significant GPU requirements, I would normally separate FastAPI from the inference process.

## 5. Approach 2: Serve the model using vLLM

For production self-hosting, an inference engine such as vLLM is often a better starting point than writing your own generation loop.

vLLM provides an OpenAI-compatible API server and implements inference optimizations, including continuous batching and efficient KV-cache management.

### Step 1 — Start the inference server

On a compatible Linux machine with a supported NVIDIA GPU and the appropriate drivers, install vLLM using its current installation instructions.

Then launch a model:

```
vllm serve Qwen/Qwen2.5-0.5B-Instruct \
  --host 0.0.0.0 \
  --port 8001
```

This command is a basic example. Check the current vLLM installation and model-compatibility documentation for the correct environment and options. You may need to configure authentication to download a gated model, specify a supported precision, or adjust memory settings.

What happens when you run this command?

1. vLLM loads the model weights into available device memory.
2. It initializes its inference engine and memory-management structures.
3. It starts an HTTP server.
4. Applications send generation requests to the server.
5. vLLM schedules incoming requests and returns generated text.

The application does not need to load the model weights itself.

### Step 2 — Call vLLM through its OpenAI-compatible API

Install the client:

```
pip install openai
```

Create `client.py`:

```
from openai import OpenAIclient = OpenAI(    base_url="http://localhost:8001/v1",    api_key="local-development-key",)response = client.chat.completions.create(    model="Qwen/Qwen2.5-0.5B-Instruct",    messages=[        {            "role": "system",            "content": "You are a helpful assistant.",        },        {            "role": "user",            "content": "Explain LLM serving in simple terms.",        },    ],    max_tokens=200,    temperature=0.2,)print(response.choices[0].message.content)
```

The API key above is a placeholder for local development. It works only if the server configuration accepts it; a real deployment needs actual authentication and network restrictions.

Notice that the application uses an HTTP client rather than `AutoModelForCausalLM`. The inference server owns the model lifecycle and handles generation.

### Step 3 — Why vLLM is useful

The most important optimizations are:

- Continuous batching: New requests can join ongoing inference work as earlier requests finish, instead of relying only on fixed batches.
- KV-cache management: The model reuses attention key/value tensors from previously processed tokens during autoregressive generation.
- Memory-aware scheduling: The engine manages GPU memory and concurrent requests more efficiently.
- Streaming: Generated tokens can be sent to the client incrementally.
- Parallel inference options: Supported configurations can distribute model computation across multiple GPUs.

These features are especially useful when many users submit requests concurrently.

They do not eliminate capacity planning. If traffic exceeds available compute or KV-cache capacity, requests can queue, slow down, or fail.

## 6. What actually happens inside an LLM inference server?

This is an important interview topic because it distinguishes someone who can call an API from someone who understands model serving.

1\. Request arrives

The server validates the request, applies scheduling rules, and tokenizes the prompt.

2\. Prefill phase

The model processes the prompt tokens and computes the initial attention key/value states. Long prompts can make this phase expensive.

3\. Decode phase

The model repeatedly predicts the next token, appends it to the sequence, and continues until a stop condition is reached.

4\. Output processing

The server decodes tokens, streams or returns the response, and records relevant usage and timing metrics.

### Why is the KV cache important?

Transformers use attention to process tokens. During autoregressive generation, previously computed key and value tensors can be reused instead of recomputing them from scratch for every new token.

That reuse is the KV cache.

It speeds up token generation, but consumes GPU memory. The cache requirement depends on factors such as model architecture, number of layers, attention heads, precision, and total active sequence length.

A model may fit into GPU memory when loaded alone, yet run out of memory when many long conversations are active because the KV cache also needs space.

### Three inference metrics you should know

| Metric                       | Meaning                                                              |
| ---------------------------- | -------------------------------------------------------------------- |
| TTFT — Time to First Token   | Time from request submission until the first generated token arrives |
| TPOT — Time Per Output Token | Average time between subsequent output tokens                        |
| Throughput                   | Number of tokens or requests served per unit of time                 |

You should also monitor end-to-end latency, queue wait time, error rates, and GPU utilization.

For interactive chat, TTFT strongly influences perceived responsiveness. For long outputs, TPOT and total generation time become important. For large batch workloads, throughput and cost per token may be more important than minimizing each individual request's latency.

## 7. Approach 3: Use a hosted LLM API

If you do not want to operate GPUs, a managed API is often the simplest way to serve an LLM.

For example, using a hosted provider's API:

```
from openai import OpenAIclient = OpenAI()def generate_with_hosted_model(question: str) -> str:    response = client.responses.create(        model="gpt-4.1-mini",        input=question,    )    return response.output_textif __name__ == "__main__":    print(        generate_with_hosted_model(            "Explain how LLM inference works."        )    )
```

This example uses a specific model identifier as an illustration; verify availability, pricing, and API compatibility for your account and deployment.

Set the provider's API key through a secure environment or secrets manager rather than hardcoding it in source code.

### What changes with a hosted API?

Your application sends a request over HTTPS. The provider operates the inference infrastructure and returns the result.

You do not directly manage:

- Model weights.
- GPU scheduling.
- Inference server upgrades.
- Hardware capacity.

You still manage:

- Request validation and authentication for your application.
- Provider credentials and access permissions.
- Rate limits, timeouts, retries, and fallback behavior.
- Data-handling and contractual compliance.
- Application-level evaluation and observability.
- Cost controls and provider dependency.

For an enterprise financial application, verify data retention, processing location, access controls, and contractual requirements before sending sensitive content to any hosted model.

## 8. Streaming responses to users

Without streaming, the client may wait until the entire answer is generated. Streaming allows it to receive partial output as tokens become available.

With vLLM's OpenAI-compatible endpoint, you can use the streaming option:

```
from openai import OpenAIclient = OpenAI(    base_url="http://localhost:8001/v1",    api_key="local-development-key",)stream = client.chat.completions.create(    model="Qwen/Qwen2.5-0.5B-Instruct",    messages=[        {            "role": "user",            "content": "Explain continuous batching.",        }    ],    stream=True,    max_tokens=200,)for chunk in stream:    if not chunk.choices:        continue    token = chunk.choices[0].delta.content    if token:        print(token, end="", flush=True)
```

Streaming is valuable for chat interfaces, long explanations, and interactive applications. It reduces perceived waiting time, but does not necessarily reduce the total compute required to generate the answer.

For production APIs, handle stream interruptions, client disconnects, cancellation, timeouts, and usage accounting. If your application requires valid structured JSON, be careful about exposing partial output before the complete result has been validated.

## 9. How do you deploy an LLM in production?

For a self-hosted model, a common deployment architecture is:

Client / API Gateway

FastAPI application replicas

Auth, RAG, business rules, model routing

Internal inference service

vLLM or another serving engine

GPU worker pool

Model replicas, KV caches, batching, GPU execution

Observability and operations

Metrics, traces, logs, alerts, evaluation, capacity planning

### What I would implement

1\. Containerization

Package the inference server and application in reproducible container images. Keep model artifacts versioned and ensure GPU runtime dependencies match the serving environment.

2\. GPU-aware deployment

Place inference workloads on nodes with compatible GPUs. Set appropriate resource requests, limits, scheduling constraints, and memory capacity. A Kubernetes deployment needs GPU device-plugin and runtime support configured correctly.

3\. Concurrency and backpressure

Set bounded request queues and concurrency limits. When capacity is exhausted, apply admission control or return a controlled overload response rather than allowing unbounded queues and memory pressure.

4\. Health checks

Distinguish process liveness from model readiness. A server process may be alive while model weights are still loading or the inference engine is unable to accept requests.

5\. Autoscaling

Scale according to suitable signals, such as queue depth, waiting requests, request rate, latency, or GPU utilization. GPU utilization alone can be misleading: a GPU can be busy while the service is already failing its latency objectives.

6\. Deployment and rollback

Pin model versions, test new versions offline, deploy gradually, compare quality and operational metrics, and retain a rollback path.

7\. Observability

Capture TTFT, TPOT, request latency, input/output tokens, queue wait time, error rates, timeouts, GPU memory, cache pressure, and cost.

For LLM applications, tracing should also cover retrieval, reranking, tool calls, and validation. Otherwise, it can be difficult to determine whether latency comes from the model or another stage of the workflow.

## 10. How do you optimize LLM serving?

Once the basic service works, senior engineers focus on throughput, latency, memory, and cost.

| Technique                | How it helps                                                                 | Trade-off                                                  |
| ------------------------ | ---------------------------------------------------------------------------- | ---------------------------------------------------------- |
| Continuous batching      | Serves multiple active requests efficiently                                  | Scheduling and capacity management become more complex     |
| Quantization             | Reduces model-weight memory and may improve inference speed                  | May reduce quality or affect hardware compatibility        |
| Tensor parallelism       | Splits model computation across GPUs                                         | Adds communication overhead and deployment complexity      |
| Prefix caching           | Reuses computed prefix state for identical reusable prefixes where supported | Cache invalidation, memory use, and prefix matching matter |
| KV-cache optimization    | Supports more active sequences within available memory                       | Memory pressure still depends on context and concurrency   |
| Streaming                | Improves perceived responsiveness                                            | Requires stream lifecycle and cancellation handling        |
| Prompt/context reduction | Reduces prefill work and input-token cost                                    | Removing useful evidence can reduce answer quality         |
| Model routing            | Sends simple tasks to cheaper models                                         | Adds routing logic and evaluation complexity               |

### Quantization example

Suppose you want to serve an open-weight model but its full-precision weights do not fit comfortably into your GPU memory.

Quantization represents model weights using fewer bits, such as 8-bit or 4-bit representations. This can substantially reduce weight memory, though the actual speed improvement depends on the hardware and inference engine.

For supported vLLM configurations, you may be able to serve a compatible quantized checkpoint or specify an appropriate quantization mode. The exact options depend on the model and vLLM version.

Do not assume 4-bit quantization will always be faster or equally accurate. Benchmark the quantized model on your actual workload, including output quality, latency, throughput, and memory usage.

### How to estimate memory

A rough first estimate for model-weight memory is:

\\[ \text{Weight memory} \approx \text{Parameter count} \times \text{Bytes per parameter} \\]

For example, a hypothetical 7-billion-parameter model using 16-bit weights requires approximately:

\\[ 7 \times 10^9 \times 2 =14 \text{ GB} \\]

This is roughly 14 GB of decimal memory for the weights alone. It excludes KV cache, activations, runtime overhead, CUDA allocations, and other memory requirements. Real deployments need additional headroom.

That is why selecting a GPU based only on parameter count is insufficient.

## 11. Production concerns interviewers expect you to understand

### Security

- Authenticate clients and authorize requests.
- Keep inference endpoints private when they are intended for internal use.
- Store credentials in a secrets manager.
- Apply request size limits, rate limits, and tenant isolation.
- Prevent unauthorized model downloads and access to private model artifacts.
- Avoid logging sensitive prompts or model outputs unnecessarily.
- Treat retrieved documents and tool results as untrusted inputs.

### Reliability

- Configure bounded timeouts and retries.
- Retry only appropriate failures; do not blindly retry every error.
- Use circuit breakers or provider fallback strategies where justified.
- Apply backpressure when queues are full.
- Cancel work when possible after clients disconnect.
- Define graceful behavior when the model server becomes unavailable.

### Model correctness

- Version model weights, prompts, tokenizers, and inference settings.
- Run regression evaluations before model upgrades.
- Validate structured output before returning it.
- Keep deterministic business rules outside the model where possible.
- Require human approval for consequential decisions.

### Capacity planning

Benchmark realistic combinations of:

- Concurrent users.
- Input and output token lengths.
- Request arrival patterns.
- Batch sizes and queue limits.
- Model precision and GPU configuration.

A service that performs well with ten short requests may perform very differently with hundreds of long-context requests.

## 12. Which serving approach would I choose for your enterprise AI project?

For an AI-powered financial compliance platform, I would consider two initial architectures.

### Option A — Managed LLM API

Simpler operations

- FastAPI for application logic.
- PostgreSQL/pgvector for policy retrieval.
- A managed LLM endpoint for generation.
- Redis for appropriate caching and coordination.
- Guardrails, structured validation, audit logging, and human approval.
- OpenTelemetry for traces and metrics.

Best starting point when managed inference meets your data-handling requirements and you want to minimize GPU operations.

### Option B — Self-hosted inference

- FastAPI for the application API and orchestration.
- vLLM for open-weight model serving.
- GPU-backed infrastructure, potentially Kubernetes.
- PostgreSQL/pgvector for retrieval.
- Redis and Celery where background work and queueing are required.
- Evaluation, monitoring, security controls, and human approval.

Best when private inference, model control, customization, or measured workload economics justify the operational investment.

I would not introduce Kubernetes and a GPU cluster merely because the application is described as enterprise-grade. I would begin with the simplest architecture that meets the security, quality, and load requirements, then add serving infrastructure when measured demand justifies it.

## 13. Interview-ready answer: “How do you serve an LLM in production?”

I think of LLM serving as two separate concerns: the application layer and the inference layer.

The application layer typically uses FastAPI or another API framework to handle authentication, input validation, business logic, RAG, model routing, and output validation. The inference layer loads the model and executes generation. For a hosted model, I call the provider's API; for a self-hosted open-weight model, I would evaluate an inference engine such as vLLM.

For a prototype, Hugging Face Transformers with `model.generate()` is sufficient. For concurrent production workloads, I generally prefer a dedicated inference server because it can provide continuous batching, efficient KV-cache management, streaming, and better control over GPU utilization.

For deployment, I containerize the application, place inference workloads on compatible GPU infrastructure, keep internal model endpoints protected, and implement health checks, bounded queues, timeouts, rate limits, monitoring, and graceful failure handling. At larger scale, I can use Kubernetes and GPU-aware autoscaling, but I choose the scaling approach based on measured demand.

I monitor time to first token, time per output token, end-to-end latency, throughput, queue depth, GPU memory, errors, and cost. I also version models and prompts and run regression evaluations before deploying changes.

Finally, serving quality is not just infrastructure performance. I validate generated outputs, measure task success and safety, and ensure that critical business decisions remain subject to deterministic controls and human review where required.

## 14. Follow-up questions you should prepare for

Why not serve a model directly inside FastAPI?

What is continuous batching?

What is the difference between TTFT and TPOT?

How do you serve a model larger than one GPU can hold?

How do you scale LLM serving?

The key takeaway: An LLM is not production-ready just because it can generate a response. Production serving requires an inference engine, an API and security boundary, capacity management, observability, failure handling, and continuous evaluation of output quality.