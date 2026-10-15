# How to Reduce LLM Inference Latency — Senior AI Engineer Guide

In production AI systems, reducing inference latency is not simply about choosing a smaller model. A senior AI engineer identifies where the latency occurs, measures the bottleneck, and optimizes the appropriate layer without sacrificing unacceptable amounts of quality, reliability, or security.

For example, in your enterprise financial compliance platform, a user submits a marketing document and expects the system to identify risky claims, retrieve relevant policies, and return a structured compliance report. The total response time may involve several stages:

- API gateway and request queuing
- Prompt construction and input tokenization
- Model inference
- RAG retrieval and reranking
- Tool calls or multiple agent steps
- Output generation and validation

Optimizing GPU generation alone will not solve the problem if most of the delay comes from serial agent calls or a slow vector database.

## 1. Understand the different latency metrics

Time to First Token (TTFT)

Time from request submission until the first generated token arrives. Important for chat responsiveness and streaming.

Time Per Output Token (TPOT)

Average time required to generate each subsequent token. Important for how quickly the response appears once generation starts.

End-to-End Latency

Total time from receiving the request to returning the complete response, including retrieval, inference, tools, and validation.

Queueing Latency and Throughput

Queueing measures time spent waiting for capacity. Throughput measures how much work the system completes per unit of time, such as output tokens per second.

These metrics are related but not interchangeable. A system can have high throughput while an individual request experiences poor latency under load.

For interactive applications, measure p50, p95, and p99 latency. Average latency alone can hide the slowest user experiences.

## 2. First identify the bottleneck

Before changing the model, instrument the request pipeline.

For a RAG application, you might measure:

| Stage                   | Example measurement |
| ----------------------- | ------------------- |
| API and queue           | 30 ms               |
| Embedding and retrieval | 120 ms              |
| Reranking               | 100 ms              |
| LLM first-token wait    | 400 ms              |
| LLM generation          | 1,000 ms            |
| Validation and response | 50 ms               |
| Total                   | 1,700 ms            |

These are illustrative numbers, not benchmark results. Here, retrieval, reranking, and generation are the major optimization targets.

### Python: measure each stage

```
from time import perf_counterfrom collections.abc import Callablefrom typing import TypeVarT = TypeVar("T")def measure_stage(    name: str,    operation: Callable[[], T],) -> T:    start = perf_counter()    try:        return operation()    finally:        elapsed_ms = (perf_counter() - start) * 1000        print(f"{name}: {elapsed_ms:.2f} ms")def answer_question(question: str) -> str:    context = measure_stage(        "retrieval",        lambda: retrieve_documents(question),    )    prompt = measure_stage(        "prompt_building",        lambda: build_prompt(question, context),    )    answer = measure_stage(        "llm_generation",        lambda: generate_answer(prompt),    )    return measure_stage(        "validation",        lambda: validate_answer(answer),    )
```

The functions `retrieve_documents`, `build_prompt`, `generate_answer`, and `validate_answer` are placeholders for your application's implementations.

In production, export measurements to OpenTelemetry or Prometheus instead of relying on `print`. Record model name, input/output token counts, queue wait, retry count, and stage durations. Avoid logging confidential prompts or financial documents.

## 3. The most effective ways to reduce inference latency

[Nvidia CEO sees $1T AI chip demand by 2027 at GTC](https://images.openai.com/static-rsc-4/iINqqRSSrTeJbw7DAA6IpTFmFXVLw7MFS_pIo9o2Hf8qR5JnyfRXK2lkrFUoSDoDub-DVLSMT4iXNurNhMN6HogEMIvqZdzIA4KPJ2wXUaY8gf5GpnsmyUb7GPAhwr_nuTtbcz4hYCPtAZSuCjrMyZl-9eumUTObFbPrGiWP8s8?purpose=inline)

1\. Use a faster or smaller model

Usually high impact

Route simple classification and extraction tasks to a smaller model, and reserve more capable models for difficult reasoning. Benchmark task quality before switching.

[How LLMs Generate Responses](https://images.openai.com/static-rsc-4/9KAu__rbT5XYT-TLt4nV3vlcE7i1YAzb4vixeNqc-ff9LL_Mt6aPOlJBpfpj3nG_04eGNBD0O5e6vVLT8Yik9yKgKg_dh6hl99XJEI8ewU4bAEtLCisf2cZN-QeBXOl_0Vh-G-NvIOBb38LXwI82xngckwo6k5u21WTZZzzyzBk?purpose=inline)

2\. Reduce input and output tokens

Often high impact

Remove irrelevant context, shorten repeated instructions, retrieve fewer but better documents, and set a sensible output-token limit.

[GPU-сервер ASUS ESC N8-E11 (HGX H100 8GPU) H100 купить в интернет-магазине Kvantech | Цена, описание, характеристики](https://images.openai.com/static-rsc-4/Eor2mXpoAf1hnP0m7f8SFBIIywVzWdiHP1ymobXmcIB_Lb_mXpwa4UO8tSZQhK0vO14Tvp7AhkXO41FT-8zUp7u4tJZobbWJFiw1kPqyl-uYkjfsntARfJdSU8T-8OBhZ87bM-aYqZlkZVzK3q_uN3dQeULTYn3PAUnBTU4OuAM?purpose=inline)

3\. Optimize inference serving

High impact at scale

Use a purpose-built inference server such as vLLM, continuous batching, efficient KV caching, and appropriate GPU capacity.

[Mango: A new way to do Bayesian optimization in Python | by Dr. Carmen Adriana Martínez Barbosa | TDS Archive | Medium](https://images.openai.com/static-rsc-4/ugqnnG-2kOhQdr5IACJVhPkG_4YBL3TpqftOOSnz4C6YgY8S-ZgKdiT6Xe_VDaaw5uXSGZKhz10_jGp5coc_0Urab74oUiXPn31gcyk2RCnQl3yG-Xa4gyUEu6Iddm8F4qk0ADa5azXv7SebHUqWvd_ReFsiN1ZvABgBjS2F4AA?purpose=inline)

4\. Parallelize independent work

High impact in agentic systems

Run independent retrievals, specialist agents, or document checks concurrently instead of waiting for each to finish serially.

[Caching in Distributed Systems - Part I](https://images.openai.com/static-rsc-4/Jyz5HMWZ-zM66cy0PHE_XuEM5YUaNOGcFRudkbzXJD9Nv2MiUBLijEZfgp80IRxOokXIM9029CIpKADay0bWBYFL_ZqJwjql4hEgCl7vVnUGQB9M5qKlKF2UxoGqydWuxFdg4mUDobsRNhOBL-1OP62jzeyebL0mYFYvYczAe3k?purpose=inline)

5\. Cache reusable work

Cache exact or safely reusable results, embeddings, retrieval results, and supported model prefixes. Apply authorization and freshness rules to every cache.

Other techniques include streaming, quantization, speculative decoding, prompt caching, optimized attention kernels, and keeping model instances warm. The best combination depends on the workload.

## 4. Optimize the model and token counts

### A. Choose the right model for the task

Suppose your compliance platform performs three tasks:

- Classify whether a statement contains a potentially risky claim.
- Extract a fee percentage or performance claim into JSON.
- Explain a complex policy conflict with supporting citations.

You might use a smaller model for the first two and a stronger model for the third.

```
def select_model(task: str) -> str:    routing_policy = {        "classification": "small-fast-model",        "structured_extraction": "small-fast-model",        "complex_policy_reasoning": "strong-reasoning-model",    }    try:        return routing_policy[task]    except KeyError:        raise ValueError(f"Unsupported task: {task}")
```

These are illustrative model identifiers, not actual deployable model names. In a real router, the mapping would point to configured providers or deployment IDs.

Senior-level consideration: route based on task complexity, risk, measured quality, and latency objectives—not simply on price. A fast model that misses a material compliance violation is not a successful optimization.

### B. Reduce input tokens

The model must process the input prompt before generating an answer. Long inputs can increase prefill latency, consume GPU memory, and increase the attention computation required.

For example, avoid placing 30 entire policy documents into every prompt when only three are relevant.

```
def build_compliance_prompt(    claim: str,    relevant_policies: list[dict],) -> str:    policy_context = "\n".join(        (            f"Policy ID: {p['policy_id']}\n"            f"Rule: {p['rule']}\n"            f"Version: {p['version']}"        )        for p in relevant_policies    )    return f"""Review the claim against the supplied policies.Identify supported violations and cite policy IDs.Do not invent policy requirements.CLAIM:{claim}RELEVANT POLICIES:{policy_context}"""
```

In production, enforce a token budget using the tokenizer for the selected model. The example assumes retrieval already selected relevant policy chunks.

### C. Limit unnecessary output generation

Generation is often sequential: the model produces a token, then uses the updated state to produce the next token. Long answers can therefore dominate latency.

```
response = client.chat.completions.create(    model="your-deployment",    messages=[        {            "role": "user",            "content": "Classify this claim and return the required fields."        }    ],    max_tokens=150,)
```

This uses an OpenAI-compatible chat-completions interface; supported parameters depend on the actual provider and model.

Use structured output when you need a predictable schema, and request only required fields. A token limit is a ceiling, not a guarantee of a complete or valid response.

Trade-off: reducing output tokens can improve speed but may truncate explanations or omit required evidence. Validate completeness.

## 5. Use an optimized inference server

A model running directly inside a Python API process is convenient for experimentation, but often not the best design for high-concurrency production workloads.

A common architecture is:

Client / API gateway

FastAPI application

Authentication · Validation · RAG · Routing

Inference server (e.g. vLLM)

Continuous batching · KV cache · GPU scheduling

GPU / model weights

The API layer handles business logic; the inference server specializes in efficiently serving model requests.

### Run a local vLLM server

For a compatible model and environment, a basic command looks like:

```
vllm serve Qwen/Qwen2.5-0.5B-Instruct \
    --host 0.0.0.0 \
    --port 8001
```

This small example model is for demonstrating the interface; choose a model that meets your actual quality and hardware requirements. Production deployment also needs access controls, capacity limits, network isolation, and pinned compatible versions.

### Send requests through the OpenAI-compatible endpoint

```
from openai import OpenAIclient = OpenAI(    base_url="http://localhost:8001/v1",    api_key="local-development-key",)response = client.chat.completions.create(    model="Qwen/Qwen2.5-0.5B-Instruct",    messages=[        {            "role": "user",            "content": "Classify the compliance risk in this statement."        }    ],    max_tokens=120,)print(response.choices[0].message.content)
```

This works when the local server exposes the compatible endpoint and recognizes that model ID. It is a starting point, not a complete production deployment.

### Why vLLM can help

- Continuous batching: dynamically schedules requests together rather than waiting for fixed batches to finish.
- Efficient KV-cache management: reuses stored attention keys and values while generating tokens, reducing redundant computation.
- Optimized kernels: use hardware efficiently for supported models and configurations.
- Concurrency: serve many requests through a shared inference engine instead of loading a separate model into every API worker.

Continuous batching is especially valuable under concurrent traffic. It may not make every isolated request faster; it often improves overall GPU utilization and performance under load.

## 6. Quantization: reduce memory pressure and sometimes improve speed

Quantization represents model weights using lower-precision numbers.

For example:

| Weight format      | Approximate memory for 7B parameters |
| ------------------ | ------------------------------------ |
| FP32 — 32-bit      | 28 GB                                |
| FP16/BF16 — 16-bit | 14 GB                                |
| 8-bit              | 7 GB                                 |
| 4-bit              | 3.5 GB                               |

These are theoretical weight-only estimates using decimal GB; actual GPU memory use is higher because of the KV cache, activations, temporary buffers, and runtime overhead.

Quantization can allow a model to fit on a smaller GPU, increase serving capacity, or reduce memory traffic. Whether it reduces latency depends on the GPU, kernels, model, batch size, and quantization format.

### Example: load a quantized model

With a supported model and compatible versions of Transformers, PyTorch, and bitsandbytes:

```
import torchfrom transformers import (    AutoTokenizer,    AutoModelForCausalLM,    BitsAndBytesConfig,)model_id = "your-compatible-model"quantization_config = BitsAndBytesConfig(    load_in_4bit=True,    bnb_4bit_quant_type="nf4",    bnb_4bit_compute_dtype=torch.float16,)tokenizer = AutoTokenizer.from_pretrained(model_id)model = AutoModelForCausalLM.from_pretrained(    model_id,    quantization_config=quantization_config,    device_map="auto",)model.eval()
```

This illustrates 4-bit loading; hardware and model support vary. It does not automatically guarantee lower end-to-end latency.

Production practice: benchmark the quantized model against the original on representative prompts, measuring task quality, p95 latency, GPU memory, throughput, and output consistency.

## 7. Parallelize independent calls in agentic workflows

This is particularly important for your LangGraph-based financial compliance and financial research projects.

Imagine you have three independent checks:

1. Performance-claim review.
2. Fee-disclosure review.
3. Risk-disclosure review.

A sequential workflow waits for each specialist before starting the next. A parallel workflow launches all three, then aggregates their results.

Sequential execution

Performance

Fees

Risk

Total ≈ sum of specialist durations, plus orchestration overhead.

Parallel execution

Performance

Fees

Risk

Deterministic aggregation

Total ≈ slowest specialist duration, plus shared overhead, assuming sufficient capacity.

### Python implementation using asyncio

```
import asynciofrom typing import Anyasync def review_performance(document: str) -> dict[str, Any]:    return await performance_service.review(document)async def review_fees(document: str) -> dict[str, Any]:    return await fee_service.review(document)async def review_risk(document: str) -> dict[str, Any]:    return await risk_service.review(document)async def review_document(document: str) -> dict[str, Any]:    performance, fees, risk = await asyncio.gather(        review_performance(document),        review_fees(document),        review_risk(document),    )    return {        "performance_findings": performance,        "fee_findings": fees,        "risk_findings": risk,    }
```

The three `*_service` objects are placeholders for your actual implementations.

`asyncio.gather` lets independent asynchronous operations overlap while waiting for I/O. It does not make synchronous model generation faster by itself, and running three LLM calls concurrently may increase GPU queueing or rate-limit pressure.

In production, add per-provider concurrency limits, timeouts, explicit failure policies, and tracing. If one specialist fails, define whether the system retries, returns a partial result, or blocks the compliance review.

Staff-level insight: reducing the number of serial LLM round trips is often as important as optimizing the speed of each individual inference.

## 8. Optimize RAG retrieval and reranking

For RAG applications, inference latency is only one component of response time.

Consider this pipeline:

`Question → Embedding → Vector search → Reranking → Prompt assembly → LLM`

Ways to reduce RAG latency:

- Apply metadata filters such as tenant, document type, and policy version before or during retrieval.
- Tune `top_k`; retrieve enough evidence to maintain recall without flooding the prompt.
- Use approximate nearest-neighbor search where appropriate.
- Keep database connections and embedding models warm.
- Cache query embeddings or retrieval results when safe.
- Avoid reranking dozens of irrelevant chunks.
- Run independent retrievals in parallel when their results do not depend on one another.
- Use hybrid search only when its quality improvement justifies the extra work, and run independent keyword/vector searches concurrently if supported.

### Example: run independent retrievals concurrently

```
import asyncioasync def retrieve_context(query: str) -> list[dict]:    vector_task = vector_search(query, top_k=5)    keyword_task = keyword_search(query, top_k=5)    vector_results, keyword_results = await asyncio.gather(        vector_task,        keyword_task,    )    return reciprocal_rank_fusion(        vector_results,        keyword_results,    )
```

Here, `vector_search`, `keyword_search`, and `reciprocal_rank_fusion` represent application-specific functions. Both searches must be asynchronous for this example to overlap effectively.

Do not simply reduce `top_k` to one or two to make the system faster. In a compliance application, missing the policy clause that changes the review outcome is much worse than saving a few milliseconds.

## 9. Cache expensive, reusable operations

Caching can avoid repeated work altogether, but it must be designed carefully.

| Cache                | What it avoids                               | Important consideration                                          |
| -------------------- | -------------------------------------------- | ---------------------------------------------------------------- |
| Embedding cache      | Recomputing embeddings                       | Key by normalized text and embedding-model version               |
| Retrieval cache      | Repeating searches                           | Include tenant, filters, policy version, and authorization scope |
| Exact response cache | Repeating an identical inference             | Only reuse when inputs, model, prompt, and permissions match     |
| Prefix/KV cache      | Recomputing supported shared prompt prefixes | Depends on inference engine and compatible prefix settings       |

### Example: cache embeddings with Redis

```
import hashlibimport jsonasync def get_embedding(text: str) -> list[float]:    normalized = text.strip()    digest = hashlib.sha256(        normalized.encode("utf-8")    ).hexdigest()    key = f"embedding:v1:{digest}"    cached = await redis_client.get(key)    if cached is not None:        return json.loads(cached)    vector = await embedding_client.embed(normalized)    await redis_client.set(        key,        json.dumps(vector),        ex=3600,    )    return vector
```

This assumes `redis_client` and `embedding_client` are already configured asynchronous clients. Production code should also handle cache failures, concurrent cache misses, and embedding-model version changes.

For confidential financial content, do not use globally shared cache keys that can leak information between tenants. Include appropriate tenant and authorization boundaries for any cached result that depends on them.

## 10. Streaming: improve perceived latency

Streaming does not necessarily reduce the time required to generate the entire answer. It allows the user to see the beginning of the answer earlier.

For example, instead of waiting two seconds for a complete compliance explanation, the UI may begin displaying text as soon as the first tokens arrive.

```
from openai import AsyncOpenAIclient = AsyncOpenAI(    base_url="http://localhost:8001/v1",    api_key="local-development-key",)async def stream_answer(question: str) -> None:    stream = await client.chat.completions.create(        model="Qwen/Qwen2.5-0.5B-Instruct",        messages=[            {"role": "user", "content": question}        ],        max_tokens=200,        stream=True,    )    async for chunk in stream:        delta = chunk.choices[0].delta.content        if delta:            print(delta, end="", flush=True)
```

This is a simplified example using an OpenAI-compatible local server. In a web application, forward chunks using Server-Sent Events (SSE) or another appropriate streaming transport.

For structured compliance JSON, streaming requires extra care: do not treat a partially generated JSON object as a validated final result. Buffer it, parse it when complete, validate the schema, and only then mark the review as finalized. You can stream progress messages separately.

## 11. Advanced inference optimizations

These techniques are useful when simpler changes are insufficient.

Continuous batching

Dynamically combines requests at the inference-server level. Particularly useful for high concurrent traffic and improved GPU utilization. Tune capacity carefully so queues do not become excessive.

KV caching and prefix caching

KV caching avoids recomputing past attention states during autoregressive generation. Prefix caching can reuse computation for shared prompt prefixes when supported. These are different from caching the final answer.

Speculative decoding

A smaller draft model proposes tokens, and a larger target model verifies them. It can reduce generation latency when the draft tokens are accepted often enough and the hardware/runtime supports the technique efficiently.

Hardware and kernel optimization

Use supported attention kernels, suitable precision, GPU-aware scheduling, and tensor parallelism when model size or throughput requires it. Multi-GPU execution can add communication overhead, so more GPUs do not automatically mean lower latency.

Warm instances and capacity planning

Keep model workers initialized, avoid loading weights per request, manage autoscaling and cold starts, and provision enough capacity for peak traffic. Extra replicas help only if the inference workload can use them effectively.

## 12. Make FastAPI concurrency work correctly

A common mistake is assuming that declaring an endpoint `async` makes model inference asynchronous or faster.

This endpoint can block the event loop if `generate_answer()` is a long-running synchronous function:

```
@app.post("/generate")async def generate(request: RequestModel):    # Potentially blocks the event loop!    answer = generate_answer(request.prompt)    return {"answer": answer}
```

For a synchronous operation in a demonstration app, you can offload it to a worker thread:

```
from fastapi import FastAPIfrom pydantic import BaseModelfrom anyio import to_threadapp = FastAPI()class RequestModel(BaseModel):    prompt: str@app.post("/generate")async def generate(request: RequestModel):    answer = await to_thread.run_sync(        generate_answer,        request.prompt,    )    return {"answer": answer}
```

This helps prevent a synchronous call from blocking the event loop, but it does not increase GPU inference speed. It also does not guarantee that concurrent access to a shared model is safe or efficiently scheduled.

For serious workloads, prefer a dedicated inference server with explicit concurrency and queue management. Do not solve GPU saturation by creating unlimited worker threads.

## 13. How to decide which optimization to apply

Use this decision guide.

| Observed problem                              | First actions to investigate                                                             |
| --------------------------------------------- | ---------------------------------------------------------------------------------------- |
| High TTFT, even at low traffic                | Shorten input, inspect prefill time, check model initialization and network overhead     |
| High time per output token                    | Check model size, output length, GPU utilization, precision, and serving kernels         |
| Good latency at low traffic, poor p95 at peak | Investigate queueing, concurrency limits, batching, GPU saturation, and scaling          |
| RAG requests are slow before generation       | Optimize vector/keyword retrieval, reranking, database connections, and parallelism      |
| Multi-agent workflow is slow                  | Remove unnecessary agent steps and parallelize independent calls                         |
| Repeated requests are expensive               | Evaluate safe embedding, retrieval, response, or prefix caching                          |
| Slow only after deployment or scale-to-zero   | Investigate cold starts, model loading, and warm capacity                                |
| Model is fast but responses are still slow    | Trace the full request path, including gateway, database, tools, validation, and network |

### Example of latency reduction through serial-call elimination

Suppose three independent specialist agents each take 800 ms, excluding shared overhead.

- Sequential execution: \\(800 + 800 + 800 = 2400\\) ms.
- Parallel execution: approximately \\(800\\) ms plus scheduling and aggregation overhead, assuming sufficient capacity.

The theoretical specialist-stage improvement is about \\(3\times\\). Real production performance will vary due to queueing, rate limits, shared GPU capacity, and differences in task duration.

This example illustrates an important point: architectural optimization can outperform low-level model optimization when the application contains unnecessary serial work.

## 14. Production benchmarking: prove that the optimization works

Never declare a latency improvement based on one local request. Benchmark under representative traffic and compare the same workload before and after the change.

Track at least these metrics:

- p50, p95, p99 end-to-end latency.
- TTFT and TPOT.
- Input and output token counts.
- Queue wait and request concurrency.
- GPU utilization and memory.
- Requests per second and output tokens per second.
- Timeout and error rates.
- Quality metrics and cost per successful task.

For an enterprise financial application, also measure citation correctness, policy coverage, false negatives, and human-review escalation rates. A faster model that misses important compliance violations is not an acceptable production improvement.

### A simple latency benchmark

```
import asyncioimport timeimport statisticsasync def benchmark(    operation,    requests: int = 50,    concurrency: int = 5,) -> dict:    semaphore = asyncio.Semaphore(concurrency)    latencies = []    async def run_one():        async with semaphore:            start = time.perf_counter()            await operation()            latencies.append(                (time.perf_counter() - start) * 1000            )    start_all = time.perf_counter()    await asyncio.gather(        *(run_one() for _ in range(requests))    )    total_seconds = time.perf_counter() - start_all    ordered = sorted(latencies)    def percentile(p: float) -> float:        index = round((len(ordered) - 1) * p)        return ordered[index]    return {        "requests": requests,        "concurrency": concurrency,        "throughput_rps": requests / total_seconds,        "p50_ms": percentile(0.50),        "p95_ms": percentile(0.95),        "p99_ms": percentile(0.99),        "mean_ms": statistics.mean(latencies),    }
```

Example use:

```
async def sample_request():    await call_your_model_endpoint("Example question")results = asyncio.run(    benchmark(sample_request, requests=100, concurrency=10))print(results)
```

This is a basic demonstration. It assumes the `operation` is an async function and that failures should fail the benchmark. For a production load test, collect failures separately, warm the model first, test multiple concurrency levels, use representative input/output lengths, and calculate percentiles from enough samples. Include TTFT measurements separately for streaming endpoints.

## 15. My recommended optimization order as a Staff AI Engineer

I would work in this order:

1. Establish a baseline. Instrument the full request path and identify whether the bottleneck is inference, retrieval, queueing, or orchestration.
2. Set an explicit latency SLO. For example, define a p95 target for a specific workload and concurrency level, along with minimum acceptable quality.
3. Remove unnecessary work. Reduce redundant model calls, shorten prompts, eliminate serial dependencies, and avoid excessive reranking.
4. Improve model selection and serving. Benchmark candidate models, inference engines, precision formats, and GPU configurations.
5. Apply caching and concurrency controls. Cache safe reusable computations and prevent overload from producing runaway queues.
6. Load-test and validate quality. Compare latency percentiles, throughput, cost, errors, and task-specific quality against the baseline.
7. Roll out safely. Use canary deployments, dashboards, alerts, and rollback criteria so that an optimization does not silently degrade output quality.

## 16. Interview-ready answer

Here is a strong answer you can give in a Senior or Staff AI Engineer interview.

I reduce LLM inference latency using a measurement-driven, end-to-end optimization strategy rather than focusing only on model generation speed.

First, I instrument the request path and measure p50, p95, and p99 latency, TTFT, TPOT, queueing time, input/output tokens, and GPU utilization. This helps distinguish model inference bottlenecks from retrieval, network, and agent-orchestration bottlenecks.

Second, I optimize the workload. I select the smallest model that meets the task's quality requirements, reduce unnecessary prompt and output tokens, use structured outputs, and route complex requests to more capable models only when needed.

Third, I optimize model serving using an inference engine such as vLLM, continuous batching, efficient KV-cache management, appropriate quantization, and GPU-aware capacity planning. I use streaming to improve perceived responsiveness and caching where reuse is safe.

Fourth, at the application level, I optimize RAG retrieval and reranking, parallelize independent tool calls and specialist agents, eliminate redundant LLM round trips, and manage concurrency, timeouts, and backpressure.

Finally, I validate every change through load testing and offline evaluation. I compare latency, throughput, cost, and task quality. For a financial compliance system, I would specifically ensure that reducing latency does not increase missed violations or reduce citation correctness.

My goal is not simply to make inference faster. It is to meet the latency SLO at the required scale while preserving quality, security, reliability, and cost efficiency.

One final distinction to remember: inference optimization has two broad dimensions. Per-request latency is how quickly an individual request completes; system throughput is how much work the service handles overall. Techniques such as smaller models and shorter outputs may improve both, while batching and additional concurrency can increase throughput but sometimes increase an individual request's waiting time. A senior engineer optimizes the metric that actually matters to the product.