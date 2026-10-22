Your code is a good starting point for benchmarking, but I would modify it before using it in a production AI project.

For your FastAPI, RAG, and multi-agent AI systems, the main issue is not the basic approach. `asyncio`, semaphores, `time.perf_counter()`, and latency percentiles are all useful. The gaps are around reliability, measurement accuracy, error handling, and realistic load testing.

## 1. What needs to change?

| Area             | Original code                         | Production improvement                                        |
| ---------------- | ------------------------------------- | ------------------------------------------------------------- |
| Concurrency      | Semaphore                             | Keep it, but validate the configuration                       |
| Request failures | One exception can fail the benchmark  | Track successes, failures, and error types                    |
| Timeouts         | No timeout                            | Add a per-operation timeout                                   |
| Latency          | Measures operation time only          | Measure both queue wait and operation latency                 |
| Percentiles      | Custom index calculation              | Use a documented percentile method                            |
| Empty results    | Mean/percentile calculations can fail | Handle zero requests and zero successful samples              |
| Throughput       | Counts all requested operations       | Report successful throughput separately                       |
| Observability    | Prints basic metrics                  | Return structured metrics suitable for logging and monitoring |
| Load testing     | One fixed concurrency level           | Test multiple levels and sustained traffic                    |

One particularly important detail: your original throughput formula counts every requested operation, even if some fail. For production analysis, you should distinguish offered request rate, completed request rate, successful request rate, and failure rate.

## 2. Production-improved implementation

This version is more suitable for internal AI performance testing. It adds timeouts, failure tracking, queue-wait measurements, safer percentile calculations, and separate successful-request metrics.

```
# app/benchmarking/benchmark.pyimport asyncioimport mathimport statisticsimport timefrom collections import Counterfrom collections.abc import Awaitable, Callablefrom typing import AnyAsyncOperation = Callable[[], Awaitable[Any]]def percentile(values: list[float], p: float) -> float | None:    """Calculate a nearest-rank percentile in milliseconds."""    if not values:        return None    if not 0 <= p <= 1:        raise ValueError("p must be between 0 and 1")    ordered = sorted(values)    index = max(0, math.ceil(p * len(ordered)) - 1)    return round(ordered[index], 2)async def benchmark(    operation: AsyncOperation,    *,    requests: int = 100,    concurrency: int = 10,    timeout_seconds: float = 30.0,) -> dict[str, Any]:    """Benchmark an async operation under bounded concurrency."""    if requests < 1:        raise ValueError("requests must be at least 1")    if concurrency < 1:        raise ValueError("concurrency must be at least 1")    if timeout_seconds <= 0:        raise ValueError("timeout_seconds must be positive")    semaphore = asyncio.Semaphore(concurrency)    operation_latencies: list[float] = []    queue_waits: list[float] = []
```

There is one further refinement I would make before calling this final production-quality code: retain separate latency samples for successful operations and failed operations. The code above explicitly measures all operation outcomes, so a timeout is included in the latency distribution, which is useful for understanding failure behavior but can distort successful-request latency. Also, the unused `successful_latencies` placeholder should be removed.

### Corrected final version

Use this complete version instead of the preceding draft. It records successful and failed latency distributions separately.

```
# app/benchmarking/benchmark.pyimport asyncioimport mathimport statisticsimport timefrom collections import Counterfrom collections.abc import Awaitable, Callablefrom typing import AnyAsyncOperation = Callable[[], Awaitable[Any]]def percentile(    values: list[float],    p: float,) -> float | None:    """Return the nearest-rank percentile."""    if not values:        return None    if not 0 <= p <= 1:        raise ValueError("p must be between 0 and 1")    ordered = sorted(values)    index = max(0, math.ceil(p * len(ordered)) - 1)    return round(ordered[index], 2)def summarize(values: list[float]) -> dict[str, float | int | None]:    """Summarize latency values in milliseconds."""    return {        "count": len(values),        "p50_ms": percentile(values, 0.50),        "p95_ms": percentile(values, 0.95),        "p99_ms": percentile(values, 0.99),        "mean_ms": (            round(statistics.mean(values), 2) if values else None        ),    }async def benchmark(    operation: AsyncOperation,    *,    requests: int = 100,
```

## 3. How to run it

Create a separate runner script:

```
# scripts/run_benchmark.pyimport asyncioimport jsonfrom app.benchmarking.benchmark import benchmarkasync def sample_request() -> None:    # Replace this with your actual asynchronous    # model client or API request.    await call_your_model_endpoint("Example question")async def main() -> None:    results = await benchmark(        sample_request,        requests=100,        concurrency=10,        timeout_seconds=30,    )    print(json.dumps(results, indent=2))if __name__ == "__main__":    asyncio.run(main())
```

`call_your_model_endpoint()` is still a placeholder. Replace it with your actual async HTTP client or model SDK call.

## 4. What still separates this from a production load-testing system?

Even with these improvements, this is an internal benchmarking utility, not a complete production load-testing framework.

Before relying on it for capacity planning:

- Add warm-up requests and repeat each benchmark several times.
- Capture successful response latency, failures, timeouts and cancellation separately.
- Use realistic request payloads and response sizes.
- Reuse an asynchronous HTTP client instead of creating a new client per request.
- Test increasing concurrency and sustained traffic, not only a single burst.
- Correlate results with OpenTelemetry traces, provider metrics, database latency and infrastructure utilization.
- Use Locust, k6 or an equivalent tool for realistic traffic patterns and distributed load generation.

One final technical detail: the semaphore bounds active operations, but `asyncio.gather()` still creates one task per requested operation. That is fine for moderate benchmark sizes; for millions of requests, use a bounded worker pool or a dedicated load-testing tool.

My recommendation: Keep this utility in your AI project's `benchmarking` or `performance_tests` package. Use it during development and regression testing, and use a dedicated load-testing framework for production capacity validation.


-------

# Production AI Benchmarking Code — Complete Explanation

Let's understand the improved version of your code as a senior AI engineer would explain it in a code review: what each component does, why it exists, how execution flows through it, and how you would use it in a real RAG or multi-agent AI system.

The goal is to understand not just Python syntax, but why we need each part in a production AI application.

## 1. First, understand the problem this code solves

Imagine you have built a financial advisor copilot using FastAPI, Qdrant, PostgreSQL, Redis, and an LLM.

A user sends this question:

> "Analyze my portfolio and explain the risk associated with my investments."

Your application retrieves financial documents, calls an LLM, processes the result, and returns an answer.

Now imagine 100 users send questions around the same time. You need to know:

- How long do requests take?
- How many requests succeed?
- How many fail or time out?
- How does latency change when concurrency increases?
- Are requests waiting for available execution slots?
- What throughput can the system sustain?

Your benchmark code measures these things for an asynchronous operation.

100 benchmark requests

Simulated requests to your AI operation

Concurrency controller

Allows at most 10 operations inside the semaphore at once

Successful

Latency recorded

Failed / timed out

Error and latency recorded

Performance report

Throughput · p50 · p95 · p99 · errors · queue wait

## 2. Imports and types

Here is the first part of the code:

```
import asyncioimport mathimport statisticsimport timefrom collections import Counterfrom collections.abc import Awaitable, Callablefrom typing import AnyAsyncOperation = Callable[[], Awaitable[Any]]
```

Let's examine each line.

| Component    | Purpose                                                       |
| ------------ | ------------------------------------------------------------- |
| `asyncio`    | Runs and coordinates asynchronous tasks.                      |
| `math`       | Provides `ceil()` for percentile calculations.                |
| `statistics` | Calculates mean latency.                                      |
| `time`       | Measures elapsed time with a high-resolution monotonic clock. |
| `Counter`    | Counts errors and outcomes by category.                       |
| `Awaitable`  | Represents an object that can be awaited.                     |
| `Callable`   | Describes a function signature.                               |
| `Any`        | Allows the operation to return different result types.        |

### What does `AsyncOperation` mean?

```
AsyncOperation = Callable[[], Awaitable[Any]]
```

This type alias says that the benchmark accepts a callable that:

1. Takes no arguments.
2. Returns an awaitable object.
3. May return any result when awaited.

For example:

```
async def sample_request() -> str:    await asyncio.sleep(0.2)    return "AI response"
```

This function is compatible with `AsyncOperation`.

The benchmark doesn't need to know whether the operation calls OpenAI, invokes a RAG pipeline, or requests a FastAPI endpoint. It only needs to know that the function can be awaited.

This is an example of dependency injection through a callable: you provide the operation from outside instead of hardcoding the AI provider inside the benchmark.

Why is this valuable? You can benchmark multiple implementations using the same measurement logic.

## 3. The `percentile()` function

```
def percentile(    values: list[float],    p: float,) -> float | None:    """Return the nearest-rank percentile."""    if not values:        return None    if not 0 <= p <= 1:        raise ValueError("p must be between 0 and 1")    ordered = sorted(values)    index = max(0, math.ceil(p * len(ordered)) - 1)    return round(ordered[index], 2)
```

This function calculates a latency percentile.

### Why do we need percentiles?

Suppose these are the response times for 10 requests, in milliseconds:

```
latencies = [    100, 110, 120, 130, 140,    150, 160, 180, 400, 1000,]
```

The mean is 249 ms, but most requests are much faster than that. The last two slow requests inflate the average.

Percentiles give you a better understanding of how latency is distributed.

- p50: The middle of the distribution.
- p95: A latency threshold below which approximately 95% of observations fall, using the selected percentile convention.
- p99: The corresponding threshold for approximately 99% of observations.

### How the function works

Step 1 — Handle empty data

```
if not values:    return None
```

If no latency measurements exist, there is no valid percentile. Returning `None` prevents a crash when every request fails before a latency sample is collected.

Step 2 — Validate the percentile

```
if not 0 <= p <= 1:    raise ValueError("p must be between 0 and 1")
```

The percentile is represented as a fraction:

- `0.50` means p50.
- `0.95` means p95.
- `0.99` means p99.

Step 3 — Sort the measurements

```
ordered = sorted(values)
```

The function needs the values in ascending order before selecting the appropriate position.

Step 4 — Calculate the nearest-rank index

```
index = max(0, math.ceil(p * len(ordered)) - 1)
```

For 10 values and p95:

\\[ \lceil 0.95 \times 10 \rceil - 1 = 9 \\]

Python lists use zero-based indexing, so index 9 selects the tenth value.

Step 5 — Return the result

```
return round(ordered[index], 2)
```

The result is rounded to two decimal places.

One important caveat: percentile conventions differ between tools. For example, some tools interpolate between observations. When comparing your benchmark with Grafana, Prometheus, or a load-testing tool, use a consistent percentile definition.

Also, 100 requests is a small sample for p99 analysis. For reliable tail-latency estimates, collect more observations and repeat the test.

## 4. The `summarize()` function

```
def summarize(    values: list[float],) -> dict[str, float | int | None]:    """Summarize latency values in milliseconds."""    return {        "count": len(values),        "p50_ms": percentile(values, 0.50),        "p95_ms": percentile(values, 0.95),        "p99_ms": percentile(values, 0.99),        "mean_ms": (            round(statistics.mean(values), 2)            if values else None        ),    }
```

This function combines several measurements into one dictionary.

For example, given a list of successful request latencies, it returns something like:

```
{    "count": 100,    "p50_ms": 120.0,    "p95_ms": 280.0,    "p99_ms": 420.0,    "mean_ms": 145.6,}
```

These are illustrative numbers.

Let's understand each field:

- `count`: Number of measurements in this group.
- `p50_ms`: Median latency.
- `p95_ms`: Tail latency at the 95th percentile.
- `p99_ms`: Tail latency at the 99th percentile.
- `mean_ms`: Average latency, if measurements exist.

The expression

```
round(statistics.mean(values), 2) if values else None
```

is a conditional expression. It calculates the mean only when the list contains values.

That is important because `statistics.mean([])` raises an exception.

The return type is a dictionary containing numbers and `None` where a measurement isn't available.

## 5. The main `benchmark()` function

Now we reach the core of the implementation.

```
async def benchmark(    operation: AsyncOperation,    *,    requests: int = 100,    concurrency: int = 10,    timeout_seconds: float = 30.0,) -> dict[str, Any]:
```

This function accepts the operation and three configuration parameters.

| Parameter         | Meaning                                                                         |
| ----------------- | ------------------------------------------------------------------------------- |
| `operation`       | The async function being tested.                                                |
| `requests`        | Total number of operations to schedule.                                         |
| `concurrency`     | Maximum number of operations permitted inside the semaphore-controlled section. |
| `timeout_seconds` | Maximum time allowed for each operation.                                        |

### Why is there a `*`?

The asterisk makes the parameters after it keyword-only.

This is valid:

```
await benchmark(    sample_request,    requests=100,    concurrency=10,    timeout_seconds=30,)
```

This is invalid:

```
await benchmark(sample_request, 100, 10, 30)
```

Keyword-only parameters make configuration more explicit and reduce the risk of accidentally swapping values.

### Step 1 — Validate the configuration

```
if requests < 1 or concurrency < 1:    raise ValueError("requests and concurrency must be >= 1")if timeout_seconds <= 0:    raise ValueError("timeout_seconds must be positive")
```

This prevents invalid configurations from silently producing meaningless results.

For example, a concurrency value of zero would never permit an operation to acquire a semaphore slot. Rejecting it immediately is much better than allowing the benchmark to hang.

## 6. Creating the semaphore and metric collections

```
semaphore = asyncio.Semaphore(concurrency)success_latencies = []failure_latencies = []queue_waits = []errors = Counter()outcomes = Counter()
```

These objects store the measurements gathered during the test.

### Why a semaphore?

If `concurrency=10`, at most 10 benchmark tasks can hold the semaphore at once.

Imagine scheduling 100 requests:

### Execution example

## 10

Operations allowed to run

## 90

Operations waiting initially

As operations finish, waiting tasks acquire the released slots. These counts describe the initial state, assuming all 100 tasks are scheduled and none has yet acquired a slot.

Why is that useful in an AI application?

You may want to simulate 100 requests without allowing all 100 to hit your LLM provider simultaneously. The semaphore gives you a controllable level of concurrency.

However, it is important to distinguish two concepts:

- Concurrency limit: How many operations may be in flight simultaneously.
- Rate limit: How many requests may begin per second or minute.

A semaphore controls concurrency, not requests per second. It does not replace a token bucket or distributed rate limiter.

### Why use `Counter`?

```
errors = Counter()outcomes = Counter()
```

`Counter` automatically counts occurrences of each key.

For example:

```
errors["TimeoutError"] += 1errors["HTTPStatusError"] += 1
```

You can later produce a summary such as:

```
{    "TimeoutError": 3,    "HTTPStatusError": 2,}
```

That tells you how many operations failed for each exception type.

The `outcomes` counter separately tracks successful operations, ordinary failures and timeouts.

## 7. The most important function: `run_one()`

This nested function represents one benchmark request.

```
async def run_one() -> None:    scheduled_at = time.perf_counter()    async with semaphore:        started_at = time.perf_counter()        queue_waits.append(            (started_at - scheduled_at) * 1000        )        outcome = "success"        try:            await asyncio.wait_for(                operation(),                timeout=timeout_seconds,            )        except asyncio.CancelledError:            raise        except TimeoutError:            outcome = "timeout"            errors["TimeoutError"] += 1        except Exception as exc:            outcome = "failure"            errors[type(exc).__name__] += 1        finally:            elapsed_ms = (                time.perf_counter() - started_at            ) * 1000            outcomes[outcome] += 1            if outcome == "success":                success_latencies.append(elapsed_ms)            else:                failure_latencies.append(elapsed_ms)
```

Let's walk through the execution in order.

### Step 1 — Record when the request is scheduled

```
scheduled_at = time.perf_counter()
```

This captures the time immediately before attempting to acquire the semaphore.

The request might have to wait because all the slots are occupied.

### Step 2 — Acquire a concurrency slot

```
async with semaphore:
```

Python waits until the semaphore allows the task to proceed.

The `async with` statement releases the semaphore automatically when the block exits, including when an exception occurs.

This is safer than manually acquiring and releasing the semaphore because you don't need to remember to release it on every execution path.

### Step 3 — Measure the waiting time

```
started_at = time.perf_counter()queue_waits.append(    (started_at - scheduled_at) * 1000)
```

The code records how long the request waited to enter the semaphore-controlled section.

For example:

- Scheduled at 1.000 seconds.
- Acquired the semaphore at 1.250 seconds.
- Queue wait: 250 milliseconds.

This can reveal that your benchmark's concurrency limit is itself creating a queue.

One caveat: this measurement also includes a small amount of task-scheduling overhead. It is an approximate queue-wait measurement, not a precise measurement of time spent inside a specific infrastructure queue.

### Step 4 — Call the operation with a timeout

```
await asyncio.wait_for(    operation(),    timeout=timeout_seconds,)
```

This is a critical improvement over your original code.

Suppose your AI provider takes too long to respond. Without a timeout, the benchmark could wait indefinitely for that operation.

With `timeout_seconds=30`, `asyncio.wait_for()` requests cancellation of the underlying awaitable when the timeout expires and raises `TimeoutError`.

For example:

```
async def sample_request():    await asyncio.sleep(60)
```

With a 30-second timeout, this operation will normally be classified as a timeout after approximately 30 seconds.

However, a timeout is not a guaranteed hard deadline: cancellation cleanup may take additional time, and an underlying operation that does not cooperate with cancellation can behave differently.

In a real AI application, also configure the HTTP client or model SDK's own connection and read timeouts. The benchmark timeout is an additional safeguard, not a substitute for network-level timeouts.

### Step 5 — Handle cancellation correctly

```
except asyncio.CancelledError:    raise
```

Cancellation is different from an ordinary request failure.

The task might be cancelled because the application is shutting down or the caller no longer needs the result. Re-raising the cancellation preserves normal asyncio cancellation behavior instead of incorrectly reporting it as a normal failure.

### Step 6 — Handle timeouts

```
except TimeoutError:    outcome = "timeout"    errors["TimeoutError"] += 1
```

This records a timeout separately from other failures.

For example, if a request takes longer than the configured limit, the benchmark increments the timeout count and includes the request in the failed latency measurements.

A subtle point: if the operation itself raises `TimeoutError`, this code will classify it as a timeout too. If you need to distinguish benchmark timeouts from provider or HTTP-client timeouts, add explicit timeout categories in the operation or client layer.

### Step 7 — Handle other exceptions

```
except Exception as exc:    outcome = "failure"    errors[type(exc).__name__] += 1
```

This prevents a typical operation exception from aborting the whole benchmark.

Examples include:

- HTTP client errors.
- Model provider errors.
- Database exceptions.
- Validation errors.

The benchmark records the exception class, not the exception message. That is a reasonable default for aggregated metrics because messages can contain sensitive information.

For production telemetry, also consider structured error codes and sanitized logs.

### Step 8 — Always record the result

```
finally:    elapsed_ms = (        time.perf_counter() - started_at    ) * 1000    outcomes[outcome] += 1    if outcome == "success":        success_latencies.append(elapsed_ms)    else:        failure_latencies.append(elapsed_ms)
```

The `finally` block runs whether the operation succeeds or raises an ordinary exception.

It records the elapsed time and assigns it to the correct outcome category.

This means the benchmark can report two distinct distributions:

- Successful operation latency.
- Failed operation latency, including timeouts.

That separation is valuable because a large number of fast failures could make an overall latency average look deceptively good.

One limitation: cancellation is deliberately re-raised. It is not counted as an ordinary failure, and the current implementation is not designed to produce a complete report if the entire benchmark is externally cancelled.

## 8. Starting and executing the entire benchmark

```
benchmark_started = time.perf_counter()await asyncio.gather(    *(run_one() for _ in range(requests)))duration = time.perf_counter() - benchmark_started
```

This is where all the individual operations are launched.

### What does the `*` operator do here?

```
*(run_one() for _ in range(requests))
```

This creates a generator of coroutine objects and unpacks them as positional arguments to `asyncio.gather()`.

For `requests=100`, it supplies 100 coroutine objects to `gather()`.

`asyncio.gather()` schedules those operations concurrently and waits for their results. The semaphore ensures that only the configured number can enter the protected section simultaneously.

### How is total duration different from latency?

Suppose you run 100 operations with concurrency 10.

- Each operation takes about 200 ms.
- Ten operations can run at once.
- The benchmark completes in roughly 2 seconds, ignoring scheduling and overhead.

The mean operation latency might be about 200 ms, while the total benchmark duration is about 2 seconds.

These are different measurements.

- Operation latency: How long an individual operation spends running.
- Total duration: How long the entire benchmark takes.
- Queue wait: How long tasks wait before entering the protected section.

This distinction is essential when analyzing AI performance under load.

## 9. Understanding throughput and failure metrics

After all tasks finish, the code calculates the duration and aggregates the outcomes.

```
duration = time.perf_counter() - benchmark_startedsuccessful = outcomes["success"]failed = requests - successful
```

`successful` counts completed operations that returned without raising an exception. `failed` is the number of requested operations that did not succeed, assuming the benchmark ran to completion.

### Completed throughput

```
"completed_rps": round(requests / duration, 2)
```

This measures the total number of benchmark operations completed per second, including failed attempts.

For example, if 100 attempts finish in 5 seconds:

\\[ \text{Completed RPS} = \frac{100}{5} = 20 \\]

### Successful throughput

```
"successful_rps": round(successful / duration, 2)
```

If 95 of those 100 operations succeed:

\\[ \text{Successful RPS} = \frac{95}{5} = 19 \\]

This is often more useful for capacity analysis because it shows the rate at which the system produces successful outcomes.

### Failure rate

```
"failure_rate": round(failed / requests, 4)
```

If five of 100 operations fail:

\\[ \text{Failure rate} = \frac{5}{100} = 0.05 \\]

That represents a 5% failure rate.

Notice that this code uses the total number of requested operations as the denominator. This is appropriate when the benchmark completes normally, but a benchmark interrupted externally needs additional accounting.

## 10. Understanding the returned report

The function returns a dictionary like this:

```
{    "requests": 100,    "concurrency": 10,    "duration_seconds": 5.0,    "successful_requests": 95,    "failed_requests": 5,    "timed_out_requests": 2,    "failure_rate": 0.05,    "completed_rps": 20.0,    "successful_rps": 19.0,    "successful_latency": {        "count": 95,        "p50_ms": 150.0,        "p95_ms": 300.0,        "p99_ms": 400.0,        "mean_ms": 175.0    },    "failed_latency": {        "count": 5,        "p50_ms": 500.0,        "p95_ms": 30000.0,        "p99_ms": 30000.0,        "mean_ms": 12500.0    },    "queue_wait": {        "count": 100,        "p50_ms": 500.0,        "p95_ms": 1500.0,        "p99_ms": 1700.0,        "mean_ms": 700.0    },    "errors_by_type": {        "TimeoutError": 2,        "HTTPStatusError": 3    }}
```

These are illustrative values, not a consistent report generated by an actual run. For instance, the failed latency distribution depends on how the five failures occurred, and the queue-wait distribution depends on the actual scheduling behavior.

Here is how to interpret the fields:

| Field                 | What it tells you                               |
| --------------------- | ----------------------------------------------- |
| `requests`            | Number of operations requested.                 |
| `concurrency`         | Maximum permitted concurrent operations.        |
| `duration_seconds`    | Total benchmark duration.                       |
| `successful_requests` | Number of successful operations.                |
| `failed_requests`     | Number of unsuccessful operations.              |
| `timed_out_requests`  | Number classified as timeouts.                  |
| `failure_rate`        | Fraction of requests that failed.               |
| `completed_rps`       | Total completed attempts per second.            |
| `successful_rps`      | Successful operations per second.               |
| `successful_latency`  | Latency distribution for successful operations. |
| `failed_latency`      | Latency distribution for failed operations.     |
| `queue_wait`          | Time spent waiting to acquire a semaphore slot. |
| `errors_by_type`      | Failure counts grouped by exception type.       |

The main benefit is that you no longer have a single average hiding all the important details.

## 11. How to call this code from your AI project

A clean project structure could look like this:

financial-advisor-copilot/

app/

benchmarking/benchmark.py

services/rag_service.py

api/routes.py

scripts/

run_benchmark.py

Your benchmark runner can call the actual RAG service or send requests to a running API.

For example, if you want to measure the full API endpoint, you can use an asynchronous HTTP client:

```
import asyncioimport httpxfrom app.benchmarking.benchmark import benchmarkasync def main():    async with httpx.AsyncClient(        base_url="http://localhost:8000",        timeout=35.0,    ) as client:        async def sample_request():            response = await client.post(                "/ask",                json={                    "question": "Explain portfolio risk."                },            )            response.raise_for_status()            return response.json()        results = await benchmark(            sample_request,            requests=100,            concurrency=10,            timeout_seconds=30.0,        )        print(results)if __name__ == "__main__":    asyncio.run(main())
```

Install the HTTP client if necessary:

```
pip install httpx
```

Here, `http://localhost:8000/ask` is an example endpoint; replace it with your actual API route. The FastAPI application must already be running.

This measures the full HTTP request, including the client-server round trip, endpoint processing and response handling. Because `raise_for_status()` raises for non-2xx responses, HTTP errors are counted as failures.

## 12. What I would improve next as a staff engineer

Before using this benchmark for serious performance decisions, I would address five additional areas.

1. Validate response correctness. An HTTP 200 response does not prove that the AI answer is correct. Add assertions or a separate evaluation suite for answer structure, citations and factual correctness.
2. Add warm-up and repeated runs. Run warm-up requests, then measure multiple independent test runs. Report sample sizes and variability.
3. Use realistic workload patterns. Test different questions, token lengths, retrieval sizes and arrival rates. This code submits a burst of tasks rather than simulating a realistic user arrival pattern.
4. Add tracing. Use OpenTelemetry to separate API, retrieval, reranking, model generation and database latency. The benchmark shows where performance degrades at the overall level; traces help explain why.
5. Protect the test environment. Use bounded request volumes, provider quotas, test credentials and cost limits. One hundred requests at concurrency 10 is a benchmark configuration, not a guarantee that the test is safe or representative.

One more technical distinction: this benchmark uses a single semaphore in one Python process. If you deploy 10 application workers, each worker's semaphore is independent. This is not a global concurrency control for your entire deployment.

## 13. Test your understanding

Suppose your AI benchmark shows p50 latency of 200 ms, p95 latency of 2 seconds, and successful throughput that stops increasing as concurrency rises. What should you investigate first?

Only optimize Python syntax.

Investigate bottlenecks, queueing, provider throttling and component-level traces.

Increase concurrency indefinitely.

Remove timeouts so requests can finish.

Check answer

The main lesson: Your benchmark is the measurement layer of an AI performance engineering workflow. The benchmark tells you what is happening; distributed tracing, load testing, error analysis and response evaluation help you understand why it is happening and what to fix.