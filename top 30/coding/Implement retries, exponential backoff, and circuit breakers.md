To build resilient production LLM applications, you must protect your system against network flakiness, rate limits (HTTP 429), and downstream service collapses.
The industry standard for achieving this in Python is tenacity for robust retry/backoff logic, combined with a Circuit Breaker pattern to completely stop traffic to a failing service before it causes cascading failures down your pipeline.

Project Dependencies

Install the required micro-framework libraries:
bash
pip install tenacity openai
Use code with caution.

Production Resilience Implementation

Below is a complete, modular, and asynchronous client architecture. It combines exponential backoff with jitter (retrying smartly on temporary glitches) and a Stateful Circuit Breaker (blocking requests if a service remains completely broken).
python
import asyncio
import logging
import time
from typing import Any, Dict, List
import openai
from openai import AsyncOpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

# Setup logging architecture
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ResilientLLMClient")

# --- 1. The Circuit Breaker Core ---
class CircuitBreakerOpenException(Exception):
    """Raised immediately when the circuit breaker is open to prevent unnecessary API calls."""
    pass

class CircuitBreaker:
    def __init__(self, failure_threshold: int = 3, recovery_time_seconds: int = 30):
        self.failure_threshold = failure_threshold
        self.recovery_time_seconds = recovery_time_seconds
        self.failure_count = 0
        self.state = "CLOSED"  # CLOSED (healthy), OPEN (failing), HALF-OPEN (testing)
        self.last_state_change = time.time()

    def __call__(self, func):
        async def wrapper(*args, **kwargs):
            self._check_state()
            
            if self.state == "OPEN":
                raise CircuitBreakerOpenException(
                    f"Circuit is OPEN. Fast-failing request. Time remaining until retry: "
                    f"{int(self.recovery_time_seconds - (time.time() - self.last_state_change))}s"
                )

            try:
                result = await func(*args, **kwargs)
                self._handle_success()
                return result
            except Exception as e:
                # Do not count expected rate limits or context length bugs as complete circuit failures
                # only count hard infrastructure dropouts or server crashes (HTTP 5xx / timeouts)
                if not isinstance(e, (openai.RateLimitError, openai.BadRequestError)):
                    self._handle_failure()
                raise e
        return wrapper

    def _check_state(self):
        """Transitions state from OPEN to HALF-OPEN if the recovery cooldown time has passed."""
        if self.state == "OPEN" and (time.time() - self.last_state_change) > self.recovery_time_seconds:
            logger.warning("🔄 Circuit Breaker transitioning from OPEN to HALF-OPEN. Testing service health...")
            self.state = "HALF-OPEN"
            self.last_state_change = time.time()

    def _handle_success(self):
        """Resets failures on a successful execution pass."""
        if self.state == "HALF-OPEN":
            logger.info("✅ Circuit Breaker transitioning from HALF-OPEN to CLOSED. Service recovered.")
        self.failure_count = 0
        self.state = "CLOSED"

    def _handle_failure(self):
        """Increments tracking counts and trips the circuit breaker open if limits are passed."""
        self.failure_count += 1
        logger.error(f"❌ Failure recorded. Total continuous failures: {self.failure_count}/{self.failure_threshold}")
        if self.failure_count >= self.failure_threshold and self.state != "OPEN":
            logger.critical(f"💥 Circuit Breaker tripped OPEN! Tripping for {self.recovery_time_seconds} seconds.")
            self.state = "OPEN"
            self.last_state_change = time.time()


# --- 2. The Resilient LLM Inference Engine ---
class ResilientLLMClient:
    def __init__(self, api_key: str = "mock-key"):
        self.client = AsyncOpenAI(api_key=api_key, base_url="http://localhost:8000/v1" if api_key == "mock-key" else None)
        self.circuit_breaker = CircuitBreaker(failure_threshold=2, recovery_time_seconds=5)

    # Tenacity handles fine-grained retry logic wrapping around the execution layer
    @retry(
        # Rerun only on retryable network exceptions, timeouts, and rate limits
        retry=retry_if_exception_type((
            openai.APITimeoutError, 
            openai.APIConnectionError, 
            openai.RateLimitError, 
            openai.InternalServerError
        )),
        # Exponential Backoff with randomized full jitter: wait 2^x * 2 seconds, capped at 10s
        wait=wait_exponential(multiplier=2, min=2, max=10),
        stop=stop_after_attempt(4),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True  # Pass final error out if all retry limits are exhausted
    )
    async def _execute_with_retry(self, prompt: str) -> str:
        """Internal runner method handling retry loops."""
        logger.info("📡 Dispatching network call to LLM Cluster endpoint...")
        response = await self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            timeout=5.0  # Strict timeout limit to prevent hanging requests
        )
        return response.choices[0].message.content

    async def generate_text(self, prompt: str) -> str:
        """
        Public execution gateway orchestrating both the Circuit Breaker layer
        and the internal retry/backoff layer together.
        """
        # Dynamically inject the instance circuit breaker wrapper context
        protected_call = self.circuit_breaker(self._execute_with_retry)
        return await protected_call(prompt)
Use code with caution.

Verification and Simulation Walkthrough

The following script simulates a production scenario: making successful calls, encountering a cluster blackout that trips the circuit breaker, verifying that incoming traffic is instantly blocked to prevent thread starvation, and recovering once the cooldown finishes.
python
import subprocess
import time as sync_time

async def main():
    # 1. Start a local mock server using our first FastAPI script to receive the calls
    # For testing simulation simplicity, we will catch what happens when the endpoint is entirely dead/down.
    llm_engine = ResilientLLMClient(api_key="sk-real-key-or-mock-key")
    
    print("🚀 Scenario 1: Sending request while downstream service is completely unreachable...")
    
    # We loop to exhaust the retry limit and trip the circuit breaker open
    for i in range(1, 4):
        print(f"\n--- Application Request Attempt #{i} ---")
        start_time = time.time()
        try:
            response = await llm_engine.generate_text("Hello Core Cluster Matrix.")
            print(f"Result: {response}")
        except CircuitBreakerOpenException as cbe:
            print(f"🛡️ Circuit Breaker Action: {cbe}")
        except Exception as general_error:
            duration = time.time() - start_time
            print(f"💥 Request Failed with: {type(general_error).__name__} (Took {duration:.2f}s)")

    print("\n⏳ Application sleeping for 6 seconds to clear the recovery window cooldown...")
    await asyncio.sleep(6)

    print("\n🔄 Scenario 2: Circuit is now HALF-OPEN. Retrying connection...")
    try:
        # This will test the service again. If it fails, it will trip OPEN immediately for another 5s.
        await llm_engine.generate_text("Testing recovery step.")
    except Exception as e:
        print(f"System Note: Request naturally failed again because backend is still down, but circuit handled it.")

if __name__ == "__main__":
    asyncio.run(main())
Use code with caution.

Core Architecture Pillars

1. Exponential Backoff with Jitter (tenacity): When thousands of requests hit an LLM API simultaneously and get rate-limited (HTTP 429), retrying at exact intervals causes a "thundering herd" problem that continues to crush the server. Adding jitter randomizes the retry math, spreading out the network load cleanly.
2. The Circuit Breaker Wrapper: If your LLM provider drops offline entirely, retry loops are useless and waste resources. The Circuit Breaker instantly blocks requests once the failure threshold is crossed. This gives your application a chance to fallback instantly to an alternative local model (like Llama 3 via Ollama) or return a clean, cached error page to the user.
To ready this architecture for production scaling, let me know:
• Do you want to implement a fallback model path (e.g., dynamically routing requests to Anthropic Claude or an internal vLLM cluster if OpenAI fails)?
• Should we integrate a central dashboard tracking mechanism like Prometheus metrics to monitor your circuit trips?
