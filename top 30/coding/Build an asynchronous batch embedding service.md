To build a production-grade asynchronous batch embedding service, you need to decouple incoming API requests from the actual GPU inference loops. Processing text embeddings one-by-one or blocking the web worker during large batch arrays leads to timeout failures and under-utilized GPU processing structures.
The industry-standard architecture uses FastAPI to receive text strings, a background consumer task loop to gather strings across incoming requests, and a vectorization worker to process text chunks using dynamic batching.

Project Dependencies

Install the required micro-framework packages:
bash
pip install fastapi uvicorn sentence-transformers torch pydantic
Use code with caution.

Production Async Batch Embedding Service

Below is a complete, scalable microservice (service.py). It utilizes an internal asyncio.Queue to gather requests concurrently and vectorizes them using a thread pool runner to prevent blocking FastAPI's async event loop.
python
import asyncio
import uuid
import torch
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException, BackgroundTasks, status
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

app = FastAPI(title="Async Batch Embedding Service")

# --- 1. Hardware & Model Configuration ---
# Resolve hardware architectures dynamically
DEVICE = (
    "cuda" if torch.cuda.is_available() 
    else "mps" if torch.backends.mps.is_available() 
    else "cpu"
)
MODEL_NAME = "BAAI/bge-small-en-v1.5"
MAX_BATCH_SIZE = 64  # Maximum array size pushed to the GPU at once
BATCH_TIMEOUT_SECS = 0.1  # Max time to wait for gathering incoming items

print(f"Loading embedding model '{MODEL_NAME}' onto hardware: {DEVICE}...")
# Load the model outside the request loop to prevent memory bloat leaks
embedding_model = SentenceTransformer(MODEL_NAME, device=DEVICE)
print("Embedding model initialization pass successful.")

# --- 2. Request / Response Data Schemas ---
class EmbeddingRequest(BaseModel):
    texts: List[str] = Field(..., min_items=1, description="Array of text inputs to embed.")

class JobStatusResponse(BaseModel):
    job_id: str
    status: str = Field(..., description="PENDING, PROCESSING, COMPLETED, or FAILED")
    results: List[List[float]] | None = None
    error: str | None = None

# Internal Global State Databases (Use Redis/PostgreSQL in a distributed production setup)
job_store: Dict[str, Dict[str, Any]] = {}
request_queue: asyncio.Queue = asyncio.Queue()

# --- 3. The Core Worker Loop (Background Consumer) ---
async def batch_processing_worker():
    """
    Continuous background loop that groups individual text arrays 
    into optimal GPU micro-batches based on size limits or time windows.
    """
    while True:
        batch_items = []
        start_time = asyncio.get_event_loop().time()

        # Gather incoming items until batch size is hit or time window expires
        while len(batch_items) < MAX_BATCH_SIZE:
            time_remaining = BATCH_TIMEOUT_SECS - (asyncio.get_event_loop().time() - start_time)
            if time_remaining <= 0 and len(batch_items) > 0:
                break

            try:
                # Wait for next text item to arrive in queue
                timeout = max(time_remaining, 0.001) if len(batch_items) > 0 else None
                queue_item = await asyncio.wait_for(request_queue.get(), timeout=timeout)
                batch_items.append(queue_item)
            except asyncio.TimeoutError:
                # Time window hit, break out and process what we have collected so far
                break

        if not batch_items:
            continue

        # Extract items for processing
        job_ids = [item["job_id"] for item in batch_items]
        texts_to_embed = [item["text"] for item in batch_items]

        # Update job statuses to PROCESSING
        for j_id in set(job_ids):
            if job_store[j_id]["status"] == "PENDING":
                job_store[j_id]["status"] = "PROCESSING"

        try:
            # sentence-transformers encode is synchronous/blocking. 
            # We wrap it in an executor thread pool to keep the async event loop free.
            loop = asyncio.get_running_loop()
            embeddings = await loop.run_in_executor(
                None, 
                lambda: embedding_model.encode(texts_to_embed, convert_to_numpy=True).tolist()
            )

            # Map vectors back to their specific parent job structures
            for idx, job_id in enumerate(job_ids):
                job_store[job_id]["results"].append(embeddings[idx])
                
            # Finalize completion status for processed jobs
            for j_id in set(job_ids):
                # Ensure all texts for a job are processed before marking completed
                if len(job_store[j_id]["results"]) == job_store[j_id]["total_items"]:
                    job_store[j_id]["status"] = "COMPLETED"

        except Exception as err:
            print(f"GPU Vectorization Worker Error: {err}")
            for j_id in set(job_ids):
                job_store[j_id]["status"] = "FAILED"
                job_store[j_id]["error"] = str(err)
        finally:
            for _ in range(len(batch_items)):
                request_queue.task_done()

# --- 4. Service Lifecycle and API Route Handlers ---
@app.on_event("startup")
async def startup_event():
    # Spin up the background orchestration worker task
    asyncio.create_task(batch_processing_worker())

@app.post("/v1/embeddings/batch", response_model=JobStatusResponse, status_code=status.HTTP_202_ACCEPTED)
async def create_batch_embedding_job(payload: EmbeddingRequest):
    """
    Ingests text lists, provisions a tracking token ID, 
    and pushes items onto the queue before returning immediately.
    """
    job_id = str(uuid.uuid4())
    
    # Initialize the structural state tracker row
    job_store[job_id] = {
        "status": "PENDING",
        "total_items": len(payload.texts),
        "results": [],
        "error": None
    }

    # Fan out texts into individual queue items for dynamic batch matching
    for text in payload.texts:
        await request_queue.put({"job_id": job_id, "text": text})

    return JobStatusResponse(job_id=job_id, status="PENDING")

@app.get("/v1/embeddings/status/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """
    Polling endpoint allowing users to check execution progress or fetch final vectors.
    """
    job = job_store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Requested batch job identifier not found.")
        
    return JobStatusResponse(
        job_id=job_id,
        status=job["status"],
        results=job["results"] if job["status"] == "COMPLETED" else None,
        error=job["error"]
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("service:app", host="127.0.0.1", port=8002, reload=False)
Use code with caution.

Production Verification Walkthrough

The following script simulates concurrent traffic hitting the batch microservice, polling until the background loop completes vectorization.
python
import asyncio
import httpx

async def simulate_client_requests():
    async with httpx.AsyncClient() as client:
        url = "http://127.0.0"
        
        # Define multi-tenant incoming mock payloads
        payload_1 = {"texts": ["The swift brown fox jumps.", "Artificial intelligence scalability."]}
        payload_2 = {"texts": ["Data structures optimization indices.", "Quantum compute nodes clusters."]}

        print("📡 Dispatching parallel batch generation jobs to service layer...")
        # Fire requests concurrently
        responses = await asyncio.gather(
            client.post(url, json=payload_1),
            client.post(url, json=payload_2)
        )

        job_ids = [res.json()["job_id"] for res in responses]
        print(f"Allocated Tracking Job Tokens: {job_ids}\n")

        # Poll status endpoints until tasks transition to COMPLETED
        for j_id in job_ids:
            status_url = f"http://127.0.0{j_id}"
            while True:
                status_check = await client.get(status_url)
                state_data = status_check.json()
                
                print(f"Job {j_id} Status state check -> {state_data['status']}")
                if state_data["status"] in ["COMPLETED", "FAILED"]:
                    if state_data["status"] == "COMPLETED":
                        print(f"🎉 Success! Extracted vector count: {len(state_data['results'])}")
                        print(f"Dimensions inside first item: {len(state_data['results'][0])}\n")
                    break
                await asyncio.sleep(0.05) # Wait before polling again

if __name__ == "__main__":
    # Ensure your local app instance runner script is executing on port 8002 prior to running this wrapper
    try:
        asyncio.run(simulate_client_requests())
    except Exception as e:
        print(f"Connection error (Is your service file running?): {e}")
Use code with caution.

Why this design scales:

• Dynamic Micro-Batching: If 20 separate users hit /v1/embeddings/batch at roughly the same time, the background worker merges their individual text inputs into a single large array. This maximizes your GPU's parallel processing matrix without forcing users to coordinate batches manually.
• Thread Offloading via run_in_executor: Because deep learning inference blocks the CPU thread, calling model.encode directly inside a normal async def route would freeze the entire server. Offloading execution to a thread pool ensures your server continues accepting incoming requests smoothly.
To optimize this setup for your production infrastructure, let me know:
• Do you want to replace the local memory dictionary with a Redis or Celery task worker queue for distributed scaling across multiple servers?
• Do you want to add token-length validation hooks using a tokenizer (like tiktoken) to catch and reject oversized inputs before they hit the queue?
