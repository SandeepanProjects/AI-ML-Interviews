# For Local Transformers implementation
pip install fastapi uvicorn transformers torch pydantic

# For API-based implementation
pip install cohere





Asynchronous Cloud Reranking Layer (FastAPI + Cohere Client)

If you do not want to manage local GPU infrastructure, you can use this lightweight, production-grade asynchronous service that uses Cohere's Rerank API.




import os
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
import cohere

app = FastAPI(title="Async Cohere Reranking Layer")

# Initialize the modern async client framework mapping
# Ensure your COHERE_API_KEY environment variable is set
cohere_client = cohere.AsyncClient(api_key=os.getenv("COHERE_API_KEY", "your-api-key"))

class CloudRerankRequest(BaseModel):
    query: str
    documents: list[str]
    top_n: int | None = None

class CloudRerankResponse(BaseModel):
    results: list[dict]

@app.post("/v1/rerank/cloud", response_model=CloudRerankResponse)
async def rerank_cloud(payload: CloudRerankRequest):
    """
    Asynchronously calls the Cohere Rerank API without blocking the FastAPI event loop.
    """
    try:
        # Non-blocking remote network request to managed API compute node zones
        response = await cohere_client.rerank(
            model="rerank-v3.5", # High-performance enterprise grade variant
            query=payload.query,
            documents=payload.documents,
            top_n=payload.top_n
        )
        
        # Parse return payload responses out systematically 
        formatted_results = [
            {
                "index": result.index,
                "document": payload.documents[result.index],
                "score": result.relevance_score
            }
            for result in response.results
        ]
        
        return CloudRerankResponse(results=formatted_results)
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"Cloud API runtime error: {str(e)}"
        )
