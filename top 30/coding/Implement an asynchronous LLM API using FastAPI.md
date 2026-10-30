import json
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

app = FastAPI(title="Streaming Async LLM API")

LLM_API_URL = "https://openai.com"
API_KEY = "your-api-key-here"

class StreamRequest(BaseModel):
    prompt: str
    model: str = "gpt-4o-mini"

@app.post("/v1/chat/stream")
async def stream_chat(payload: StreamRequest):
    """
    Streams LLM tokens asynchronously to the client using Server-Sent Events (SSE).
    """
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": payload.model,
        "messages": [{"role": "user", "content": payload.prompt}],
        "stream": True  # Instructs provider to stream back chunks
    }

    async def token_generator():
        async with httpx.AsyncClient(timeout=60.0) as client:
            # Connect to stream
            async with client.stream("POST", LLM_API_URL, json=data, headers=headers) as response:
                if response.status_code != 200:
                    yield f"data: {json.dumps({'error': 'Failed to connect to LLM'})}\n\n"
                    return

                # Iterate through incoming data lines asynchronously
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    
                    # Clean the SSE prefix coming from the provider
                    if line.startswith("data: "):
                        line = line[6:]
                        
                    if line == "[DONE]":
                        break
                        
                    try:
                        parsed = json.loads(line)
                        chunk = parsed["choices"][0]["delta"].get("content", "")
                        if chunk:
                            # Format to strict SSE specifications
                            yield f"data: {json.dumps({'token': chunk})}\n\n"
                    except Exception:
                        continue
                        
        yield "data: [DONE]\n\n" # Signal complete to client

    return StreamingResponse(token_generator(), media_type="text/event-stream")
