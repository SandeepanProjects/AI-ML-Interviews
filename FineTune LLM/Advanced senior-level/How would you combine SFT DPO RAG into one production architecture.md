# How would you combine SFT + DPO + RAG into one production architecture?

This is an excellent **Senior AI Engineer interview question** because it tests whether you understand that:

> **SFT, DPO, and RAG solve different problems and should work together rather than replace each other.**

The high-level architecture is:

```text
                         OFFLINE TRAINING
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  Base Model                                                     │
│      │                                                          │
│      ▼                                                          │
│  SFT: Teach behavior and domain tasks                           │
│      │                                                          │
│      ▼                                                          │
│  DPO: Improve preferences, helpfulness and response quality     │
│      │                                                          │
│      ▼                                                          │
│  Production Model                                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘


                         ONLINE INFERENCE
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│ User Query                                                      │
│      │                                                          │
│      ▼                                                          │
│ Query Processing                                                │
│      │                                                          │
│      ▼                                                          │
│ Embedding Model                                                 │
│      │                                                          │
│      ▼                                                          │
│ Vector / Hybrid Search                                          │
│      │                                                          │
│      ▼                                                          │
│ Reranker                                                        │
│      │                                                          │
│      ▼                                                          │
│ Relevant Context                                                │
│      │                                                          │
│      ├─────────────────────────────────────┐                    │
│      ▼                                     │                    │
│ Fine-tuned SFT + DPO Model ◄───────────────┘                    │
│      │                                                          │
│      ▼                                                          │
│ Grounded Response                                               │
│      │                                                          │
│      ▼                                                          │
│ Evaluation + Monitoring                                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

# 1. What problem does each component solve?

## SFT

**Supervised Fine-Tuning** teaches the model how to perform your task.

Example:

```text
User:
Explain how to reset my password.

Ideal Answer:
Go to Settings → Security → Reset Password...
```

SFT teaches:

```text
Task behavior
Domain terminology
Output format
Tone
Instruction following
Tool calling
Structured output
```

---

## DPO

**Direct Preference Optimization** teaches the model which answer is better.

Example:

```text
Prompt:
How do I reset my password?

Chosen:
Go to Settings → Security → Reset Password.

Rejected:
I don't know. Contact support.
```

DPO improves:

```text
Helpfulness
Answer quality
Preference alignment
Style
Safety behavior
Conciseness
```

---

## RAG

**Retrieval-Augmented Generation** provides current and private knowledge.

```text
User:
What is our company's leave policy?

        ↓

Retrieve policy document

        ↓

Provide context to LLM

        ↓

Generate grounded answer
```

RAG handles:

```text
Private knowledge
Frequently changing information
Large document collections
Current policies
Source grounding
```

---

# 2. Why combine all three?

Consider an enterprise support assistant.

### Without SFT

The base model might not understand your domain well.

### With SFT only

The model knows the domain but may give answers in an undesirable style.

### With SFT + DPO

The model behaves well but its knowledge can become outdated.

### With RAG only

The model has current knowledge but may not know:

```text
How to answer
What tone to use
How to cite sources
When to refuse
How to handle ambiguity
```

Therefore:

```text
SFT
 │
 ├── Teach WHAT TO DO
 │
 ▼
DPO
 │
 ├── Teach HOW TO PREFER ANSWERS
 │
 ▼
RAG
 │
 └── Provide WHAT THE MODEL SHOULD KNOW NOW
```

A useful mental model is:

```text
SFT = Capability and behavior

DPO = Preference and quality alignment

RAG = External knowledge
```

---

# 3. Production architecture

A realistic architecture could be:

```text
                         CLIENT
                            │
                            ▼
                       API Gateway
                            │
                            ▼
                    Authentication / RBAC
                            │
                            ▼
                        FastAPI API
                            │
              ┌─────────────┼──────────────┐
              │             │              │
              ▼             ▼              ▼
         Query Cache    RAG Pipeline    Observability
                              │
                              ▼
                        Query Rewriter
                              │
                              ▼
                          Embedding
                              │
                    ┌─────────┴──────────┐
                    │                    │
                    ▼                    ▼
                PostgreSQL             Qdrant
                  Metadata          Vector Search
                    │                    │
                    └─────────┬──────────┘
                              │
                              ▼
                         Hybrid Search
                              │
                              ▼
                           Reranker
                              │
                              ▼
                       Context Builder
                              │
                              ▼
                   Fine-tuned LLM
                    (SFT + DPO)
                              │
                              ▼
                       Output Validator
                              │
                              ▼
                         API Response
```

---

# PART 1 — OFFLINE TRAINING PIPELINE

# 4. Stage 1: Start with a base model

For example:

```text
Llama
Qwen
Mistral
Gemma
```

Conceptually:

```python
MODEL_NAME = "your-base-model"
```

The architecture should separate:

```text
Base Model
    │
    ▼
SFT Model
    │
    ▼
DPO Model
    │
    ▼
Production Model
```

---

# 5. Stage 1: SFT

Suppose you are building an enterprise financial assistant.

Your SFT data might look like:

```json
{
  "instruction": "Explain the company's expense reimbursement policy.",
  "response": "Employees can submit eligible expenses within 30 days..."
}
```

Dataset:

```text
datasets/
├── sft/
│   ├── train.jsonl
│   └── validation.jsonl
│
└── dpo/
    ├── train.jsonl
    └── validation.jsonl
```

---

## SFT dataset preparation

```python
from datasets import Dataset


sft_examples = [
    {
        "prompt": (
            "How do I submit an expense claim?"
        ),
        "response": (
            "You can submit an expense claim through "
            "the employee portal by uploading receipts."
        )
    },
    {
        "prompt": (
            "What expenses are eligible?"
        ),
        "response": (
            "Eligible expenses include approved business "
            "travel, accommodation and meals."
        )
    }
]


dataset = Dataset.from_list(
    sft_examples
)
```

---

## Format the SFT dataset

```python
def format_sft_example(
    example,
    tokenizer
):

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful enterprise assistant."
            )
        },
        {
            "role": "user",
            "content": example["prompt"]
        },
        {
            "role": "assistant",
            "content": example["response"]
        }
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False
    )
```

---

# 6. Fine-tune with SFT

Example using LoRA:

```python
from peft import LoraConfig


lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj"
    ],
    task_type="CAUSAL_LM"
)
```

Load a quantized model:

```python
import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)


MODEL_NAME = "your-base-model"


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto"
)
```

Train conceptually:

```python
from trl import SFTTrainer


trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    peft_config=lora_config,
    args=training_args
)


trainer.train()

trainer.save_model(
    "models/sft_adapter"
)
```

After SFT:

```text
Base Model
     │
     ▼
SFT Adapter
```

---

# 7. Stage 2: Create DPO preference data

Now collect examples where you have:

```text
Prompt
  │
  ├── Chosen answer
  │
  └── Rejected answer
```

Example:

```python
dpo_examples = [
    {
        "prompt": (
            "How do I submit an expense claim?"
        ),

        "chosen": (
            "Submit the expense through the employee "
            "portal, attach the required receipts, and "
            "provide the business purpose."
        ),

        "rejected": (
            "Expenses are important. Ask your manager."
        )
    }
]
```

The chosen response should be preferred because it is:

```text
More accurate
More actionable
More complete
More aligned with your style
```

---

# 8. Where does DPO data come from?

In production, preference data can come from:

```text
                    User Queries
                         │
                         ▼
                 Multiple Responses
                         │
             ┌───────────┴───────────┐
             │                       │
             ▼                       ▼
        Human Ranking           LLM Judge
             │                       │
             └───────────┬───────────┘
                         │
                         ▼
                   Preference Dataset
```

Example:

```text
Prompt:
How do I reset my password?

Response A:
Go to Settings → Security → Reset Password.

Response B:
You can probably reset it somehow.

Human chooses A
```

Then:

```python
{
    "prompt": "...",
    "chosen": "Response A",
    "rejected": "Response B"
}
```

---

# 9. Fine-tune using DPO

Conceptually:

```text
SFT Model
    │
    ▼
DPO Training
    │
    ▼
Preference-Aligned Model
```

Example:

```python
from trl import DPOTrainer
```

A typical setup conceptually uses:

```python
dpo_trainer = DPOTrainer(
    model=policy_model,
    ref_model=reference_model,
    train_dataset=dpo_train_dataset,
    eval_dataset=dpo_validation_dataset,
    processing_class=tokenizer,
    args=dpo_training_args
)

dpo_trainer.train()
```

The exact argument names vary across TRL versions, so in a real project you should pin compatible versions of `trl`, `transformers`, and `peft`.

After DPO:

```text
Base Model
     │
     ▼
SFT
     │
     ▼
DPO
     │
     ▼
Production Model
```

---

# 10. Important: should DPO train on RAG context?

This is a very important architectural question.

## Bad approach

Train DPO with:

```text
Question
  +
Entire production database
```

That creates problems:

```text
Data leakage
Stale training data
Huge training context
Privacy concerns
```

## Better approach

Train the model to behave correctly **when retrieved context is supplied**.

Example:

```text
SYSTEM:
Answer only using the provided context.

CONTEXT:
Employees may submit expense claims within 30 days.

QUESTION:
How long do I have to submit an expense?

CHOSEN:
You have 30 days to submit an expense claim.

REJECTED:
You have 90 days to submit an expense claim.
```

This teaches **RAG-aware behavior**.

---

# 11. RAG-aware SFT dataset

A better SFT example:

```json
{
  "context": "Employees must submit expense claims within 30 days of the expense date.",
  "question": "How long do I have to submit an expense claim?",
  "answer": "You must submit an expense claim within 30 days of the expense date."
}
```

Formatter:

```python
def format_rag_sft_example(
    example,
    tokenizer
):

    messages = [
        {
            "role": "system",
            "content": (
                "Answer using only the provided context. "
                "If the answer is not present, say that "
                "you do not have enough information."
            )
        },
        {
            "role": "user",
            "content": f"""
Context:
{example["context"]}

Question:
{example["question"]}
"""
        },
        {
            "role": "assistant",
            "content": example["answer"]
        }
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False
    )
```

This is powerful because the model learns:

```text
Context available
      ↓
Use context

Context unavailable
      ↓
Don't invent
```

---

# PART 2 — BUILD THE RAG PIPELINE

# 12. Document ingestion

A production ingestion pipeline:

```text
Documents
   │
   ▼
Loader
   │
   ▼
Text Cleaning
   │
   ▼
Chunking
   │
   ▼
Embedding
   │
   ▼
Vector Database
```

Example:

```python
class Document:
    def __init__(
        self,
        content: str,
        metadata: dict
    ):
        self.content = content
        self.metadata = metadata
```

---

# 13. Chunking

Simple chunking example:

```python
def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50
):

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end]

        chunks.append(chunk)

        start += (
            chunk_size - overlap
        )

    return chunks
```

Example:

```python
text = """
Employees must submit expenses
within 30 days.
"""

chunks = chunk_text(text)
```

In production, use token-aware or structure-aware chunking rather than simple character slicing.

---

# 14. Generate embeddings

Example:

```python
from sentence_transformers import (
    SentenceTransformer
)


embedding_model = SentenceTransformer(
    "BAAI/bge-small-en-v1.5"
)


def generate_embedding(
    text: str
):

    return embedding_model.encode(
        text
    ).tolist()
```

---

# 15. Store documents in Qdrant

Conceptually:

```python
from qdrant_client import QdrantClient


qdrant = QdrantClient(
    url="http://localhost:6333"
)
```

Store:

```python
from qdrant_client.models import (
    PointStruct
)


def index_document(
    document_id: int,
    text: str,
    metadata: dict
):

    embedding = generate_embedding(
        text
    )

    point = PointStruct(
        id=document_id,
        vector=embedding,
        payload={
            "text": text,
            **metadata
        }
    )

    qdrant.upsert(
        collection_name="enterprise_docs",
        points=[point]
    )
```

---

# 16. Retrieval

When a user asks a question:

```text
User:
How long do I have to submit expenses?
```

Generate a query embedding:

```python
def retrieve_documents(
    query: str,
    limit: int = 10
):

    query_vector = (
        generate_embedding(query)
    )

    results = qdrant.query_points(
        collection_name="enterprise_docs",
        query=query_vector,
        limit=limit
    )

    return results.points
```

Conceptually:

```text
Query
  │
  ▼
Embedding
  │
  ▼
Vector Search
  │
  ▼
Top 10 Chunks
```

---

# 17. Add metadata filtering

For enterprise systems:

```text
User
 │
 ▼
Tenant ID
 │
 ▼
Metadata Filter
 │
 ▼
Only tenant documents
```

Conceptual example:

```python
def retrieve_for_tenant(
    query: str,
    tenant_id: str
):

    query_vector = generate_embedding(
        query
    )

    # Conceptual metadata filter
    results = search_vector_db(
        vector=query_vector,
        filter={
            "tenant_id": tenant_id
        }
    )

    return results
```

This is essential for:

```text
Multi-tenancy
Security
Data isolation
RBAC
```

---

# 18. Add a reranker

Initial vector search may retrieve:

```text
Top 20 documents
```

A reranker selects the best:

```text
Top 3–5 documents
```

```text
User Query
    │
    ▼
Vector Search
Top 20
    │
    ▼
Reranker
    │
    ▼
Top 5
```

Conceptually:

```python
def rerank(
    query: str,
    documents: list[str]
):

    pairs = [
        [query, document]
        for document in documents
    ]

    scores = reranker.predict(
        pairs
    )

    ranked = sorted(
        zip(documents, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return ranked
```

---

# PART 3 — COMBINING THE FINE-TUNED MODEL WITH RAG

# 19. Production request flow

Now we combine everything.

```text
                  User Query
                       │
                       ▼
                  FastAPI API
                       │
                       ▼
              Authentication / RBAC
                       │
                       ▼
                  Query Analysis
                       │
                       ▼
                  RAG Retrieval
                       │
                       ▼
                    Reranking
                       │
                       ▼
                  Build Context
                       │
                       ▼
           SFT + DPO Fine-tuned Model
                       │
                       ▼
               Grounded Generation
                       │
                       ▼
                 Output Validation
                       │
                       ▼
                    Response
```

---

# 20. Build the RAG prompt

The prompt should be consistent with your training data.

```python
def build_rag_messages(
    question: str,
    documents: list[str]
):

    context = "\n\n".join(
        [
            f"[Document {i + 1}]\n{doc}"
            for i, doc in enumerate(documents)
        ]
    )

    messages = [
        {
            "role": "system",
            "content": """
You are an enterprise assistant.

Rules:
1. Answer only using the provided context.
2. Do not invent information.
3. If the answer is not in the context, say:
   "I don't have enough information in the available documents."
4. Be concise and clear.
"""
        },
        {
            "role": "user",
            "content": f"""
Context:
{context}

Question:
{question}
"""
        }
    ]

    return messages
```

---

# 21. Load the SFT + DPO model

Conceptually:

```python
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)


MODEL_PATH = (
    "models/production_dpo_model"
)


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH
)


model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    torch_dtype=torch.bfloat16
)
```

If using LoRA:

```text
Base Model
+
SFT Adapter
+
DPO Adapter / Final Adapter
```

In practice, you usually simplify deployment by validating the final model artifact and then serving either:

```text
Merged model
```

or:

```text
Base model + production adapter
```

rather than dynamically chaining multiple experimental adapters.

---

# 22. Generate the final answer

```python
import torch


def generate_answer(
    question: str,
    documents: list[str]
):

    messages = build_rag_messages(
        question=question,
        documents=documents
    )

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(model.device)

    with torch.inference_mode():

        output = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False
        )

    generated_tokens = output[
        0,
        inputs["input_ids"].shape[1]:
    ]

    return tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    )
```

---

# 23. Full RAG service

Now let's combine retrieval and generation.

```python
class RAGService:

    def __init__(
        self,
        retriever,
        generator
    ):
        self.retriever = retriever
        self.generator = generator


    def answer(
        self,
        question: str,
        tenant_id: str
    ):

        # 1. Retrieve
        documents = (
            self.retriever.retrieve(
                query=question,
                tenant_id=tenant_id,
                limit=10
            )
        )


        # 2. Rerank
        documents = (
            self.retriever.rerank(
                query=question,
                documents=documents,
                top_k=5
            )
        )


        # 3. Generate
        answer = (
            self.generator.generate(
                question=question,
                documents=documents
            )
        )


        return {
            "answer": answer,
            "sources": [
                document.metadata
                for document in documents
            ]
        }
```

---

# 24. FastAPI production API

```python
from fastapi import (
    FastAPI,
    Depends
)

from pydantic import BaseModel


app = FastAPI()


class QueryRequest(
    BaseModel
):
    question: str


class QueryResponse(
    BaseModel
):
    answer: str
    sources: list[dict]
```

Endpoint:

```python
@app.post(
    "/query",
    response_model=QueryResponse
)
async def query(
    request: QueryRequest,
    current_user=Depends(
        get_current_user
    )
):

    result = rag_service.answer(
        question=request.question,
        tenant_id=current_user.tenant_id
    )

    return result
```

This gives:

```text
Client
  │
  ▼
FastAPI
  │
  ▼
Authentication
  │
  ▼
Tenant Isolation
  │
  ▼
RAG
  │
  ▼
SFT + DPO Model
  │
  ▼
Response
```

---

# PART 4 — THE MOST IMPORTANT ARCHITECTURAL DECISION

# 25. Don't put company knowledge into SFT unnecessarily

A common mistake is:

```text
Company Documents
       ↓
Fine-tuning
       ↓
Model
```

Problems:

```text
Knowledge becomes stale
Retraining is expensive
Hard to delete information
Potential data leakage
Model may memorize sensitive information
```

Instead:

```text
Stable behavior
    ↓
SFT + DPO

Dynamic knowledge
    ↓
RAG
```

For example:

```text
HOW to answer?
    → Fine-tuning

WHAT is currently true?
    → RAG
```

This is one of the strongest interview answers.

---

# 26. How SFT, DPO, and RAG interact

Let's take an example.

User:

```text
What is the company's current travel reimbursement limit?
```

### RAG retrieves:

```text
Travel Policy 2026:

Maximum hotel reimbursement:
₹8,000 per night.
```

### SFT taught:

```text
Answer clearly.
Use enterprise terminology.
Follow instructions.
Use provided context.
```

### DPO taught:

```text
Prefer concise, accurate answers.
Avoid unnecessary text.
Don't hallucinate.
```

Final:

```text
According to the current travel policy, the maximum
hotel reimbursement is ₹8,000 per night.
```

So:

```text
             RAG
              │
       Provides facts
              │
              ▼
SFT + DPO Model
              │
      Applies learned behavior
              │
              ▼
        Final Answer
```

---

# PART 5 — EVALUATION

# 27. Evaluate each layer separately

Do not only evaluate:

```text
Final answer quality
```

Evaluate each stage.

## Retrieval

```text
Recall@K
Precision@K
MRR
NDCG
Context recall
```

---

## Generation

```text
Answer correctness
Faithfulness
Hallucination rate
Relevance
```

---

## SFT/DPO model

```text
Instruction following
Preference win rate
Style compliance
Safety
```

---

# 28. End-to-end evaluation

Create a golden dataset:

```python
evaluation_examples = [
    {
        "question":
            "How long do I have to submit expenses?",

        "expected_context":
            "30 days",

        "expected_answer":
            "30 days"
    }
]
```

Evaluate:

```text
Question
   │
   ▼
Retrieval Correct?
   │
   ├── No → Retrieval failure
   │
   ▼ Yes
Model Answer Correct?
   │
   ├── No → Generation failure
   │
   ▼ Yes
Success
```

This distinction is critical.

---

# 29. Monitoring in production

I would monitor:

```text
RAG Metrics
────────────────────
Retrieval latency
Empty retrieval rate
Similarity scores
Reranker latency
Context size


LLM Metrics
────────────────────
TTFT
Tokens/sec
Total latency
Input tokens
Output tokens
Cost
GPU utilization


Quality Metrics
────────────────────
Hallucination rate
User feedback
Groundedness
Answer relevance
Citation accuracy


System Metrics
────────────────────
Error rate
Timeout rate
Cache hit rate
Qdrant latency
Redis latency
```

Example middleware:

```python
import time


async def monitor_request(
    request,
    call_next
):

    start = time.perf_counter()

    response = await call_next(
        request
    )

    duration = (
        time.perf_counter()
        - start
    )

    print(
        {
            "path": request.url.path,
            "latency_seconds": duration,
            "status_code": response.status_code
        }
    )

    return response
```

In production, send these metrics to systems such as OpenTelemetry + Prometheus + Grafana.

---

# 30. Production feedback loop

The architecture should continuously improve.

```text
                Production Traffic
                        │
                        ▼
                  User Feedback
                        │
                        ▼
              Evaluation Pipeline
                        │
             ┌──────────┴──────────┐
             │                     │
             ▼                     ▼
      Retrieval Failures      Generation Failures
             │                     │
             ▼                     ▼
      Improve Documents       SFT / DPO Data
             │                     │
             └──────────┬──────────┘
                        │
                        ▼
                  New Model Version
                        │
                        ▼
                 Offline Evaluation
                        │
                        ▼
                    A/B Testing
                        │
                        ▼
                   Production
```

Example feedback data:

```python
feedback_example = {
    "question": "How do I submit expenses?",
    "retrieved_documents": [
        "doc_123",
        "doc_456"
    ],
    "answer": "Submit through the portal.",
    "rating": 1
}
```

If a response is bad, investigate:

```text
Wrong retrieval?
        ↓
Improve RAG

Correct retrieval but bad answer?
        ↓
Improve SFT/DPO
```

---

# 31. Recommended project structure

For a production implementation:

```text
enterprise_ai/
│
├── app/
│   │
│   ├── api/
│   │   └── routes.py
│   │
│   ├── auth/
│   │   ├── jwt.py
│   │   └── rbac.py
│   │
│   ├── rag/
│   │   ├── ingestion.py
│   │   ├── chunker.py
│   │   ├── embeddings.py
│   │   ├── retriever.py
│   │   ├── reranker.py
│   │   └── service.py
│   │
│   ├── llm/
│   │   ├── model_loader.py
│   │   ├── generator.py
│   │   └── prompts.py
│   │
│   ├── evaluation/
│   │   ├── retrieval_eval.py
│   │   ├── generation_eval.py
│   │   └── end_to_end_eval.py
│   │
│   ├── monitoring/
│   │   ├── metrics.py
│   │   └── tracing.py
│   │
│   └── main.py
│
├── training/
│   ├── sft/
│   │   └── train.py
│   │
│   ├── dpo/
│   │   └── train.py
│   │
│   ├── datasets/
│   │   ├── prepare_sft.py
│   │   └── prepare_dpo.py
│   │
│   └── configs/
│       ├── sft.yaml
│       └── dpo.yaml
│
├── models/
│   ├── sft/
│   └── dpo/
│
├── tests/
│
├── docker-compose.yml
└── pyproject.toml
```

---

# 32. Complete mental model

The complete architecture is:

```text
                   ┌──────────────────────┐
                   │      BASE MODEL      │
                   └──────────┬───────────┘
                              │
                              ▼
                   ┌──────────────────────┐
                   │         SFT          │
                   │                      │
                   │ Task capability      │
                   │ Domain behavior      │
                   │ Instruction following│
                   └──────────┬───────────┘
                              │
                              ▼
                   ┌──────────────────────┐
                   │         DPO          │
                   │                      │
                   │ Preference alignment │
                   │ Better responses     │
                   │ Quality              │
                   └──────────┬───────────┘
                              │
                              ▼
                     PRODUCTION MODEL
                              │
                              │
                              ▼
User ───► Query ───► RAG ───► Context
                              │
                              ▼
                    ┌───────────────────┐
                    │  FINE-TUNED LLM   │
                    │     SFT + DPO     │
                    └─────────┬─────────┘
                              │
                              ▼
                         Response
                              │
                              ▼
                      Evaluation
                              │
                              ▼
                       Feedback Loop
```

---

# The best interview answer

> **I would separate model behavior from enterprise knowledge. I would use SFT to teach the model the core task, domain behavior, instruction following, output format, and how to behave when context is present or missing. Then I would use DPO on preference pairs to improve answer quality, helpfulness, conciseness, and grounded behavior.**
>
> **I would not fine-tune dynamic enterprise knowledge into the model. Instead, I would keep documents in a RAG system using ingestion, chunking, embeddings, vector or hybrid search, metadata filtering, and reranking. At inference time, the RAG pipeline retrieves relevant context and passes it to the SFT+DPO-aligned model.**
>
> **For production, I would evaluate retrieval and generation separately using metrics such as Recall@K, MRR, faithfulness, correctness, and hallucination rate. I would also implement observability, RBAC, tenant isolation, caching, tracing, feedback collection, offline evaluation, and canary deployment.**
>
> **The key principle is: SFT teaches capability, DPO teaches preference, and RAG provides current and private knowledge.**

## Easy way to remember

```text
SFT → What should the model learn to do?

DPO → Which answers should the model prefer?

RAG → What information should the model use right now?
```

This is the architecture I would recommend for a real enterprise AI application rather than treating fine-tuning and RAG as competing alternatives.
