# Can you combine RAG and fine-tuning?

**Yes.** In fact, combining **Fine-tuning + RAG** is often one of the best architectures for enterprise LLM applications.

The simple idea is:

> **Fine-tuning teaches the model how to behave. RAG gives the model what it needs to know right now.**

---

# 1. The core difference

## Fine-tuning

Fine-tuning changes the behavior of the model.

```text
Base Model
    │
    ▼
Fine-tuning
    │
    ├── Company terminology
    ├── Response format
    ├── Coding style
    ├── Domain behavior
    └── Task specialization
```

Example:

```text
User:
Explain our customer churn.

Fine-tuned model learns:
Use company terminology.
Explain clearly.
Follow company-approved style.
```

---

## RAG

RAG provides **external, current knowledge** at runtime.

```text
User Question
      │
      ▼
Retrieve relevant documents
      │
      ▼
Add documents to prompt
      │
      ▼
LLM generates answer
```

Example:

```text
User:
What is our latest refund policy?

RAG retrieves:
refund_policy_2026.pdf

LLM:
Uses the latest policy to answer.
```

---

# 2. Why combine them?

Imagine an enterprise support assistant.

You want:

### Behavior

```text
Always be polite
Use company terminology
Follow response format
Classify support priority
Ask clarifying questions correctly
```

This is relatively stable.

Use:

```text
Fine-tuning
```

But you also need:

```text
Current policies
Latest product documentation
Customer-specific information
Current database records
New support articles
```

These change constantly.

Use:

```text
RAG
```

Therefore:

```text
             FINE-TUNING
                  │
                  │ teaches
                  ▼
          HOW TO RESPOND
                  │
                  │
                  ▼
User ───► Fine-tuned LLM ◄──── RAG
                              │
                              │ provides
                              ▼
                        WHAT TO KNOW
```

---

# 3. Production architecture

A realistic architecture:

```text
                        ┌─────────────────────┐
                        │    User Question    │
                        └──────────┬──────────┘
                                   │
                                   ▼
                         ┌───────────────────┐
                         │ Query Processing  │
                         │ - Auth            │
                         │ - Tenant          │
                         │ - Validation      │
                         └─────────┬─────────┘
                                   │
                                   ▼
                         ┌───────────────────┐
                         │ Query Embedding   │
                         └─────────┬─────────┘
                                   │
                                   ▼
                        ┌──────────────────────┐
                        │ Vector Database       │
                        │ Qdrant / pgvector     │
                        └──────────┬───────────┘
                                   │
                                   ▼
                        Relevant Documents
                                   │
                                   ▼
                         ┌───────────────────┐
                         │ Reranker          │
                         └─────────┬─────────┘
                                   │
                                   ▼
                          Relevant Context
                                   │
                                   │
                    ┌──────────────▼──────────────┐
                    │     Fine-tuned LLM           │
                    │                              │
                    │ Base Model + LoRA Adapter    │
                    └──────────────┬──────────────┘
                                   │
                                   ▼
                               Answer
                                   │
                                   ▼
                         Citation / Validation
```

---

# 4. A real-world example

Let's build a simplified enterprise assistant.

Assume the company has these documents:

```text
documents/
│
├── refund_policy.txt
├── customer_policy.txt
├── premium_support.txt
└── security_policy.txt
```

The model has been fine-tuned to:

* use company terminology
* produce structured answers
* refuse unsupported claims
* answer like a support agent

RAG provides the actual documents.

---

# 5. Step 1: Define the project structure

```text
enterprise_rag/
│
├── app/
│   ├── config.py
│   │
│   ├── models/
│   │   └── llm.py
│   │
│   ├── rag/
│   │   ├── chunker.py
│   │   ├── embeddings.py
│   │   ├── vector_store.py
│   │   ├── retriever.py
│   │   └── reranker.py
│   │
│   ├── services/
│   │   └── rag_service.py
│   │
│   └── api/
│       └── routes.py
│
├── data/
│   └── documents/
│
├── train/
│   └── fine_tune.py
│
└── requirements.txt
```

---

# 6. Step 2: The documents

Example:

```text
refund_policy.txt

Customers can request a refund within 30 days
of purchase.

Refunds are processed within 5 business days.

Premium users may contact priority support.
```

Another document:

```text
premium_support.txt

Premium Users receive priority support.

A Premium User is defined as a customer with
annual spending greater than ₹50,000.
```

---

# 7. Step 3: Chunk the documents

We should not send entire documents to the LLM.

```python
# app/rag/chunker.py

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)


def chunk_documents(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    return splitter.split_documents(
        documents
    )
```

Why overlap?

Suppose:

```text
Chunk 1:
Refunds are available within 30 days.

Chunk 2:
Refund processing takes 5 business days.
```

Without overlap, context may be separated.

Overlap helps preserve context:

```text
Chunk 1:
Refunds are available within 30 days.

Chunk 2:
Refunds are available within 30 days.
Refund processing takes 5 business days.
```

---

# 8. Step 4: Generate embeddings

```python
# app/rag/embeddings.py

from sentence_transformers import (
    SentenceTransformer
)


class EmbeddingService:

    def __init__(self):

        self.model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )


    def embed_text(
        self,
        text: str
    ) -> list[float]:

        embedding = self.model.encode(
            text
        )

        return embedding.tolist()


    def embed_query(
        self,
        query: str
    ) -> list[float]:

        return self.embed_text(
            query
        )
```

Example:

```text
"What is the refund period?"
            ↓
Embedding model
            ↓
[0.12, -0.44, 0.88, ...]
```

---

# 9. Step 5: Store embeddings in Qdrant

```python
# app/rag/vector_store.py

from qdrant_client import (
    QdrantClient
)

from qdrant_client.models import (
    PointStruct
)


class VectorStore:

    def __init__(self):

        self.client = QdrantClient(
            host="localhost",
            port=6333
        )

        self.collection = (
            "company_documents"
        )


    def upsert(
        self,
        documents,
        embeddings
    ):

        points = []

        for index, (
            document,
            embedding
        ) in enumerate(
            zip(
                documents,
                embeddings
            )
        ):

            point = PointStruct(

                id=index,

                vector=embedding,

                payload={
                    "text":
                        document.page_content,

                    "source":
                        document.metadata.get(
                            "source"
                        )
                }
            )

            points.append(
                point
            )

        self.client.upsert(
            collection_name=self.collection,
            points=points
        )
```

---

# 10. Step 6: Retrieve relevant context

```python
# app/rag/retriever.py


class Retriever:

    def __init__(
        self,
        vector_store,
        embedding_service
    ):

        self.vector_store = (
            vector_store
        )

        self.embedding_service = (
            embedding_service
        )


    def retrieve(
        self,
        query: str,
        limit: int = 5
    ):

        query_vector = (
            self.embedding_service
            .embed_query(query)
        )

        results = (
            self.vector_store
            .client
            .query_points(
                collection_name=
                    self.vector_store.collection,

                query=query_vector,

                limit=limit
            )
        )

        return results.points
```

User asks:

```text
What is the refund period?
```

Retriever returns:

```text
Chunk 1:
Customers can request a refund within 30 days.

Chunk 2:
Refunds are processed within 5 business days.
```

---

# 11. Step 7: Fine-tune the model

Now we train the model to use retrieved context properly.

This is important.

A normal fine-tuning example might be:

```text
User:
What is our refund period?

Assistant:
Customers can request a refund within 30 days.
```

But for a RAG system, a better dataset teaches the model to work with context.

Example:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a company support assistant. Answer only using the provided context. If the answer is not present, say that you do not have enough information."
    },
    {
      "role": "user",
      "content": "Context:\nCustomers can request a refund within 30 days of purchase.\n\nQuestion:\nWhat is the refund period?"
    },
    {
      "role": "assistant",
      "content": "Customers can request a refund within 30 days of purchase."
    }
  ]
}
```

The fine-tuned model learns:

```text
Context
   ↓
Extract relevant facts
   ↓
Answer correctly
```

---

# 12. Fine-tuning dataset for RAG behavior

Let's create examples.

```python
import json
```

```python
training_examples = [

    {
        "messages": [

            {
                "role": "system",

                "content": (
                    "You are an enterprise "
                    "support assistant. "
                    "Answer only using the "
                    "provided context. "
                    "If information is missing, "
                    "say you do not have enough "
                    "information."
                )
            },

            {
                "role": "user",

                "content": """
Context:

Customers can request a refund within
30 days of purchase.

Question:

What is the refund period?
"""
            },

            {
                "role": "assistant",

                "content": (
                    "Customers can request a "
                    "refund within 30 days "
                    "of purchase."
                )
            }
        ]
    },


    {
        "messages": [

            {
                "role": "system",

                "content": (
                    "Answer only using the "
                    "provided context."
                )
            },

            {
                "role": "user",

                "content": """
Context:

Premium Users have annual spending
greater than ₹50,000.

Question:

Who qualifies as a Premium User?
"""
            },

            {
                "role": "assistant",

                "content": (
                    "A Premium User is a "
                    "customer whose annual "
                    "spending is greater "
                    "than ₹50,000."
                )
            }
        ]
    }
]
```

Save:

```python
with open(
    "rag_finetuning.jsonl",
    "w",
    encoding="utf-8"
) as file:

    for example in training_examples:

        file.write(
            json.dumps(example)
            + "\n"
        )
```

---

# 13. Fine-tune with LoRA

```python
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

from peft import (
    LoraConfig,
    get_peft_model
)

from trl import (
    SFTTrainer,
    SFTConfig
)
```

Load:

```python
MODEL_NAME = "your-base-instruct-model"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME
)
```

LoRA:

```python
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

Apply:

```python
model = get_peft_model(
    model,
    lora_config
)
```

Format dataset:

```python
def format_example(example):

    text = tokenizer.apply_chat_template(

        example["messages"],

        tokenize=False,

        add_generation_prompt=False
    )

    return {
        "text": text
    }
```

Training configuration:

```python
training_args = SFTConfig(

    output_dir="./rag_finetuned_model",

    num_train_epochs=2,

    learning_rate=2e-4,

    per_device_train_batch_size=2,

    gradient_accumulation_steps=8,

    max_length=2048,

    warmup_ratio=0.05,

    lr_scheduler_type="cosine",

    bf16=True,

    logging_steps=10,

    report_to="none"
)
```

Trainer:

```python
trainer = SFTTrainer(

    model=model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=validation_dataset,

    processing_class=tokenizer,

    dataset_text_field="text"
)
```

Train:

```python
trainer.train()
```

Save:

```python
trainer.save_model(
    "./rag_lora_adapter"
)
```

---

# 14. Step 8: Build the final prompt

This is where RAG and fine-tuning actually meet.

```python
def build_prompt(
    question: str,
    documents: list[str]
):

    context = "\n\n".join(

        [
            f"[Source {index + 1}]\n{doc}"

            for index, doc
            in enumerate(documents)
        ]
    )

    return f"""
Use only the provided context to answer.

If the answer is not present in the context,
say:

"I don't have enough information in the
provided documents."

Context:
{context}

Question:
{question}

Answer:
"""
```

---

# 15. Complete RAG + fine-tuned model service

```python
# app/services/rag_service.py


class RAGService:

    def __init__(
        self,
        retriever,
        tokenizer,
        model
    ):

        self.retriever = retriever

        self.tokenizer = tokenizer

        self.model = model


    def answer(
        self,
        question: str
    ):

        # ----------------------
        # 1. Retrieve
        # ----------------------

        results = (
            self.retriever.retrieve(
                query=question,
                limit=5
            )
        )


        # ----------------------
        # 2. Extract documents
        # ----------------------

        documents = [

            result.payload["text"]

            for result in results
        ]


        # ----------------------
        # 3. Build context
        # ----------------------

        context = "\n\n".join(
            documents
        )


        # ----------------------
        # 4. Create messages
        # ----------------------

        messages = [

            {
                "role": "system",

                "content": (
                    "You are a company support "
                    "assistant. Use only the "
                    "provided context. "
                    "Do not invent information."
                )
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


        # ----------------------
        # 5. Tokenize
        # ----------------------

        inputs = (
            self.tokenizer
            .apply_chat_template(

                messages,

                add_generation_prompt=True,

                return_tensors="pt"
            )
            .to(self.model.device)
        )


        # ----------------------
        # 6. Generate
        # ----------------------

        outputs = self.model.generate(

            inputs,

            max_new_tokens=512,

            do_sample=False
        )


        # ----------------------
        # 7. Decode
        # ----------------------

        new_tokens = outputs[
            0,
            inputs.shape[1]:
        ]

        answer = (
            self.tokenizer.decode(

                new_tokens,

                skip_special_tokens=True
            )
        )


        return {

            "answer": answer,

            "sources": [

                result.payload.get(
                    "source"
                )

                for result in results
            ]
        }
```

This is the core combination:

```text
             RAG
              │
              ▼
        Retrieved Context
              │
              ▼
      ┌─────────────────┐
      │ Fine-tuned LLM  │
      │                 │
      │ Learned behavior│
      └────────┬────────┘
               │
               ▼
             Answer
```

---

# 16. Example request flow

User:

```text
What benefits do Premium Users receive?
```

### Step 1: Retrieve

Qdrant finds:

```text
Premium Users receive priority support.

Premium Users are customers with annual
spending greater than ₹50,000.
```

### Step 2: Build prompt

```text
SYSTEM:
You are a company assistant.
Answer only from context.

CONTEXT:
Premium Users receive priority support.

Premium Users are customers with annual
spending greater than ₹50,000.

QUESTION:
What benefits do Premium Users receive?
```

### Step 3: Fine-tuned model responds

```text
Premium Users receive priority support.
```

The **RAG system provides the fact**.

The **fine-tuned model knows how to answer according to company behavior and terminology**.

---

# 17. Add a reranker

Vector search can retrieve:

```text
Top 10 approximate matches
```

But not all are equally relevant.

Use:

```text
Query
  │
  ▼
Vector Search (Top 20)
  │
  ▼
Reranker
  │
  ▼
Top 5 highly relevant chunks
```

Example using a cross-encoder:

```python
from sentence_transformers import (
    CrossEncoder
)


class Reranker:

    def __init__(self):

        self.model = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )


    def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int = 5
    ):

        pairs = [

            (query, document)

            for document in documents
        ]

        scores = self.model.predict(
            pairs
        )

        ranked = sorted(

            zip(
                documents,
                scores
            ),

            key=lambda item: item[1],

            reverse=True
        )

        return ranked[:top_k]
```

---

# 18. Add metadata filtering

In an enterprise system, you must prevent cross-tenant data leakage.

Example:

```text
tenant_id = company_A
```

The user must only retrieve:

```text
tenant_id = company_A
```

Never:

```text
tenant_id = company_B
```

Conceptually:

```python
results = client.query_points(

    collection_name="documents",

    query=query_vector,

    query_filter={
        "must": [
            {
                "key": "tenant_id",

                "match": {
                    "value": tenant_id
                }
            }
        ]
    },

    limit=10
)
```

Production architecture:

```text
User
 │
 ▼
Authentication
 │
 ▼
Extract Tenant ID
 │
 ▼
Vector Search
 │
 ▼
Filter:
tenant_id
permissions
document classification
 │
 ▼
Only authorized documents
```

---

# 19. Fine-tuning should not memorize your documents

This is one of the biggest mistakes.

Bad approach:

```text
10,000 company documents
        ↓
Fine-tune model
        ↓
Hope it remembers everything
```

Problems:

```text
Information changes
Need retraining
No citations
Can hallucinate
Expensive
Hard to delete information
```

Better:

```text
Stable patterns
      ↓
Fine-tuning

Dynamic documents
      ↓
RAG
```

---

# 20. What should be fine-tuned?

Good candidates:

```text
✓ Company terminology
✓ Response style
✓ Output JSON format
✓ Support workflow
✓ Classification behavior
✓ Tool selection
✓ SQL generation style
✓ Code generation patterns
✓ Domain-specific reasoning patterns
```

---

# 21. What should go into RAG?

```text
✓ Policies
✓ Product documentation
✓ Internal knowledge base
✓ Current pricing
✓ Current regulations
✓ Frequently updated data
✓ Customer information
✓ Current API documentation
```

---

# 22. Advanced production architecture

For a real enterprise application:

```text
                           User
                            │
                            ▼
                     FastAPI Gateway
                            │
                    ┌───────┴────────┐
                    ▼                ▼
                  Auth              Rate Limit
                    │
                    └────────┬───────┘
                             ▼
                      Query Service
                             │
             ┌───────────────┼────────────────┐
             ▼               ▼                ▼
        Query Router      Cache         Query Classifier
             │               │                │
             └───────────────┼────────────────┘
                             ▼
                       Hybrid Search
                   ┌─────────┴─────────┐
                   ▼                   ▼
              Vector Search       Keyword Search
                   │                   │
                   └─────────┬─────────┘
                             ▼
                           Reranker
                             │
                             ▼
                      Context Builder
                             │
                             ▼
                   Fine-tuned LLM
                    (LoRA / QLoRA)
                             │
                             ▼
                     Output Validator
                             │
                    ┌────────┴─────────┐
                    ▼                  ▼
                Response           Citations
```

---

# 23. Interview-ready answer

> **Yes, I would combine RAG and fine-tuning because they solve different problems. Fine-tuning adapts the model's behavior, style, terminology, output format, and domain-specific task patterns. RAG provides current, factual, and enterprise-specific knowledge at inference time.**
>
> **I would first ingest documents, clean and chunk them, generate embeddings, and store them in a vector database such as Qdrant. At query time, I would embed the user's question, retrieve relevant chunks using vector or hybrid search, optionally rerank them, and add the best context to the prompt.**
>
> **The final prompt would be sent to a fine-tuned model, typically using a LoRA or QLoRA adapter. The model would be trained to correctly use provided context, follow company terminology, and say it lacks sufficient information when the answer is not supported.**
>
> **In production, I would add metadata filtering for multi-tenancy and RBAC, caching, observability, hallucination checks, output validation, citations, and evaluation metrics such as context precision, context recall, faithfulness, and answer correctness.**

## Best one-line answer

> **RAG provides the model with current knowledge; fine-tuning teaches the model how to use that knowledge and how to behave.**

For an enterprise production system, the combination is usually:

```text
Fine-tuned Model
      +
RAG
      +
Reranking
      +
Authorization
      +
Output Validation
      +
Evaluation
      =
Production Enterprise AI System
```
