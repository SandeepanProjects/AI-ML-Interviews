This is another **top interview topic** for Senior AI/ML Engineer, LLM Engineer, and Applied Scientist roles.

Interviewers are not looking for:

> "Embeddings convert words into vectors."

They expect you to explain:

* Why embeddings were invented
* Why one-hot encoding fails
* How embedding matrices work internally
* Static vs contextual embeddings
* Why BERT changed embeddings forever
* What the `[CLS]` token is and why it exists
* Code implementation from scratch
* Tensor shapes at every step

Let's build embeddings from first principles.

---

# Why Do We Need Embeddings?

Suppose we have a vocabulary:

```text
Dog
Cat
Apple
Car
```

Neural networks cannot understand words.

The simplest idea is to assign IDs.

```text
Dog   → 0
Cat   → 1
Apple → 2
Car   → 3
```

Can we feed these IDs into a neural network?

No.

Why?

Because the network assumes numbers have mathematical meaning.

For example

```text
Dog = 0

Cat = 1

Apple = 2
```

The network might incorrectly learn

```text
Apple > Dog
```

or

```text
Cat ≈ Dog
```

simply because

```text
1 is closer to 0
```

These IDs have **no semantic meaning**.

---

# One-Hot Encoding

The first solution was One-Hot Encoding.

Vocabulary

```text
Dog
Cat
Apple
Car
```

Represent each word

```text
Dog

[1 0 0 0]
```

```text
Cat

[0 1 0 0]
```

```text
Apple

[0 0 1 0]
```

```text
Car

[0 0 0 1]
```

Every word is unique.

Great.

Problem solved?

Not even close.

---

# Why One-Hot Fails

Imagine vocabulary

```text
50,000 words
```

Every vector becomes

```text
50000 dimensions
```

For example

```text
Dog

[0 0 0 0 0 ... 1 ... 0]
```

Mostly zeros.

Very sparse.

Memory waste.

---

Even worse

Distance

```text
Dog

[1 0 0 0]
```

Cat

```text
[0 1 0 0]
```

Cosine similarity

```text
0
```

Dog and Cat appear completely unrelated.

Likewise

```text
Dog

Car

Apple
```

All equally different.

The model learns no semantic relationships.

---

# We Need Dense Representations

Instead of

```text
Dog

[1 0 0 0 0 0 ...]
```

Use

```text
Dog

[
0.81
-0.43
1.20
...
]
```

Suppose

```text
Embedding Size

768
```

Now every word becomes

```text
768 numbers
```

Those numbers are learned during training.

---

# Intuition

Imagine

```text
Dog
```

Embedding

```text
[
0.83
0.24
-1.12
...
]
```

Cat

```text
[
0.80
0.20
-1.05
...
]
```

Very similar.

Apple

```text
[
-0.9
1.4
0.2
...
]
```

Very different.

Now

Distance

actually means

Semantic similarity.

---

# Embedding Matrix

This is the heart of embeddings.

Suppose

Vocabulary

```text
50000 words
```

Embedding Dimension

```text
768
```

Embedding Matrix

```text
50000 × 768
```

Think of it like a huge lookup table.

```text
             768 numbers
          ──────────────────────►

Word0

Word1

Word2

Word3

...

Word49999
```

Each row

is one word.

---

Example

Vocabulary

```text
Dog

Cat

Apple

Car
```

Matrix

```text
Dog

0.3 0.5 -1.2

Cat

0.2 0.6 -1.1

Apple

1.8 -0.4 0.7

Car

-0.5 1.1 0.2
```

---

# Internally

Suppose

```text
Dog

Token ID

15
```

Embedding Layer

simply returns

```text
Row 15
```

Nothing more.

No multiplication.

No neural network.

Just

```text
Lookup
```

---

PyTorch

```python
import torch
import torch.nn as nn

embedding = nn.Embedding(
    num_embeddings=50000,
    embedding_dim=768
)

tokens = torch.tensor([
    [15,29,100]
])

x = embedding(tokens)

print(x.shape)
```

Output

```text
torch.Size([1,3,768])
```

Meaning

```text
Batch =1

Sequence =3

Embedding =768
```

---

# How Are Embeddings Learned?

Many people think embeddings are predefined.

They are not.

Initially

```python
Embedding Matrix

Random Numbers
```

Example

```text
Dog

0.01

-0.08

0.12
```

During training

Loss

↓

Backpropagation

↓

Embedding Updates

Eventually

```text
Dog

0.81

-0.22

1.15
```

Similar words gradually move closer together.

---

Example

Sentence

```text
Dogs chase cats
```

Prediction wrong.

Gradient

flows

all the way back

into

Embedding Matrix.

Embeddings improve.

---

# Static Embeddings

Before Transformers

People used

* Word2Vec
* GloVe
* FastText

These produce one vector

per word.

Example

```text
Bank
```

Always

```text
[
0.82
-1.1
...
]
```

Even here

```text
River bank
```

Same vector.

And here

```text
Bank account
```

Still

same vector.

Impossible.

---

# Static Embedding Problem

Sentence

```text
I sat near the bank.
```

Embedding

```text
Bank

Vector A
```

Sentence

```text
I deposited money in the bank.
```

Still

```text
Bank

Vector A
```

Meaning changed.

Embedding didn't.

---

# Contextual Embeddings

Transformers solved this.

Word meaning depends on context.

Sentence

```text
River bank
```

After Self-Attention

```text
Bank

Vector

[
0.2
1.4
...
]
```

Sentence

```text
Bank loan
```

Different vector

```text
[
-1.3
0.6
...
]
```

Same word.

Different meaning.

Different embedding.

This is called

Contextual Embedding.

---

# How Does BERT Create Contextual Embeddings?

Input

```text
The bank approved my loan.
```

Embedding Layer

↓

```text
Bank

Random Learned Vector
```

After

12 Transformer Layers

↓

```text
Bank

Financial Meaning
```

Different sentence

```text
The fisherman sat on the bank.
```

After Attention

↓

```text
Bank

River Meaning
```

The embedding changes because surrounding words change the attention weights.

---

# Static vs Contextual

| Static Embeddings            | Contextual Embeddings               |
| ---------------------------- | ----------------------------------- |
| One vector per word          | One vector per occurrence           |
| Context ignored              | Context incorporated                |
| Word2Vec, GloVe, FastText    | BERT, GPT, Llama                    |
| Fast                         | More expensive                      |
| Cannot disambiguate polysemy | Handles multiple meanings naturally |

---

# The `[CLS]` Token

One of the most common BERT interview questions.

Suppose

```text
I love Transformers
```

BERT actually receives

```text
[CLS]

I

love

Transformers

[SEP]
```

Question

Why add

```text
[CLS]
```

It is not a real word.

---

# What Does `[CLS]` Do?

During self-attention

Every token talks to every other token.

Including

```text
[CLS]
```

Initially

```text
[CLS]

Random Vector
```

After many Transformer layers

```text
[CLS]

Contains information about

the entire sentence.
```

Think of it as

A summary vector.

---

Visualization

```text
Sentence

[CLS]

The

Movie

Was

Amazing
```

Attention

```text
CLS

↔ The

↔ Movie

↔ Was

↔ Amazing
```

After many layers

CLS has aggregated information from every token.

---

# Why Use `[CLS]`?

Suppose task

```text
Positive

Negative
```

Instead of averaging

all token embeddings

BERT simply uses

```text
CLS Vector
```

Feed into classifier

```text
CLS

↓

Linear Layer

↓

Softmax

↓

Positive
```

---

# PyTorch Example

```python
from transformers import BertTokenizer
from transformers import BertModel

import torch

tokenizer = BertTokenizer.from_pretrained(
    "bert-base-uncased"
)

model = BertModel.from_pretrained(
    "bert-base-uncased"
)

inputs = tokenizer(
    "I love transformers",
    return_tensors="pt"
)

outputs = model(**inputs)

last_hidden = outputs.last_hidden_state

print(last_hidden.shape)
```

Output

```text
torch.Size([1,6,768])
```

Sequence

```text
CLS

I

love

transformers

SEP
```

Every token

has

768 dimensions.

---

Extract CLS

```python
cls_embedding = last_hidden[:,0,:]

print(cls_embedding.shape)
```

Output

```text
torch.Size([1,768])
```

That single vector represents the whole sequence.

---

# From Scratch — Simple Embedding Layer

```python
import torch
import torch.nn as nn

class SimpleEmbedding(nn.Module):

    def __init__(self, vocab_size, d_model):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            d_model
        )

    def forward(self, token_ids):
        return self.embedding(token_ids)

model = SimpleEmbedding(
    vocab_size=50000,
    d_model=768
)

tokens = torch.tensor([[10, 15, 22, 41]])

embeddings = model(tokens)

print(embeddings.shape)
```

Output

```text
torch.Size([1,4,768])
```

---

# End-to-End Flow

```text
Text
   │
   ▼
Tokenizer
   │
   ▼
Token IDs
   │
   ▼
Embedding Lookup
   │
   ▼
(Batch, Sequence, Hidden Size)
   │
   ▼
+ Positional Encoding
   │
   ▼
Transformer Layers
   │
   ▼
Contextual Embeddings
   │
   ├── Token embeddings for token-level tasks
   │
   └── [CLS] embedding for sequence-level tasks (in BERT-style models)
```

---

# Senior AI Engineer Interview Answer (2–3 minutes)

> "Embeddings are dense, learnable vector representations that map discrete token IDs into continuous semantic spaces. Unlike one-hot encoding, which is sparse and cannot capture relationships between words, embeddings allow semantically similar words to occupy nearby regions in the vector space. An embedding layer is implemented as a learnable lookup table of shape `(vocabulary_size × embedding_dimension)`, and backpropagation updates its rows during training. Early approaches such as Word2Vec and GloVe produced static embeddings, meaning each word always had the same vector regardless of context. Transformer-based models introduced contextual embeddings, where self-attention updates each token's representation based on surrounding words, allowing the same token—such as *bank*—to have different vectors in financial and river contexts. In BERT, the special `[CLS]` token participates in self-attention like every other token, and its final hidden state becomes a learned summary representation of the entire sequence, making it suitable for sentence-level tasks such as classification.

This is one of the **most important production RAG interview questions**.

Many candidates answer:

> "I use `all-MiniLM-L6-v2`."

That is **not** a senior-level answer.

A Senior AI Engineer explains:

* Why embeddings exist
* How embedding models work internally
* How to choose an embedding model
* What metrics matter
* Trade-offs
* Production deployment decisions
* Real-world examples
* Implementation

---

# Interview Question

> **How do you choose an embedding model for RAG?**

A senior answer is:

> "The embedding model determines how documents and queries are represented in vector space. Choosing the wrong embedding model results in poor retrieval regardless of the LLM quality. I select embedding models based on semantic accuracy, domain coverage, multilingual support, embedding dimension, latency, cost, and compatibility with my vector database."

Let's understand why.

---

# Why Do We Need an Embedding Model?

Suppose our document says

```text
Employees receive 20 annual leaves.
```

User asks

```text
Vacation policy
```

Keyword search

```text
Vacation
```

≠

```text
Annual Leave
```

SQL search fails.

Embedding model converts both sentences into vectors.

```text
Document
      │
      ▼
Embedding Model
      │
      ▼
[0.24, -1.12, 0.55, ...]
```

Query

```text
Vacation Policy
```

↓

```text
[0.28,-1.05,0.59,...]
```

The vectors become close because they have similar meanings.

---

# Internally How Embedding Models Work

Suppose input

```text
Employees receive annual leave.
```

Flow

```text
Sentence
      │
      ▼
Tokenizer
      │
      ▼
Token IDs
      │
      ▼
Transformer Encoder
      │
      ▼
Hidden States
      │
      ▼
Pooling
      │
      ▼
Embedding Vector
```

Notice

Unlike GPT

Embedding models stop here.

They don't generate text.

They only produce vectors.

---

# Example

Sentence

```text
The employee gets annual leave.
```

Tokenizer

↓

```text
[
The,
employee,
gets,
annual,
leave
]
```

Embedding Layer

↓

```text
768-dimensional vectors
```

Transformer Layers

↓

```text
Contextual representations
```

Pooling

↓

```text
384-dimensional vector
```

That vector is stored.

---

# What Makes a Good Embedding Model?

A good embedding model should place semantically similar sentences close together.

Example

Sentence A

```text
Employee leave policy
```

Sentence B

```text
Vacation rules
```

Desired similarity

```text
0.94
```

Sentence C

```text
Pizza recipe
```

Desired similarity

```text
0.08
```

A good model creates this separation.

---

# Factors for Choosing an Embedding Model

There is **no single best embedding model**.

It depends on your application.

Let's examine every factor.

---

# 1. Domain

The first question I ask is:

> **What type of documents am I embedding?**

Examples

Healthcare

```text
Diagnosis

Medication

Prescription

Radiology
```

Legal

```text
Contracts

Court Orders

Case Laws
```

Finance

```text
Invoices

Tax

Trading

Balance Sheet
```

General-purpose embeddings may struggle with specialized terminology.

Sometimes you need domain-specific models or fine-tuned embeddings.

---

# Example

General model

```text
Transformer
```

Could mean

```text
Electrical Equipment
```

or

```text
Neural Network
```

A domain-specific model better captures the intended meaning.

---

# 2. Embedding Dimension

Every embedding has a size.

Examples

```text
384

768

1024

1536

3072
```

Question

Should we always choose larger vectors?

No.

---

Smaller vectors

```text
384
```

Advantages

* Faster search
* Less RAM
* Smaller index
* Lower storage

Disadvantages

Slightly lower accuracy.

---

Larger vectors

```text
3072
```

Advantages

* Richer representation
* Better semantic accuracy

Disadvantages

* Higher latency
* More storage
* Larger vector indexes
* More expensive ANN search

---

Example

Suppose

One million chunks.

384 dimensions

```text
1M × 384
```

Memory

≈

1.5 GB (float32)

1536 dimensions

```text
1M × 1536
```

Memory

≈

6 GB

Same data.

4× memory.

---

# 3. Retrieval Accuracy

This is the most important metric.

Suppose

User asks

```text
Annual Leave
```

Top Results

Model A

```text
Leave Policy

Travel Policy

Insurance
```

Model B

```text
Leave Policy

Vacation Rules

Carry Forward
```

Model B is better because the retrieved documents are more relevant.

---

Evaluation Metrics

Production teams measure

* Recall@K
* Precision@K
* MRR (Mean Reciprocal Rank)
* nDCG
* Hit Rate

Don't choose models only by benchmark scores—evaluate on your own documents.

---

# 4. Latency

Suppose

You have

```text
50 million documents
```

Embedding speed matters.

Large models

```text
300 ms/document
```

Small models

```text
20 ms/document
```

For ingestion,

speed matters.

For live queries,

query latency matters.

---

# 5. Multilingual Support

Suppose documents contain

```text
English

French

German

Hindi

Japanese
```

Need multilingual embeddings.

Otherwise

English query

↓

Japanese document

↓

Poor similarity.

---

# Example

English

```text
Annual Leave
```

Japanese

```text
有給休暇
```

A multilingual embedding model can place these close together in vector space.

---

# 6. Cost

Open-source model

```text
Run locally
```

Pros

* Free inference
* Data stays private
* No API costs

Cons

Need GPUs or CPUs to host.

---

API embeddings

Pros

* Easy to use
* Strong quality
* Managed infrastructure

Cons

* Ongoing cost
* Network latency
* Data governance considerations

---

# 7. Context Length

Some embedding models support

```text
512 tokens
```

Others

```text
8000 tokens
```

Suppose chunk size

```text
3000 tokens
```

Model supports

```text
512
```

Text gets truncated.

Information lost.

Always align your chunk size with the embedding model's maximum input length.

---

# 8. Compatibility

The **same embedding model** must usually be used for:

* Document embeddings
* Query embeddings

Why?

Because both vectors must live in the **same vector space**.

Document

```text
Annual Leave
```

↓

Model A

↓

```text
[0.3,-0.5,...]
```

Query

```text
Vacation Policy
```

↓

Model B

↓

```text
[-0.8,1.1,...]
```

Different vector spaces.

Similarity becomes meaningless.

---

# Popular Embedding Models

| Model                           | Dimension | Best For                       |
| ------------------------------- | --------- | ------------------------------ |
| `all-MiniLM-L6-v2`              | 384       | Fast local RAG, prototypes     |
| `bge-base-en-v1.5`              | 768       | High-quality English retrieval |
| `bge-large-en-v1.5`             | 1024      | Enterprise English RAG         |
| `e5-base-v2`                    | 768       | Query-document retrieval       |
| `e5-large-v2`                   | 1024      | Higher retrieval accuracy      |
| OpenAI `text-embedding-3-small` | 1536      | Managed API, strong quality    |
| OpenAI `text-embedding-3-large` | 3072      | Maximum semantic quality       |

---

# Example Using Sentence Transformers

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "BAAI/bge-base-en-v1.5"
)

documents = [
    "Employees receive 20 annual leaves.",
    "Medical insurance covers family."
]

vectors = model.encode(
    documents,
    normalize_embeddings=True
)

print(vectors.shape)
```

Output

```text
(2, 768)
```

---

# Similarity Search Example

```python
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer(
    "BAAI/bge-base-en-v1.5"
)

docs = [
    "Employees receive annual leave.",
    "Medical insurance is provided."
]

query = "Vacation policy"

doc_vectors = model.encode(
    docs,
    normalize_embeddings=True
)

query_vector = model.encode(
    [query],
    normalize_embeddings=True
)

scores = cosine_similarity(
    query_vector,
    doc_vectors
)

print(scores)
```

Example output

```text
[[0.91 0.18]]
```

The first document is much more relevant.

---

# Production Considerations

In a production RAG system, choosing an embedding model also involves infrastructure trade-offs.

```text
Documents
      │
      ▼
Embedding Service
      │
      ▼
Vector DB
      │
      ▼
Retriever
```

Questions I ask:

* Can I run the model on CPU, or do I need GPUs?
* How many documents will I index?
* How often will new documents arrive?
* What's the acceptable query latency?
* Do I need multilingual support?
* Is data allowed to leave my environment?
* What's my storage budget for vectors?

---

# Decision Matrix

| Requirement                       | Recommended Choice                           |
| --------------------------------- | -------------------------------------------- |
| Fast local prototype              | `all-MiniLM-L6-v2`                           |
| Best English enterprise retrieval | `bge-large-en-v1.5`                          |
| Strong query-document retrieval   | `e5-large-v2`                                |
| Multilingual RAG                  | Multilingual E5 or multilingual BGE variants |
| Highest managed quality           | OpenAI `text-embedding-3-large`              |
| Strict privacy/on-premises        | Open-source Sentence Transformers            |

---

# Senior AI Engineer Interview Answer (3–5 Minutes)

> "Choosing an embedding model is one of the most important decisions in a RAG system because retrieval quality depends on the embedding space. I first evaluate the document domain, whether it's general, legal, medical, or financial, since domain-specific terminology can significantly affect retrieval performance. Next, I consider embedding dimension because larger vectors generally improve semantic representation but increase storage, memory usage, and vector search latency. I also evaluate retrieval quality using metrics such as Recall@K, Precision@K, MRR, and nDCG on my own dataset rather than relying solely on public benchmarks. Additional considerations include multilingual support, maximum input length, inference latency, infrastructure cost, and data privacy. In production, I always ensure that both document and query embeddings are generated by the same model so they occupy the same vector space. For a lightweight local system I might choose `all-MiniLM-L6-v2`, while for enterprise-grade English retrieval I'd typically use a BGE or E5 model, and for managed cloud deployments I would consider high-quality API-based embedding models."
