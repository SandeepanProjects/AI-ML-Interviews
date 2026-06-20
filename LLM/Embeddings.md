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
