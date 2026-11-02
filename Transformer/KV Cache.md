# KV Cache in Transformers — explained simply and in depth

KV Cache stands for Key-Value Cache. It is an optimization technique used during LLM inference (text generation) to avoid recalculating attention keys and values for tokens that the model has already processed.

It is one of the most important techniques behind making GPT, Llama, and other decoder-only LLMs generate text efficiently.

Let's understand it step by step, from the basic problem to the production implementation.

## 1. First, understand how GPT generates text

Suppose we give a model this prompt:

> Explain artificial intelligence

The model generates text one token at a time:

1. Input: `Explain artificial intelligence`
2. Output: `AI`
3. Next output: `is`
4. Next output: `a`
5. Next output: `technology`
6. And so on.

This is called autoregressive generation.

At every generation step, the Transformer uses self-attention to understand the relationship between the current token and previous tokens.

Remember the attention formula:

\\[ Attention(Q,K,V)=Softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V \\]

Where:

- Q (Query): What information is the current token looking for?
- K (Key): What information does each token represent for matching?
- V (Value): What information does each token provide?

The problem is that, without caching, the model repeatedly calculates keys and values for the previous tokens.

Let's see why that is inefficient.

## 2. The problem without KV Cache

Imagine the model generates the following sequence:

`I → love → artificial → intelligence`

At each generation step, it processes the entire sequence generated so far.

Without KV Cache

Step 1 — Generate "I"

`I` → Calculate Q, K, V

Step 2 — Generate "love"

`I love` → Recalculate Q, K, V for both tokens

Step 3 — Generate "artificial"

`I love artificial` → Recalculate Q, K, V for all three tokens

Step 4 — Generate "intelligence"

`I love artificial intelligence` → Recalculate Q, K, V for all four tokens

Blue blocks represent the tokens processed again at each step.

Notice the repeated work:

- At step 1, calculate K and V for `I`.
- At step 2, calculate K and V for `I` again, even though they have not changed.
- At step 3, calculate them again for `I` and `love`.
- At step 4, calculate them again for `I`, `love`, and `artificial`.

Why is this wasteful?

Because the Key and Value representations of previous tokens do not need to be recalculated when we append a new token. Their inputs and positions have not changed.

This is the problem KV Cache solves.

## 3. What is KV Cache, and how does it solve the problem?

Instead of recalculating Key and Value for previous tokens, we calculate them once and store them in memory.

When a new token arrives:

1. Calculate Query, Key, and Value for the new token.
2. Store its Key and Value in the cache.
3. Use the new Query to attend to all cached Keys and Values.
4. Generate the next token.

The important point is that we still calculate a new Query for the current token. We reuse the previous tokens' Keys and Values.

With KV Cache

Step 1 — Generate "I"

Calculate Q, K, V for `I`.

Cache stores K₁, V₁

Step 2 — Generate "love"

Calculate Q₂, K₂, V₂ only for `love`.

Reuse K₁, V₁; store K₂, V₂

Step 3 — Generate "artificial"

Calculate Q₃, K₃, V₃ only for `artificial`.

Reuse previous cache; store K₃, V₃

Step 4 — Generate "intelligence"

Calculate Q₄, K₄, V₄ only for `intelligence`.

Reuse previous cache; store K₄, V₄

Green blocks show that previous K/V values are reused instead of recalculated.

So, at the fourth step:

- New Query: \\(Q_4\\)
- Cached Keys: \\(K_1,K_2,K_3\\)
- New Key: \\(K_4\\)
- Cached Values: \\(V_1,V_2,V_3\\)
- New Value: \\(V_4\\)

The attention calculation is effectively:

\\[ Attention(Q_4,[K_1,K_2,K_3,K_4], [V_1,V_2,V_3,V_4]) \\]

The model can now attend to all previous tokens without recomputing their K and V projections.

## 4. Where exactly is KV Cache stored?

This is an important interview question.

A Transformer has multiple layers. For example, imagine a GPT-style model with 32 Transformer layers.

Each layer has its own self-attention mechanism, and each layer maintains its own Key and Value cache.

[Large Language Model — LLM Model Efficient Inference | by Ling Huang | Medium](https://images.openai.com/static-rsc-4/oZDiog8S7t1T9RiNkCJiCErXSWAec4HlKYZe3sM988ZsnrPiWsGVy-Jc3Gxh3kL0FH0MD6mwXhk2dQ-a65rY90gKvNsPTqCky0BYrSOCtZ_SwBvsxiDjcqIhk0fpW8eFgN_QQxo6pXDQm_7YZKxF03GCkK1mt55y_glZND5aX-U?purpose=inline)

[medium.com](https://medium.com/%40linghuang_76674/large-language-model-llm-model-efficient-inference-c9b05fce9ec1?utm_source=chatgpt.com)

Conceptually, the cache looks like this:

```
Transformer
│
├── Layer 1
│   ├── Key Cache
│   └── Value Cache
│
├── Layer 2
│   ├── Key Cache
│   └── Value Cache
│
├── Layer 3
│   ├── Key Cache
│   └── Value Cache
│
│   ...
│
└── Layer 32
    ├── Key Cache
    └── Value Cache
```

Why does every layer need its own cache?

Because every Transformer layer produces different hidden representations.

For example:

- Layer 1 produces its own K and V.
- Layer 2 receives the output of Layer 1 and produces different K and V.
- Layer 3 receives the output of Layer 2 and produces another set of K and V.

Therefore, we cannot use Layer 1's cached Keys and Values as a replacement for Layer 2's cache.

In real implementations, these tensors are generally stored in GPU memory because the attention computation is performed on the GPU.

## 5. Understand the two phases: Prefill and Decode

This is especially important for Senior AI Engineer interviews.

LLM inference normally has two major phases.

Phase 1

Prefill phase

The user sends a prompt:

`Explain how transformers work`

The model processes all prompt tokens together and computes their representations, including Keys and Values at every layer.

These Keys and Values are stored in the KV Cache.

[Wafer-Scale AI Compute: A System Software Perspective | USENIX](https://images.openai.com/static-rsc-4/szlZ1ZbRe9E9JxB6Wr-vmrBYT_pEDn4vtcyWByQmtzxStIQWYZRBfIZQIQnwXGIETw2dxesToRruZsxov8Nqrx9j3qvurx3nch7WSXWSGN6D7XCrnXtkZ5_CmQ2g264MvIxFomKioaOXgJGWU_Ze7ULH7zp-roKHbf3X9LsS8yo?purpose=inline)

[usenix.org](https://www.usenix.org/publications/loginonline/wafer-scale-ai-compute-system-software-perspective?utm_source=chatgpt.com)

Phase 2

Decode phase

The model generates tokens one by one.

For every new token:

- Compute its Q, K and V.
- Append its K and V to the cache.
- Attend to the cached K and V.
- Predict the next token.

[KV Caching in Autoregressive Transformers - FeynmanWiki](https://images.openai.com/static-rsc-4/WQwWpbZZdMkQshIElpsHZek1rkI2JlQpA4KwjcUJseoity3vBoztsrezdGK3PJUDGmLfMssQEvSZ1MHwicIkquT2k576DLFZeuWMfpecxgGdwRu9IEIJ8A-ZY_7-Rec-h1q1WuPWGyQN3GS48qrDX1PwQlvJ2U92O_IDLEcxLaY?purpose=inline)

[feynmanwiki.com](https://www.feynmanwiki.com/library/kv-caching-pvo5?utm_source=chatgpt.com)

Let's use an example.

Prompt:

`I love AI`

During prefill, the model processes all three tokens and caches their K and V.

During decoding:

| Generation step | New token being processed | Keys and Values used        |
| --------------- | ------------------------- | --------------------------- |
| Prefill         | `I love AI`               | K/V for all 3 prompt tokens |
| Decode 1        | `is`                      | Cached 3 + new 1            |
| Decode 2        | `powerful`                | Cached 4 + new 1            |
| Decode 3        | `technology`              | Cached 5 + new 1            |

Notice that the model doesn't need to recalculate K and V for `I love AI` when generating `powerful` or `technology`.

One subtle but important detail: during prefill, the prompt tokens can be processed in parallel because causal masking prevents each token from looking at future prompt tokens. During decoding, each newly generated token is processed incrementally.

## 6. Does KV Cache store the Query?

No. It normally stores only Keys and Values, not Queries.

Why?

The Query is used to determine what information the current token needs from previous tokens.

Once the current token's attention calculation is complete, its Query is generally not needed for the next generation step.

But its Key and Value are needed because future tokens may attend to it.

| Component | Purpose                                   | Cached?           |
| --------- | ----------------------------------------- | ----------------- |
| Query (Q) | What is the current token looking for?    | No, generally not |
| Key (K)   | What can this token be matched against?   | Yes               |
| Value (V) | What information does this token provide? | Yes               |

This is a useful way to remember it:

- Query is needed for the current attention calculation.
- Key and Value are retained for future attention calculations.

## 7. Simple mathematical explanation

Suppose the input sequence has \\(n\\) tokens.

Without KV Cache, when generating each new token, the model recalculates K and V for the entire prefix.

Across generation of \\(n\\) tokens, the amount of K/V projection work can grow approximately as:

\\[ 1+2+3+\cdots+n \\]

\\[ =\frac{n(n+1)}{2}=O(n^2) \\]

With KV Cache, each token's K and V are calculated once:

\\[ 1+1+1+\cdots+n=O(n) \\]

This is the reduction in repeated K/V projection work.

Important nuance: KV Cache does not make the entire attention computation linear in sequence length. A new query still needs to attend to the previous cached keys. Across all generated tokens, full attention still has quadratic total attention work in sequence length. KV Cache avoids recomputing old K/V projections and old token representations, making incremental decoding much more efficient.

## 8. How much GPU memory does KV Cache consume?

This is a very important production-level question.

KV Cache consumes GPU memory, and its size increases as the context grows.

A useful approximate formula is:

\\[ \boxed{ Memory=2\times L\times B\times S\times H\_{kv}\times D_h\times Bytes } \\]

Where:

| Symbol        | Meaning                                  |
| ------------- | ---------------------------------------- |
| 2             | Key and Value                            |
| L             | Number of Transformer layers             |
| B             | Batch size                               |
| S             | Number of cached tokens                  |
| \\(H\_{kv}\\) | Number of KV heads                       |
| \\(D_h\\)     | Dimension per head                       |
| Bytes         | Bytes per stored value (2 for FP16/BF16) |

Let's calculate a practical example.

Suppose we have:

- Transformer layers = 32
- Batch size = 1
- Sequence length = 8,192 tokens
- KV heads = 32
- Head dimension = 128
- Data type = FP16 (2 bytes)

\\[ 2\times32\times1\times8192\times32\times128\times2 \\]

# 4 GiB

Approximate KV Cache memory for this configuration.

That is just the KV Cache. It does not include model weights, activations, temporary attention buffers, or other GPU memory allocations.

Imagine serving 100 concurrent requests with this same context length. The cache requirement could approach 400 GiB, before accounting for other memory usage.

This is why KV Cache management is extremely important when deploying LLMs at scale.

## 9. What happens when the context becomes very large?

Suppose a user has a long conversation:

- First message: 2,000 tokens
- Second message: 3,000 tokens
- Third message: 4,000 tokens
- Fourth message: 5,000 tokens

The model's context keeps growing. Therefore, its KV Cache also grows.

This creates a problem:

GPU memory is limited, and KV Cache can become a bottleneck when serving many concurrent users.

Production systems use several techniques to address this.

### A. Grouped Query Attention (GQA)

Traditional Multi-Head Attention may have a separate Key and Value head for every Query head.

For example:

- 32 Query heads
- 32 Key heads
- 32 Value heads

This can consume considerable cache memory.

GQA reduces the number of KV heads and allows multiple Query heads to share the same KV heads.

[Understanding and Implementing Qwen3 From Scratch](https://images.openai.com/static-rsc-4/yRLmuRXGoYxiPKL3AZ9OH24BWLAqG3NQX1R88MyAdNO1qwqn8UTUhWnS7sGDixymC_DzqduT1N2XONrUfYHCTv3x-SCLw_3NSv6Q0KSfqkOk_q1cZij-XXkdRrjXBukP5ubE1okpHig4Td_xRTlDNY7uuEudXNXAt1-K1660_Gk?purpose=inline)

[magazine.sebastianraschka.com](https://magazine.sebastianraschka.com/p/qwen3-from-scratch?utm_source=chatgpt.com)

For example:

- 32 Query heads
- 8 KV heads

Now four Query heads share one KV head.

This reduces KV Cache memory substantially while retaining good model quality in many architectures.

### B. Multi-Query Attention (MQA)

MQA takes this further:

- Multiple Query heads.
- Only one shared Key head.
- Only one shared Value head.

[How to Optimize LLM Inference ?. In this blog, we’ll explore how to… | by Maninder Singh | Medium](https://images.openai.com/static-rsc-4/0RVKHEOHp_gFTh61shh1UoMwTxdq6RuKBWtlPdetJ7aLmHkts4loGQB_IDI7WzG2hcSwJhiEnNcF61SNyUe_1lGGK3SRvo9dR4QE6BQdMXFAVHsjHAxIMsY3LbexyGTsJ1oX7BmcEkakbgBzkpUACwhMfCYPMhw8N60186Viffw?purpose=inline)

[medium.com](https://medium.com/%40manindersingh120996/how-to-optimize-llm-inference-0076d9f3dd1b?utm_source=chatgpt.com)

This reduces memory usage even more, although it can involve quality and architectural trade-offs.

### C. PagedAttention

PagedAttention is an important technique used by vLLM.

Think about how an operating system manages virtual memory in pages.

Instead of requiring one large, continuous memory allocation for each request, PagedAttention stores KV Cache in smaller fixed-size blocks.

[\<Inference, KV Cache> \[vLLM\] Efficient Memory Management for Large Language Model Serving with PagedAttention (2023.09) — chanmuzi](https://images.openai.com/static-rsc-4/YNYi1L4Gr4Sv5fqLNCcWmD2eQbXus2EiKsLhJ8BZrrbZA6eAy2jVPkj2mnBp2oQ0sJV494_QtyB0OSE83Fx8aZSKvOqlhw9YZeVPgsSI18h1ohETxC5KDLbXUZMeoZMSPNj8lDRoB4EtCYySazCiKMNAdJ4qBnJhFu7ErTyYpks?purpose=inline)

[chanmuzi.tistory.com](https://chanmuzi.tistory.com/510?utm_source=chatgpt.com)

For example:

- Request A needs 1,000 tokens.
- Request B needs 3,000 tokens.
- Request C needs 500 tokens.

The system allocates cache blocks as needed rather than reserving one large maximum-size region for every request.

Benefits:

1. Better GPU memory utilization.
2. Less memory fragmentation.
3. More requests can run concurrently.
4. Cache blocks can be shared in suitable cases, such as common prompt prefixes.

Interview point: PagedAttention improves KV Cache memory management. It does not eliminate the KV Cache or make attention computation free.

### D. Sliding Window Attention

Some models use a sliding attention window.

For example, instead of attending to all 100,000 previous tokens, the model may attend only to the most recent 4,096 tokens in a particular attention layer.

This can limit the amount of cache needed for that layer.

However, it changes which past tokens are directly available to attention. It is an architectural choice, not simply a universally safe cache deletion strategy.

## 10. How does KV Cache work in actual Python code?

Let's first understand a simplified example using PyTorch.

This example demonstrates incremental self-attention. It is an educational implementation of the core mechanism, not a complete Transformer inference engine.

```
import torchimport torch.nn as nnimport torch.nn.functional as Fclass KVCacheAttention(nn.Module):    def __init__(self, d_model, num_heads):        super().__init__()        assert d_model % num_heads == 0        self.num_heads = num_heads        self.head_dim = d_model // num_heads        self.q_proj = nn.Linear(d_model, d_model)        self.k_proj = nn.Linear(d_model, d_model)        self.v_proj = nn.Linear(d_model, d_model)        self.out_proj = nn.Linear(d_model, d_model)        # Cached keys and values        self.key_cache = None        self.value_cache = None    def split_heads(self, x):        batch, seq_len, _ = x.shape        x = x.view(            batch,            seq_len,            self.num_heads,            self.head_dim        )
```

### Understand the important lines

1\. Calculate Q, K, V

```
q = self.q_proj(x)k = self.k_proj(x)v = self.v_proj(x)
```

These projections create the Query, Key, and Value representations for the newly supplied tokens.

2\. Append K and V to the cache

```
self.key_cache = torch.cat(    [self.key_cache, k],    dim=2)
```

The sequence dimension is `dim=2` because the tensors have shape:

```
(batch_size, num_heads, sequence_length, head_dim)
```

For example:

Before:

```
Key cache: [1, 8, 100, 64]
```

After processing one new token:

```
Key cache: [1, 8, 101, 64]
```

3\. Use cached K and V

```
output = F.scaled_dot_product_attention(    q,    self.key_cache,    self.value_cache)
```

The new Query attends to all available cached Keys and Values.

### Important production caveats

The code above demonstrates the concept, but a production implementation needs additional handling:

- The cache must be isolated per request and per layer. The example's instance-level cache is not safe for concurrent requests.
- A real Transformer has multiple layers, each maintaining its own cache.
- Position IDs and positional encodings such as RoPE must be handled correctly.
- Prompt prefill and incremental decoding need appropriate attention-mask handling. For a single new query attending to the entire past plus itself, a standard causal mask over the query's local length can be incorrect; the cache-aware attention path must use the correct positions.
- Repeated `torch.cat` operations can cause memory copying and allocation overhead. Production systems use preallocated or paged cache storage.
- Cache must be reset or released when a request finishes, subject to any intentional prefix-cache reuse.
- Production implementations also handle batching, variable sequence lengths, beam search, memory limits, and cache eviction.

The main thing to understand is the mechanism, not to memorize this simplified implementation as production code.

## 11. What happens to KV Cache when a request finishes?

Suppose 100 users send requests to your LLM application.

Each request has its own context and its own KV Cache.

```
LLM Serving System
│
├── Request A
│   └── KV Cache A
│
├── Request B
│   └── KV Cache B
│
├── Request C
│   └── KV Cache C
│
└── Request D
    └── KV Cache D
```

When Request A finishes, its cache can be released so that GPU memory can be reused.

There are some advanced optimizations:

- Prefix caching: Reuse KV Cache for identical shared prompt prefixes when the serving engine supports it.
- Continuous batching: Add and remove requests dynamically while the GPU is generating tokens for other requests.
- Cache eviction: Remove cache blocks that are no longer needed when memory pressure occurs.

One important security consideration: cache reuse must be carefully scoped. A system must not accidentally reuse another tenant's private context or expose its cached representations.

## 12. How does KV Cache improve LLM performance?

Let's connect this to production AI systems.

There are two important inference performance measurements:

| Metric | Meaning                                                                  |
| ------ | ------------------------------------------------------------------------ |
| TTFT   | Time to First Token — how long until the model produces its first token  |
| TPOT   | Time Per Output Token — average time between subsequent generated tokens |

KV Cache primarily improves the efficiency of the decode phase, reducing repeated computation and generally improving token-generation throughput and latency.

However, it does not eliminate all bottlenecks.

For example, an LLM might still be slow because:

- GPU memory bandwidth is insufficient.
- The model is too large.
- There are too many concurrent requests.
- Context length is too large.
- The model has a high number of attention heads or layers.
- GPU memory is exhausted by model weights and KV Cache.

This is why production serving engines combine KV Cache with techniques such as continuous batching, optimized attention kernels, quantization, and efficient memory management.

## 13. KV Cache vs FlashAttention — what's the difference?

This is another common interview question.

These two techniques solve different problems.

| KV Cache                                        | FlashAttention                                       |
| ----------------------------------------------- | ---------------------------------------------------- |
| Avoids recomputing previous tokens' K and V     | Optimizes the attention computation itself           |
| Primarily useful during autoregressive decoding | Useful in training and inference                     |
| Stores K/V in memory                            | Uses IO-aware computation to reduce data movement    |
| Requires additional memory for cached tensors   | Reduces intermediate memory traffic and memory usage |
| Improves incremental generation efficiency      | Improves attention kernel efficiency                 |

Simple way to remember:

- KV Cache = Don't calculate old Keys and Values again.
- FlashAttention = Calculate attention more efficiently by reducing expensive memory transfers.

They can be used together.

## 14. Interview-ready answer: What is KV Cache?

Here is how I would explain it in a Senior AI Engineer interview.

Question: What is KV Cache in Transformers, and why do we need it?

KV Cache stands for Key-Value Cache. It is an inference optimization used in autoregressive Transformer models such as GPT and Llama.

During text generation, the model generates tokens one at a time. At every generation step, self-attention requires Keys and Values from previous tokens.

Without KV Cache, the model repeatedly computes Keys and Values for the entire previous sequence, resulting in redundant computation.

To solve this, we store previously calculated Keys and Values in GPU memory.

During the next decoding step, we calculate Q, K and V only for the new token, append its Key and Value to the cache, and use the new Query to attend to all cached Keys and Values.

KV Cache is maintained separately for every Transformer layer.

It reduces repeated computation and improves decoding efficiency. However, the trade-off is increased GPU memory consumption, which grows with sequence length, batch size, number of layers, and number of KV heads.

In production, we use techniques such as GQA, MQA, PagedAttention, and continuous batching to manage KV Cache memory and improve LLM serving throughput.

## 15. Common follow-up interview questions

Q1. Why do we cache only K and V, not Q?

Because the Query is used for the current token's attention calculation. Future tokens need the current token's Key and Value to attend to it, but they do not need its previous Query.

Q2. Is KV Cache used during training?

Generally, standard full-sequence training does not need KV Cache. Training processes the sequence in parallel with causal masking. KV Cache is primarily used during autoregressive inference.

Q3. Does KV Cache reduce model size?

No. It does not reduce model weights. It stores intermediate Key and Value tensors to avoid redundant computation.

Q4. Why does KV Cache increase GPU memory usage?

Because the model stores Key and Value tensors for every cached token, for every layer, and for every active request.

Q5. What happens when KV Cache exceeds GPU memory?

The serving engine may need to evict cache blocks, offload data, limit context length, reduce concurrency, or reject requests. PagedAttention and memory-aware scheduling help manage this situation.

Q6. What is the difference between MHA, MQA and GQA?

- MHA: Each Query head has its own Key and Value heads.
- MQA: All Query heads share one Key head and one Value head.
- GQA: Groups of Query heads share Key and Value heads.

MQA and GQA reduce the number of KV heads and therefore reduce KV Cache memory requirements.

Q7. Does KV Cache make attention O(n) instead of O(n²)?

No. It reduces repeated K/V projection and prefix-processing work. With full attention, each new token still attends to all previous tokens. Total attention work across a sequence of length \\(n\\) remains approximately quadratic.

## 16. Test your understanding

KV Cache knowledge check

0 / 4 answered

1\. What does KV Cache store?

Query and output tensors

Key and Value tensors

Model weights

Only token embeddings

2\. Why don't we usually cache Query?

Query is never calculated

Query is only needed for the current attention calculation

Query is stored in model weights

Query is identical for every token

3\. Which technique is used by vLLM for efficient KV Cache memory management?

Dropout

PagedAttention

Batch Normalization

Fine-tuning

4\. Does KV Cache eliminate quadratic attention work?

Yes, completely

No, it avoids repeated K/V computation but full attention still has quadratic total work

Only during training

Only when using MQA

Check answers

My recommendation for your Senior AI Engineer preparation: After KV Cache, study FlashAttention, GQA/MQA, continuous batching, and PagedAttention together. These topics help you explain not just how Transformers work, but how to serve LLMs efficiently in production.