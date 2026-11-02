# Multi-Head Attention in Transformers – Complete Explanation for Senior AI Engineer Interviews

Multi-Head Attention is one of the most important components of the Transformer architecture.

You already understand:

- What self-attention is.
- What Query, Key and Value are.
- How attention scores are calculated.
- How Softmax generates attention weights.
- How Value vectors produce contextual representations.

Now let's understand why Transformers use multiple attention heads instead of just one attention mechanism.

I'll explain it from the basics, with simple examples, architecture diagrams, mathematical formulas, Python implementation, and interview questions.

# 1. What is Multi-Head Attention?

Multi-Head Attention is a mechanism that allows a Transformer to learn different types of relationships between tokens simultaneously using multiple attention heads.

Let's understand with an example.

Consider this sentence:

```
The developer built an AI application using Python.
```

To understand this sentence, the model needs to learn different relationships:

- `developer` is related to `built`.
- `built` is related to `application`.
- `AI` describes `application`.
- `Python` is related to the development technology.

One attention head might learn to focus on subject-verb relationships.

Another head might learn to focus on object relationships.

Another head might learn to focus on technology-related context.

These are illustrative possibilities, not fixed roles assigned to individual heads.

The important idea is that multiple heads allow the model to learn different attention patterns in parallel.

[Transformers & Attention Mechanism - Complete NLP Series Part 8 - Wasil Zafar](https://images.openai.com/static-rsc-4/sXKWwFTfbfiwH3JSdQgKx_I3rFkDfSHlOKC2gMCxPQvvBrhWjlDVQrVgY05bjPEXB24oKKf2TAi5klgRS1XLQtF32OPrVCAYgfevW_63npiEzHLdwSyr8aEeqYVlFHmZlANKpMkK56gs_RV27Mm4TImTaR40cSFTs_RjxHiUTxY?purpose=inline)

[wasilzafar.com](https://www.wasilzafar.com/pages/series/nlp/nlp-transformers-attention.html?utm_source=chatgpt.com)

Think about it this way:

Imagine you are reading a financial report.

You might read it from different perspectives:

1. Financial perspective – revenue, profit and expenses.
2. Risk perspective – debt, volatility and exposure.
3. Business perspective – products, customers and competition.

Each perspective gives you different information about the same report.

Similarly, Multi-Head Attention allows the Transformer to examine the same input from multiple learned representation spaces.

# 2. Why Do We Need Multi-Head Attention?

First, remember how single-head attention works.

In single-head attention:

```
Input Embeddings
       |
       v
Query, Key, Value
       |
       v
Attention Calculation
       |
       v
Attention Output
```

It performs one attention operation.

The limitation is that this provides only one learned attention representation at that layer.

But language contains many different types of relationships.

For example:

```
The bank approved the loan because it
had sufficient collateral.
```

The model needs to understand:

- Which entity does `it` refer to?
- What is the relationship between `bank` and `approved`?
- How does `collateral` relate to `loan`?
- Which words provide relevant context?

Using multiple heads allows the model to learn multiple attention patterns at the same time.

Single-Head Attention

Input tokens

One attention head

One learned attention pattern

One attention output

Multi-Head Attention

Input tokens

Head 1

Attention

Head 2

Attention

Head 3

Attention

Head 4

Attention

Concatenate head outputs

Final linear projection

The advantage is that the model can learn different relationships simultaneously.

# 3. How Does Multi-Head Attention Work?

This is the most important part.

Let's take a simple example:

```
The developer builds AI applications.
```

Assume:

- Model dimension = 512.
- Number of attention heads = 8.
- Dimension per head = 64.

Why 64?

Because:

\\[ d\_{head}=\frac{d\_{model}}{h} \\]

\\[ d\_{head}=\frac{512}{8}=64 \\]

Each attention head works with its own Query, Key and Value projections.

Let's understand the complete process.

## Step 1: Input Embeddings

First, the input sentence is converted into token embeddings.

```
The          → 512-dimensional vector
developer    → 512-dimensional vector
builds       → 512-dimensional vector
AI           → 512-dimensional vector
applications → 512-dimensional vector
```

The input matrix has the shape:

\\[ X \in \mathbb{R}^{5\times512} \\]

There are five tokens, each represented by 512 dimensions.

## Step 2: Create Separate Q, K and V Projections for Each Head

This is where Multi-Head Attention starts.

In single-head attention, we have one set of Q, K and V projection weights.

In Multi-Head Attention, each head has its own learned projection matrices.

For example:

```
Input Embeddings (512 dimensions)
              |
      +-------+-------+
      |       |       |
      v       v       v
     Q1      K1      V1
     Q2      K2      V2
     Q3      K3      V3
     ...     ...     ...
     Q8      K8      V8
```

Each head learns its own projection parameters.

For eight heads, there are eight sets of Q, K and V projections.

Technically, implementations usually calculate these using three larger linear layers and then split the results into heads. They do not necessarily execute eight independent layers.

## Step 3: Each Head Performs Its Own Attention Calculation

Every head independently calculates attention.

The formula for one head is:

\\[ head_i=Attention(Q_i,K_i,V_i) \\]

Or:

\\[ head_i = Softmax\left( \frac{Q_iK_i^T}{\sqrt{d_k}} \right)V_i \\]

For example:

Eight attention heads (illustrative)

Head 1

Subject relationships

Head 2

Verb relationships

Head 3

Object relationships

Head 4

Context relationships

Head 5

Long-distance relationships

Head 6

Semantic relationships

Head 7

Other patterns

Head 8

Other patterns

These labels are conceptual examples. Real attention heads do not have predefined responsibilities, and their learned patterns can overlap.

Each head produces its own output representation.

## Step 4: Concatenate All Head Outputs

After all heads finish their calculations, the outputs are combined.

Suppose each head produces 64-dimensional output.

We have eight heads.

Therefore:

\\[ 8\times64=512 \\]

We concatenate the outputs:

\\[ Concat(head_1,head_2,\ldots,head_8) \\]

The combined output has 512 dimensions again.

[Transformer Architecture: From Concepts to Code | by Sachin Pant | Data And Beyond | Dec, 2025 | Medium](https://images.openai.com/static-rsc-4/379cRA8xu8nt-t3P0fApef3yvJVaaaQ1_ROVQG0nBQPttDzIv0Eq6E1xpf16KS7OCmljWH74IW7OqhtL4txJLqDgmlgDGwnnWwOO4T8HyRIExFfucX6tiZQoSY1A7F2B4i5tNVp9yb58pHEmGiz04KuoMV9KweYLyWjV-3OquHI?purpose=inline)

[medium.com](https://medium.com/data-and-beyond/transformer-architecture-from-concepts-to-code-fb3ac99c7396?utm_source=chatgpt.com)

## Step 5: Apply Final Linear Projection

After concatenation, the combined representation passes through another learned linear transformation.

\\[ MultiHead(Q,K,V)=Concat(head_1,\ldots,head_h)W_O \\]

Where:

- `head_i` = Output of each attention head.
- `Concat` = Combines all head outputs.
- `W_O` = Output projection matrix.

Why do we need this final projection?

It allows the model to learn how to combine information from different heads into a unified representation.

This final output is passed to the next component of the Transformer block.

# 4. Understand Multi-Head Attention With Actual Dimensions

This is particularly important for senior-level interviews.

Let's assume:

| Parameter          | Value |
| ------------------ | ----- |
| Number of tokens   | 5     |
| Model dimension    | 512   |
| Number of heads    | 8     |
| Dimension per head | 64    |

Let's calculate the tensor shapes.

Input:

\\[ X=(5,512) \\]

Q, K and V projections:

\\[ Q=(5,512) \\]

\\[ K=(5,512) \\]

\\[ V=(5,512) \\]

We split each into eight heads:

\\[ Q=(8,5,64) \\]

\\[ K=(8,5,64) \\]

\\[ V=(8,5,64) \\]

For each head, attention scores are calculated:

\\[ QK^T \\]

The shape of the attention matrix for each head is:

\\[ (5,64)\times(64,5)=(5,5) \\]

So each head produces a 5 × 5 attention matrix.

For eight heads:

\\[ (8,5,5) \\]

Each head then multiplies its attention weights by its Value matrix:

\\[ (5,5)\times(5,64)=(5,64) \\]

After calculating all eight heads, we concatenate:

\\[ 8\times64=512 \\]

Final output:

\\[ (5,512) \\]

Complete dimension flow

Input X

# 5 × 512

Linear projections

Q, K, V: 5 × 512 each

Split into 8 heads

# 8 × 5 × 64

Attention scores per head

# 5 × 5

Attention output per head

5 × 64

Concatenate 8 heads

# 5 × 512

Final output projection

# 5 × 512

Notice something important:

The model dimension remains 512 before and after Multi-Head Attention.

The attention heads divide the representation into smaller dimensions, perform attention independently, and then combine their outputs.

# 5. Mathematical Formula of Multi-Head Attention

You should understand and memorize these two formulas.

Formula 1: Scaled Dot-Product Attention

\\[ Attention(Q,K,V)= Softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V \\]

Formula 2: Multi-Head Attention

\\[ MHA(Q,K,V)= Concat(head_1,\ldots,head_h)W_O \\]

Where:

\\[ head_i=Attention(QW_i^Q,KW_i^K,VW_i^V) \\]

Here:

- \\(W_i^Q\\): Query projection for head i.
- \\(W_i^K\\): Key projection for head i.
- \\(W_i^V\\): Value projection for head i.
- \\(W_O\\): Final output projection.
- \\(h\\): Number of attention heads.

The most important distinction:

- Scaled Dot-Product Attention calculates attention within one head.
- Multi-Head Attention combines the outputs of multiple heads.

# 6. Single-Head vs Multi-Head Attention

| Feature                | Single-Head                          | Multi-Head                                 |
| ---------------------- | ------------------------------------ | ------------------------------------------ |
| Attention operations   | One                                  | Multiple                                   |
| Learned projections    | One Q/K/V set                        | Separate projections per head              |
| Attention patterns     | One learned attention representation | Multiple learned attention representations |
| Information processing | One representation space             | Multiple representation spaces             |
| Output                 | One head output                      | Concatenated and projected outputs         |
| Transformer usage      | Basic attention mechanism            | Standard Transformer attention design      |

Interview explanation:

"Single-head attention performs one attention operation over the input, whereas Multi-Head Attention performs several attention operations using different learned projections. The outputs of these heads are concatenated and passed through a final linear projection. This allows the Transformer to learn multiple types of relationships between tokens simultaneously."

# 7. Python Implementation – Multi-Head Attention Using PyTorch

Now let's see how this is implemented in actual deep learning code.

We will use PyTorch because it is commonly used for building Transformer architectures.

The following is a complete educational implementation of Multi-Head Attention.

```
import torchimport torch.nn as nnimport torch.nn.functional as Fclass MultiHeadAttention(nn.Module):    def __init__(self, embed_dim=512, num_heads=8):        super().__init__()        assert embed_dim % num_heads == 0        self.embed_dim = embed_dim        self.num_heads = num_heads        self.head_dim = embed_dim // num_heads        # Separate learned projections        self.query = nn.Linear(embed_dim, embed_dim)        self.key = nn.Linear(embed_dim, embed_dim)        self.value = nn.Linear(embed_dim, embed_dim)        # Final output projection        self.output = nn.Linear(embed_dim, embed_dim)    def forward(self, x, causal_mask=None):        batch_size, seq_len, _ = x.shape        # 1. Generate Q, K, V        Q = self.query(x)        K = self.key(x)        V = self.value(x)
```

Expected shapes:

```
Input:
torch.Size([2, 5, 512])

Output:
torch.Size([2, 5, 512])

Attention weights:
torch.Size([2, 8, 5, 5])
```

Let's understand the most important lines:

| Code                              | Purpose                                         |
| --------------------------------- | ----------------------------------------------- |
| `nn.Linear(embed_dim, embed_dim)` | Learns Q/K/V projections                        |
| `view()`                          | Separates the embedding dimension into heads    |
| `transpose()`                     | Rearranges dimensions for attention calculation |
| `Q @ K.transpose()`               | Calculates attention scores                     |
| `scores / sqrt(head_dim)`         | Scales scores                                   |
| `softmax()`                       | Generates attention weights                     |
| `weights @ V`                     | Calculates context representations              |
| `transpose()` and `view()`        | Combines all heads                              |
| `self.output()`                   | Final linear projection                         |

Note: This example uses self-attention, so Q, K and V are all projected from `x`. For cross-attention, the Query input and Key/Value input would be different. It also omits padding-mask handling and uses an explicit implementation for clarity; production systems commonly use optimized attention kernels such as PyTorch SDPA.

# 8. Where Does Multi-Head Attention Fit in Transformer Architecture?

Let's connect this with the complete Transformer block.

[Transformers: Attention is all you need — Layer Normalization | by Shravan Kumar | Medium](https://images.openai.com/static-rsc-4/ucdlLSLPR43uzQ-JcfFlaneEdD9AzCwJ6Smn7m7XTFfn2b_YrIwYlNRn4cIdHeWVGLPhLOGq7KMjKA69d9iFtkltHxzlw9u94Drgx_QyLiHiIaMrMtkYC94laJN7Iqao6MFgs_yLjZgWfq8SFBTHEInk70l8LK1fjWYxKsFspo0?purpose=inline)

[medium.com](https://medium.com/%40shravankoninti/transformers-attention-is-all-you-need-layer-normalization-1435248866d6?utm_source=chatgpt.com)

The flow is:

```
Input Embeddings
       |
       v
Multi-Head Attention
       |
       v
Residual Connection
       |
       v
Layer Normalization
       |
       v
Feed Forward Network
       |
       v
Residual Connection
       |
       v
Layer Normalization
       |
       v
Output
```

In the original Transformer, this is the general encoder block structure.

In the decoder, Multi-Head Attention is used for masked self-attention, and the original encoder-decoder architecture also has a separate cross-attention sublayer.

Important: Multi-Head Attention itself does not include the residual connections, normalization or feed-forward network. Those are surrounding components of the Transformer block.

# 9. What Happens If We Increase the Number of Attention Heads?

Suppose we have a model with:

\\[ d\_{model}=512 \\]

Let's compare different numbers of heads.

| Number of heads | Dimension per head |
| --------------- | ------------------ |
| 1               | 512                |
| 4               | 128                |
| 8               | 64                 |
| 16              | 32                 |
| 32              | 16                 |

As the number of heads increases, the dimension allocated to each head decreases, assuming the model dimension stays fixed.

Does increasing the number of heads always improve performance?

No.

There are trade-offs:

- More heads provide more independent learned attention projections.
- Smaller head dimensions may limit what each head can represent.
- More heads increase attention computation and memory overhead in some implementations.
- Too many heads may not provide meaningful additional information.

The right number depends on model architecture, training data, hardware and performance requirements.

# 10. What Are MQA and GQA? (Advanced Interview Topic)

As you move towards Senior AI Engineer and LLM architecture interviews, you should also know these concepts.

Standard Multi-Head Attention uses separate Query, Key and Value projections for every head.

But large language models can have significant KV-cache memory requirements during inference.

Two approaches help reduce this cost.

1\. Multi-Head Attention (MHA)

Each attention head has its own Query, Key and Value projections.

```
Head 1 → Q1, K1, V1
Head 2 → Q2, K2, V2
Head 3 → Q3, K3, V3
Head 4 → Q4, K4, V4
```

2\. Multi-Query Attention (MQA)

Every head has its own Query, but all heads share the same Key and Value projections.

```
Head 1 → Q1 ─┐
Head 2 → Q2 ─┼→ Shared K, V
Head 3 → Q3 ─┤
Head 4 → Q4 ─┘
```

This reduces KV-cache memory requirements.

3\. Grouped-Query Attention (GQA)

Multiple Query heads share one Key/Value head within a group.

```
Q1 ─┐
Q2 ─┴→ K1, V1

Q3 ─┐
Q4 ─┴→ K2, V2
```

It provides a compromise between standard MHA and MQA.

Why does this matter in production?

Imagine serving thousands of concurrent requests to an LLM.

The model needs to retain Key and Value representations for previous tokens during generation.

Reducing KV-cache memory can help:

- Increase concurrent requests.
- Improve memory utilization.
- Support longer context windows.
- Improve inference throughput.

This is especially relevant when deploying large decoder-only models on GPUs.

# 11. Top Interview Questions on Multi-Head Attention

Here are the questions I recommend preparing.

1\. What is Multi-Head Attention?

2\. Why do we need multiple attention heads?

3\. What is the difference between self-attention and Multi-Head Attention?

4\. Why do we concatenate the attention heads?

5\. Why do we need the final linear projection?

6\. Why is the embedding dimension divided by the number of heads?

7\. Does every attention head learn a different relationship?

8\. What happens if we increase the number of heads?

9\. What is GQA?

10\. How does Multi-Head Attention help LLMs?

# 12. Two-Minute Interview Answer

If the interviewer asks, "Explain Multi-Head Attention in Transformer architecture", you can use this answer.

Interview-ready answer

"Multi-Head Attention is a mechanism in Transformer architecture that allows the model to learn different relationships between tokens simultaneously.

In normal single-head attention, we generate Query, Key and Value vectors and calculate attention once.

However, language contains different types of relationships. Therefore, Transformers use multiple attention heads.

Each attention head has its own learned Query, Key and Value projections. Each head independently calculates scaled dot-product attention using the formula Softmax of Q multiplied by K transpose divided by the square root of the Key dimension, multiplied by V.

After all attention heads calculate their outputs, we concatenate them and pass the combined representation through a final linear projection.

For example, suppose the model dimension is 512 and we have eight attention heads. Each head operates on 64 dimensions. After calculating attention independently, we concatenate all eight outputs to get back 512 dimensions.

The main advantage is that the model can learn different attention patterns and relationships between tokens in parallel, rather than relying on just one attention representation.

This helps Transformers build richer contextual representations and is one of the key reasons behind their effectiveness in language modeling."

&#x20;Copy interview answer

# 13. Test Your Understanding

Multi-Head Attention quiz

0/5 answered

1\. What is the main purpose of Multi-Head Attention?

Reduce vocabulary size

Learn multiple attention patterns simultaneously

Convert words into token IDs

2\. If model dimension is 512 and number of heads is 8, what is the head dimension?

128

64

32

3\. What happens after all attention heads calculate their outputs?

They are concatenated and linearly projected

They are discarded

They are converted into token IDs

4\. What does W_O represent?

Input embedding weights

Output projection weights

Tokenizer weights

5\. What is the main advantage of GQA?

Eliminates attention completely

Reduces KV-cache memory requirements

Removes tokenization

Submit answers

## What you should study next

To build a strong understanding of Transformer architecture for Senior AI Engineer interviews, I recommend studying in this sequence:

1.

   Covered

   &#x20;Self-Attention and Query, Key, Value.



2.

   Covered

   &#x20;Multi-Head Attention.

3. Positional Encoding and RoPE (Rotary Positional Embeddings).

4. Feed-Forward Network (FFN).

5. Residual Connections and Layer Normalization.

6. Complete Transformer Encoder and Decoder.

7. KV Cache and its role in LLM inference.

8. FlashAttention and optimization of attention computation.

My recommendation: Study RoPE next, followed by KV Cache. These two concepts will help you move from understanding the original Transformer paper to explaining how modern LLMs actually work in production.