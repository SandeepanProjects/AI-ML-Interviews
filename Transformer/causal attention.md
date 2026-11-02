# What is Causal Attention (Masked Self-Attention) in Transformers?

Causal attention, also called masked self-attention, is a mechanism used in decoder-based Transformers to ensure that a token can only look at previous tokens and itself, not future tokens.

It is one of the most important concepts behind GPT, ChatGPT and other autoregressive Large Language Models (LLMs).

Let's understand it step by step.

## 1. First understand the problem

Imagine we want an LLM to complete this sentence:

"I love machine learning because..."

Suppose the complete sentence is:

"I love machine learning because it is interesting."

The model generates text one token at a time.

At each step, it should predict the next token using only the tokens it has already seen.

For example:

| Step | Tokens available                | What should the model predict? |
| ---- | ------------------------------- | ------------------------------ |
| 1    | I                               | love                           |
| 2    | I love                          | machine                        |
| 3    | I love machine                  | learning                       |
| 4    | I love machine learning         | because                        |
| 5    | I love machine learning because | it                             |

Notice something important:

When predicting `machine`, the model must not see `learning`, `because`, `it` or `interesting`.

Why?

Because if the model could see future tokens during training, it would get information about the answer it is supposed to predict.

This is called information leakage.

Causal attention solves this problem.

## 2. What exactly is causal attention?

Causal attention restricts the attention mechanism so that each token can attend only to:

- Previous tokens.
- The current token itself.

It cannot attend to future tokens.

For example, consider:

`I love AI models`

The allowed attention looks like this:

Causal attention visibility

Rows represent the token doing the attending. Columns represent the token being attended to. Green = allowed; grey = blocked.

|        | I | love | AI | models |
| ------ | - | ---- | -- | ------ |
| I      |   |      |    |        |
| love   |   |      |    |        |
| AI     |   |      |    |        |
| models |   |      |    |        |

Allowed: current and previous tokens.

Blocked: future tokens.

Let's understand each row:

- `I` can attend only to `I`.
- `love` can attend to `I` and `love`.
- `AI` can attend to `I`, `love` and `AI`.
- `models` can attend to all four tokens.

This is the basic idea behind causal attention.

## 3. How does the attention mask actually work?

Remember the standard scaled dot-product attention formula:

\\[ Attention(Q,K,V)=Softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V \\]

Here:

- Q = Query vectors.
- K = Key vectors.
- V = Value vectors.
- \\(QK^T\\) = attention scores.
- Softmax = converts scores into attention probabilities.

Normally, attention can calculate scores between every pair of tokens.

But for GPT-style models, we don't want future tokens to be visible.

So we introduce a causal mask.

The mask is applied to the attention scores before softmax.

Let's use a four-token example.

Sentence:

`I love AI models`

The causal mask is:

\\[ M= \begin{bmatrix} 0 & -\infty & -\infty & -\infty\\\ 0 & 0 & -\infty & -\infty\\\ 0 & 0 & 0 & -\infty\\\ 0 & 0 & 0 & 0 \end{bmatrix} \\]

What does this matrix mean?

- `0` means attention is allowed.
- \\(-\infty\\) means attention is blocked.

We add this mask to the attention scores:

\\[ Scores=\frac{QK^T}{\sqrt{d_k}}+M \\]

Then we apply softmax:

\\[ AttentionWeights=Softmax(Scores) \\]

Why use negative infinity?

Because:

\\[ e^{-\infty}=0 \\]

Therefore, after softmax, blocked future positions receive zero attention probability.

The model cannot use information from those future tokens.

Example: Attention scores before and after masking

Illustrative values for the token "AI".

Before masking

## 0.2

I

## 0.5

love

## 0.8

AI

## 1.2

models

After causal masking

## 0.2

I

## 0.5

love

## 0.8

AI

Blocked

The future token "models" is blocked when calculating attention for "AI".

Notice that the values above are illustrative attention scores, not probabilities. Softmax is applied after masking to produce the actual attention probabilities.

## 4. Why is causal attention important during LLM training?

This is a very important interview point.

You might wonder:

"If the model generates one token at a time, why do we need masking during training?"

The answer is that training and inference work differently.

During training:

We usually provide the entire training sequence at once.

Example:

`I love machine learning`

The model is trained to predict the next token at every position:

| Input token             | Target token     |
| ----------------------- | ---------------- |
| I                       | love             |
| I love                  | machine          |
| I love machine          | learning         |
| I love machine learning | Next token / end |

The model can calculate attention for all positions in parallel, but the causal mask ensures that each position only sees its permitted context.

This allows efficient parallel training without leaking future information.

During inference:

The model generates tokens autoregressively, one token at a time.

Example:

## Autoregressive generation

1. Input: I

   Predict

   &#x20;love

2. Input: I love

   Predict

   &#x20;machine

3. Input: I love machine

   Predict

   &#x20;learning

4. Input: I love machine learning

   Predict

   &#x20;is



Each new prediction uses the previously available tokens.

The causal mask enforces the same left-to-right prediction rule during training that the model follows during generation.

## 5. How is causal attention different from normal self-attention?

This is a common interview question.

| Feature                  | Bidirectional self-attention | Causal self-attention |
| ------------------------ | ---------------------------- | --------------------- |
| Future tokens visible?   | Yes                          | No                    |
| Previous tokens visible? | Yes                          | Yes                   |
| Current token visible?   | Yes                          | Yes                   |
| Attention pattern        | Full sequence                | Lower-triangular      |
| Typical use              | BERT encoder                 | GPT decoder           |
| Main purpose             | Understand context           | Generate next token   |

For example:

- BERT uses bidirectional self-attention. It can look at tokens on both sides.
- GPT uses causal self-attention. It cannot look at future tokens.

Why does BERT need bidirectional attention?

Because BERT is designed to understand text using context from both directions. For example, when learning the meaning of a masked word, it can use words both before and after that word.

Why does GPT need causal attention?

Because GPT predicts the next token and must not use future tokens that it is supposed to generate.

## 6. How do we implement causal attention in Python?

Let's implement a simplified causal self-attention layer using PyTorch.

```
import torchimport torch.nn as nnimport torch.nn.functional as Fimport mathclass CausalSelfAttention(nn.Module):    def __init__(self, embed_dim):        super().__init__()        self.query = nn.Linear(embed_dim, embed_dim)        self.key = nn.Linear(embed_dim, embed_dim)        self.value = nn.Linear(embed_dim, embed_dim)        self.output = nn.Linear(embed_dim, embed_dim)    def forward(self, x):        batch_size, seq_len, embed_dim = x.shape        # 1. Generate Query, Key and Value        Q = self.query(x)        K = self.key(x)        V = self.value(x)        # 2. Calculate attention scores        scores = Q @ K.transpose(-2, -1)        scores = scores / math.sqrt(embed_dim)        # 3. Create causal mask        # True = future position, which must be blocked        mask = torch.triu(            torch.ones(                seq_len,
```

Output:

```
Output: torch.Size([2, 4, 128])
Attention: torch.Size([2, 4, 4])
```

Let's understand the important lines:

```
mask = torch.triu(    torch.ones(seq_len, seq_len),    diagonal=1)
```

This creates an upper-triangular mask.

For a sequence of four tokens:

```
0 1 1 1
0 0 1 1
0 0 0 1
0 0 0 0
```

Then:

```
scores = scores.masked_fill(    mask,    float("-inf"))
```

All future positions receive negative infinity.

After softmax, those positions get zero attention probability.

Production note: This implementation is for learning. Real LLM implementations commonly use optimized scaled dot-product attention kernels, such as PyTorch's `torch.nn.functional.scaled_dot_product_attention`, with `is_causal=True` where appropriate. They also handle padding, batching, head dimensions and other details.

For example:

```
output = F.scaled_dot_product_attention(    Q,    K,    V,    is_causal=True)
```

Here, PyTorch handles the causal masking internally. In this API, don't also pass a separate causal mask unless you have a specific reason and understand the mask semantics.

## 7. Causal mask vs padding mask

Another question that interviewers sometimes ask is whether causal masking and padding masking are the same.

No. They solve different problems.

Causal mask: Prevents a token from looking at future tokens.

Padding mask: Prevents the model from attending to artificial padding tokens added to make sequences in a batch the same length.

Example:

```
Sentence 1: I love AI
Sentence 2: I love Python
```

If sequences have different lengths, we may pad the shorter sequence:

```
I love AI <PAD>
I love Python
```

The padding mask prevents the model from treating `<PAD>` as meaningful content.

In a decoder, causal masking and padding masking may both be needed. In practice, some implementations combine their restrictions into one attention mask.

## 8. What happens inside a complete GPT Transformer block?

A GPT-style Transformer block typically works like this:

Input token representations

RoPE positional information

Applied to Q and K in attention

Multi-Head Causal Self-Attention

Each token can see only previous and current tokens

Residual Connection + Layer Normalization

Feed-Forward Network (MLP)

Residual Connection + Layer Normalization

Next Transformer Block

The diagram is conceptual: actual modern GPT implementations often use pre-normalization, and the precise normalization and positional-encoding placement varies by architecture.

Remember: causal masking is applied inside the self-attention operation. It is not a separate neural network layer.

## 9. Important advanced interview points

Q1. Does causal attention mean the model processes tokens one by one during training?

No. During training, the model can process the entire sequence in parallel. The causal mask restricts which positions each token can attend to.

During inference, next-token generation is autoregressive, so tokens are generated sequentially.

Q2. Does causal attention prevent the model from seeing the current token?

No. A token can attend to itself and all previous tokens. It is only future tokens that are blocked.

Q3. Is causal attention the same as autoregressive generation?

They are related, but not identical.

- Causal attention is the mechanism that prevents access to future tokens.
- Autoregressive generation is the process of predicting one token at a time using previously available tokens.

Q4. Does the original Transformer use causal masking?

Yes. Its decoder uses masked self-attention to prevent looking at future output tokens. Its encoder uses bidirectional self-attention. The decoder's separate encoder-decoder cross-attention is not restricted by this causal mask.

## 10. Interview-ready answer

If the interviewer asks, "Explain causal or masked attention," you can answer:

> Causal attention, also known as masked self-attention, is used in decoder-based Transformer architectures such as GPT.
>
> Its main purpose is to prevent a token from attending to future tokens.
>
> For example, when the model is predicting the next word in "I love machine learning", it should only use the tokens that have already appeared, not future words.
>
> Technically, we calculate the normal attention scores using Q, K and V. Then we apply a lower-triangular causal mask by adding negative infinity to future positions before applying softmax.
>
> Since the softmax of negative infinity is zero, future tokens receive zero attention probability.
>
> During training, the model can process the entire sequence in parallel while the causal mask prevents information leakage. During inference, it generates tokens autoregressively, one at a time.
>
> This mechanism is fundamental to GPT-style language models because it ensures that next-token prediction follows the correct left-to-right generation process.

## 11. Test yourself

Interview knowledge check

0/4 answered

1\. If a sequence has 5 tokens, how many tokens can position 3 attend to (assuming zero-based indexing)?

2

3

4

5

2\. Why do we add negative infinity to future attention scores?

To increase attention

To make softmax probability zero

To remove token embeddings

To increase sequence length

3\. Can Transformer training happen in parallel with causal attention?

No, never

Yes, using a causal mask

Only with BERT

Only during inference

4\. What is the difference between causal masking and padding masking?

They are identical

Causal blocks future tokens; padding blocks padding tokens

Causal blocks previous tokens

Padding blocks future tokens

Check answers

Remember these three concepts:

1. Self-attention learns relationships between tokens.
2. Causal masking prevents future-token information leakage.
3. Autoregressive generation predicts the next token using the available previous context.

The next concept I recommend learning is KV Cache—it explains how GPT avoids repeatedly calculating attention for all previous tokens during generation and improves LLM inference performance.