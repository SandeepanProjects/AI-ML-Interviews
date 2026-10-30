# Complete LLM Architecture: How Transformers Work Internally

I'll explain this from a Senior AI Engineer / Staff AI Engineer perspective, starting from the fundamentals and moving towards production-level architecture.

The most important thing to understand is:

An LLM (Large Language Model) is generally a neural network built using the Transformer architecture. Its primary job is to predict the next token based on the tokens that came before it.

For example:

Input:

`Explain how a transformer works`

The model processes the input and predicts the next token, repeatedly, until it generates a complete response.

However, a production LLM involves much more than a Transformer. It includes tokenization, embeddings, attention, neural network layers, model weights, training, inference, decoding, and serving infrastructure.

Let's break down the complete architecture.

# 1. Complete architecture of an LLM

First, understand the high-level architecture.

[IrukaDark｜PCの雑務を超高速で片付けるスーパーサブAI](https://images.openai.com/static-rsc-4/bFrAvp8tpdVhmkss4zRs3y7fOeyvaXR32_2046KUmq1BuW0MmHK7j_2a_Q3tthNS6X4aWfpooTOLOxDWfVic8Q0GmRRM9cjY0i2_6SPRZ5yDEzTRG-i97mWFdCu78wYtMznfo7VL9nK6wQb7ygAkzU1F47m5wRhEOY5hq1rp760?purpose=inline)

[irukadark.com](https://irukadark.com/ja?utm_source=chatgpt.com)

A typical decoder-only LLM, such as GPT-style models, follows this architecture:

1\. Input text

Example: Explain how transformers work

2\. Tokenizer

Converts text into token IDs

3\. Token embeddings + positional information

Converts token IDs into numerical vectors and represents position

Core neural network

4\. Transformer blocks (repeated N times)

Multi-Head Self-Attention

Add & Norm / Residual Connection

Feed Forward Neural Network (MLP)

Residual Connection + Layer Normalization

This block repeats many times, depending on the model architecture.

5\. Final normalization + Linear projection

Produces vocabulary-sized logits

6\. Softmax + Decoding

Converts scores into token probabilities and selects the next token

7\. Generated token

Append token and repeat generation

There are seven major stages. Let's understand each one.

# 2. Tokenization – How text becomes numbers

Neural networks cannot directly understand text.

They operate on numerical representations.

Therefore, the first step is tokenization.

Consider:

`I am building an AI application.`

A tokenizer splits the text into smaller units called tokens.

For illustration:

| Token       | Token ID |
| ----------- | -------- |
| I           | 40       |
| am          | 620      |
| building    | 1824     |
| an          | 276      |
| AI          | 955      |
| application | 3021     |

These IDs are illustrative, not actual IDs from a specific tokenizer.

The tokenizer might split a word into multiple subword tokens. For example, an uncommon word might be divided into smaller pieces.

Popular tokenization approaches:

- BPE (Byte Pair Encoding)
- WordPiece
- Unigram Language Model
- SentencePiece

Why do we need tokenization?

1. Convert text into numerical IDs.
2. Handle unknown and uncommon words.
3. Reduce vocabulary size compared with word-level tokenization.
4. Provide consistent input to the neural network.

The output is a sequence of token IDs:

```
input_ids = [40, 620, 1824, 276, 955, 3021]
```

These IDs are not embeddings. They are just identifiers.

That distinction is important.

# 3. Embedding layer – Converting token IDs into vectors

Now the model needs to convert each token ID into a meaningful numerical representation.

This is done using an embedding layer.

Suppose the vocabulary contains 50,000 tokens and the embedding dimension is 768.

The embedding matrix has this shape:

\\[ E \in \mathbb{R}^{50000 \times 768} \\]

Each token ID selects one row from this matrix.

For example:

```
token_id = 955embedding = embedding_matrix[token_id]
```

The result is a vector containing 768 floating-point values.

Conceptually:

```
AI = [    0.21,    -0.42,    0.81,    0.16,    ...]
```

Every token has its own learned vector.

These vectors are learned during model training.

Why embeddings?

Because numerical vectors allow the model to learn relationships between tokens and their contexts.

For example:

- King and Queen
- Paris and France
- Python and Programming
- Apple and Technology

However, an important distinction:

An embedding alone does not fully represent the contextual meaning of a word.

The Transformer layers create contextual representations.

For example, the word `bank` can mean a financial institution or a riverbank. Its final representation depends on the surrounding words.

# 4. Positional encoding – How does the model understand word order?

This is one of the most important concepts.

Self-attention by itself does not inherently understand the sequential order of tokens.

Consider:

- The dog chased the cat.
- The cat chased the dog.

The same tokens are present, but their order changes the meaning.

Therefore, the model needs positional information.

There are different approaches:

1. Sinusoidal positional encoding.
2. Learned positional embeddings.
3. Rotary Positional Embeddings (RoPE).

Modern decoder-only LLMs frequently use RoPE or related positional mechanisms.

[Tutorial 9: Introduction to Large Language Models and the Transformer Architecture - ADSC 4720 - Data Mining for Data Science](https://images.openai.com/static-rsc-4/WsQLsFxpj9xnkGKmZPi7NZ8CDOeWxGcPw-ETCd2W2RiOWS5aZhhXopbJ3GqebfhBqIBv8NTvRIgUDd9jdB2138lXBAQ6SPyEzLYyKeFi_FxX6g7UKi9PHkAGFNoBptGzwbE1NKTrhbLrybP4hft_WqdYW9swZbogsLID33Us3dQ?purpose=inline)

[tru-pbads.github.io](https://tru-pbads.github.io/ADSC_4720_instructors/tutorial9/?utm_source=chatgpt.com)

Conceptually, the model combines token representations with positional information:

\\[ X = E + P \\]

This equation illustrates additive positional embeddings. It is not the exact operation used by every modern architecture, particularly RoPE.

Why is this necessary?

Without positional information, the model would have difficulty distinguishing different token orders.

# 5. Self-attention – The heart of the Transformer

Most important interview topic

This is the most important component of the Transformer architecture.

Self-attention allows each token to examine other relevant tokens in the input sequence and determine how much attention it should give them.

Consider:

`The developer deployed the application because it was ready.`

When processing the word `it`, the model needs to understand what `it` refers to.

It can use relationships with other tokens, particularly `application`.

Self-attention helps the model learn these relationships.

## 5.1 How does self-attention work?

Self-attention uses three important vectors:

- Query (Q)
- Key (K)
- Value (V)

Let's understand these individually.

| Component | Meaning                                        |
| --------- | ---------------------------------------------- |
| Query (Q) | What information am I looking for?             |
| Key (K)   | What information can I offer or match against? |
| Value (V) | What information should I contribute?          |

Think about a search engine.

- Query: You search for a particular topic.
- Key: Each document has searchable characteristics.
- Value: The actual information retrieved from the document.

Self-attention works using a similar conceptual mechanism.

### Step 1: Generate Q, K and V

The input embedding matrix is multiplied by three different learned weight matrices.

\\[ Q=XW_Q \\]

\\[ K=XW_K \\]

\\[ V=XW_V \\]

Where:

- X = input representation.
- WQ = Query weight matrix.
- WK = Key weight matrix.
- WV = Value weight matrix.

These weight matrices are learned during training.

Let's take an example.

Input:

`The cat is sleeping`

The Transformer generates Query, Key and Value vectors for every token.

[What is Transformers in Deep Learning: A Guide 2026](https://images.openai.com/static-rsc-4/xO4fPox9ELTrYHdqurOpfbPtoHytHgEgmh5g8rkWfhUFfpkH9hgzv4zmeWEnGHh6LGEIkw3D6VcTWPZ3RiFzyq_UH16e76U4NB1nyfR4-DpZQ3PFTFCPpFlP8Y0-0F4Qn-xpySScL0lnZIl-65d5pOF-xekJxjxZSWjZYNm4X1o?purpose=inline)

[deeplearningcourse.co.in](https://www.deeplearningcourse.co.in/blog/transformers-in-deep-learning?utm_source=chatgpt.com)

For a sequence of four tokens, each token gets its own Q, K and V vector.

### Step 2: Calculate attention scores

Now the model compares Query vectors with Key vectors.

The attention score is calculated using the dot product.

\\[ Score = QK^T \\]

This measures how strongly one token relates to another in the learned attention space.

For example, when processing `sleeping`, the model might assign different attention scores to:

- The
- cat
- is
- sleeping

These scores indicate the relative relevance of the tokens.

### Step 3: Scale the attention scores

The scores are divided by the square root of the key dimension.

\\[ AttentionScore = \frac{QK^T}{\sqrt{d_k}} \\]

Why?

When the dimensions become large, dot products can become large as well.

This can cause the softmax function to produce extremely concentrated probabilities and make optimization difficult.

Scaling helps stabilize the computation.

### Step 4: Apply Softmax

Softmax converts attention scores into normalized probabilities.

\\[ AttentionWeights = Softmax\left(\frac{QK^T}{\sqrt{d_k}}\right) \\]

Example:

| Token    | Attention weight |
| -------- | ---------------- |
| The      | 0.05             |
| cat      | 0.15             |
| is       | 0.20             |
| sleeping | 0.60             |
| Total    | 1.00             |

These numbers are illustrative.

The model now has normalized weights describing how much information to take from each position.

### Step 5: Multiply attention weights by Values

Now we multiply the attention weights with the Value vectors.

\\[ Output = Softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V \\]

This generates a contextual representation for each token.

The complete self-attention equation is:

\\[ \boxed{ Attention(Q,K,V)= Softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V } \\]

This is one of the most important equations in LLM architecture.

## 5.2 Visualizing the attention process

[Artificial Neural Networks: A Complete Beginner's Guide - Wasil Zafar](https://images.openai.com/static-rsc-4/m8rCrTUOus6loVB2rfRm_ZuT4nSs7ZVVMFL1uOlyeFBSW6l91FKQi7COcyt53UpK_3CQmX47grjsmKlqoLH-4b8T8QNobbaGKbDmzDFomxI8FJpyc9335fMw1Nct27uzoPrmiX7_utFO93yXq9eYkBNKXP2Vn-47oEtnBK8RFrs?purpose=inline)

[wasilzafar.com](https://www.wasilzafar.com/pages/series/ai-data-science/artificial-neural-networks-guide.html?utm_source=chatgpt.com)

The complete sequence is:

1. Receive input representations.
2. Generate Query, Key and Value.
3. Calculate Query-Key similarity.
4. Scale the scores.
5. Apply softmax.
6. Multiply the attention weights with Values.
7. Produce contextual representations.

This happens across all tokens in the sequence.

Senior engineer insight: Attention is not a database search or a fixed semantic lookup. The model learns projections and attention patterns through training. The attention weights are dynamically calculated from the current input.

# 6. Multi-Head Attention – Why do we need multiple attention heads?

A single attention operation can learn certain relationships between tokens.

But language has many different types of relationships.

For example:

`The engineer who developed the application fixed the production issue.`

The model might need to learn:

- Relationship between engineer and developed.
- Relationship between developed and application.
- Relationship between fixed and issue.
- Relationship between engineer and fixed.

Using multiple attention heads allows the network to learn different projections and relationships in parallel.

[Transformers & Attention Mechanism - Complete NLP Series Part 8 - Wasil Zafar](https://images.openai.com/static-rsc-4/sXKWwFTfbfiwH3JSdQgKx_I3rFkDfSHlOKC2gMCxPQvvBrhWjlDVQrVgY05bjPEXB24oKKf2TAi5klgRS1XLQtF32OPrVCAYgfevW_63npiEzHLdwSyr8aEeqYVlFHmZlANKpMkK56gs_RV27Mm4TImTaR40cSFTs_RjxHiUTxY?purpose=inline)

[wasilzafar.com](https://www.wasilzafar.com/pages/series/nlp/nlp-transformers-attention.html?utm_source=chatgpt.com)

### How does it work?

Instead of calculating one Q, K and V set, the model calculates multiple sets.

For example, assume:

- Model dimension = 768.
- Number of attention heads = 12.
- Dimension per head = 64.

Then:

\\[ 768 = 12 \times 64 \\]

Each attention head operates on its own 64-dimensional projections.

Conceptually:

```
head_1 = attention(Q1, K1, V1)head_2 = attention(Q2, K2, V2)head_3 = attention(Q3, K3, V3)# ... up to 12 headscombined = concatenate([    head_1,    head_2,    head_3])
```

The actual implementation concatenates all heads, not just these three.

Then a final learned projection is applied:

\\[ MultiHead(Q,K,V)= Concat(head_1,\ldots,head_h)W_O \\]

Why is this useful?

Different heads can learn different contextual relationships.

Do not assume that every head has a clean, human-interpretable purpose. The learned behavior can be distributed and more complicated.

# 7. Causal Attention Masking – How GPT prevents looking into the future

This is another critical concept.

Consider this input:

`I am building an AI application`

During autoregressive generation, the model should not use future tokens to predict the current token.

For example:

When predicting the token after `I am`, it should not see `building an AI application`.

This is achieved through causal masking.

[Attention Is All You Need: The Original Transformer Architecture](https://images.openai.com/static-rsc-4/YHgV5b_n-tYkQofGJay5n1CnwTIbkecPSmnksMMqcMwNj-zIN7V9QKf-RyYedI29JZW2Wjd-CdsuWTzb0dIFBO1UA_okSSkee-yoJidnARz9icotUVjbkG69CmKDY7T2EJD1ErJYlS9AyliZR49juNa1QJWa6lrppkT8aLwLgaQ?purpose=inline)

[newsletter.theaiedge.io](https://newsletter.theaiedge.io/p/attention-is-all-you-need-the-original?utm_source=chatgpt.com)

The attention matrix is masked so that each position can attend only to itself and preceding positions.

For a sequence of four tokens:

| Query / Key | I | am   | building | AI   |
| ----------- | - | ---- | -------- | ---- |
| I           | ✓ | Mask | Mask     | Mask |
| am          | ✓ | ✓    | Mask     | Mask |
| building    | ✓ | ✓    | ✓        | Mask |
| AI          | ✓ | ✓    | ✓        | ✓    |

Mathematically, the masked attention operation can be represented as:

\\[ Attention = Softmax\left( \frac{QK^T}{\sqrt{d_k}}+M \right)V \\]

Where M contains zero for permitted positions and negative infinity for masked positions.

After softmax, masked positions receive zero probability.

Why does GPT need this?

Because GPT-style models are autoregressive language models. They generate the next token using only the preceding context.

Note that encoder models such as BERT generally use bidirectional attention instead of this GPT-style causal mask.

# 8. Feed Forward Neural Network (FFN / MLP)

After the attention operation, the Transformer applies a feed-forward neural network.

This is another fundamental component.

Attention determines how information is mixed across tokens.

The feed-forward network performs nonlinear transformations on the representations at each token position.

A traditional feed-forward network can be represented as:

\\[ FFN(x)=W_2 \cdot GELU(W_1x+b_1)+b_2 \\]

Here:

- W1 = first learned weight matrix.
- W2 = second learned weight matrix.
- GELU = activation function.
- b1 and b2 = biases.

[Feed-Forward Networks in Transformers: Architecture, Parameters & Efficiency - Interactive | Michael Brenndoerfer | Michael Brenndoerfer](https://images.openai.com/static-rsc-4/LOm0sf371SiK8wfDo__y9Z4wnsZXbIcGdpiYQCHucbsQQkNalWua59EPvtg2xFHXUbvVjW18Y31kHp9Vy8tuXpcHxC0GOzmFY4Q4lzxHubUKK3WnhF45uGLwS1gU4saBwItl7A7IhyALpzPyFetZaTPycaZIuMBiL6gCBYzmTrA?purpose=inline)

[mbrenndoerfer.com](https://mbrenndoerfer.com/writing/transformer-feed-forward-networks?utm_source=chatgpt.com)

### Example

Assume:

- Input dimension = 768.
- Hidden dimension = 3072.
- Output dimension = 768.

The process:

```
Input: 768 dimensions
       |
       v
Linear transformation
       |
       v
3072 dimensions
       |
       v
GELU activation
       |
       v
3072 dimensions
       |
       v
Linear transformation
       |
       v
Output: 768 dimensions
```

Why expand from 768 to 3072?

The larger intermediate representation provides the network with additional capacity to perform nonlinear feature transformations.

Modern architectures can use other FFN variants, including SwiGLU and gated MLPs.

Important distinction for interviews:

- Attention mixes information across sequence positions.
- FFN transforms information within each position.

# 9. Residual Connections and Layer Normalization

These two components help make deep Transformer networks trainable.

Imagine a Transformer containing 32, 48, or more layers.

Without appropriate architectural mechanisms, training very deep networks can become difficult.

## 9.1 Residual connections

A residual connection allows the original input to bypass a subnetwork.

Conceptually:

\\[ Output = x + F(x) \\]

Where F(x) is the output of the attention or FFN subnetwork.

[Gemma-3n Deep Dive: The Compact Genius of Google’s Next-Gen AI | by advi | Medium](https://images.openai.com/static-rsc-4/SmeHPsLZvUAuEaGYy38zsB_RTXVnzx-5Q7KYKYuBys2KGAfa7vAUMoVonknTeKjyLnM6QsYXQNEMYiHMCGAxd7HZopXXauB4rJI0Ot7UdtSN_jjFLwVg6lysGgwzWkt0-uO1JG7pxavRBDAZe7akOeV-BHXM3S0Mv4jmcTqZrwM?purpose=inline)

[medium.com](https://medium.com/%40learnwithadvi/gemma-3n-deep-dive-the-compact-genius-of-googles-next-gen-ai-b4e8c02cecb2?utm_source=chatgpt.com)

Why do we need it?

1. Helps information flow across layers.
2. Improves gradient propagation.
3. Makes deeper networks easier to optimize.
4. Allows a layer to learn modifications to its input instead of reconstructing everything.

## 9.2 Layer Normalization

Layer normalization normalizes activations across the feature dimension of each token representation.

It helps stabilize the distribution of activations during training.

A simplified representation is:

\\[ LayerNorm(x)= \gamma\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta \\]

Where:

- μ = mean.
- σ² = variance.
- γ = learned scale.
- β = learned shift.
- ε = numerical stability constant.

### Pre-Norm versus Post-Norm

There are two common arrangements.

| Architecture | Processing                     |
| ------------ | ------------------------------ |
| Post-Norm    | Sublayer → Residual Add → Norm |
| Pre-Norm     | Norm → Sublayer → Residual Add |

Many modern decoder-only LLMs use Pre-Norm-style designs, often with RMSNorm instead of LayerNorm.

RMSNorm normalizes using the root mean square rather than subtracting the mean and computing variance.

# 10. Complete Transformer Block

Now let's combine everything.

A typical Pre-Norm decoder-only Transformer block looks like this:

[Generalized Language Models | Lil'Log](https://images.openai.com/static-rsc-4/RzKEmtj9D6Jbj_2GMlVVdpzZNU06xojykb3-0dOLuMCQYJGCdyZyT_bpt3bVEWXqZXMSMl9gMY90PJGA7oAFJgwbs9HqmiX5fl7n6SUwCz8d5a-zRxPI8zdG3VIouHemu3IoaT-kE-aqmXck5KNAWAiiHCFiC1GMjC9xjVTfXq0?purpose=inline)

[lilianweng.github.io](https://lilianweng.github.io/posts/2019-01-31-lm/?utm_source=chatgpt.com)

The processing flow is:

```
Input hidden states
       |
       v
Normalization
       |
       v
Causal Multi-Head Self-Attention
       |
       v
Residual Connection
       |
       v
Normalization
       |
       v
Feed Forward Network (MLP)
       |
       v
Residual Connection
       |
       v
Output hidden states
```

This block is repeated multiple times.

For example, imagine a hypothetical LLM containing 32 Transformer blocks.

```
Token Embeddings
       |
       v
Transformer Block 1
       |
       v
Transformer Block 2
       |
       v
Transformer Block 3
       |
       |
       v
Transformer Block 32
       |
       v
Final Normalization
       |
       v
Output Projection
       |
       v
Next Token Prediction
```

Each block transforms the hidden representations further.

The model learns increasingly sophisticated representations across its depth, although it is not accurate to assume that each layer has one specific human-defined linguistic responsibility.

# 11. Output Projection – How does the LLM predict the next token?

We have now passed through all the Transformer blocks.

The model has a final hidden representation for each token position.

But this representation is not yet a predicted word.

We need to convert it into vocabulary-sized scores.

This is done using a linear output projection, also called the language-model head.

Suppose:

- Vocabulary size = 50,000.
- Hidden dimension = 768.

The output projection maps the hidden representation into 50,000 scores.

\\[ Logits = hW\_{vocab}^{T}+b \\]

Where h is the final hidden representation.

The output shape for a sequence of length T is:

\\[ [T, 50000] \\]

Each score is called a logit.

## 11.1 Softmax

The model converts logits into probabilities using softmax.

\\[ P(token_i)= \frac{e^{z_i}}{\sum_j e^{z_j}} \\]

For example:

| Candidate token | Probability |
| --------------- | ----------- |
| AI              | 0.45        |
| Machine         | 0.25        |
| Software        | 0.15        |
| Technology      | 0.10        |
| Other           | 0.05        |

These are illustrative probabilities.

The model now has a probability distribution over the possible next tokens.

## 11.2 Decoding – Selecting the next token

The model needs a strategy to select a token from the probability distribution.

Popular decoding methods include:

| Method          | How it works                              |
| --------------- | ----------------------------------------- |
| Greedy decoding | Selects the highest-probability token     |
| Temperature     | Controls randomness in the distribution   |
| Top-K sampling  | Samples from the K most likely tokens     |
| Top-P sampling  | Samples from a probability mass threshold |
| Beam search     | Maintains multiple candidate sequences    |

For example:

```
next_token = argmax(probabilities)
```

This is greedy decoding.

However, production LLMs often use sampling strategies depending on the use case.

For example:

- Code generation may use relatively conservative sampling.
- Creative writing may use more varied sampling.
- Structured JSON generation may use constrained decoding.

# 12. How does an LLM actually generate a complete response?

Let's understand the entire inference process.

User sends:

`What is machine learning?`

The tokenizer converts the text into token IDs.

The model processes those IDs and predicts the next token.

Let's simplify the process:

Autoregressive token generation

1. Input

   `What is machine learning?`
2. First prediction

   `Machine`
3. Second prediction

   `learning`
4. Third prediction

   `is`
5. Fourth prediction

   `a`
6. Fifth prediction

   `field`
7. Continue

   The model keeps predicting tokens until it reaches a stopping condition.

Final generated response

Machine learning is a field of artificial intelligence that enables systems to learn patterns from data.

The actual model predicts the next token based on the entire available context, not by looking at one word in isolation.

## 12.1 Why is inference expensive?

Because generating a response requires repeated model computation.

For example, if an LLM generates 500 tokens, it must perform 500 sequential next-token prediction steps.

There is an important optimization called KV caching.

### What is KV Cache?

During attention computation, the model generates Key and Value representations for tokens.

When generating the next token, previously calculated Keys and Values do not need to be recomputed.

They can be cached.

[Vizuara AI Pods — Learn AI from First Principles](https://images.openai.com/static-rsc-4/9r-oehVuQbz4TWKgVGrZCzcn63T6CBy1-xzDjETnQfOpBvuwgTR-S33Za16Vqd-gnPquKw21EfQ-IQy9lkK9AaCol9q47MGrEbAyNa0cbJ4GBSjwN_GdereqaDS-EtDuHXxiuwARiiT0OesXBIlOZ59DJrEZKc5yQvcSUCAZSQ4?purpose=inline)

[pods.vizuara.ai](https://pods.vizuara.ai/courses/ai-in-production/kv-cache-the-necessary-evil-of-llm-inference/article?utm_source=chatgpt.com)

Without KV caching:

- Repeatedly recompute previous token representations.
- More computation.
- Higher latency.

With KV caching:

- Store previous Keys and Values.
- Calculate new Keys and Values for the newly generated token.
- Reuse the cached representations.

This significantly reduces redundant computation during autoregressive decoding.

However, KV cache consumes GPU memory, and its memory requirements increase with context length, batch size, and number of layers.

Staff-level insight: KV caching improves decoding efficiency, but it does not eliminate the sequential dependency between generated tokens. This is why optimizing time-to-first-token and time-per-output-token requires different techniques.

# 13. How is an LLM trained?

So far, we have discussed inference: how an already-trained LLM generates responses.

Now let's understand how the model learns.

LLM development generally involves multiple stages.

[AI Engineering (1/3): Foundation Models, Evaluation, and Model Selection | by Marina Wyss | Data Science Collective | Medium](https://images.openai.com/static-rsc-4/-ib5PgT1WkvUcXihBodiwNRcSXFEi34o-X2GmO-9N1CpAWovV9wHoCcLIMzvbtAfvKbYDHWa86ZOeFMJclWIWTdr8mPSlqtlC8OmM-CsUilIW_05hTLkRalPksvHvJ0t23TPF6RJrbLRd0FBOLrR2ssFLcG7ZaO6vOoWAciiPMo?purpose=inline)

[medium.com](https://medium.com/data-science-collective/ai-engineering-1-3-foundation-models-evaluation-and-model-selection-889afd06b0ba?utm_source=chatgpt.com)

## Stage 1: Pretraining

This is where the model learns general language patterns from massive datasets.

Training data may include:

- Books.
- Articles.
- Public web data.
- Code.
- Technical documentation.
- Other licensed or appropriately sourced datasets.

For an autoregressive language model, the main objective is next-token prediction.

Example:

Input:

`The capital of France is`

Expected next token:

`Paris`

The model predicts a probability distribution.

The training process calculates the loss between its predictions and the actual next token.

### Cross-Entropy Loss

A common objective is:

\\[ L=-\sum\_{i=1}^{N}y_i\log(p_i) \\]

Where:

- y = target distribution.
- p = predicted probability distribution.
- N = number of prediction positions.

The model updates its weights using backpropagation and an optimizer such as AdamW.

Training cycle:

```
Training Dataset
      |
      v
Tokenization
      |
      v
Forward Pass
      |
      v
Next Token Prediction
      |
      v
Calculate Loss
      |
      v
Backpropagation
      |
      v
Calculate Gradients
      |
      v
Optimizer Updates Weights
      |
      v
Repeat over training batches
```

Over many training iterations, the model learns patterns in language and data.

## Stage 2: Supervised Fine-Tuning (SFT)

Pretraining does not automatically make a model a good conversational assistant.

SFT trains the pretrained model using curated input-output examples.

Example:

| Input          | Expected output                |
| -------------- | ------------------------------ |
| Explain Python | A useful Python explanation    |
| Write SQL      | A valid SQL query              |
| Explain RAG    | A RAG architecture explanation |

The model learns to follow instructions and produce more useful responses.

## Stage 3: Preference Optimization

Now the model needs to learn which responses humans prefer.

For example:

Prompt:

`Explain database indexing.`

Response A: Accurate, clear, well-structured.

Response B: Incorrect and confusing.

Human preference data can indicate that A is better than B.

Common approaches include:

- RLHF (Reinforcement Learning from Human Feedback).
- DPO (Direct Preference Optimization).
- Other preference-optimization methods.

In a traditional RLHF pipeline, a reward model can learn from preference comparisons, and reinforcement learning can optimize the language model against that reward.

DPO directly optimizes the model using preference pairs without requiring the same separate reward-model-plus-RL optimization pipeline.

## Stage 4: Evaluation and Deployment

Before production deployment, models are evaluated for:

- Accuracy.
- Reasoning capabilities.
- Instruction following.
- Hallucination.
- Safety.
- Latency.
- Token generation throughput.
- Cost.

Then they are deployed using model-serving infrastructure.

Important: Not every LLM uses precisely these training stages. Training pipelines vary by model provider and architecture.

# 14. Different types of Transformer architectures

This is a frequently asked interview question.

There are three major architectural categories.

[The Illustrated Transformer – Jay Alammar – Visualizing machine learning one concept at a time.](https://images.openai.com/static-rsc-4/b1wZ30_ofAWmZD1o2sO2FFsbv5dZOXWREMRjIyOPBl02GfYLS0axCjORwPaBVuhDfPnHXSKIvL8KWwWB6dR-5cbm1H0BQy4oVz8vlPCY_MX0PJlI4-MY40weVC60CGswMUoDpFoOd4IXTVV0kEOQqGybQOixW_Q4SgLlvZFF3gM?purpose=inline)

1\. Encoder-only Transformer

Example: BERT.

- Bidirectional attention.
- Understands surrounding context.
- Commonly used for classification, embeddings and language understanding.

[Understanding the Open Pre-Trained Transformers (OPT) Library | by Cameron R. Wolfe, Ph.D. | TDS Archive | Medium](https://images.openai.com/static-rsc-4/M0c0bCRB0Q2_pRTmnnEzuPtse7wvuTnYd1Ds8jnqQOS7PDhL8RZMFfASWABeCK33QH96JjUqWz3_fGohQAj5ZnKu_UbwaIpyX8eifwxWciFVulgNzaOMzrPkibnOxo399KRcuPOhwgegel-qOcLzTcx1PcmL0KtZHMABtmxYPhQ?purpose=inline)

2\. Decoder-only Transformer

Examples: GPT, Llama.

- Causal self-attention.
- Autoregressive next-token prediction.
- Used for text generation and conversational assistants.

[GeoAI for Large-Scale Image Analysis and Machine Vision: Recent Progress of Artificial Intelligence in Geography | MDPI](https://images.openai.com/static-rsc-4/uC2PzLwBJU-2tpS4MvIXuw0M1VyVvc0u6Z13a9afTyhJ_OQ_yYoWduG7V6cl6C3KsNPWaZAu85fhJtiqGKVJv0yXPut8XzAj0hC_LzWXT_1mOSvQlV-7BbgWngzNiz3boewqjnKf2wcMSLmp6McEdO1Ouh6Hb2maBqzxmRBbyaU?purpose=inline)

3\. Encoder-Decoder Transformer

Example: T5.

- Encoder processes input.
- Decoder generates output.
- Cross-attention connects decoder representations to encoder outputs.
- Common in translation and sequence-to-sequence tasks.

### Difference between self-attention and cross-attention

This is important.

| Self-attention                                        | Cross-attention                                 |
| ----------------------------------------------------- | ----------------------------------------------- |
| Q, K and V come from the same sequence representation | Q comes from decoder; K and V come from encoder |
| Models relationships within a sequence                | Connects two different sequences                |
| Used in encoder and decoder blocks                    | Used in encoder-decoder architectures           |

# 15. How does the complete LLM architecture look in production?

Now let's move beyond the neural network itself.

As a Senior AI Engineer, you should understand the difference between:

1. LLM model architecture.
2. LLM training infrastructure.
3. LLM inference infrastructure.
4. LLM application architecture.

These are related, but they are not the same thing.

[A survey on Inference Serving for Large Language Models | hello@woonyee.org \~ $](https://images.openai.com/static-rsc-4/mtI2Ym9aLJNhSDss-rjN1nDnpPN-x-172qUsrx18LLKPhGxMMrXKNTuAzSO3i0lEj80kHwGcVJMF1dcDX_0Y6RyW40amlHOHzrwN9fw-t3-zVAd0ZPsV9XTN22k_G4ehLgsgnGcZjlrOFAoEaRQmhB3XPhNuxkagDDIu2bPiYSg?purpose=inline)

[woonyee.org](https://woonyee.org/projects/ml-inference-sys-review/?utm_source=chatgpt.com)

A typical production inference system looks like this:

Client / Application

Web application, mobile app, AI agent

API Gateway / Load Balancer

Authentication, rate limiting, routing

LLM Serving Layer

Request validation, batching, scheduling, streaming

Inference Engine

Tokenizer

Scheduler

KV Cache

Decoding

GPU / Accelerator Model Execution

Transformer layers, attention and MLP computation

Output Processing

Detokenization, structured output validation, streaming

Response to Client

Observability runs across the complete system: metrics, logs, traces, GPU utilization, token usage, latency and errors.

## 15.1 Important production components

| Component           | Responsibility                       |
| ------------------- | ------------------------------------ |
| API Gateway         | Authentication and rate limiting     |
| Load Balancer       | Distributes requests                 |
| Tokenizer           | Converts text to token IDs           |
| Scheduler           | Manages concurrent requests          |
| Continuous Batching | Combines requests efficiently        |
| KV Cache            | Stores previously calculated K/V     |
| GPU Inference       | Executes Transformer computation     |
| Decoder             | Selects the next token               |
| Streaming           | Sends generated tokens progressively |
| Observability       | Tracks latency, errors and GPU usage |

### Popular LLM serving technologies

- vLLM – efficient serving, continuous batching and KV-cache management.
- Hugging Face TGI – LLM inference serving.
- TensorRT-LLM – GPU-optimized inference.
- SGLang – efficient LLM and structured generation serving.

These are inference frameworks, not different Transformer architectures.

# 16. How does the Transformer run mathematically? (Tensor dimensions)

This is where you move from conceptual understanding to engineering understanding.

Let's take a hypothetical decoder-only LLM.

| Parameter                  | Value  |
| -------------------------- | ------ |
| Vocabulary size            | 50,000 |
| Sequence length            | 1,024  |
| Hidden dimension           | 768    |
| Transformer layers         | 32     |
| Attention heads            | 12     |
| Head dimension             | 64     |
| FFN intermediate dimension | 3,072  |

Now follow the tensor shapes.

| Operation          | Tensor shape          |
| ------------------ | --------------------- |
| Input token IDs    | `[1, 1024]`           |
| Token embeddings   | `[1, 1024, 768]`      |
| Query              | `[1, 12, 1024, 64]`   |
| Key                | `[1, 12, 1024, 64]`   |
| Value              | `[1, 12, 1024, 64]`   |
| Attention scores   | `[1, 12, 1024, 1024]` |
| Attention output   | `[1, 1024, 768]`      |
| FFN intermediate   | `[1, 1024, 3072]`     |
| Transformer output | `[1, 1024, 768]`      |
| Vocabulary logits  | `[1, 1024, 50000]`    |

Here, `1` represents the batch size.

Notice something important:

The attention scores have dimensions:

\\[ [Batch,Heads,Sequence,Sequence] \\]

This is why self-attention can become computationally expensive for long sequences.

For standard dense attention, the attention computation scales approximately as:

\\[ O(n^2d) \\]

Where:

- n = sequence length.
- d = hidden dimension.

If you double the sequence length, the attention score matrix becomes four times larger.

This is one of the reasons long-context LLM inference is challenging.

# 17. Python implementation – Simplified Transformer components

Let's connect the concepts to actual engineering code using PyTorch.

This is an educational implementation of scaled dot-product attention and a Transformer block. It is not a complete production LLM.

```
import torchimport torch.nn as nnimport torch.nn.functional as Fclass MultiHeadSelfAttention(nn.Module):    def __init__(self, d_model=768, num_heads=12):        super().__init__()        assert d_model % num_heads == 0        self.num_heads = num_heads        self.head_dim = d_model // num_heads        self.q_proj = nn.Linear(d_model, d_model)        self.k_proj = nn.Linear(d_model, d_model)        self.v_proj = nn.Linear(d_model, d_model)        self.out_proj = nn.Linear(d_model, d_model)    def forward(self, x):        batch, seq_len, d_model = x.shape        # Generate Query, Key and Value        q = self.q_proj(x)        k = self.k_proj(x)        v = self.v_proj(x)        # Split into attention heads        q = q.view(            batch, seq_len,            self.num_heads, self.head_dim
```

What does this implementation demonstrate?

1. Token embedding.
2. Positional embedding.
3. Query, Key and Value projection.
4. Multi-head attention.
5. Causal masking.
6. Feed-forward neural network.
7. Residual connections.
8. Layer normalization.
9. Output projection.

For an actual GPT-style implementation, you would also need to address RoPE or another positional mechanism, a suitable normalization design, model initialization, training, weight tying where appropriate, efficient attention, KV caching, and decoding.

# 18. Important differences you should understand as a Senior AI Engineer

These concepts are often confused during interviews.

| Concept              | What it actually does                      |
| -------------------- | ------------------------------------------ |
| Tokenization         | Converts text into token IDs               |
| Embedding            | Converts token IDs into vectors            |
| Positional encoding  | Provides sequence-position information     |
| Self-attention       | Mixes information across tokens            |
| Multi-head attention | Performs multiple attention projections    |
| Causal masking       | Prevents access to future tokens           |
| FFN / MLP            | Transforms token representations           |
| Residual connection  | Provides skip connections                  |
| LayerNorm / RMSNorm  | Normalizes hidden representations          |
| Logits               | Raw vocabulary prediction scores           |
| Softmax              | Converts scores into probabilities         |
| Decoding             | Selects the next token                     |
| KV cache             | Reuses previously computed Keys and Values |

# 19. Top 15 Transformer and LLM interview questions

For senior and staff-level AI engineering interviews, I would prioritize these questions.

Interview preparation

0/15 explored

1.What is the Transformer architecture?

2.Why did Transformers replace RNNs and LSTMs in many NLP tasks?

3.Explain Query, Key and Value.

4.What is scaled dot-product attention?

5.What is multi-head attention?

6.What is causal masking?

7.What is the difference between self-attention and cross-attention?

8.Why do Transformers need positional encoding?

9.What is the difference between LayerNorm and RMSNorm?

10.Why do Transformers use residual connections?

11.What is KV caching?

12.Why does attention have quadratic complexity?

13.What is the difference between pretraining and fine-tuning?

14.What is the difference between temperature, Top-K and Top-P?

15.How would you optimize LLM inference latency?

# 20. What you should learn next

Given your focus on building production AI systems with FastAPI, RAG, LangGraph, and vLLM, I recommend learning the following sequence:

## LLM engineering learning roadmap

1. Transformer internals

   Tokenization, embeddings, attention, RoPE, RMSNorm, FFN, residuals.

   Start here
2. LLM training

   Pretraining, cross-entropy, backpropagation, AdamW, SFT, LoRA, QLoRA, DPO.
3. LLM inference

   Prefill, decoding, KV cache, continuous batching, quantization, speculative decoding.
4. LLM infrastructure

   vLLM, GPU memory, tensor parallelism, pipeline parallelism, Kubernetes GPU deployment.
5. Production LLM applications

   FastAPI, RAG, tool calling, structured output, evaluation, tracing, model routing and fallbacks.

My recommendation: Don't stop at understanding what Query, Key and Value mean. You should be able to explain how the tensors are created, what their dimensions are, how attention consumes GPU memory, why KV caching improves inference, and how a serving engine schedules concurrent requests.

That is the difference between knowing LLM terminology and understanding LLM engineering at a senior/staff level.