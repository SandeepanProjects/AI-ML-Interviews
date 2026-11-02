# Why do Transformers need positional information?

This is one of the most important concepts in Transformer architecture and a common Senior AI Engineer interview question.

Let's understand it from the basics.

## 1. The main problem: Transformers don't naturally understand word order

Consider these two sentences:

1. The dog chased the cat.
2. The cat chased the dog.

Both sentences contain exactly the same words:

- The
- dog
- chased
- cat

But their meanings are completely different because the word order has changed.

The dog is chasing the cat in the first sentence, whereas the cat is chasing the dog in the second.

Now, here's the problem:

Self-attention, by itself, does not understand the position or order of tokens.

It calculates relationships between tokens using Query, Key and Value vectors. Without additional positional information, it has no inherent indication that `dog` appeared before `chased` or that `cat` appeared after it.

This is why Transformers need positional information.

## 2. Why does self-attention not understand word order?

Remember the self-attention formula:

\\[ Attention(Q,K,V)=Softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V \\]

The attention mechanism calculates how strongly each token should pay attention to other tokens.

But notice something: the formula does not directly contain token positions such as:

- Position 0
- Position 1
- Position 2
- Position 3

It processes token representations and their relationships, not their original sequence indices.

For example:

Sentence A

The

Pos 0

dog

Pos 1

chased

Pos 2

cat

Pos 3

Sentence B

The

Pos 0

cat

Pos 1

chased

Pos 2

dog

Pos 3

Same vocabulary, different order and meaning.

If we only give the model token representations without position information, it cannot reliably distinguish the roles that depend on word order.

There is a deeper mathematical reason: standard self-attention without positional information is permutation-equivariant. If you reorder the input tokens, the output representations reorder in the same way. It doesn't independently know that the order changed.

## 3. How do we solve this problem?

We provide positional information to the Transformer along with token information.

The original Transformer uses Positional Encoding.

The idea is simple:

- Token embedding tells the model what the token is.
- Positional encoding tells the model where the token is located.

We combine them.

\\[ \boxed{Input\ Representation = Token\ Embedding + Positional\ Encoding} \\]

For example:

## How positional information is added

Token Embedding

Meaning / identity of the token

Positional Encoding

Position / order of the token

Final Input Representation

Token meaning + position information

This is the input passed into the first Transformer layer.

Let's take a small example:

| Token  | Token embedding     | Position |
| ------ | ------------------- | -------- |
| The    | Meaning of "The"    | 0        |
| dog    | Meaning of "dog"    | 1        |
| chased | Meaning of "chased" | 2        |
| cat    | Meaning of "cat"    | 3        |

Each token gets its own positional vector, which is added to its embedding.

Now the Transformer can use both the token's meaning and its location when calculating attention.

## 4. How does positional encoding work in the original Transformer?

The original Transformer paper introduced Sinusoidal Positional Encoding.

It uses sine and cosine mathematical functions to generate a unique positional vector for each position.

The formulas are:

For even dimensions:

\\[ PE(pos,2i)=\sin\left(\frac{pos}{10000^{2i/d\_{model}}}\right) \\]

For odd dimensions:

\\[ PE(pos,2i+1)=\cos\left(\frac{pos}{10000^{2i/d\_{model}}}\right) \\]

Don't worry if these formulas look complicated. Let's understand the intuition.

- `pos` = position of the token in the sequence.
- `i` = dimension index.
- `d_model` = embedding dimension.
- `sin` and `cos` = functions that generate values between -1 and +1.

Why use sine and cosine?

Because they generate different patterns at different frequencies.

- Some dimensions change quickly as position increases.
- Other dimensions change slowly.
- Together, these patterns give each position a distinctive representation.

[Transformer Architecture: From Concepts to Code | by Sachin Pant | Data And Beyond | Dec, 2025 | Medium](https://images.openai.com/static-rsc-4/cOeseE0M7d_nZW5u81ytO6UVxocg3fSAGvuBDDdvLRYOOY_bHdQ51tVkwV4hiPp2ndu5n0fETJ7g1ZmBV6uF-GL_vHT2BsaoPT-5-Dj3P65spLFvySu1YXucKNNIQoNcV3CtglBFQF874PZ1fuWmIKmEhsDKdQaQgnR59TmQCjs?purpose=inline)

[medium.com](https://medium.com/data-and-beyond/transformer-architecture-from-concepts-to-code-fb3ac99c7396?utm_source=chatgpt.com)

Think of it like this:

Imagine assigning a unique mathematical fingerprint to every position in a sentence.

Position 0 gets one fingerprint, position 1 gets another, position 2 gets another, and so on.

This gives the model information about token positions and helps it learn relationships between nearby and distant positions.

Important: Sinusoidal encoding does not explicitly tell the model "this is the subject" or "this is the verb." It provides position-related signals; the model learns how to use them during training.

## 5. Different types of positional information

There are several approaches used in Transformer models.

### A. Absolute positional encoding

This tells the model the actual position of each token.

For example:

| Token | Position |
| ----- | -------- |
| I     | 0        |
| love  | 1        |
| AI    | 2        |

The model receives the token embedding plus a positional representation for position 0, 1 or 2.

There are two common approaches:

- Sinusoidal positional encoding: Generated using sine and cosine functions. Used in the original Transformer.
- Learned positional embeddings: Position vectors are learned during model training, just like other model parameters.

For example, BERT uses learned absolute position embeddings.

### B. Relative positional encoding

Instead of focusing only on the absolute position of a token, the model receives information about the distance or relative position between tokens.

For example:

`The dog chased the cat`

The distance between `dog` and `chased` is one position, while the distance between `dog` and `cat` is two positions.

This helps the model reason about relationships such as:

- Which token is immediately before another token?
- Which token is five positions away?
- Is another token to the left or right?

### C. RoPE (Rotary Positional Embedding)

Important for modern LLM interviews

RoPE is a positional encoding technique used in many modern decoder-only LLM architectures.

Examples include Llama-family models.

Instead of simply adding a positional vector to token embeddings, RoPE applies position-dependent rotations to the Query and Key vectors inside the attention mechanism.

The intuition:

Imagine Query and Key vectors as arrows in a mathematical space.

RoPE rotates these vectors based on their token positions.

When the attention mechanism calculates the dot product between Query and Key, the result incorporates their relative positional relationship.

In simplified terms:

\\[ Q'=R_m Q \\]

\\[ K'=R_n K \\]

Here:

- \\(R_m\\) = rotation associated with Query position \\(m\\).
- \\(R_n\\) = rotation associated with Key position \\(n\\).
- \\(Q'\\) and \\(K'\\) = position-aware Query and Key vectors.

The dot product between these rotated vectors depends on the relative position \\(m-n\\).

Why is RoPE useful?

1. It incorporates positional information directly into attention.
2. It naturally represents relative position relationships.
3. It works well with autoregressive language models.
4. It can support longer contexts when combined with appropriate context-extension techniques.

[RoPE（Rotary Position Embedding）を理解するまで](https://images.openai.com/static-rsc-4/-yXjZ6wiYIl46LlkzTIa3up8WMaOQX5ZtUV5bBt7GcPcSxx_rhQ94H0vv7qHpO09wdQBfkQaCt3TWuaFeypy9KvtjXCYhmXTqB9hCI3I0dYEoYBKdEUnKfuQtuHV298V4EOzapTZ-DwZIx7TsbQ6gDOXwil7VMyaY0BiB9HtlNc?purpose=inline)

[zenn.dev](https://zenn.dev/amasashi/articles/6d2d380e483ffc?utm_source=chatgpt.com)

One important distinction: RoPE rotates Q and K, not V, in the standard implementation.

## 6. What happens if we remove positional information?

Let's consider:

Sentence 1: The boy loves the girl.

Sentence 2: The girl loves the boy.

Without positional information, self-attention has no built-in knowledge of the sequence order.

The model would struggle to distinguish these sentences because the same token representations are present, just in a different arrangement.

With positional information:

- The model gets token identity.
- The model gets token position.
- Attention can learn which tokens relate to each other and how their order affects meaning.

This is particularly important for:

- Language translation.
- Text summarization.
- Question answering.
- Code generation.
- Conversational AI.
- Long-context document understanding.

## 7. Where exactly is positional information used in a Transformer?

For the original Transformer architecture, the flow is:

Input tokens

The | dog | chased | cat

Token Embeddings

Positional Encoding

Combined Input Representation

Multi-Head Self-Attention

Feed-Forward Network

Output representations

The Transformer layer also includes residual connections and normalization around its sublayers.

For a GPT-style decoder-only model using RoPE, the implementation is different: token embeddings enter the Transformer, and RoPE is applied to the Query and Key vectors within each attention layer.

## 8. Simple Python example

Here's a small example showing how token embeddings and positional embeddings are combined.

```
import torchimport torch.nn as nnvocab_size = 10000embedding_dim = 128max_sequence_length = 512# Token embeddingtoken_embedding = nn.Embedding(    vocab_size,    embedding_dim)# Learned positional embeddingposition_embedding = nn.Embedding(    max_sequence_length,    embedding_dim)# Example token IDsinput_ids = torch.tensor([    [10, 25, 30, 45]])# Generate position IDs: 0, 1, 2, 3positions = torch.arange(    input_ids.shape[1]).unsqueeze(0)# Convert tokens into embeddingstoken_vectors = token_embedding(input_ids)# Convert positions into embeddingsposition_vectors = position_embedding(positions)# Combine both representations
```

Output:

```
torch.Size([1, 4, 128])
```

The shape remains the same because both embeddings have the same dimensions.

This is a simplified example of learned absolute positional embeddings, not the sinusoidal implementation or RoPE.

## 9. Interview-ready answer

If an interviewer asks, "Why do Transformers need positional encoding?", you can answer:

Your interview answer

Transformers use self-attention to understand relationships between tokens. However, standard self-attention does not inherently capture the order of tokens in a sequence.

For example, "The dog chased the cat" and "The cat chased the dog" contain the same words but have different meanings because their word order is different.

To solve this problem, Transformers incorporate positional information along with token embeddings.

In the original Transformer, sinusoidal positional encoding was used. It generates position-dependent vectors using sine and cosine functions, which are added to token embeddings.

There are also learned positional embeddings and relative positional encoding techniques.

Modern LLMs such as Llama use RoPE, or Rotary Positional Embedding, which applies position-dependent rotations to Query and Key vectors inside the attention mechanism.

So, positional information helps Transformers understand token order, relative positions and sequence structure.

## 10. Important interview questions

Test your understanding

1\. Does self-attention understand word order by itself?

2\. What is the difference between token embedding and positional encoding?

3\. Why does the original Transformer use sine and cosine?

4\. What is the difference between absolute and relative positional encoding?

5\. What is RoPE?

6\. Is positional encoding the same as causal masking?

Remember this one-line summary for interviews:

Token embedding tells the Transformer what a token is, while positional information tells it where the token is. Both are necessary for understanding sequence structure.

Recommended next topic: Learn RoPE in depth, including how rotating Q and K produces relative position information. This is particularly important when explaining the architecture of modern LLMs such as Llama.