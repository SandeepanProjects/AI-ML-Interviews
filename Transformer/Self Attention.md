# Self-Attention, Query, Key and Value (Q, K, V) – Complete Explanation for Senior AI Engineer Interviews

This is probably the most important concept to understand in Transformer architecture.

If an interviewer asks you about Transformers, there is a good chance they will ask:

1. What is self-attention?
2. Why do we need self-attention?
3. What are Query, Key and Value?
4. How are Q, K and V calculated?
5. How does the attention score work?
6. Why do we apply Softmax?
7. Why do we divide by the square root of the dimension?
8. What is the difference between self-attention and cross-attention?

Let's understand everything from the beginning, using simple examples, diagrams, mathematics and Python code.

# 1. What is Self-Attention?

Self-attention is a mechanism that allows every token in a sentence to understand its relationship with other tokens in the same sentence.

Let's take an example:

```
The animal didn't cross the road because it was tired.
```

What does the word `it` refer to?

It refers to `animal`.

But how does a machine understand that?

It needs to understand the relationship between:

- `it`
- `animal`
- `tired`

Self-attention helps the model calculate how much attention one token should give to other tokens.

For example:

[Transformers & Attention Mechanism - Complete NLP Series Part 8 - Wasil Zafar](https://images.openai.com/static-rsc-4/RIIl0vd6gwNHo6wO_qxGIA_i5C65Bedc5hDDFrcj46yh9xSeD8Mpnp44pkE5zTCFd2kmCJUR7rSjCc_UIxA9VYgGY6TrrUme3IK_dbLaXgimYdfiro3_IckR7cBfZNCHlSeHBKsRjiXmvgXjB6Bsn4BtKVnMet2V2taPqUSepSA?purpose=inline)

[wasilzafar.com](https://www.wasilzafar.com/pages/series/nlp/nlp-transformers-attention.html?utm_source=chatgpt.com)

The word `it` can assign a higher attention weight to `animal` because that token is more relevant to understanding its context.

This is the basic idea behind self-attention.

## Another simple example

Consider:

```
I deposited money in the bank.
```

And:

```
I sat near the river bank.
```

The word `bank` has different meanings in these sentences.

In the first sentence, `bank` means a financial institution.

In the second sentence, `bank` means the land beside a river.

Self-attention helps the model use surrounding words to construct a context-aware representation of `bank`.

Important interview point: The same token can have different contextual representations depending on the surrounding tokens.

# 2. Why Do We Need Self-Attention?

Before Transformers, models such as RNNs and LSTMs processed sequences sequentially.

For example:

```
I → love → machine → learning
```

This created challenges:

- Long-range dependencies could be difficult to learn.
- Processing was sequential.
- Training parallelization was limited.

Transformers introduced self-attention to address these limitations.

Consider:

```
The developer created an application using Python.
```

The model needs to understand relationships such as:

- Developer → created.
- Created → application.
- Application → Python.

Self-attention allows each token to calculate relationships with other tokens directly.

[I Finally Understood Transformers After This One Simple Visualization (You Will Too) | by Abduldattijo | ILLUMINATION | May, 2026 | Medium](https://images.openai.com/static-rsc-4/vrKU215OolrsL0Kc-Co_2dOLLane0e1GB1p5TREOg0Cmadgx06VxUXGQh7klxTHVx9-7Dyb2-xymZkI5VYK9HuHT9F77QrsNi2eFP_OeevhXalB82bSYgGGdwWaVnBPXppH05amJ9DEuSZnKbEheffRsjQgedu9PRLvn-C2xzlg?purpose=inline)

[medium.com](https://medium.com/illumination/i-finally-understood-transformers-after-this-one-simple-visualization-you-will-too-14adf4661a22?utm_source=chatgpt.com)

Benefits:

1. Captures relationships between distant tokens.
2. Builds contextual representations.
3. Allows parallel computation across input positions during training.
4. Supports multiple types of relationships through attention heads.

# 3. What Are Query, Key and Value?

Most important interview concept

Now let's understand Q, K and V.

Imagine you are searching for a document in a database.

You enter a search query:

```
Find documents related to AI architecture.
```

The database has multiple documents.

Each document has:

- A description of what it contains.
- Actual content.

The system compares your query with the document descriptions to determine which documents are relevant.

Then it retrieves the actual content from those documents.

This is a useful analogy for understanding Q, K and V.

| Component | Simple meaning                     | Analogy                    |
| --------- | ---------------------------------- | -------------------------- |
| Query (Q) | What information am I looking for? | Search query               |
| Key (K)   | What information do I contain?     | Document index/description |
| Value (V) | What information can I provide?    | Actual document content    |

Let's understand each one.

## A. Query (Q)

Query represents what a particular token is looking for from other tokens.

For example:

```
The developer builds AI applications.
```

Suppose the current token is `developer`.

Its Query represents the kind of contextual information it needs from other tokens.

Think of it as asking:

"What other tokens are relevant to understanding me?"

## B. Key (K)

Key represents the information used to determine how relevant each token is to a Query.

For example:

The token `builds` has its own Key representation.

The model compares the Query of `developer` with the Keys of all tokens.

It calculates which tokens are relevant.

Think of it as:

"What information do I contain that might match another token's Query?"

## C. Value (V)

Value contains the information that will be combined based on the attention scores.

Once the model determines which tokens are relevant using Query and Key, it uses their Values to construct the final attention output.

Think of it as:

"Here is the actual information I contribute."

[Understanding The Self-Attention Mechanism](https://images.openai.com/static-rsc-4/MOIsSxCB3GJaU3pf2k-gtPzFHJdgPP2utFWbw1WaJjoyS9EVD4gu_58qlohSHDmeNRaIX1ZIo3kfSUYql8FqrlnSvL24ykDG3MuYO0ZTOP3X7MdlLethaKseUqj3V4-f94cGZw2VG-87IBJ-p2opqzhnAW5fqJ5z75xbCSBW2o4?purpose=inline)

[newsletter.theaiedge.io](https://newsletter.theaiedge.io/p/understanding-the-self-attention?utm_source=chatgpt.com)

Remember this simple explanation:

- Query = What am I looking for?
- Key = How relevant am I?
- Value = What information do I contribute?

One technical clarification: Q, K and V are not manually assigned meanings or database records. They are learned vector representations generated by linear transformations of the input.

# 4. How Are Query, Key and Value Generated?

This is where we move from conceptual understanding to the actual mathematics.

Suppose our input is:

```
I love AI
```

After tokenization and embedding, the input becomes a matrix of numerical vectors.

Let's call this matrix:

\\[ X \\]

The Transformer has three trainable weight matrices:

\\[ W_Q,\quad W_K,\quad W_V \\]

The model multiplies the input matrix by these three weight matrices.

\\[ Q=XW_Q \\]

\\[ K=XW_K \\]

\\[ V=XW_V \\]

This creates three different representations of the same input.

[Q, K, V and W in Transformers — A Complete Step-by-Step Numeric Walkthrough | by Dewasheesh Rana | Feb, 2026 | Medium](https://images.openai.com/static-rsc-4/OwAezAwo3HrQSlVgUWs97fWU0Cy0su7ubY1kJ1P2meIieRuYNJ9uwsSjyLn17AZoCaskxAKL5Wxe8IJpj33IgrhD3IrLcOx7NVQUsIeyNIJHOFHYYUWShxe4nGbgo7zZyQ97rAX7adGkDS63G79EQ8Ezz1RwdToEwpJYI3d-3L0?purpose=inline)

[medium.com](https://medium.com/%40dewasheesh.rana/q-k-v-and-w-in-transformers-a-complete-step-by-step-numeric-walkthrough-3288b95827ae?utm_source=chatgpt.com)

Let's understand with a simple example.

Suppose the embedding of the word `AI` is:

```
AI = [0.2, 0.5, 0.8, 0.3]
```

The model transforms this vector using three different weight matrices.

Illustrative output:

```
Query = [0.4, 0.6]Key   = [0.3, 0.7]Value = [0.9, 0.2]
```

Notice something important:

The input is the same, but the three projections produce different vectors.

Why?

Because each projection has a different set of learned weights.

Interview answer:

"Query, Key and Value are generated by applying three separate learned linear transformations to the input embeddings. The Query and Key vectors are used to calculate attention scores, while the Value vectors contain the information that gets combined using those attention scores."

# 5. How Does Self-Attention Work Step by Step?

Now let's connect all the concepts.

Consider:

```
I love AI
```

We will go through the complete calculation.

Input sentence

# I love AI

Token Embeddings

Convert each token into a vector

Query

Q = XWQ

Key

K = XWK

Value

V = XWV

Calculate Q × K transpose

Calculate relevance scores between tokens

Divide by √dk and apply Softmax

Normalize attention scores into weights

Attention Weights × V

Combine Value vectors according to relevance

Final Contextual Representations

Let's understand each calculation.

## Step 1: Generate Q, K and V

Every token gets its own Query, Key and Value vectors.

```
Token       Query       Key         Value

I           Q1          K1          V1
love        Q2          K2          V2
AI          Q3          K3          V3
```

## Step 2: Calculate Attention Scores

Now the model compares each Query with all the Keys.

The formula is:

\\[ Scores=QK^T \\]

Why?

Because the dot product measures how compatible a Query is with a Key in the learned attention space.

For example, the Query of `AI` is compared with:

- Key of `I`.
- Key of `love`.
- Key of `AI`.

This produces an attention score for each token.

## Step 3: Scale the Attention Scores

The model divides the attention scores by the square root of the Key dimension.

\\[ ScaledScores=\frac{QK^T}{\sqrt{d_k}} \\]

Why?

When the dimensions of the vectors become large, their dot products can also become large.

Large scores can cause Softmax to produce very concentrated probabilities, which can make gradient optimization difficult.

Scaling helps stabilize the attention calculation.

## Step 4: Apply Softmax

Now we apply Softmax to convert the scores into normalized attention weights.

For example:

Suppose the attention scores are:

```
[2.0, 1.0, 0.5]
```

After Softmax:

```
[0.63, 0.23, 0.14]
```

These values represent how much attention should be given to each corresponding token.

The weights add up to 1.

## Step 5: Multiply Attention Weights with Values

This is the final step.

\\[ Output=AttentionWeights\times V \\]

Suppose the attention weights are:

```
I       → 0.63
love    → 0.23
AI      → 0.14
```

And the Value vectors are:

```
I       → [1, 0]
love    → [0, 2]
AI      → [2, 1]
```

The final attention output is:

\\[ Output=0.63[1,0]+0.23[0,2]+0.14[2,1] \\]

\\[ Output=[0.91,0.60] \\]

This is a simplified illustrative calculation.

The result is a weighted combination of the Value vectors.

What have we achieved?

The model has combined information from different tokens according to their attention weights.

This produces a context-aware representation for the token being processed.

# 6. Understanding the Complete Mathematical Formula

You should memorize this formula for interviews.

Important formula

\\[ Attention(Q,K,V)=Softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V \\]

Meaning of each component:

| Component    | Meaning                                 |
| ------------ | --------------------------------------- |
| Q            | Query matrix                            |
| K            | Key matrix                              |
| Kᵀ           | Transpose of Key matrix                 |
| QKᵀ          | Similarity/compatibility scores         |
| dk           | Key vector dimension                    |
| √dk          | Scaling factor                          |
| Softmax      | Converts scores into normalized weights |
| V            | Value matrix                            |
| Final output | Weighted combination of Values          |

The entire mechanism can be summarized in one sentence:

Query and Key determine where to pay attention, and Value determines what information is retrieved.

This is probably the most important sentence to remember.

# 7. What Does the Attention Matrix Actually Look Like?

Let's understand this with a simple example.

Sentence:

```
I love AI
```

Suppose the model calculates the following attention weights:

| Query token | I    | love | AI   |
| ----------- | ---- | ---- | ---- |
| I           | 0.70 | 0.20 | 0.10 |
| love        | 0.20 | 0.50 | 0.30 |
| AI          | 0.10 | 0.30 | 0.60 |

[Transformer Block Assembly: Building Complete Encoder & Decoder Blocks from Components - Interactive | Michael Brenndoerfer | Michael Brenndoerfer](https://images.openai.com/static-rsc-4/43DAG6b-W4-ObL991gCij2wUMvwdcGVymbva2_h8B2rvziuYtKSoj_MXukhBb1T2GdE01oJNMcRMg2PL0d9c_d_RKqpafR6Qipsxl6NjZDsphXCWyjijEIHxzH5WHoX97-kieA2Vwnmj-fX3tT_uqwltTKKo1cluoCe7FU-iX_0?purpose=inline)

[mbrenndoerfer.com](https://mbrenndoerfer.com/writing/transformer-block-assembly?utm_source=chatgpt.com)

How do you read this matrix?

Take the row for `love`:

```
love → I      = 0.20
love → love   = 0.50
love → AI     = 0.30
```

It means the Query for `love` assigns these attention weights to the Keys of `I`, `love`, and `AI`.

The output for `love` is:

\\[ Output\_{love} = 0.2V_I+0.5V\_{love}+0.3V\_{AI} \\]

This is how the representation of a token incorporates information from other tokens.

Note: These numbers are illustrative. Real attention weights are learned dynamically from the input, model parameters and attention layer.

# 8. What Is the Difference Between Self-Attention and Cross-Attention?

This is another very common interview question.

## Self-Attention

In self-attention, Query, Key and Value are generated from the same input sequence.

Example:

```
Input:
I love AI
```

All three projections come from the same input representations.

\\[ Q=XW_Q \\]

\\[ K=XW_K \\]

\\[ V=XW_V \\]

The tokens attend to other tokens in their own sequence.

## Cross-Attention

In cross-attention, Query comes from one sequence, while Key and Value come from another sequence.

For example, in an encoder-decoder Transformer:

```
English Input
     |
     v
   Encoder
     |
     v
Encoder Representations
     |
     +------------------+
                        |
                        v
                 Cross-Attention
                        ^
                        |
                  Decoder Query
```

The formula:

\\[ CrossAttention= Softmax\left(\frac{Q_dK_e^T}{\sqrt{d_k}}\right)V_e \\]

Where:

- Qd = Decoder Query.
- Ke = Encoder Key.
- Ve = Encoder Value.

| Self-Attention                     | Cross-Attention                          |
| ---------------------------------- | ---------------------------------------- |
| Q, K, V from same sequence         | Q from one sequence, K/V from another    |
| Models internal relationships      | Models relationships across sequences    |
| Used in encoder and decoder blocks | Used in original encoder-decoder decoder |

Example: In translation, the decoder uses cross-attention to access the encoder's representation of the English sentence while generating French.

# 9. What Is the Difference Between Self-Attention and Multi-Head Attention?

You may be asked this immediately after explaining Q, K and V.

Self-attention calculates relationships between tokens using one attention operation.

Multi-head attention performs multiple attention operations using different learned projections.

For example:

```
Input Embeddings
       |
       +-------------------+
       |                   |
       v                   v
   Attention Head 1    Attention Head 2
       |                   |
       v                   v
   Attention Head 3    Attention Head 4
       |                   |
       +---------+---------+
                 |
                 v
            Concatenate
                 |
                 v
         Linear Projection
                 |
                 v
             Output
```

Different heads can learn different relationships.

The formula:

\\[ head_i=Attention(Q_i,K_i,V_i) \\]

\\[ MHA=Concat(head_1,\ldots,head_h)W_O \\]

For example, if the model dimension is 512 and we have 8 attention heads, each head commonly uses a dimension of:

\\[ 512/8=64 \\]

This allows the model to learn multiple attention patterns in parallel.

# 10. Python Implementation of Self-Attention

Now let's understand how you would implement self-attention in Python.

This is a simplified educational implementation using NumPy. It demonstrates the core mathematical operations used inside a Transformer.

```
import numpy as npdef softmax(x):    # Numerically stable softmax    x = x - np.max(x, axis=-1, keepdims=True)    exp_x = np.exp(x)    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)def self_attention(X, Wq, Wk, Wv):    """    X:  Input embeddings (sequence_length, model_dim)    Wq: Query projection weights    Wk: Key projection weights    Wv: Value projection weights    """    # 1. Generate Query, Key, Value    Q = X @ Wq    K = X @ Wk    V = X @ Wv    # 2. Calculate attention scores    scores = Q @ K.T    # 3. Scale attention scores    dk = K.shape[-1]    scores = scores / np.sqrt(dk)    # 4. Apply softmax    attention_weights = softmax(scores)    # 5. Combine Value vectors    output = attention_weights @ V
```

Let's understand what happens:

| Code              | Responsibility                      |
| ----------------- | ----------------------------------- |
| `Q = X @ Wq`      | Generates Query vectors             |
| `K = X @ Wk`      | Generates Key vectors               |
| `V = X @ Wv`      | Generates Value vectors             |
| `Q @ K.T`         | Calculates attention scores         |
| `/ np.sqrt(dk)`   | Scales attention scores             |
| `softmax(scores)` | Calculates attention weights        |
| `weights @ V`     | Produces the final attention output |

This is the core scaled dot-product self-attention mechanism.

A production implementation also needs to handle batches, multiple heads, masks, padding, efficient GPU kernels and numerical precision.

# 11. How Does Self-Attention Work in GPT?

Now let's connect this with the GPT architecture.

GPT is a decoder-only Transformer, so it uses causal self-attention.

Consider:

```
I love machine learning
```

During next-token prediction, the model cannot look at future tokens.

Its attention matrix follows this pattern:

| Query    | I   | love | machine | learning |
| -------- | --- | ---- | ------- | -------- |
| I        | Yes | No   | No      | No       |
| love     | Yes | Yes  | No      | No       |
| machine  | Yes | Yes  | Yes     | No       |
| learning | Yes | Yes  | Yes     | Yes      |

Why?

Because GPT must predict the next token using only the available context.

For example:

```
Input: I love

Prediction: machine
```

The model cannot use `machine` or `learning` as future information when predicting `machine`.

This is implemented using a causal attention mask.

Important distinction:

- BERT encoder: Bidirectional self-attention.
- GPT decoder: Causal self-attention.
- Original Transformer decoder: Causal self-attention plus encoder-decoder cross-attention.

# 12. What Happens During Training vs Inference?

This is a good senior-level question.

| Training                                        | Inference                                     |
| ----------------------------------------------- | --------------------------------------------- |
| Q, K and V are calculated                       | Q, K and V are calculated                     |
| Attention is calculated for training positions  | Attention is calculated for available context |
| Causal mask is applied in GPT                   | Causal mask is applied in GPT                 |
| Gradients update projection weights             | Weights generally remain fixed                |
| Multiple positions can be processed in parallel | New tokens are generated autoregressively     |

During inference, modern LLMs use KV Cache to reuse previously calculated Key and Value vectors.

For example:

```
Prompt: I love AI

First generation:
Calculate Q, K, V

Next generation:
Reuse previous K and V
Calculate Q, K, V for new token

Next generation:
Reuse cached K and V again
```

This reduces repeated attention computation.

# 13. Common Mistakes to Avoid in Interviews

| Incorrect explanation                      | Correct explanation                                                                                   |
| ------------------------------------------ | ----------------------------------------------------------------------------------------------------- |
| Query is the input question.               | Query is a learned projection of a token representation.                                              |
| Key stores actual documents.               | Key is a learned vector used to calculate attention compatibility.                                    |
| Value is the final answer.                 | Value is information combined using attention weights.                                                |
| Q, K, V are manually defined.              | They are generated using learned projection matrices.                                                 |
| Self-attention only looks at nearby words. | It can attend to all permitted positions in its sequence.                                             |
| Softmax generates the final output.        | Softmax generates attention weights; multiplying by V produces the attention output.                  |
| Every Transformer uses causal masking.     | Encoder self-attention is generally bidirectional; causal masking is used in autoregressive decoders. |

# 14. Interview-Ready Answer: What Is Self-Attention and What Are Q, K, V?

This is the answer I recommend practising until you can explain it naturally.

Senior AI Engineer interview answer

"Self-attention is a mechanism in Transformer architecture that allows each token to understand its relationship with other tokens in the same sequence.

For example, consider the sentence 'The animal didn't cross the road because it was tired'. The model needs to understand that 'it' refers to 'animal'. Self-attention helps it learn these contextual relationships.

Internally, the model generates three vectors for every token: Query, Key and Value.

These vectors are generated by multiplying the input embeddings with three different learned weight matrices.

Query represents what information a token is looking for. Key represents the information used to determine how relevant a token is to that Query. Value contains the information that will be combined to produce the attention output.

The model calculates attention scores by multiplying the Query matrix with the transpose of the Key matrix.

It then divides the scores by the square root of the Key dimension to stabilize their scale and applies Softmax to obtain normalized attention weights.

Finally, these attention weights are multiplied with the Value matrix to generate context-aware representations.

In simple terms, Query and Key determine where to pay attention, while Value determines what information is retrieved.

This allows Transformers to capture relationships between tokens and understand context effectively."

&#x20;Copy interview answer

# 15. Test Your Understanding

Try answering these six questions without referring to the explanation.

Self-attention quiz

0/6 completed

1\. What is the primary purpose of self-attention?

Convert text into token IDs

Understand relationships between tokens

Generate embeddings only

2\. How are Query, Key and Value generated?

Three separate learned linear projections of input representations

Three separate tokenizers

Three different neural networks processing different sentences

3\. What does Query represent?

The final answer

What information a token is looking for

The actual document content

4\. What is the purpose of Softmax?

Generate token IDs

Normalize attention scores into weights

Calculate embeddings

5\. Which formula represents scaled dot-product attention?

Q + K + V

Softmax(QKᵀ / √dk) × V

Q × V + K

6\. In cross-attention, where do Q, K and V come from?

All from encoder

All from decoder

Q from decoder; K and V from encoder

Check my answers

My advice for your Senior AI Engineer interviews: Spend extra time understanding the actual matrix calculation, not just the definitions of Q, K and V. If you can explain `Q × Kᵀ → Scaling → Softmax → Multiply by V` and demonstrate why causal masking is needed in GPT, you will be able to handle most foundational self-attention questions confidently.