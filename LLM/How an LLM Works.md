# How an LLM Works — From Tokens to Production AI Systems

To understand Large Language Models (LLMs) at a senior AI engineer or staff AI engineer level, you need to understand more than prompting an API. You need to know how text becomes tokens, how embeddings represent those tokens, how the Transformer processes context, how attention works, how the model predicts the next token, and how inference systems serve requests efficiently.

We'll build the understanding from first principles, then connect it to production AI architectures such as RAG, tool calling, and multi-agent systems.

## 1. The complete LLM architecture

An LLM is a neural network trained on large amounts of data to predict tokens. Modern generative LLMs commonly use the Transformer architecture, although not every language model uses exactly the same architecture.

At a high level, the generation process looks like this:

1\. Input text

"Explain Python memory management"

2\. Tokenizer

Text → token IDs

3\. Token embeddings + positional information

Token IDs → vectors containing learned representations

4\. Transformer layers

Self-attention → MLP → residual connections → normalization

5\. Output projection + logits

Scores for possible next tokens

6\. Sampling / token selection

Choose the next token, append it, and repeat

The key idea is that an autoregressive LLM generates text incrementally. It predicts a next token, adds that token to the context, and predicts again until it reaches a stopping condition.

One important distinction: this describes the model's inference process. Training is a separate process that adjusts the model's parameters so its predictions improve.

## 2. What exactly is a token?

An LLM generally does not process text as whole words or individual characters. It processes tokens, which may represent complete words, word fragments, punctuation, whitespace patterns, or other text units.

For example, the sentence:

```
Python is useful for AI.
```

Might be represented by a tokenizer as:

```
["Python", " is", " useful", " for", " AI", "."]
```

This is illustrative, not the exact output of every tokenizer.

A tokenizer maps the text to integer IDs:

```
["Python", " is", " useful", " for", " AI", "."]
                ↓
[  1245,    318,    827,   369,  9552,  13]
```

The numbers above are illustrative IDs, not actual IDs from a particular model.

Why do we need tokenization?

- Neural networks operate on numerical tensors, not raw strings.
- A subword vocabulary can represent common words efficiently while also handling unfamiliar words.
- Token IDs let the model look up learned vector representations.

### Code: build a tiny tokenizer

This example uses a simple word-level tokenizer so you can see the mapping. Production LLMs generally use more sophisticated tokenization algorithms.

```
text = "python is useful for ai"vocabulary = {    "<UNK>": 0,    "python": 1,    "is": 2,    "useful": 3,    "for": 4,    "ai": 5,}tokens = text.split()token_ids = [    vocabulary.get(token, vocabulary["<UNK>"])    for token in tokens]print("Tokens:", tokens)print("Token IDs:", token_ids)
```

Output:

```
Tokens: ['python', 'is', 'useful', 'for', 'ai']
Token IDs: [1, 2, 3, 4, 5]
```

A real tokenizer would also need to handle punctuation, casing, Unicode, special tokens, and unseen word forms.

Senior-engineer takeaway: tokenization is part of the model's interface contract. Using the wrong tokenizer or mismatched vocabulary can make the model's input and output meaningless.

## 3. What happens after tokenization? Embeddings

Token IDs are just integers. The number `5` is not inherently more meaningful than the number `2`.

The model therefore uses an embedding matrix to map every token ID to a learned vector.

Suppose our vocabulary contains 1,000 tokens and the embedding dimension is 4.

The embedding matrix has shape:

\\[ E \in \mathbb{R}^{1000 \times 4} \\]

Each row corresponds to a token's learned representation.

For example, the token IDs `[1, 2, 5]` might map to:

Token embedding lookup

Token ID 1

`[0.2, 0.8, -0.1, 0.4]`

Token ID 2

`[0.5, -0.3, 0.7, 0.1]`

Token ID 5

`[-0.2, 0.4, 0.6, 0.9]`

Illustrative values only. Actual embedding vectors are learned during training.

The output is a matrix of shape:

\\[ X \in \mathbb{R}^{L \times d\_{\text{model}}} \\]

Where:

- \\(L\\) is the number of input tokens.
- \\(d\_{\text{model}}\\) is the hidden representation size.

### Code: embedding lookup using PyTorch

```
import torchimport torch.nn as nnvocab_size = 1000embedding_dim = 4embedding = nn.Embedding(vocab_size, embedding_dim)token_ids = torch.tensor([1, 2, 5])vectors = embedding(token_ids)print(vectors.shape)
```

Output:

```
torch.Size([3, 4])
```

The embedding layer contains trainable parameters. During training, gradient descent updates them along with the rest of the model.

Important: embeddings inside an LLM are not necessarily the same as embeddings produced by a dedicated sentence-embedding model for semantic search. They serve related but different purposes.

## 4. How does the model understand token order?

Consider:

```
The dog chased the cat.
The cat chased the dog.
```

Many of the same tokens occur in both sentences, but the meaning differs.

The Transformer must represent token position as well as token identity.

A common conceptual formula is:

\\[ X = E\_{\text{token}} + E\_{\text{position}} \\]

Here, positional information is combined with token embeddings. Different architectures implement position differently.

Examples include:

- Learned absolute positional embeddings.
- Sinusoidal positional encodings.
- Rotary positional embeddings (RoPE), widely used in modern decoder-only LLMs.

### Why is this important?

Without any mechanism to encode position or order, ordinary self-attention would not inherently distinguish the same tokens in different positions.

In practice, a Transformer uses both token content and positional information to construct contextual representations.

## 5. The most important concept: self-attention

Self-attention allows each token to use information from other tokens in its context.

Consider this sentence:

```
The bank approved the loan because it had sufficient funds.
```

The model needs to process relationships among tokens. For example, the meaning of "it" depends on the surrounding context.

Attention calculates how much information one position should take from other positions. It does not use a fixed dictionary of grammatical rules; it learns attention transformations from training data.

### Query, Key, and Value

For the input representation \\(X\\), the attention layer computes three projections:

\\[ Q = XW_Q,\qquad K = XW_K,\qquad V = XW_V \\]

- Query (Q): what information a position is looking for.
- Key (K): what information each position offers for matching.
- Value (V): the information that gets combined.

These are learned linear transformations. The query/key/value analogy is useful, but they are not literal natural-language questions, labels, or database fields.

### Scaled dot-product attention

The core equation is:

\\[ \operatorname{Attention}(Q,K,V) = \operatorname{softmax} \left( \frac{QK^\top}{\sqrt{d_k}}+M \right)V \\]

Where:

- \\(QK^\top\\) computes similarity scores between positions.
- \\(d_k\\) is the query/key dimension.
- The division by \\(\sqrt{d_k}\\) helps control the scale of the scores.
- \\(M\\) is an optional mask, including a causal mask for autoregressive generation.
- Softmax converts scores into normalized weights.
- Multiplying by \\(V\\) produces a weighted combination of information.

Let's walk through the math with a tiny example.

Suppose a query matches two keys with scores:

\\[ [2.0,\ 1.0] \\]

Softmax converts them into approximately:

\\[ [0.731,\ 0.269] \\]

The first position contributes more to the output, though the second still contributes.

These weights are learned indirectly through the model's training; they aren't manually assigned linguistic importance.

### Code: implement attention from scratch

This example uses small random tensors to demonstrate the calculation.

```
import torchimport mathtorch.manual_seed(42)# 3 tokens, each represented by 4 dimensionsX = torch.randn(3, 4)# Learned projections, represented here by random matricesWq = torch.randn(4, 4)Wk = torch.randn(4, 4)Wv = torch.randn(4, 4)Q = X @ WqK = X @ WkV = X @ Wvscores = Q @ K.T / math.sqrt(K.shape[-1])weights = torch.softmax(scores, dim=-1)output = weights @ Vprint("Input:", X.shape)print("Attention weights:", weights.shape)print("Output:", output.shape)print(weights)
```

Expected tensor shapes:

```
Input: torch.Size([3, 4])
Attention weights: torch.Size([3, 3])
Output: torch.Size([3, 4])
```

The attention-weight matrix has three rows and three columns because each of the three tokens computes scores against all three positions in this example.

The example is not yet a language model: it has random weights, no training, and no causal mask.

## 6. Why do LLMs use causal attention?

For a GPT-style decoder-only LLM, the model learns to predict the next token using the preceding context. It must not see future tokens when calculating predictions during training.

Suppose the training sentence is:

```
Python is a powerful language
```

The training targets are conceptually:

| Input context        | Target next token |
| -------------------- | ----------------- |
| Python               | is                |
| Python is            | a                 |
| Python is a          | powerful          |
| Python is a powerful | language          |

A causal mask prevents a position from attending to future positions.

Causal attention mask

Rows are the current token positions; columns are positions they can attend to. ✓ = allowed, × = blocked.

|            | 1 | 2 | 3 | 4 |
| ---------- | - | - | - | - |
| Position 1 | ✓ | × | × | × |
| Position 2 | ✓ | ✓ | × | × |
| Position 3 | ✓ | ✓ | ✓ | × |
| Position 4 | ✓ | ✓ | ✓ | ✓ |

The first token can attend only to itself. The second can attend to the first and second. The fourth can attend to all four positions.

### Code: causal masking

```
import torchsequence_length = 4scores = torch.zeros(sequence_length, sequence_length)# True where future positions must be blockedfuture_mask = torch.triu(    torch.ones(sequence_length, sequence_length, dtype=torch.bool),    diagonal=1,)scores = scores.masked_fill(future_mask, float("-inf"))weights = torch.softmax(scores, dim=-1)print(weights)
```

The output has the following pattern:

```
tensor([
    [1.0000, 0.0000, 0.0000, 0.0000],
    [0.5000, 0.5000, 0.0000, 0.0000],
    [0.3333, 0.3333, 0.3333, 0.0000],
    [0.2500, 0.2500, 0.2500, 0.2500]
])
```

These particular values arise because the example starts with zero scores. A real model produces different attention weights.

Staff-level point: causal masking prevents information leakage from future target tokens during training. It also enables the autoregressive generation objective. Padding masks and other attention masks serve different purposes.

## 7. What is multi-head attention?

A single attention mechanism learns one set of attention projections. Multi-head attention uses multiple attention heads in parallel.

Each head can learn a different representation or pattern of relationships. However, don't assume that a specific head always corresponds to a human-defined concept such as grammar, names, or dates.

The outputs are concatenated and projected back into the model's hidden dimension.

\\[ \operatorname{MultiHead}(Q,K,V) = \operatorname{Concat}(head_1,\ldots,head_h)W_O \\]

Here \\(h\\) is the number of attention heads.

### Example

Suppose:

- Hidden dimension = 512
- Number of heads = 8
- Dimension per head = 64

Then:

\\[ 512 = 8 \times 64 \\]

Each head operates on its own projected query, key, and value representations. Their outputs are combined to produce a 512-dimensional representation per token.

Modern architectures may also use grouped-query attention (GQA) or multi-query attention (MQA), where query heads share key/value heads to reduce inference memory and bandwidth costs.

## 8. What happens inside a Transformer block?

Attention is only one part of a Transformer. A typical Transformer block also includes a feed-forward network, residual connections, and normalization.

Input hidden states

Normalization + self-attention

Residual connection

Combine attention output with the input

Normalization + feed-forward network

Residual connection

Output hidden states

The diagram shows a common pre-normalization arrangement; exact block designs vary by model.

### The feed-forward network (MLP)

After attention mixes information across token positions, the MLP transforms each position's hidden representation independently, using learned parameters shared across positions.

A simplified formula is:

\\[ \operatorname{MLP}(x) = W_2\\,\sigma(W_1x+b_1)+b_2 \\]

Here \\(\sigma\\) is an activation function, often GELU or a gated activation such as SwiGLU in modern models.

The attention layer enables information exchange between positions. The MLP performs additional learned transformations on the resulting representations.

### Why residual connections?

A residual connection adds a block's input to its output:

\\[ y = x + F(x) \\]

This helps information and gradients flow through deep networks and makes optimization easier.

### Why normalization?

Layer normalization or RMSNorm helps stabilize activation scales. This is especially useful when stacking many layers.

### How does the complete model use these blocks?

A large LLM stacks many Transformer blocks. At each layer, the token representations are progressively transformed using learned attention and feed-forward computations.

The resulting representations contain contextual information that helps the model predict the next token.

## 9. How does the model predict the next token?

After the final Transformer block, the model applies an output projection to produce a score for every token in its vocabulary.

These scores are called logits.

Suppose the vocabulary contains 50,000 tokens. For one position, the model might produce 50,000 logits:

\\[ z \in \mathbb{R}^{50000} \\]

The logits are not probabilities. Softmax can convert them into a probability distribution:

\\[ P(t_i \mid context) = \frac{e^{z_i}}{\sum_j e^{z_j}} \\]

Consider a tiny illustrative vocabulary:

| Candidate token | Logit | Approximate probability |
| --------------- | ----- | ----------------------- |
| `Python`        | 2.0   | 66.5%                   |
| `Java`          | 1.0   | 24.5%                   |
| `banana`        | -0.5  | 5.5%                    |
| `database`      | -1.0  | 3.4%                    |

Probabilities are rounded. The actual probabilities depend on the complete vocabulary and all logits.

The model can select the token with the highest probability, or sample from the distribution according to the decoding configuration.

### Code: logits to probabilities

```
import torchlogits = torch.tensor([2.0, 1.0, -0.5, -1.0])probabilities = torch.softmax(logits, dim=-1)tokens = ["Python", "Java", "banana", "database"]for token, probability in zip(tokens, probabilities.tolist()):    print(f"{token}: {probability:.3f}")
```

The important point is that the model scores candidate tokens based on the context. It doesn't look up a predetermined sentence and simply copy it.

## 10. Greedy decoding, temperature, and top-p sampling

The same model can generate different outputs depending on the decoding strategy.

Greedy decoding

Selects the token with the highest probability at each step. It is deterministic for fixed logits and tie-breaking behavior.

Temperature

Changes the sharpness of the probability distribution. Lower positive temperatures concentrate probability on higher-scoring tokens; higher temperatures flatten it.

Top-p sampling

Selects from a dynamically chosen set of the highest-probability tokens whose cumulative probability reaches a configured threshold.

### Code: temperature sampling

```
import torchdef sample_next_token(logits, temperature=1.0):    if temperature <= 0:        raise ValueError("Temperature must be positive")    probabilities = torch.softmax(logits / temperature, dim=-1)    next_token_id = torch.multinomial(        probabilities,        num_samples=1,    )    return next_token_id.item()logits = torch.tensor([2.0, 1.0, 0.0, -1.0])print(sample_next_token(logits, temperature=0.5))print(sample_next_token(logits, temperature=1.5))
```

The sampled IDs are stochastic and may differ between runs. A lower temperature makes the distribution more concentrated, but it doesn't guarantee that every sampled token is the most likely one.

In production, decoding can also include top-k filtering, repetition penalties, stop sequences, structured-output constraints, and other model-specific options.

## 11. How does an LLM generate a complete response?

Suppose the prompt is:

```
The capital of France is
```

A simplified generation loop might work like this:

1. Tokenize the prompt.
2. Run the model over the token IDs.
3. Obtain the next-token logits.
4. Select the next token.
5. Append the token to the context.
6. Repeat until the model emits an end-of-sequence token or reaches a configured limit.

For illustration, the generated tokens might be:

```
Paris → . → <EOS>
```

The actual response depends on the model and decoding settings.

### A simplified autoregressive generation loop

The following example is deliberately tiny. It demonstrates the mechanics of repeated next-token prediction, not a real trained LLM.

```
import torchimport torch.nn as nnclass TinyNextTokenModel(nn.Module):    def __init__(self, vocab_size, hidden_size):        super().__init__()        self.embedding = nn.Embedding(vocab_size, hidden_size)        self.output = nn.Linear(hidden_size, vocab_size)    def forward(self, token_ids):        # Simplified context representation:        # average all token embeddings        hidden = self.embedding(token_ids).mean(dim=1)        return self.output(hidden)def generate(model, token_ids, max_new_tokens=5):    model.eval()    with torch.no_grad():        for _ in range(max_new_tokens):            logits = model(token_ids)            next_id = logits.argmax(dim=-1, keepdim=True)            token_ids = torch.cat([token_ids, next_id], dim=1)    return token_idsmodel = TinyNextTokenModel(vocab_size=20, hidden_size=8)prompt_ids = torch.tensor([[1, 2, 3]])generated_ids = generate(model, prompt_ids)print(generated_ids)
```

This runs, but because the model is untrained, its generated tokens have no meaningful language quality. It also uses mean pooling instead of causal attention.

A real decoder-only LLM uses a causal Transformer and is trained on next-token prediction. Its generation loop also typically uses a KV cache, discussed below.

Why is this useful? It separates two things that are often confused: the neural network computes scores; the decoding algorithm decides which token to output.

## 12. How is an LLM trained?

So far, we've discussed inference: giving a trained model input and generating an answer.

Training is different. The model begins with parameter values that don't yet encode useful language behavior. It processes training examples, predicts tokens, calculates a loss, and updates its parameters.

The standard language-model training objective is often next-token prediction.

For a sequence \\(x_1, x_2, \ldots, x_T\\), the model learns to maximize:

\\[ \sum\_{t=1}^{T} \log P\_\theta(x_t \mid x\_{\<t}) \\]

Equivalently, training minimizes the negative log-likelihood, commonly implemented as cross-entropy loss.

### The training pipeline

1\. Training data

Text, code, and other permitted data

2\. Tokenization

Convert examples into token IDs

3\. Forward pass

Calculate next-token logits

4\. Loss calculation

Compare predictions with target tokens

5\. Backpropagation

Calculate gradients of model parameters

6\. Optimizer step

Update weights, for example with AdamW

7\. Repeat

Train across batches and many steps

### Code: calculate next-token prediction loss

This is a minimal training example using PyTorch. It demonstrates a forward pass, cross-entropy loss, backpropagation, and a parameter update. The neural network itself is intentionally simple, not a full Transformer.

```
import torchimport torch.nn as nnimport torch.optim as optimtorch.manual_seed(42)vocab_size = 10embedding_dim = 8model = nn.Sequential(    nn.Embedding(vocab_size, embedding_dim),    nn.Flatten(),    nn.Linear(embedding_dim * 3, vocab_size),)optimizer = optim.AdamW(model.parameters(), lr=0.001)loss_fn = nn.CrossEntropyLoss()# Context: three token IDsinput_ids = torch.tensor([[1, 2, 3]])# The correct next token is ID 4target_ids = torch.tensor([4])model.train()logits = model(input_ids)loss = loss_fn(logits, target_ids)optimizer.zero_grad()loss.backward()optimizer.step()print("Training loss:", loss.item())
```

What each operation does:

- `model(input_ids)`: calculates the logits.
- `loss_fn(...)`: measures how poorly the model predicts the target.
- `optimizer.zero_grad()`: clears gradients from the previous step.
- `loss.backward()`: calculates gradients using backpropagation.
- `optimizer.step()`: updates the trainable parameters.

This is only one optimization step, and the model is too simple to represent an actual LLM. A production training pipeline would also need batches, correct causal sequence alignment, attention masks, validation, checkpointing, distributed training where needed, and robust data handling.

### What happens after pretraining?

Many chat LLMs go through additional training stages, which may include:

- Supervised fine-tuning (SFT): train on instruction-and-response examples.
- Preference optimization: use human or synthetic preference data with methods such as DPO, or use reinforcement learning approaches.
- Safety and behavior tuning: improve adherence to policies, refusal behavior, and task-specific reliability.

These stages are not identical across all model providers. They influence how the model follows instructions and behaves as an assistant.

## 13. Why can an LLM hallucinate?

An LLM generates outputs based on learned patterns, context, and its training and inference setup. Its token probabilities do not directly represent verified truth.

A model can produce a plausible but incorrect statement when:

- Its learned knowledge is incomplete or outdated.
- The prompt is ambiguous or missing key facts.
- The model extrapolates beyond reliable evidence.
- The context contains misleading information.
- The model generates a citation, number, or fact without verifying it.

For example, a model might confidently invent an API method or a financial regulation clause.

Increasing model size or lowering temperature does not guarantee factual correctness.

### How production systems reduce hallucinations

RAG (retrieval-augmented generation)

Retrieve relevant source documents and provide them as context for the model's answer.

Grounding and validation

Require claims to be supported by evidence, validate citations, and reject unsupported structured output.

Guardrails and human review

Enforce permissions and domain policies, and require approval for high-impact actions.

Evaluation and monitoring

Measure factual correctness, groundedness, retrieval quality, latency, and failure rates using representative test cases.

For a financial compliance platform, for example, the LLM should retrieve the applicable policy version, identify supporting passages, classify the claim, and produce a review result with citations. Deterministic authorization and audit logic should remain outside the model.

RAG reduces some grounding failures, but it doesn't eliminate hallucinations. Retrieved documents can be incomplete or wrong, and the model can still misinterpret them.

## 14. What happens inside a production LLM serving system?

Knowing the Transformer is only half the job for a senior AI engineer. You also need to understand how the model is deployed, how requests flow through it, and how latency and GPU memory are managed.

Client / API gateway

Authentication, rate limits, request validation

Inference server

Continuous batching, scheduling, tokenization

GPU model execution

Transformer layers, attention, KV cache, logits

Token decoding and streaming

Next-token selection, stop conditions, response delivery

### A. Why does the KV cache matter?

During autoregressive generation, the model repeatedly processes a growing context. Without caching, the model would unnecessarily recompute key and value projections for earlier tokens at every generation step.

The KV cache stores previously computed attention keys and values so that the next decoding step can reuse them.

This reduces redundant computation during generation, although each new token still requires model computation. The cache itself consumes memory, and its size grows with the number of cached tokens, layers, and key/value heads.

This is one reason long-context generation and high concurrency can be expensive.

### B. What is quantization?

Quantization represents model weights, and sometimes activations or cached values, with lower-precision numerical formats.

Examples include FP16, BF16, INT8, and INT4-based approaches.

Benefits can include:

- Lower model memory requirements.
- Higher inference throughput on suitable hardware.
- Ability to serve larger models on constrained GPUs.

Trade-offs include possible quality degradation, hardware-specific performance differences, and implementation complexity.

Quantization is not the same as reducing the model's number of parameters. It changes how numerical values are represented or computed.

### C. What is batching?

Instead of processing every request independently, an inference server can combine work from multiple requests.

Continuous batching allows new requests to join as other requests finish, improving GPU utilization under variable workloads.

However, batching decisions must balance throughput, latency, context length, and KV-cache capacity.

### D. What metrics would you monitor?

| Metric                     | Why it matters                                        |
| -------------------------- | ----------------------------------------------------- |
| Time to first token (TTFT) | Measures initial response delay                       |
| Inter-token latency        | Measures the time between generated tokens            |
| Tokens per second          | Measures generation throughput                        |
| GPU utilization and memory | Helps identify hardware bottlenecks                   |
| KV-cache utilization       | Indicates pressure from active contexts               |
| Queue time                 | Reveals scheduling and capacity issues                |
| Error and timeout rate     | Shows reliability problems                            |
| Cost per request           | Connects infrastructure consumption to business value |

A production inference stack may use a server such as vLLM or another optimized runtime. The best choice depends on model architecture, hardware, throughput, latency requirements, and deployment constraints.

## 15. How do LLMs fit into RAG and multi-agent AI?

This distinction is critical for your AI engineering work: the LLM is a reasoning and generation component, not the entire application.

Consider a financial compliance review system.

FastAPI

Authentication, RBAC, request validation

Workflow orchestrator

Planner / supervisor, routing, state, retries

Retrieval

PostgreSQL / pgvector

LLM

Analysis and structured output

Validation and evaluation

Evidence checks, schema validation, policy rules

Human approval + audit

Persist decisions and resume the workflow when approved

The components have distinct responsibilities:

- LLM: interprets context and generates candidate analysis.
- RAG: retrieves relevant evidence to ground that analysis.
- Planner/supervisor: decides which workflow step or specialist should run.
- Tools: execute explicit operations, such as retrieving policy versions or calculating deterministic risk metrics.
- Guardrails: constrain inputs, outputs, and permitted actions.
- Human-in-the-loop (HITL): pauses a workflow for an authorized human decision.
- Database and audit layer: persist policy versions, review state, and decisions.
- Observability: tracks failures, latency, token usage, and quality.

A staff-level design doesn't assume the LLM will always make the correct decision. It treats model output as something that must be validated, authorized, and monitored.

## 16. Test your understanding

1\. What does an autoregressive LLM predict?

The entire response in one operation

A probability distribution over the next token

The correct answer from a database

2\. What is the primary purpose of a causal attention mask?

Reduce the model's vocabulary

Prevent access to future token positions

Convert logits into probabilities

3\. Why is the KV cache useful?

It stores every possible answer

It replaces model training

It reuses previously computed attention keys and values during generation

4\. Does RAG guarantee that the LLM cannot hallucinate?

Yes, if retrieval succeeds

No, retrieved evidence can be incomplete or misinterpreted

Yes, if temperature is zero

0 of 4 answered

Reset quiz

## 17. What you should master next

For senior/staff-level AI engineering, study these topics in this order:

1. Transformer implementation: multi-head attention, causal masks, residual connections, normalization, MLPs, and positional encodings.
2. Training: cross-entropy, backpropagation, AdamW, learning-rate schedules, pretraining versus fine-tuning, and evaluation.
3. Inference optimization: KV caching, batching, quantization, tensor parallelism, and GPU memory planning.
4. Model adaptation: SFT, LoRA/QLoRA, preference optimization, and model evaluation.
5. Production integration: RAG, tool calling, structured outputs, guardrails, observability, and workflow orchestration.

The central mental model to remember is: training teaches the model its parameters; inference uses those parameters to predict tokens; decoding turns those predictions into text; and the production system makes that process useful, secure, reliable, and cost-effective.