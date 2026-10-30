# Temperature, Top-K, Top-P and Token Generation in LLMs

These are fundamental concepts in LLM inference and decoding.

As a Senior AI Engineer, you should understand how an LLM selects its next token, how temperature changes the probability distribution, and how Top-K and Top-P control which tokens can be selected.

Let's use one example throughout.

Suppose a user asks an LLM:

"What is the best programming language for AI?"

The model processes the prompt through its Transformer layers and generates a probability distribution for the next token.

For example:

| Candidate token | Probability |
| --------------- | ----------- |
| Python          | 0.55        |
| Java            | 0.20        |
| C++             | 0.12        |
| Rust            | 0.08        |
| Swift           | 0.05        |

These probabilities are illustrative.

The model must now select one token.

This selection process is called decoding or token sampling.

The four concepts we will cover are:

1. Token generation – how the model produces tokens.
2. Temperature – controls randomness.
3. Top-K – limits the number of candidate tokens.
4. Top-P – limits candidates based on cumulative probability.

# 1. Token Generation – How does an LLM generate text?

First, understand this important concept:

An LLM does not generate an entire paragraph in one operation. It generates one token at a time, autoregressively.

Let's understand the complete process.

## Autoregressive token generation

User prompt

What is machine learning?

Tokenizer

Converts text into token IDs

Transformer

Processes the input and produces next-token logits

Decoding strategy

Temperature + Top-K + Top-P

Select next token

`Machine`

Append token to context

`What is machine learning? Machine`

Repeat Transformer inference and decoding until a stopping condition is reached.

## Step-by-step example

Suppose the prompt is:

`The capital of India is`

The model predicts:

| Generation step | Predicted token |
| --------------- | --------------- |
| 1               | New             |
| 2               | Delhi           |
| 3               | EOS             |

The final output is:

`The capital of India is New Delhi.`

Notice that after generating `New`, the model uses the updated context to predict `Delhi`.

This is called autoregressive generation.

## What happens internally?

There are four major operations:

1. Transformer processes the input.
2. Output projection generates logits.
3. Decoding converts logits into a probability distribution and applies the selection strategy.
4. The selected token is appended to the input sequence.

Mathematically:

\\[ P(x_t|x_1,\ldots,x\_{t-1}) \\]

This represents the probability of generating the next token given all previously generated tokens.

The process continues until:

- An EOS (end-of-sequence) token is generated.
- Maximum output tokens are reached.
- A configured stopping condition is met.

# 2. Temperature – How does it control randomness?

Important interview concept

Temperature controls how concentrated or distributed the token probability distribution is.

In simple terms:

- Low temperature = more predictable generation.
- High temperature = more varied generation.

Temperature is applied to the model's logits before softmax.

## 2.1 Mathematical explanation

Suppose the model generates these logits:

| Token  | Logit |
| ------ | ----- |
| Python | 3.0   |
| Java   | 2.0   |
| C++    | 1.0   |

Normally, softmax calculates probabilities:

\\[ P_i = \frac{e^{z_i}}{\sum_j e^{z_j}} \\]

With temperature:

\\[ \boxed{ P_i = \frac{e^{z_i/T}} {\sum_j e^{z_j/T}} } \\]

Where:

- \\(z_i\\) = token logit.
- \\(T\\) = temperature.

Temperature changes the relative concentration of the distribution.

## 2.2 Temperature = 0.2

When temperature is low, differences between logits become more pronounced.

The model becomes more likely to select the highest-scoring token.

Temperature: 0.2

More deterministic

Python

99.3%

Java

0.670%

C++

0.005%

## 2.3 Temperature = 1.0

At temperature 1, the original softmax distribution remains unchanged.

Temperature: 1.0

Original distribution

Python

66.5%

Java

24.5%

C++

9.0%

## 2.4 Temperature = 2.0

When temperature increases, the probability distribution becomes flatter.

Temperature: 2.0

More random

Python

50.6%

Java

30.7%

C++

18.6%

The probabilities above are calculated from the example logits, with rounding.

## 2.5 When should you use different temperatures?

| Use case               | Suggested starting point |
| ---------------------- | ------------------------ |
| Code generation        | 0–0.3                    |
| Structured JSON        | 0–0.2                    |
| Financial analysis     | 0–0.3                    |
| Technical explanations | 0.2–0.5                  |
| General conversation   | 0.5–0.8                  |
| Creative writing       | 0.8–1.2                  |

These are starting points, not universal optimal settings.

Senior engineer insight: Temperature = 0 usually means greedy decoding in common inference APIs, but exact behavior depends on the serving implementation. A low temperature does not guarantee factual accuracy.

# 3. Top-K Sampling – Limit the number of candidate tokens

Top-K is a decoding technique that restricts token selection to the K highest-probability tokens.

In simple terms:

The model considers only the K most likely next tokens and ignores all remaining candidates.

Let's use our previous example.

Original probability distribution:

| Token  | Probability |
| ------ | ----------- |
| Python | 55%         |
| Java   | 20%         |
| C++    | 12%         |
| Rust   | 8%          |
| Swift  | 5%          |

## 3.1 Top-K = 3

If we set K = 3, only these tokens remain:

- Python
- Java
- C++

Rust and Swift are removed.

Top-K = 3

3 candidates retained

Python

55%

Keep

Java

20%

Keep

C++

12%

Keep

Rust

8%

Remove

Swift

5%

Remove

After filtering and renormalization

The retained probabilities are divided by their sum (87%).

| Token  | New probability |
| ------ | --------------- |
| Python | 63.22%          |
| Java   | 22.99%          |
| C++    | 13.79%          |

Notice that the probabilities are renormalized.

Python's probability changes from 55% to approximately 63.22% because the removed candidates no longer participate in sampling.

## 3.2 What happens with different K values?

| Top-K  | Behavior                                   |
| ------ | ------------------------------------------ |
| K = 1  | Only the highest-probability token remains |
| K = 3  | Three highest-probability tokens remain    |
| K = 5  | Five highest-probability tokens remain     |
| K = 50 | Fifty highest-probability tokens remain    |

If K = 1, sampling becomes equivalent to greedy selection, assuming no additional tie-breaking or decoding constraints.

### When should you use Top-K?

- Restrict unlikely token choices.
- Reduce extremely unusual generations.
- Control the candidate set.
- Improve predictability in some generation tasks.

Limitation: Top-K always retains a fixed number of candidates, regardless of how confident the model is.

For example, sometimes the model is extremely confident about one token, while in another situation it is uncertain across many tokens. Top-K does not adapt to that difference.

This is where Top-P becomes useful.

# 4. Top-P Sampling (Nucleus Sampling)

Important interview concept

Top-P is also called nucleus sampling.

Unlike Top-K, it does not retain a fixed number of tokens.

Instead:

Top-P retains the smallest set of highest-probability tokens whose cumulative probability reaches or exceeds the configured threshold P.

Let's understand with an example.

Original distribution:

| Token  | Probability |
| ------ | ----------- |
| Python | 55%         |
| Java   | 20%         |
| C++    | 12%         |
| Rust   | 8%          |
| Swift  | 5%          |

Suppose:

\\[ TopP=0.80 \\]

We calculate cumulative probabilities.

| Token  | Probability | Cumulative probability |
| ------ | ----------- | ---------------------- |
| Python | 55%         | 55%                    |
| Java   | 20%         | 75%                    |
| C++    | 12%         | 87%                    |
| Rust   | 8%          | 95%                    |
| Swift  | 5%          | 100%                   |

We need to reach at least 80%.

Therefore, we retain:

- Python
- Java
- C++

Because cumulative probability reaches 87%.

The remaining tokens are removed.

Top-P = 0.80

Adaptive candidate selection

Python

55% cumulative

Keep

Java

75% cumulative

Keep

C++

87% cumulative

Keep

Rust

95% cumulative

Remove

Swift

100% cumulative

Remove

The cumulative threshold is crossed at C++. All three retained tokens are renormalized for sampling.

## 4.1 What happens with different Top-P values?

| Top-P | Behavior                                        |
| ----- | ----------------------------------------------- |
| 0.20  | Retains only enough leading tokens to reach 20% |
| 0.50  | Retains enough tokens to reach 50%              |
| 0.80  | Retains enough tokens to reach 80%              |
| 0.95  | Retains enough tokens to reach 95%              |
| 1.00  | No probability-based filtering                  |

A smaller P generally produces a smaller candidate set.

A larger P generally produces a larger candidate set.

### Why is Top-P useful?

Consider two situations.

Situation A: The model is confident

| Token  | Probability |
| ------ | ----------- |
| Python | 90%         |
| Java   | 5%          |
| C++    | 3%          |
| Rust   | 2%          |

With Top-P = 0.95, the model needs only Python and Java.

Situation B: The model is uncertain

| Token  | Probability |
| ------ | ----------- |
| Python | 25%         |
| Java   | 23%         |
| C++    | 20%         |
| Rust   | 17%         |
| Swift  | 15%         |

With Top-P = 0.95, it needs almost all five tokens.

This is the main advantage of Top-P: the number of candidate tokens adapts to the probability distribution.

# 5. Top-K vs Top-P – What is the difference?

This is a very common interview question.

| Feature              | Top-K                  | Top-P                                |
| -------------------- | ---------------------- | ------------------------------------ |
| Filtering method     | Fixed number of tokens | Cumulative probability               |
| Candidate count      | Fixed K                | Variable                             |
| Adapts to confidence | No                     | Yes                                  |
| Main parameter       | K                      | P                                    |
| Example              | Keep top 10 tokens     | Keep tokens covering 90% probability |
| Common name          | Top-K sampling         | Nucleus sampling                     |

Think about it this way:

- Top-K asks: How many candidates should I consider?
- Top-P asks: How much of the probability distribution should I consider?

# 6. Can we use Temperature, Top-K and Top-P together?

Yes.

They are decoding controls and can be combined in an inference pipeline.

A common conceptual processing sequence is:

Transformer output logits

Temperature scaling

Adjusts probability concentration

Top-K filtering (optional)

Keeps the highest-scoring K candidates

Top-P filtering (optional)

Keeps the nucleus of probability mass

Renormalize and sample

Select one token from the remaining distribution

Append token and repeat

This is a common conceptual ordering, not a universal implementation requirement. Actual inference engines can apply filtering operations in different orders.

For example, changing temperature before Top-P can change which tokens belong to the nucleus.

Important: Top-K and Top-P are not necessarily both enabled. Many applications use Top-P without Top-K, or Top-K without Top-P.

# 7. Python implementation – Temperature, Top-K and Top-P

Now let's implement these concepts using PyTorch.

This is the kind of code you should understand as an AI engineer because it demonstrates how an LLM selects the next token.

## 7.1 Complete sampling implementation

```
import torchimport torch.nn.functional as Fdef sample_next_token(    logits,    temperature=1.0,    top_k=None,    top_p=None):    """    Select the next token using:    - Temperature    - Top-K    - Top-P    """    # Ensure we work on the last token's logits    logits = logits[:, -1, :]    # 1. Greedy decoding    if temperature <= 0:        return torch.argmax(            logits,            dim=-1,            keepdim=True        )    # 2. Temperature scaling    logits = logits / temperature    # 3. Top-K filtering    if top_k is not None and top_k > 0:        k = min(top_k, logits.size(-1))
```

## 7.2 How to call this function

```
# Example model output logitslogits = torch.randn(1, 10, 50000)# Temperature samplingtoken = sample_next_token(    logits,    temperature=0.7)# Top-K samplingtoken = sample_next_token(    logits,    temperature=0.7,    top_k=50)# Top-P samplingtoken = sample_next_token(    logits,    temperature=0.7,    top_p=0.9)# Combined samplingtoken = sample_next_token(    logits,    temperature=0.7,    top_k=50,    top_p=0.9)print(token)
```

This returns a token ID, not a readable word. You would pass that token ID to the tokenizer's decoding functionality to obtain text.

The function demonstrates the sampling logic, but a production implementation also needs input validation, repetition controls where appropriate, EOS handling, special-token handling, deterministic random seeds where required, and efficient GPU execution.

# 8. Complete token generation loop

Now let's understand how this sampling function fits into actual LLM inference.

The following is a simplified autoregressive generation loop.

```
import torch@torch.no_grad()def generate(    model,    input_ids,    max_new_tokens=100,    temperature=0.7,    top_k=50,    top_p=0.9,    eos_token_id=None):    model.eval()    generated_ids = input_ids    for _ in range(max_new_tokens):        # Forward pass through Transformer        logits = model(generated_ids)        # Select the next token        next_token = sample_next_token(            logits,            temperature=temperature,            top_k=top_k,            top_p=top_p        )        # Append new token        generated_ids = torch.cat(            [generated_ids, next_token],            dim=-1
```

This is educational code. It recomputes the entire sequence on every iteration and does not implement KV caching. A production inference engine avoids much of this repeated computation and manages multiple concurrent generation requests.

# 9. What is the difference between Greedy Decoding and Sampling?

You should also understand greedy decoding.

Greedy decoding always selects the token with the highest probability.

Example:

| Token  | Probability |
| ------ | ----------- |
| Python | 55%         |
| Java   | 20%         |
| C++    | 15%         |
| Rust   | 10%         |

Greedy decoding selects Python every time.

Sampling selects a token according to the probability distribution.

For example:

- Python: 55% chance.
- Java: 20% chance.
- C++: 15% chance.
- Rust: 10% chance.

Over multiple independent draws from the same distribution, the frequencies tend towards these probabilities.

| Decoding approach    | Behavior                             |
| -------------------- | ------------------------------------ |
| Greedy               | Always chooses highest probability   |
| Random sampling      | Samples from the full distribution   |
| Temperature sampling | Changes distribution before sampling |
| Top-K                | Samples from K candidates            |
| Top-P                | Samples from a probability nucleus   |

One important point: Greedy decoding can produce repetitive or less diverse responses. Sampling can improve diversity, but can also introduce less predictable outputs.

# 10. Practical production settings for different AI applications

As a Senior AI Engineer, you should not use the same decoding configuration for every application.

Here are sensible starting configurations.

| Application            | Temperature | Top-K | Top-P |
| ---------------------- | ----------- | ----- | ----- |
| Code generation        | 0.1–0.3     | 40    | 0.9   |
| Financial assistant    | 0.1–0.3     | Off   | 0.9   |
| RAG question answering | 0.1–0.3     | Off   | 0.9   |
| JSON generation        | 0–0.2       | Off   | 1.0   |
| General chatbot        | 0.5–0.8     | Off   | 0.9   |
| Creative writing       | 0.8–1.0     | Off   | 0.95  |

These are illustrative starting configurations, not universal benchmarks. You should tune them against your own evaluation dataset.

For structured JSON generation, use constrained decoding or grammar-based output validation rather than relying on low temperature alone.

# 11. Important Senior AI Engineer concepts related to decoding

There are a few more concepts worth understanding.

| Concept              | Explanation                                                                             |
| -------------------- | --------------------------------------------------------------------------------------- |
| Repetition penalty   | Penalizes previously generated tokens                                                   |
| Frequency penalty    | Penalizes tokens based on their frequency                                               |
| Presence penalty     | Penalizes tokens that have already appeared                                             |
| Min-P                | Removes tokens whose probability is too small relative to the highest-probability token |
| Typical sampling     | Selects tokens based on their information-theoretic typicality                          |
| Beam search          | Explores multiple candidate sequences                                                   |
| Speculative decoding | Uses a smaller draft model to accelerate generation with a larger model                 |

These are additional decoding or inference techniques, not all variations of Top-K and Top-P.

# 12. Top 10 interview questions

These are questions I would prepare for a Senior AI Engineer interview.

1. What is autoregressive token generation?
2. What is the difference between logits and probabilities?
3. How does temperature affect the softmax distribution?
4. What happens when temperature is set to zero?
5. What is the difference between Top-K and Top-P?
6. Can Top-K and Top-P be applied together?
7. Why is Top-P called nucleus sampling?
8. Why can high temperature produce hallucinations or unpredictable answers?
9. How do you select decoding parameters for a RAG application?
10. How would you optimize token generation latency in production?

# 13. Test your understanding

Decoding knowledge check

0/5 answered

1\. What does temperature control?

Number of Transformer layers

Randomness of token selection

Vocabulary size

Context window



2\. What does Top-K = 5 mean?

Generate five tokens

Keep five highest-scoring candidates

Keep 5% probability

Use five attention heads



3\. What does Top-P = 0.9 mean?

Keep exactly 90 tokens

Keep candidates covering at least 90% cumulative probability

Generate 90% of the response

Set temperature to 0.9



4\. What is autoregressive generation?

Generating all tokens simultaneously

Predicting the next token using previous context

Training a reward model

Converting text into embeddings



5\. Which approach always selects the highest-probability token?

Top-P

Random sampling

Greedy decoding

Beam sampling



Check answers

## The five concepts to remember

| Concept          | Remember this                                |
| ---------------- | -------------------------------------------- |
| Token generation | Predict one token at a time                  |
| Temperature      | Controls randomness                          |
| Top-K            | Controls number of candidates                |
| Top-P            | Controls cumulative probability              |
| Greedy decoding  | Always selects the highest-probability token |

My recommendation for your next learning step: Learn LLM inference internals in depth: Prefill vs Decode, KV Cache, Continuous Batching, PagedAttention, Quantization, Speculative Decoding and GPU memory management. These concepts connect token generation directly to production LLM serving with vLLM and are particularly important for Senior/Staff AI Engineer interviews.