# Top-K Sampling in LLMs — Explained with Python Code

Top-K sampling is a text-generation strategy that restricts an LLM to choosing its next token from the K highest-scoring candidate tokens. The model then samples one token from that smaller set according to their probabilities.

It is used to control the balance between predictable and diverse text generation.

To understand it properly as a senior AI engineer, we'll cover:

1. Why Top-K sampling is needed.
2. How it works step by step.
3. How logits become probabilities.
4. A complete Python implementation using PyTorch.
5. How Top-K compares with greedy decoding, temperature, and Top-P.
6. How these techniques are used in production LLM systems.

## 1. Why do we need Top-K sampling?

Suppose an LLM receives this prompt:

```
The developer opened the
```

The model might assign the following probabilities to possible next tokens:

| Candidate token | Probability |
| --------------- | ----------- |
| `laptop`        | 35%         |
| `computer`      | 25%         |
| `terminal`      | 18%         |
| `application`   | 12%         |
| `banana`        | 5%          |
| `mountain`      | 3%          |
| Other tokens    | 2%          |

These values are illustrative.

Without filtering, sampling could select any token in the vocabulary, including a very unlikely token such as `mountain`.

Top-K sampling restricts the candidates to the K tokens with the highest scores.

For example, if \\(K=3\\), the candidate set becomes:

Top-K candidates when K = 3

laptop

35%

computer

25%

terminal

18%

The remaining tokens are excluded from sampling for this step.

The model can now sample only `laptop`, `computer`, or `terminal`.

Notice that Top-K doesn't necessarily select `laptop`. It selects randomly according to the normalized probabilities of the remaining candidates.

That is the key difference between Top-K sampling and greedy decoding.

- Greedy decoding always selects the highest-scoring token.
- Top-K sampling randomly selects among the K highest-scoring tokens.

## 2. How does Top-K sampling work internally?

Let's walk through the exact steps.

Assume the model has a vocabulary of 50,000 tokens. After processing the prompt, the model produces 50,000 logits.

A logit is a raw score, not a probability.

The Top-K algorithm works as follows:

Step 1: LLM produces logits

50,000 candidate scores

Step 2: Find the K largest logits

For K = 3, retain the three highest scores

Step 3: Mask all other logits

Set excluded scores to negative infinity

Step 4: Apply softmax

Normalize probabilities over the retained candidates

Step 5: Sample one token

Use the resulting probability distribution

The masking step is especially important. If excluded logits were left unchanged, those tokens could still be sampled.

## 3. Understand the mathematics

Suppose the model generates these logits:

\\[ z = [5.0,\ 4.0,\ 3.0,\ 1.0,\ 0.5] \\]

Let's choose \\(K=3\\).

The three largest logits are:

\\[ z\_{\text{top-3}} = [5.0,\ 4.0,\ 3.0] \\]

We mask the other two:

\\[ z' = [5.0,\ 4.0,\ 3.0,\ -\infty,\ -\infty] \\]

Then apply softmax:

\\[ P_i = \frac{e^{z'\_i}}{\sum_j e^{z'\_j}} \\]

The resulting probabilities are approximately:

| Token               | Original softmax probability | Probability after Top-K |
| ------------------- | ---------------------------- | ----------------------- |
| Token A (logit 5.0) | 82.7%                        | 84.4%                   |
| Token B (logit 4.0) | 11.2%                        | 11.4%                   |
| Token C (logit 3.0) | 4.1%                         | 4.2%                    |
| Token D (logit 1.0) | 0.6%                         | 0%                      |
| Token E (logit 0.5) | 0.4%                         | 0%                      |

Values are rounded. The exact probabilities depend on the logits.

Notice two things:

1. Tokens D and E become impossible to select in this step.
2. The remaining probabilities are renormalized to sum to 100%.

Top-K doesn't simply remove low-probability tokens after sampling. It filters the logits before sampling.

A small numerical correction to the illustrative table above: for logits `[5.0, 4.0, 3.0, 1.0, 0.5]`, the original softmax probabilities are approximately `65.3%, 24.0%, 8.8%, 1.2%, 0.7%`. After Top-3 filtering and renormalization, the probabilities are approximately `66.5%, 24.5%, 9.0%, 0%, 0%`.

## 4. Implement Top-K sampling in Python

Let's implement the algorithm using PyTorch, the library commonly used to build and run neural networks.

### Complete working example

```
import torchdef top_k_sampling(logits, k=3, temperature=1.0):    """    Sample one token from the K highest-scoring logits.    Args:        logits: 1D tensor of next-token scores.        k: Number of candidate tokens to retain.        temperature: Positive temperature for sampling.    Returns:        Selected token ID and filtered probability distribution.    """    if logits.ndim != 1:        raise ValueError("logits must be a 1D tensor")    if not 1 <= k <= logits.numel():        raise ValueError("k must be between 1 and vocabulary size")    if temperature <= 0:        raise ValueError("temperature must be positive")    # 1. Adjust logits by temperature    scaled_logits = logits / temperature    # 2. Find the K largest logits    top_values, top_indices = torch.topk(scaled_logits, k=k)    # 3. Exclude all tokens outside the top K    filtered_logits = torch.full_like(        scaled_logits, float("-inf")    )    filtered_logits[top_indices] = top_values
```

The selected token can vary between runs because sampling is random. The probabilities should be approximately:

```
laptop    : 0.665
computer  : 0.245
terminal  : 0.090
banana    : 0.000
mountain  : 0.000
```

### Let's understand each important line

`torch.topk(scaled_logits, k=k)`

Finds the K highest values and their original indices. We need the indices because they tell us which vocabulary tokens to keep.

`torch.full_like(scaled_logits, float("-inf"))`

Creates a tensor filled with negative infinity. Softmax assigns zero probability to those excluded candidates.

`filtered_logits[top_indices] = top_values`

Restores the original scores for the selected tokens, leaving every other candidate masked.

`torch.softmax(filtered_logits, dim=-1)`

Converts the filtered scores into probabilities summing to one.

`torch.multinomial(probabilities, num_samples=1)`

Samples one token according to the resulting distribution. It does not always select the token with the highest probability.

One important edge case: if \\(K=1\\), Top-K sampling becomes equivalent to greedy selection for a finite set of logits, because only the highest-scoring token remains.

## 5. Experiment: What happens when you change K?

Try changing K to see how the candidate set affects generation.

Top-K value

## 3

1

2

3

4

5

Temperature

## 1.0

0.5

1.0

2.0

Resulting sampling distribution

laptop

66.5%

computer

24.5%

terminal

9.0%

banana

0.0%

Excluded by Top-K

mountain

0.0%

Excluded by Top-K

Interactive illustration based on the five example logits. A real model normally has thousands or more candidate tokens.

What should you observe?

- K = 1: only the highest-scoring token can be selected.
- K = 2: the model chooses between the two highest-scoring candidates.
- K = 3: it can choose from three candidates.
- K = 5: all five example candidates remain eligible.
- Lower temperature: probability concentrates more strongly on the highest-scoring candidates.
- Higher temperature: probability becomes more evenly distributed among the retained candidates.

Temperature and Top-K solve different problems. Temperature changes the probability distribution; Top-K changes which candidates are eligible.

## 6. Top-K vs Top-P vs greedy decoding

| Method                   | How it selects candidates                                                        | Main behavior                  |
| ------------------------ | -------------------------------------------------------------------------------- | ------------------------------ |
| Greedy decoding          | Keeps only the highest-scoring token                                             | Most predictable               |
| Top-K                    | Keeps a fixed number of highest-scoring tokens                                   | Fixed candidate count          |
| Top-P (nucleus sampling) | Keeps enough high-probability tokens to reach a cumulative probability threshold | Candidate count varies         |
| Temperature              | Rescales logits before sampling                                                  | Controls probability sharpness |

For example, Top-K with \\(K=3\\) always keeps three candidates when the vocabulary has at least three valid candidates.

Top-P with \\(p=0.90\\) may keep two candidates for a sharply concentrated distribution or many candidates for a flatter distribution.

Top-K and Top-P can also be combined in one decoding pipeline.

## 7. How is Top-K used in production LLMs?

Top-K is a decoding parameter, not a model-training technique. You generally configure it in the inference server or model-generation API, if the runtime supports it.

For example, a generation configuration might conceptually look like this:

```
generation_config = {    "temperature": 0.7,    "top_k": 40,    "max_new_tokens": 300,}
```

This is an illustrative configuration dictionary, not a universal API. Actual support and defaults vary across inference libraries and model providers.

Typical considerations:

- Creative writing: a larger candidate set may permit more variety.
- Code generation: a smaller candidate set or lower temperature can make output more predictable, but neither guarantees correctness.
- Financial compliance: validate claims against evidence, enforce schemas, and use deterministic rules for consequential decisions. Top-K alone is not a safety mechanism.
- High-throughput serving: filtering is only one part of decoding; batching, KV-cache memory, GPU execution, and streaming often have much larger performance implications.

One subtle but important detail: because Top-K is based on logits, dividing all logits by a positive temperature does not change their ranking. Therefore, temperature usually changes the sampling probabilities without changing which tokens Top-K retains. Tie-breaking can affect the exact selected set when logits are equal.

## 8. Senior AI engineer summary

Remember this exact sequence:

1. The LLM produces next-token logits.
2. Top-K finds the K largest logits.
3. The other logits are masked to negative infinity.
4. Softmax renormalizes the retained scores into probabilities.
5. The decoder samples a token from those probabilities.
6. The token is appended to the context, and generation continues.

The most important distinction: Top-K controls the size of the candidate set, while temperature controls how probability is distributed across that set. Neither one improves the model's underlying knowledge or guarantees factual correctness.