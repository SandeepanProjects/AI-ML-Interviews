# What is FFN (Feed-Forward Network) in Transformers?

FFN stands for Feed-Forward Network. It is also called the Position-wise Feed-Forward Network (FFN) or, in modern LLMs, the MLP (Multi-Layer Perceptron) block.

It is one of the two major computational components inside a standard Transformer block:

1. Self-Attention: Helps tokens understand relationships with other tokens.
2. FFN: Processes and transforms the information within each token representation.

Let's understand it from the basics.

## 1. First, why does a Transformer need an FFN?

Remember what we learned about self-attention.

Self-attention helps a token gather information from other tokens.

For example:

Sentence: "The dog is running because it is energetic."

When processing the word `it`, self-attention can help the model connect it with `dog`.

So, attention helps answer:

Which other tokens are relevant to me?

But after gathering information from other tokens, the model needs to process that information and learn useful patterns.

That's where FFN comes in.

Think of it this way:

- Attention = Information gathering.
- FFN = Information processing and transformation.

An easy real-world analogy:

Imagine a team meeting.

- Attention is like listening to what everyone in the meeting says.
- FFN is like individually thinking about that information, analyzing it and transforming it into a useful understanding.

Both are necessary.

## 2. What exactly does an FFN do?

An FFN is a small neural network applied independently to each token representation.

In the original Transformer, it consists of two linear layers with a ReLU activation function between them.

The formula is:

\\[ FFN(x)=max(0,xW_1+b_1)W_2+b_2 \\]

Let's break it down.

- \\(x\\): Input token representation.
- \\(W_1\\): First layer's learned weights.
- \\(b_1\\): First layer's bias.
- ReLU: Activation function.
- \\(W_2\\): Second layer's learned weights.
- \\(b_2\\): Second layer's bias.

The FFN first expands the representation into a larger dimension, applies a nonlinear transformation, and then projects it back to the original dimension.

## 3. Understand FFN with an example

Suppose our Transformer has:

- Model dimension (`d_model`) = 512
- FFN hidden dimension (`d_ff`) = 2048

The input representation of one token has 512 values.

The FFN processes it like this:

## FFN architecture

Input representation

# 512

One token's vector

Linear Layer 1

# 512 → 2048

Expands the representation

ReLU Activation

Introduces non-linearity

Linear Layer 2

# 2048 → 512

Projects back to model dimension

Output representation

# 512

Transformed token representation

Why do we expand from 512 to 2048 and then reduce it back to 512?

Because the larger hidden space gives the network more capacity to learn complex nonlinear transformations.

Think of it like this:

You have 512 pieces of information. You temporarily transform them into a larger workspace with 2048 dimensions, process them, and then compress the result back into 512 dimensions.

The output dimension stays the same so it can work with the Transformer block's residual connection.

Important: 2048 is just an example. In the original Transformer, the FFN dimension was four times the model dimension. Modern LLMs can use different sizes and different FFN architectures.

## 4. Why do we need the ReLU activation function?

This is an important interview question.

ReLU stands for Rectified Linear Unit.

Its formula is:

\\[ ReLU(x)=max(0,x) \\]

It works very simply:

| Input | ReLU output |
| ----- | ----------- |
| -5    | 0           |
| -2    | 0           |
| 0     | 0           |
| 3     | 3           |
| 8     | 8           |

So:

- Negative values become zero.
- Positive values remain unchanged.

But why is this necessary?

Because without a nonlinear activation function, two linear layers combined would still behave like one linear transformation.

For example:

\\[ y=(xW_1)W_2 \\]

This can be simplified to:

\\[ y=x(W_1W_2) \\]

That means two linear layers without a nonlinear activation would not give the network the additional nonlinear expressive power we want.

ReLU introduces nonlinearity, helping the model learn more complex patterns.

[FFN Activation Functions: ReLU, GELU, and SiLU for Transformer Models - Interactive | Michael Brenndoerfer | Michael Brenndoerfer](https://images.openai.com/static-rsc-4/Q9_-9iN4PdpYC-JpW04yBhCXo1NxVgCSnppreTaLnwNlvPr4ASa_sNGuTuvaaADqw2qJG8Vm6t77HL38n20-dbqGLvhm_U_e7iO0iZ-ZifGpdq3gdCXGXgoFW2PVgXnzkiipTy8ItbTmmAAE76cWzuqTLlo2Aeh9IEkNRpHOEMM?purpose=inline)

[mbrenndoerfer.com](https://mbrenndoerfer.com/writing/ffn-activation-functions?utm_source=chatgpt.com)

## 5. Why is FFN called a Position-wise Feed-Forward Network?

This is a very important concept.

Suppose our input sentence has four tokens:

`I love machine learning`

After embedding and self-attention, we have four token representations.

Assume each representation has 512 dimensions.

Our input tensor has this shape:

\\[ X=(1,4,512) \\]

Here:

- 1 = batch size.
- 4 = sequence length.
- 512 = embedding dimension.

The FFN processes each token representation independently using the same network weights.

Position-wise processing

Illustration: each token goes through the same FFN, independently.

I

Same FFN

Output

love

Same FFN

Output

machine

Same FFN

Output

learning

Same FFN

Output

Each token is transformed independently, but all tokens share the same FFN parameters.

There are two important things to remember:

1. The FFN does not directly mix information between different token positions.
2. Every position uses the same weights and biases.

But remember something interesting:

Although the FFN processes tokens independently, each token already contains contextual information gathered through self-attention.

For example, the representation of `dog` may already contain information about `chased` and `cat`.

The FFN then transforms that context-rich representation.

This is why both attention and FFN work together.

## 6. What happens inside a complete Transformer block?

Let's connect FFN with the self-attention mechanism we studied earlier.

A standard Transformer block has two major sublayers:

1. Multi-Head Self-Attention.
2. Feed-Forward Network.

Each sublayer is surrounded by a residual connection and normalization.

Input representations

Multi-Head Self-Attention

Collects information from other tokens

Add + Layer Normalization

Residual connection and normalization

Feed-Forward Network

Transforms each token's representation

Add + Layer Normalization

Another residual connection and normalization

Output of Transformer block

The above is the original Post-LayerNorm-style arrangement.

Modern decoder-only LLMs such as GPT and Llama commonly use Pre-LayerNorm or RMSNorm-style arrangements, where normalization occurs before the attention and FFN sublayers.

The important distinction is:

- Attention gathers contextual information.
- FFN transforms the contextual information.
- Residual connections help preserve information and support training.
- Normalization helps stabilize the network.

## 7. What type of FFN is used in modern LLMs?

This is especially important for Senior AI Engineer interviews.

The original Transformer uses:

Linear → ReLU → Linear

But modern LLMs often use more advanced activation functions and FFN designs.

For example:

| Architecture         | Common FFN design                                |
| -------------------- | ------------------------------------------------ |
| Original Transformer | ReLU FFN                                         |
| GPT-style models     | GELU or other MLP variants, depending on version |
| Llama-family models  | SwiGLU FFN                                       |
| Some modern LLMs     | Gated MLP variants                               |

### What is SwiGLU?

SwiGLU is a gated feed-forward network used in many modern LLM architectures.

Instead of just applying one activation between two linear layers, it uses a gating mechanism.

A simplified formula is:

\\[ SwiGLU(x)= (SiLU(xW_1)\odot xW_3)W_2 \\]

Here:

- \\(W_1\\) and \\(W_3\\) are separate learned projections.
- SiLU is an activation function.
- \\(\odot\\) means element-wise multiplication.
- \\(W_2\\) projects the result back to the model dimension.

Let's understand the intuition.

Imagine the FFN has two information paths:

- One path generates information.
- Another path generates a gate that controls how much of that information passes through.

The outputs are multiplied together.

This gives the network a more flexible way to control information flow.

[拆解 LLM 中的 SwiGLU - 有何m不可 - 博客园](https://images.openai.com/static-rsc-4/EjfN1uWWqUOiE50pdYMlYJ367K9OO_0MDs_idK_exHdwP8saz5NuOkq9WkamE6EHdISD71RDw32lPMRPJdjgRXU99cozbQCYMFXOoXLGiIG0LUvfnc-HmCMpNXU53cJab0y5eZdP5eKNHXcXBSBxI_lo__KXrLANiAdxHeQFtjs?purpose=inline)

[cnblogs.com](https://www.cnblogs.com/gongzb/p/19112194?utm_source=chatgpt.com)

Interview point: If asked what FFN is used in Llama, you can say that Llama-family models use a SwiGLU-based gated MLP, rather than the original Transformer’s simple ReLU FFN.

## 8. Implement FFN using PyTorch

Let's implement the original Transformer-style FFN.

```
import torchimport torch.nn as nnclass FeedForwardNetwork(nn.Module):    def __init__(        self,        d_model: int,        d_ff: int    ):        super().__init__()        self.linear1 = nn.Linear(            d_model,            d_ff        )        self.activation = nn.ReLU()        self.linear2 = nn.Linear(            d_ff,            d_model        )    def forward(self, x):        # Expand embedding dimension        x = self.linear1(x)        # Apply non-linearity        x = self.activation(x)        # Project back to original dimension        x = self.linear2(x)
```

Output:

```
Input shape: torch.Size([2, 5, 512])
Output shape: torch.Size([2, 5, 512])
```

Why does the output shape remain the same?

Because the FFN expands the last dimension internally from 512 to 2048 and then projects it back to 512.

The batch size and sequence length remain unchanged.

Production note: This is an educational implementation of the original FFN. Production implementations may use fused linear operations, optimized activation kernels, gated MLPs, tensor parallelism and other performance optimizations.

## 9. Why is FFN important in large language models?

There are three important reasons.

1\. It increases the model's learning capacity.

The FFN has millions or billions of trainable parameters across all Transformer layers. These parameters help the model learn complex transformations and patterns.

2\. It introduces nonlinearity.

Without nonlinear activation functions, stacked linear transformations would collapse into a single linear transformation. Nonlinearity allows the model to learn more complex functions.

3\. It transforms contextual representations.

Attention gathers information from different tokens. FFN processes that information and transforms each token's representation into a more useful feature representation.

### Does FFN consume a lot of computation?

Yes.

In a standard Transformer, the FFN can account for a significant portion of the computation and parameters.

For the original two-layer FFN, ignoring biases:

\\[ Parameters = 2 \times d\_{model}\times d\_{ff} \\]

For example, with:

- \\(d\_{model}=512\\)
- \\(d\_{ff}=2048\\)

\\[ Parameters=2\times512\times2048 \\]

\\[ =2,097,152 \\]

That's approximately 2.1 million parameters in just one FFN.

This is one reason FFN optimization is important when building and serving large language models.

## 10. FFN vs Self-Attention

This is a common comparison question.

| Feature                      | Self-Attention                                   | FFN                                                            |
| ---------------------------- | ------------------------------------------------ | -------------------------------------------------------------- |
| Main purpose                 | Gather information from tokens                   | Transform token representations                                |
| Token interaction            | Interacts across positions                       | Processes each position independently                          |
| Parameters                   | Q, K, V and output projections                   | Linear/gating projections                                      |
| Input                        | Token representations                            | Contextual token representations                               |
| Output                       | Context-enriched representations                 | Transformed representations                                    |
| Computational characteristic | Attention interaction grows with sequence length | Computation scales approximately linearly with sequence length |

A simple way to remember:

Self-attention answers:

What information should I collect from other tokens?

FFN answers:

How should I process and transform the information I have collected?

## 11. Interview-ready answer

If an interviewer asks, "Explain the Feed-Forward Network in Transformers," you can give this answer:

Senior AI Engineer interview answer

A Feed-Forward Network, or FFN, is a neural network component inside every standard Transformer block.

Its main responsibility is to transform the contextual representations generated by the self-attention mechanism.

Self-attention allows tokens to gather information from other tokens, whereas the FFN processes each token representation independently.

In the original Transformer architecture, the FFN consists of two linear layers with a ReLU activation function in between.

The first linear layer expands the embedding dimension, for example, from 512 to 2048. The ReLU introduces nonlinearity, and the second linear layer projects the representation back to 512.

The same FFN parameters are shared across all token positions.

Modern LLMs often use more advanced gated FFN architectures, such as SwiGLU, which is used in Llama-family models.

So, attention is responsible for gathering contextual information, and FFN is responsible for processing and transforming that information.

## 12. Test your understanding

FFN interview quiz

0/5 answered

1\. What is the primary responsibility of FFN?

Gather information from other tokens

Transform each token representation

Tokenize the input

Apply causal masking

2\. Why does the FFN expand 512 dimensions to 2048 and then reduce them to 512?

To increase sequence length

To temporarily provide a larger hidden representation for learning transformations

To remove attention

To generate new tokens

3\. What does position-wise mean?

Every token uses different FFN weights

The FFN operates independently at each token position using shared weights

The FFN only processes the first token

The FFN processes future tokens only

4\. Which FFN architecture is commonly used in Llama-family models?

ReLU only

SwiGLU

Causal attention

Softmax

5\. What happens if two linear layers have no nonlinear activation between them?

They become self-attention

They can be combined into one linear transformation

They automatically become SwiGLU

They perform causal masking

Submit answers

What you should learn next: Residual connections and Layer Normalization. These explain how Transformers preserve information, stabilize training and successfully train very deep neural networks.