# What are Residual Connections in Transformers?

Residual connections, also called Skip Connections, are a mechanism that allows information to bypass a neural network layer and get added directly to its output.

They are extremely important in deep neural networks, especially Transformers, because they help:

1. Prevent information from getting lost as it passes through many layers.
2. Improve gradient flow during backpropagation.
3. Make deep networks easier to train.
4. Allow each layer to learn modifications to existing representations rather than rebuilding everything from scratch.

Let's understand this with a simple example.

## 1. First understand the problem: Why do we need residual connections?

Imagine a Transformer with 12, 24, 48 or even 80 layers.

The information passes through many layers:

```
Input
  |
  v
Layer 1
  |
  v
Layer 2
  |
  v
Layer 3
  |
  v
Layer 4
  |
  v
 ...
  |
  v
Layer 48
  |
  v
Output
```

Every layer transforms the input representation.

But here's a problem:

As information passes through many transformations, the network can become difficult to train.

Two major problems can occur:

- Vanishing gradients: Gradients can become extremely small during backpropagation, making earlier layers difficult to update.
- Degradation problem: Adding more layers can make optimization harder, and a deeper network may perform worse than a shallower one.

We need a mechanism that allows information and gradients to travel more directly through the network.

That's where residual connections help.

## 2. What exactly is a residual connection?

Let's first understand a normal neural network layer.

Suppose the input is `x`.

A normal layer performs some transformation:

\\[ Output=F(x) \\]

Here, \\(F(x)\\) represents the transformation performed by the layer.

For example, it could be:

- A linear layer.
- An attention operation.
- A feed-forward network.

With a residual connection, instead of using only the transformed output, we add the original input to it.

\\[ \boxed{Output=x+F(x)} \\]

This is the fundamental residual connection formula.

Let's understand what this means:

- `x` = Original input.
- `F(x)` = Information transformed by the layer.
- `x + F(x)` = Original information plus the transformation.

In simple terms:

The layer gets the original information as a shortcut, while also learning what additional information or modification it should contribute.

## 3. Understand residual connections with a simple example

Imagine the input representation of a token is:

\\[ x=[2,4,6] \\]

A neural network layer transforms this input and produces:

\\[ F(x)=[1,3,2] \\]

Without a residual connection:

\\[ Output=F(x) \\]

\\[ Output=[1,3,2] \\]

With a residual connection:

\\[ Output=x+F(x) \\]

\\[ Output=[2,4,6]+[1,3,2] \\]

\\[ \boxed{Output=[3,7,8]} \\]

Notice what happened.

The output contains the original input information along with the transformation learned by the network.

## How a residual connection works

Original input (x)

# [2, 4, 6]

Neural network transformation F(x)

# [1, 3, 2]

Add original input

Final output: x + F(x)

# [3, 7, 8]

The important idea is that the original input has a direct path to the output.

This direct path is called the skip connection or shortcut connection.

## 4. Where exactly are residual connections used in Transformers?

This is especially important for interviews.

A standard Transformer block has two major sublayers:

1. Multi-Head Self-Attention.
2. Feed-Forward Network (FFN).

Both sublayers have residual connections.

Let's understand them separately.

### A. Residual connection around self-attention

Suppose the input to the attention sublayer is `x`.

The attention mechanism calculates contextual information:

\\[ AttentionOutput=Attention(x) \\]

The residual connection adds the original input:

\\[ \boxed{Output=x+Attention(x)} \\]

This means the model preserves the original representation while adding the contextual information gathered from other tokens.

### B. Residual connection around FFN

The output from the attention sublayer goes into the FFN.

The FFN transforms the representation:

\\[ FFNOutput=FFN(x) \\]

The residual connection adds the input to the FFN:

\\[ \boxed{Output=x+FFN(x)} \\]

Here, `x` means the input to that FFN sublayer, not necessarily the original input to the whole Transformer block.

### Complete Transformer block

Input x

Multi-Head Self-Attention

Add original input + Attention output

Residual connection

Feed-Forward Network

Add FFN input + FFN output

Residual connection

Output of Transformer block

In practice, LayerNorm or RMSNorm is also used around these sublayers. We'll discuss its exact placement shortly.

## 5. How do residual connections help with vanishing gradients?

Very important interview concept

First, let's understand backpropagation.

During training:

1. The model makes a prediction.
2. We calculate the loss.
3. Backpropagation calculates gradients.
4. The optimizer updates model parameters.

Gradients tell the network how to change its weights to reduce the loss.

The problem is that when gradients pass backward through many layers, they can become very small.

This is called the vanishing gradient problem.

Residual connections provide a direct path for gradients.

Let's understand mathematically.

Without a residual connection:

\\[ y=F(x) \\]

The gradient is:

\\[ \frac{\partial y}{\partial x}=F'(x) \\]

With a residual connection:

\\[ y=x+F(x) \\]

Now:

\\[ \boxed{\frac{\partial y}{\partial x}=1+F'(x)} \\]

Notice the important difference.

There is an additional `1` in the gradient.

This comes from the direct connection between the input and output.

Even if the gradient through the transformation \\(F(x)\\) becomes small, the direct path can still carry gradient information backward.

For example, suppose:

\\[ F'(x)=0.001 \\]

Without residual connection:

\\[ \frac{\partial y}{\partial x}=0.001 \\]

With residual connection:

\\[ \frac{\partial y}{\partial x}=1.001 \\]

This illustrates why residual connections can improve gradient flow.

Important: Residual connections reduce the risk of vanishing gradients; they do not guarantee that gradients will never vanish or explode.

## 6. Why are residual connections especially important in large LLMs?

Imagine a GPT model containing dozens of Transformer blocks.

Without residual connections, every block would have to transform the representation and pass only that transformed result to the next block.

With residual connections, every block has a shortcut path that carries its input forward.

This makes it easier to train deep networks.

Another important benefit is that a layer can learn a relatively small modification to the existing representation rather than having to learn a completely new representation.

For example:

\\[ Output=Input+LearnedModification \\]

This is called residual learning.

This idea became particularly popular through ResNet (Residual Networks), introduced for training very deep neural networks.

Transformers also use this concept extensively.

## 7. What is the difference between Pre-LayerNorm and Post-LayerNorm?

This is an advanced interview question because residual connections are closely related to normalization.

There are two common ways to combine residual connections and normalization.

### A. Post-LayerNorm (original Transformer)

In the original Transformer, normalization happens after adding the residual.

\\[ \boxed{Output=LayerNorm(x+F(x))} \\]

The sequence is:

1. Pass input through attention or FFN.
2. Add the original input.
3. Apply LayerNorm.

### B. Pre-LayerNorm (common in modern LLMs)

In Pre-LayerNorm, normalization happens before the transformation.

\\[ \boxed{Output=x+F(LayerNorm(x))} \\]

The sequence is:

1. Normalize the input.
2. Pass it through attention or FFN.
3. Add the original, unnormalized input.

Post-LayerNorm

Input x

Transformation F(x)

Add x + F(x)

LayerNorm

Pre-LayerNorm

Input x

LayerNorm

Transformation F(LayerNorm(x))

Add original x + transformed output

Why is Pre-LayerNorm popular in modern LLMs?

It provides a relatively direct residual path through the network, which often makes optimization and training of deep Transformers more stable.

For example, modern Llama-family architectures use RMSNorm-based pre-normalization.

Remember:

- Post-LN: Normalize after residual addition.
- Pre-LN: Normalize before the transformation, then add the original input.

## 8. Implement residual connections using PyTorch

Let's implement a simple Transformer-style block containing self-attention and FFN residual connections.

For simplicity, we'll use PyTorch's built-in multi-head attention.

```
import torchimport torch.nn as nnclass TransformerBlock(nn.Module):    def __init__(        self,        d_model=512,        num_heads=8,        d_ff=2048    ):        super().__init__()        # Multi-head attention        self.attention = nn.MultiheadAttention(            embed_dim=d_model,            num_heads=num_heads,            batch_first=True        )        # Feed-forward network        self.ffn = nn.Sequential(            nn.Linear(d_model, d_ff),            nn.GELU(),            nn.Linear(d_ff, d_model)        )        # Layer normalization        self.norm1 = nn.LayerNorm(d_model)        self.norm2 = nn.LayerNorm(d_model)    def forward(self, x):        # Pre-LayerNorm attention        normalized = self.norm1(x)
```

Output:

```
Input shape: torch.Size([2, 5, 512])
Output shape: torch.Size([2, 5, 512])
```

Look at these two lines:

```
x = x + attention_output
```

And:

```
x = x + ffn_output
```

These are the residual connections.

Notice that the input and output dimensions must match for direct addition. That's why the FFN projects its hidden representation back to `d_model`.

Production note: This is an illustrative pre-LayerNorm block. It does not include a causal mask, positional encoding or optimized attention kernels. A production GPT implementation would include the appropriate causal attention behavior.

## 9. Residual connection vs skip connection

You may hear both terms in interviews.

They are commonly used interchangeably when discussing Transformer residual paths.

The idea is:

- A transformation processes the input.
- A shortcut carries the original input around that transformation.
- The two paths are combined, usually through addition.

One distinction: a residual connection is not the same as concatenating the original and transformed vectors. Standard Transformer residual connections use element-wise addition.

Also, both vectors must have compatible dimensions. If their dimensions differ, a projection may be needed before adding them.

## 10. Interview-ready answer

If an interviewer asks, "What are residual connections in Transformers?", you can answer:

Senior AI Engineer interview answer

Residual connections, also known as skip connections, allow the original input of a neural network sublayer to bypass its transformation and be added directly to the sublayer output.

Mathematically, they work as:

\\[ Output=x+F(x) \\]

Here, x is the original input and F(x) is the transformed output.

In a Transformer block, residual connections are used around both the multi-head self-attention and feed-forward network sublayers.

They are important because they provide a direct path for information and gradients to flow through deep networks. This helps reduce optimization difficulties associated with vanishing gradients and makes training deep Transformers more stable.

Modern LLMs commonly use Pre-LayerNorm or RMSNorm-based architectures, where the input is normalized before the attention or FFN operation and the original input is added back afterward.

In simple terms, residual connections allow each layer to learn modifications to existing representations instead of having to completely replace them.

## 11. Important interview questions to prepare

| Interview question                    | Key point                                        |
| ------------------------------------- | ------------------------------------------------ |
| Why do we need residual connections?  | Better information and gradient flow             |
| What is the residual formula?         | \\(Output=x+F(x)\\)                              |
| Where are they used in Transformers?  | Around attention and FFN                         |
| How do they help vanishing gradients? | Provide a direct gradient path                   |
| What is residual learning?            | Learning modifications to the input              |
| What is Pre-LayerNorm?                | Normalize before the sublayer                    |
| What is Post-LayerNorm?               | Normalize after residual addition                |
| Why must dimensions match?            | Element-wise addition requires compatible shapes |

Remember these three concepts:

1. Self-attention gathers information from other tokens.
2. FFN transforms the gathered information.
3. Residual connections preserve a direct path for the original representation and help gradients flow.

Recommended next topic: Layer Normalization and RMSNorm. These are the other essential components of Transformer blocks, and interviewers often ask how residual connections and normalization work together.