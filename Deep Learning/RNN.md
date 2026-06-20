If you're interviewing for a **Senior AI/ML Engineer (5–10+ years)**, the interviewer usually doesn't want textbook definitions. They expect you to understand:

* Mathematical intuition
* Internal implementation
* Why each architecture was invented
* Tradeoffs
* Production implications
* When NOT to use them
* Code implementation from scratch

Let's go through each topic as if we're designing these models ourselves.

---

# 1. Recurrent Neural Network (RNN)

## Why do we even need an RNN?

Suppose we want to predict the next word.

Sentence:

```
I love deep learning because it is ______
```

The prediction depends on previous words.

A normal neural network does this

```
Input → Hidden → Output
```

Every sample is independent.

But language isn't independent.

```
Word1 affects Word2
Word2 affects Word3
Word3 affects Word4
...
```

We need memory.

That's exactly why RNN was invented.

---

# Normal Neural Network

```
Dog
↓

NN

↓

Animal
```

Another example

```
Cat
↓

NN

↓

Animal
```

No relationship.

---

But language

```
I
love
deep
learning
```

Each word depends on previous words.

Need memory.

---

# RNN Idea

Instead of forgetting previous computation...

We carry it.

```
Hidden State

h0

↓

Word1

↓

h1

↓

Word2

↓

h2

↓

Word3

↓

h3
```

Hidden state stores memory.

---

# Internal Equation

At every timestep

```
h_t = tanh(
Wx*x_t
+
Wh*h_(t-1)
+
b
)
```

Output

```
y_t = Wy*h_t
```

---

Where

```
x_t
Current input

h_(t-1)
Previous memory

Wx
Input weights

Wh
Memory weights

Wy
Output weights
```

---

# Example

Sentence

```
The cat sat on the mat
```

Processing

```
Input

The

↓

Hidden State

↓

cat

↓

Updated Hidden State

↓

sat

↓

Updated Hidden State

↓

on

↓

Updated Hidden State

↓

the

↓

Updated Hidden State

↓

mat
```

Memory keeps growing.

---

# Code from Scratch

```python
import numpy as np

input_size = 5
hidden_size = 4

Wx = np.random.randn(hidden_size, input_size)
Wh = np.random.randn(hidden_size, hidden_size)
b = np.zeros((hidden_size, 1))

def rnn_step(x, h_prev):
    return np.tanh(
        Wx @ x +
        Wh @ h_prev +
        b
    )

h = np.zeros((hidden_size,1))

for t in range(6):
    x = np.random.randn(input_size,1)
    h = rnn_step(x,h)

print(h)
```

This is exactly what happens inside every timestep.

---

# Why RNN Fails

This is one of the most asked interview questions.

Suppose sentence is

```
I grew up in France.

...

...

...

I speak fluent French.
```

To predict

```
French
```

Model needs to remember

```
France
```

which appeared 50 words earlier.

RNN cannot.

Why?

---

# Hidden State Update

```
h1

↓

h2

↓

h3

↓

h4

↓

...

↓

h100
```

Every timestep updates memory.

Old information slowly disappears.

Like repeatedly mixing paint.

Eventually original color disappears.

---

# Bigger Problem

Backpropagation.

Gradient

```
∂Loss
---------
∂Weights
```

During training

Gradient flows backwards.

```
Output

↓

h100

↓

h99

↓

...

↓

h2

↓

h1
```

At every layer

Gradient multiplied.

```
Gradient

×

Weight

×

Weight

×

Weight

×

Weight

×

Weight
```

If weights are

```
0.8

0.8

0.8

0.8

0.8
```

Gradient becomes

```
0.8^100
```

Almost zero.

This is

# Vanishing Gradient

---

If weights

```
1.5

1.5

1.5

1.5
```

Gradient becomes

```
1.5^100
```

Huge.

Training explodes.

This is

# Exploding Gradient

---

Result

RNN forgets long-term information.

---

# Solution

Need selective memory.

Don't overwrite everything.

This led to

# LSTM

---

# 2. Long Short-Term Memory (LSTM)

Instead of one hidden state

LSTM introduces

```
Hidden State

+

Cell State
```

Think

```
Hidden State

=
Current thinking

Cell State

=
Long-term memory
```

---

Architecture

```
Input

↓

Forget Gate

↓

Input Gate

↓

Cell Update

↓

Output Gate

↓

Hidden State
```

---

There are FOUR neural networks inside one LSTM cell.

---

# Forget Gate

Question

```
What should I forget?
```

Equation

```
f = sigmoid(
Wx*x
+
Wh*h
)
```

Sigmoid outputs

```
0

Forget

1

Keep
```

Example

Sentence

```
I live in India.

Yesterday I visited Japan.
```

Forget gate removes

```
Japan
```

when predicting

```
Indian citizen
```

---

# Input Gate

Question

```
What new information should I store?
```

```
i = sigmoid(...)
```

Candidate

```
g = tanh(...)
```

---

# Cell Update

```
C_new

=

Forget

×

Old Cell

+

Input

×

Candidate
```

Equation

```
Ct

=

ft*Ct-1

+

it*gt
```

Old memory survives.

---

# Output Gate

Question

```
What should I expose?
```

```
ot = sigmoid(...)
```

Hidden state

```
ht

=

ot

×

tanh(Ct)
```

---

Diagram

```
Old Memory

↓

Forget

↓

Remaining Memory

↓

Add New Memory

↓

Updated Cell

↓

Output Gate

↓

Hidden State
```

---

# Code

```python
import torch
import torch.nn as nn

lstm = nn.LSTM(
    input_size=128,
    hidden_size=256,
    num_layers=2,
    batch_first=True
)

x = torch.randn(32,50,128)

output,(h,c)=lstm(x)

print(output.shape)
```

Output

```
32
×

50
×

256
```

---

# Why LSTM Works Better

Instead of

```
Overwrite memory
```

It says

```
Forget 10%

Keep 90%

Add 5%

Output 30%
```

Much smarter.

---

# Why LSTM is Slow

Every timestep

```
Forget Gate

+

Input Gate

+

Candidate

+

Output Gate
```

Four matrix multiplications.

Very expensive.

---

Hence

GRU.

---

# 3. GRU (Gated Recurrent Unit)

Google asked

```
Can we simplify LSTM?
```

Answer

GRU.

---

GRU removes

```
Cell State
```

Only hidden state remains.

---

Instead of 4 gates

GRU has only

```
Reset Gate

Update Gate
```

---

Architecture

```
Input

↓

Reset Gate

↓

Candidate

↓

Update Gate

↓

Hidden State
```

---

Update Gate

```
How much old memory should remain?
```

Reset Gate

```
Ignore previous memory?
```

---

Equation

```
z = sigmoid(...)

r = sigmoid(...)

h~

= tanh(...)
```

Final

```
h

=

(1-z)

×

Old

+

z

×

Candidate
```

---

Advantages

```
Faster

Less memory

Almost same accuracy

Fewer parameters
```

---

Code

```python
gru = nn.GRU(
    input_size=128,
    hidden_size=256,
    batch_first=True
)

x = torch.randn(16,40,128)

output,h = gru(x)
```

---

# LSTM vs GRU

| LSTM                              | GRU                  |
| --------------------------------- | -------------------- |
| 4 gates                           | 2 gates              |
| Cell state                        | No cell state        |
| More parameters                   | Less parameters      |
| Slower                            | Faster               |
| Better for very long dependencies | Good enough for most |

---

# Biggest Limitation of RNN/LSTM/GRU

Everything happens sequentially.

```
Word1

↓

Word2

↓

Word3

↓

Word4
```

GPU waits.

Cannot parallelize.

Training becomes slow.

---

Example

Sentence

```
The movie was not very good.
```

To understand

```
good
```

Need

```
not
```

which appeared earlier.

RNN only remembers compressed hidden state.

Information gets lost.

---

Researchers asked

Why compress entire sentence into one vector?

Why not directly look back?

This created

# Attention

---

# 4. Attention

Instead of remembering everything...

Model simply looks wherever needed.

---

Sentence

```
The animal didn't cross the road because it was tired.
```

Question

```
Who was tired?
```

Attention looks directly at

```
animal
```

instead of hoping memory retained it.

---

Think

Teacher in classroom.

Student asks

```
Who invented Python?
```

Teacher doesn't memorize every book.

She looks at the relevant page.

That's Attention.

---

# Query

Current word.

```
What am I looking for?
```

---

# Key

Address of every word.

```
Where is information stored?
```

---

# Value

Actual information.

```
What information exists?
```

---

Every word becomes

```
Query

Key

Value
```

---

Example

Sentence

```
The cat drank milk.
```

Word

```
drank
```

creates Query.

Looks at Keys

```
The

cat

drank

milk
```

Scores

```
The   0.01

cat   0.40

drank 0.10

milk  0.49
```

Softmax

```
0.01

0.40

0.10

0.49
```

Weighted sum

```
Context

=

0.01*The

+

0.40*cat

+

0.10*drank

+

0.49*milk
```

Now model knows

```
cat drank milk
```

---

# Attention Equation

```
Score

=

QKᵀ
```

Normalize

```
Softmax
```

Context

```
Attention

=

Softmax(QKᵀ)

V
```

Scaled version used in transformers:

[
\text{Attention}(Q,K,V)=\text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
]

The scaling factor (\sqrt{d_k}) prevents the dot products from becoming too large, which would otherwise make the softmax saturate and produce tiny gradients.

---

# Code

```python
import torch
import torch.nn.functional as F

Q = torch.randn(4,64)
K = torch.randn(4,64)
V = torch.randn(4,64)

scores = Q @ K.T

weights = F.softmax(scores, dim=-1)

context = weights @ V

print(context.shape)
```

---

# Why Attention Changed AI

Attention solved several limitations of recurrent networks:

* **Direct access to any token:** Every token can attend to every other token without compressing everything into a single hidden state.
* **Parallel computation:** All tokens are processed simultaneously, making excellent use of GPUs.
* **Long-range dependencies:** A word at the beginning of a document can directly influence a word thousands of tokens later.
* **Better gradient flow:** Information doesn't need to pass through hundreds of recurrent steps.

These advantages are why modern large language models rely on attention-based architectures rather than RNNs.

---

# Complexity Comparison

| Model     | Long-term Memory | Parallelizable | Training Speed   | Parameters            | Main Limitation                                                  |
| --------- | ---------------- | -------------- | ---------------- | --------------------- | ---------------------------------------------------------------- |
| RNN       | ❌ Poor           | ❌ No           | Slow             | Low                   | Vanishing/exploding gradients                                    |
| LSTM      | ✅ Good           | ❌ No           | Slower           | High                  | Computationally expensive                                        |
| GRU       | ✅ Good           | ❌ No           | Faster than LSTM | Medium                | Slightly less expressive than LSTM                               |
| Attention | ✅ Excellent      | ✅ Yes          | Fast on GPUs     | Depends on model size | Quadratic memory/time with sequence length in standard attention |

---

# Senior AI Engineer Interview Takeaway

A concise explanation that interviewers appreciate is:

> "RNN introduced recurrence by carrying a hidden state across time, allowing sequential information to be modeled. However, because gradients must propagate through many recurrent steps, RNNs suffer from vanishing and exploding gradients, making long-term dependencies difficult to learn. LSTM addressed this by introducing a persistent cell state and gated mechanisms—forget, input, and output gates—to control information flow. GRU simplified LSTM by combining some gates and removing the separate cell state, reducing parameters while maintaining similar performance. Ultimately, transformers replaced recurrent architectures in many NLP tasks because attention allows every token to directly interact with every other token, enabling parallel training and much better modeling of long-range dependencies."
