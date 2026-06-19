# He Initialization (Kaiming Initialization) — Internal Working (Senior AI Engineer Level)

If Xavier Initialization solved one of the biggest problems in deep learning, **He Initialization solved the next one.**

Today, almost every modern CNN and many MLPs that use **ReLU** rely on **He Initialization**.

Interviewers often ask:

* **Why was Xavier not enough?**
* **Why do we need He Initialization?**
* **Why is the variance (2/\text{fan_in})?**
* **What happens internally?**

Most candidates answer:

> "He Initialization is used for ReLU."

That is correct but **doesn't explain the mathematics**.

Let's understand it from first principles.

---

# 1. Recap: What problem were we trying to solve?

Consider a deep neural network.

```text
Input
   │
Layer 1
   │
Layer 2
   │
Layer 3
   │
Layer 4
   │
Output
```

During forward propagation, every layer computes

[
z = Wx + b
]

Then applies an activation.

For ReLU,

[
a = \max(0,z)
]

The goal is:

> **Keep the activations and gradients at roughly the same scale across all layers.**

Otherwise,

* activations explode
* activations vanish
* gradients explode
* gradients vanish

---

# 2. Why wasn't Xavier enough?

Recall Xavier Initialization:

[
Var(W)=\frac{2}{fan_{in}+fan_{out}}
]

This assumes the activation function is approximately **symmetric around zero**.

Examples:

* tanh ✅
* sigmoid (near initialization) ✅

---

But ReLU is different.

ReLU is

[
f(x)=\max(0,x)
]

Graphically

```text
      /
     /
    /
___/
```

Everything below zero becomes

```text
0
```

---

# 3. What does ReLU do internally?

Suppose pre-activation values are

```text
[-5,-2,-1,0,3,8]
```

After ReLU

```text
[0,0,0,0,3,8]
```

Half the neurons disappear.

Approximately

```text
50%
```

of activations become zero (assuming inputs are centered around zero).

This changes the variance.

---

# 4. Why does variance shrink?

Suppose activations before ReLU are

```text
-4
-2
0
2
4
```

Mean

```text
0
```

Variance

```text
High
```

After ReLU

```text
0
0
0
2
4
```

Half the information disappears.

Variance becomes much smaller.

Now imagine repeating this.

```text
Layer1

Variance = 1
```

↓

After ReLU

```text
0.5
```

↓

Next layer

```text
0.25
```

↓

Next

```text
0.125
```

↓

Next

```text
0.062
```

Eventually,

everything becomes tiny.

Even though Xavier preserved variance before the activation, **ReLU halves it**.

---

# 5. He Initialization's idea

The authors asked:

> **If ReLU removes about half of the activations, shouldn't we start with larger weights?**

Answer:

Yes.

Instead of

[
Var(W)=\frac{2}{fan_{in}+fan_{out}}
]

He proposed

[
\boxed{
Var(W)=\frac{2}{fan_{in}}
}
]

Notice

there is **no fan_out**.

Only

```text
fan_in
```

is used.

This gives slightly larger initial weights.

Those larger weights compensate for ReLU zeroing roughly half the activations.

---

# 6. Why exactly (2/fan_{in})?

Let's derive the intuition.

Suppose

Input variance

```text
1
```

ReLU keeps roughly

```text
50%
```

of activations.

Therefore

Output variance becomes approximately

```text
0.5
```

To restore it back to

```text
1
```

we multiply by

```text
2
```

Hence

[
Var(W)=\frac{2}{fan_{in}}
]

The factor of **2** compensates for the variance lost due to ReLU.

---

# 7. He Uniform Initialization

Weights are sampled from

[
U(-a,a)
]

where

[
a=
\sqrt{\frac{6}{fan_{in}}}
]

---

Suppose

```text
fan_in = 100
```

Python

```python
import numpy as np

fan_in = 100

limit = np.sqrt(6 / fan_in)

print(limit)
```

Output

```text
0.2449
```

Weights are sampled from

```text
[-0.2449, +0.2449]
```

Notice this range is **wider** than Xavier for the same `fan_in`, because He is compensating for ReLU.

---

# 8. He Normal Initialization

Instead of a uniform distribution,

sample from

[
N
\left(
0,
\frac{2}{fan_{in}}
\right)
]

Python

```python
import numpy as np

fan_in = 100

std = np.sqrt(2 / fan_in)

W = np.random.normal(
    0,
    std,
    size=(100,50)
)

print(W.mean())
print(W.var())
```

Typical output

```text
Mean

≈0

Variance

≈0.02
```

Exactly as expected.

---

# 9. Forward Propagation Example

Without He

Layer 1 variance

```text
1
```

↓

ReLU

```text
0.5
```

↓

Layer 2

```text
0.25
```

↓

Layer 3

```text
0.12
```

↓

Layer 4

```text
0.06
```

Signal keeps shrinking.

---

With He Initialization

Layer 1

```text
1
```

↓

ReLU

```text
0.5
```

↓

Larger weights amplify appropriately

↓

Back to approximately

```text
1
```

↓

Next ReLU

```text
0.5
```

↓

Back to approximately

```text
1
```

Variance remains stable throughout the network.

---

# 10. Backpropagation

He Initialization also helps gradients.

Recall

[
\frac{\partial L}{\partial W}
=============================

\frac{\partial L}{\partial a_n}
\times
...
\times
\frac{\partial a_1}{\partial W}
]

If activation variance stays stable,

gradient variance also remains much more stable.

This reduces both

* vanishing gradients
* exploding gradients

especially in deep ReLU networks.

---

# 11. From-Scratch Implementation

```python
import numpy as np

fan_in = 128
fan_out = 64

std = np.sqrt(2 / fan_in)

weights = np.random.normal(
    0,
    std,
    (fan_in, fan_out)
)

print(weights.mean())
print(weights.var())
```

---

# 12. PyTorch Implementation

```python
import torch
import torch.nn as nn

layer = nn.Linear(128,64)

nn.init.kaiming_normal_(layer.weight)
```

or

```python
nn.init.kaiming_uniform_(layer.weight)
```

PyTorch computes

```text
fan_in
```

internally,

then calculates

[
\sqrt{\frac{2}{fan_{in}}}
]

automatically.

---

# 13. TensorFlow

```python
from tensorflow.keras import layers

layer = layers.Dense(
    256,
    kernel_initializer="he_normal"
)
```

or

```python
kernel_initializer="he_uniform"
```

---

# 14. Why is it called Kaiming Initialization?

The method was proposed by **Kaiming He** and colleagues in the paper:

> *Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification* (2015)

Therefore,

He Initialization is also called

* **Kaiming Initialization**
* **MSRA Initialization**

---

# 15. Xavier vs He

| Property              | Xavier                         | He                             |
| --------------------- | ------------------------------ | ------------------------------ |
| Designed for          | Sigmoid, tanh                  | ReLU, Leaky ReLU, ELU          |
| Variance              | (\frac{2}{fan_{in}+fan_{out}}) | (\frac{2}{fan_{in}})           |
| Compensates for ReLU? | No                             | Yes                            |
| Weight magnitude      | Smaller                        | Slightly larger                |
| Modern CNNs           | Rare                           | Standard                       |
| Deep ReLU Networks    | Can lose variance              | Preserves variance much better |

---

# 16. Real Production Example

A ResNet block typically looks like

```python
class Block(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv1 = nn.Conv2d(64,64,3,padding=1)

        nn.init.kaiming_normal_(self.conv1.weight)

        self.bn1 = nn.BatchNorm2d(64)

        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(64,64,3,padding=1)

        nn.init.kaiming_normal_(self.conv2.weight)
```

Every convolution layer uses

```text
He Initialization
```

because every convolution is followed by

```text
ReLU
```

---

# 17. Why doesn't He solve everything?

Initialization only ensures that **training starts in a stable regime**.

As training progresses:

* weights change,
* activations shift,
* gradients evolve.

That's why modern networks also rely on:

* Batch Normalization
* Residual Connections
* Adam/AdamW optimizers
* Learning-rate schedules
* Gradient clipping (when appropriate)

Initialization is the **starting point**, not the entire optimization strategy.

---

# 18. Interview Answer (Senior AI Engineer)

> **He (Kaiming) Initialization is a weight initialization strategy specifically designed for ReLU-based neural networks. Since ReLU sets approximately half of its inputs to zero, it reduces the variance of activations. He Initialization compensates for this by initializing weights with variance (2/\text{fan_in}), preserving the scale of activations and gradients across layers. This leads to more stable optimization, faster convergence, and significantly reduces the risk of vanishing gradients in deep ReLU networks.**

---

# 19. Staff AI Engineer Perspective

He Initialization is fundamentally a **variance-preserving initialization tailored to rectified activation functions**.

For a ReLU network,

[
a^{(l)} = \text{ReLU}(W^{(l)}a^{(l-1)})
]

the ReLU nonlinearity removes roughly half of the signal energy by zeroing negative activations. If weights were initialized using Xavier's assumptions, activation variance would progressively decrease with depth. He Initialization analytically compensates for this by choosing

[
Var(W)=\frac{2}{fan_{in}},
]

so that the expected activation variance remains approximately constant after the ReLU transformation.

The broader principle is that **weight initialization should match the statistical behavior of the activation function**. Xavier matches symmetric activations like tanh, while He matches rectified activations like ReLU. This alignment keeps both forward activations and backward gradients well-conditioned, enabling the successful training of very deep neural networks.
