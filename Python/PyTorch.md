These are among the **most frequently asked PyTorch interview questions** for **Senior AI Engineer**, **Staff AI Engineer**, **ML Engineer**, and **LLM Engineer** roles. Interviewers expect not just definitions, but an understanding of **how PyTorch works internally**, **performance implications**, and **production best practices**.

---

# 1. Tensor vs NumPy ndarray

## Interview Question

> What is the difference between a PyTorch Tensor and a NumPy ndarray?

## Short Answer

A **Tensor** is PyTorch's primary data structure for deep learning.

An **ndarray** is NumPy's data structure for numerical computing.

The biggest difference is:

* PyTorch supports **automatic differentiation**
* PyTorch supports **GPU execution**
* NumPy does not.

---

## NumPy Example

```python
import numpy as np

a = np.array([[1,2],[3,4]])

print(type(a))
```

Output

```
<class 'numpy.ndarray'>
```

---

## Tensor Example

```python
import torch

x = torch.tensor([[1,2],[3,4]])

print(type(x))
```

Output

```
<class 'torch.Tensor'>
```

---

## GPU Support

NumPy

```
CPU only
```

PyTorch

```python
device = torch.device("cuda")

x = torch.tensor([1,2,3]).to(device)

print(x.device)
```

Output

```
cuda:0
```

---

## Automatic Gradient

Tensor

```python
x = torch.tensor(2.0, requires_grad=True)

y = x ** 2

y.backward()

print(x.grad)
```

Output

```
tensor(4.)
```

NumPy cannot compute gradients automatically.

---

## Shared Memory

Convert Tensor → NumPy

```python
x = torch.tensor([1,2,3])

a = x.numpy()

a[0] = 100

print(x)
```

Output

```
tensor([100,2,3])
```

They share memory (CPU tensors).

---

## Interview Table

| Feature         | Tensor | ndarray |
| --------------- | ------ | ------- |
| GPU             | Yes    | No      |
| Autograd        | Yes    | No      |
| Deep Learning   | Yes    | No      |
| Neural Networks | Yes    | No      |
| CUDA            | Yes    | No      |
| Backpropagation | Yes    | No      |

---

# 2. Autograd

## Interview Question

> Explain Autograd internally.

Autograd is PyTorch's automatic differentiation engine.

It builds a **dynamic computation graph** while executing operations.

Example

```python
import torch

x = torch.tensor(3.0, requires_grad=True)

y = x * x + 2

print(y)
```

Graph

```
x
│
│
square
│
│
+
│
y
```

---

### Backpropagation

```python
y.backward()

print(x.grad)
```

Output

```
6
```

Because

```
y = x² + 2

dy/dx = 2x

=6
```

---

## Computation Graph

```
Tensor
↓

Operation

↓

Operation

↓

Loss

↓

backward()

↓

Gradient
```

---

## Multiple Operations

```python
x = torch.tensor(2.0, requires_grad=True)

y = x*x

z = y+3

w = z*5

w.backward()

print(x.grad)
```

Derivative

```
dw/dx

=5

×

2x

=20
```

---

## Why Dynamic Graph?

TensorFlow 1.x

```
Build graph

Run graph
```

PyTorch

```
Build graph while executing
```

Advantages

* Easier debugging
* Pythonic
* Dynamic models
* Variable-length sequences

---

# 3. Dataset

Interview Question

> Why use Dataset?

Dataset abstracts loading data.

Example

```python
from torch.utils.data import Dataset

class MyDataset(Dataset):

    def __init__(self):

        self.data = [1,2,3,4]

    def __len__(self):

        return len(self.data)

    def __getitem__(self, idx):

        return self.data[idx]

dataset = MyDataset()

print(dataset[0])
```

---

## Real Dataset

```python
class ImageDataset(Dataset):

    def __getitem__(self, idx):

        image = load_image(idx)

        label = load_label(idx)

        return image, label
```

---

# 4. DataLoader

Interview Question

> Why DataLoader?

DataLoader loads batches efficiently.

Example

```python
from torch.utils.data import DataLoader

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True
)

for batch in loader:
    print(batch)
```

---

## Internally

```
Dataset

↓

Sampler

↓

Workers

↓

Batch

↓

Training Loop
```

---

## Parallel Loading

```python
loader = DataLoader(
    dataset,
    batch_size=64,
    num_workers=8
)
```

Benefits

* Parallel loading
* Faster GPU utilization
* Prefetching

---

# 5. torch.no_grad()

Interview Question

> Why use no_grad()?

It disables gradient computation.

Without it,

PyTorch stores every operation.

```python
with torch.no_grad():

    y = model(x)
```

Advantages

* Faster
* Lower memory
* No computation graph

---

## Example

```python
model.eval()

with torch.no_grad():

    output = model(images)
```

---

Without no_grad()

```
Forward

↓

Store graph

↓

More memory
```

With no_grad()

```
Forward only
```

---

# 6. model.train()

Interview Question

> What does model.train() do?

It enables training mode.

Important layers affected

* Dropout
* BatchNorm

Example

```python
model.train()

output = model(x)
```

Dropout

```
Random neurons dropped
```

BatchNorm

```
Update running mean

Update variance
```

---

# 7. model.eval()

Interview Question

> Why use eval()?

Evaluation mode.

```python
model.eval()
```

Effects

Dropout

```
Disabled
```

BatchNorm

```
Uses stored statistics
```

---

Typical inference

```python
model.eval()

with torch.no_grad():

    prediction = model(x)
```

---

## Interview Question

> What happens if you forget eval()?

Dropout remains active.

Predictions become inconsistent.

BatchNorm updates statistics.

Inference accuracy decreases.

---

# 8. Difference between train() and eval()

| Feature   | train()      | eval()       |
| --------- | ------------ | ------------ |
| Dropout   | Active       | Disabled     |
| BatchNorm | Update stats | Frozen stats |
| Training  | Yes          | No           |
| Inference | No           | Yes          |

---

# 9. state_dict()

Interview Question

> What is state_dict()?

It stores

* Model weights
* Biases
* BatchNorm parameters
* Running statistics

Example

```python
model.state_dict()
```

Output

```
layer1.weight

layer1.bias

layer2.weight
...
```

---

## Save Model

```python
torch.save(model.state_dict(), "model.pth")
```

---

## Load

```python
model.load_state_dict(
    torch.load("model.pth")
)
```

---

## Why not save the entire model?

Saving only the `state_dict()` is preferred because it:

* Produces smaller files.
* Avoids tight coupling to the original Python class definition.
* Is more portable across environments and PyTorch versions.

---

# 10. Checkpoint

Interview Question

> What is a checkpoint?

A checkpoint saves the entire training state so training can be resumed after interruption.

---

## Save

```python
checkpoint = {
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "loss": loss,
}

torch.save(checkpoint, "checkpoint.pth")
```

---

## Resume

```python
checkpoint = torch.load("checkpoint.pth")

model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

epoch = checkpoint["epoch"]
loss = checkpoint["loss"]
```

---

## Why save the optimizer state?

Optimizers like **Adam**, **AdamW**, and **RMSprop** maintain internal state (e.g., momentum and moving averages). Restoring only the model weights but not the optimizer state changes the optimization trajectory and can slow or destabilize resumed training.

---

# Production Training Loop

```python
model.train()

for epoch in range(num_epochs):
    for images, labels in train_loader:
        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

# Evaluation
model.eval()

with torch.no_grad():
    for images, labels in val_loader:
        outputs = model(images)
        predictions = outputs.argmax(dim=1)
```

---

# Senior AI Engineer Interview Tips

Be prepared to explain these concepts beyond their APIs:

* **Autograd internals:** Dynamic computation graph, leaf tensors, gradient accumulation, and why `optimizer.zero_grad()` is needed.
* **Tensor memory:** CPU vs GPU tensors, shared memory with NumPy, views vs copies, and contiguous vs non-contiguous tensors.
* **DataLoader performance:** `num_workers`, `pin_memory`, `persistent_workers`, `prefetch_factor`, custom `collate_fn`, and handling variable-length batches.
* **Training vs inference:** The distinct roles of `model.train()`, `model.eval()`, and `torch.no_grad()`.
* **Checkpointing:** Saving and restoring model weights, optimizer state, scheduler state, random seeds, epoch number, and mixed-precision scaler state for reproducible training.
* **Distributed training:** How `state_dict()` behaves with `DistributedDataParallel` and best practices for saving checkpoints in multi-GPU training.

Mastering both the API usage and the underlying mechanics is what typically distinguishes senior-level PyTorch candidates from intermediate ones.

These are **core PyTorch interview questions** that are asked in almost every **Senior AI Engineer, ML Engineer, LLM Engineer, and GenAI Engineer** interview. A senior candidate is expected to explain not only **what** these APIs do, but also **how they work internally**, **when to use them**, and **common production pitfalls**.

---

# 1. Tensor vs NumPy ndarray

## Interview Questions

* What is a Tensor?
* How is it different from a NumPy ndarray?
* When would you choose one over the other?
* Can they share memory?

---

## What is a Tensor?

A **Tensor** is the fundamental data structure in PyTorch. It is a multidimensional array that supports:

* GPU acceleration
* Automatic differentiation (Autograd)
* Efficient tensor operations
* Deep learning workflows

Think of it as:

```
Scalar → 0D Tensor

Vector → 1D Tensor

Matrix → 2D Tensor

Cube → 3D Tensor

Higher dimensions → nD Tensor
```

---

## NumPy ndarray

NumPy provides multidimensional arrays optimized for numerical computing.

Example:

```python
import numpy as np

arr = np.array([[1,2],[3,4]])

print(type(arr))
```

Output

```
<class 'numpy.ndarray'>
```

---

## PyTorch Tensor

```python
import torch

tensor = torch.tensor([[1,2],[3,4]])

print(type(tensor))
```

Output

```
<class 'torch.Tensor'>
```

---

# Main Differences

| Feature               | Tensor | ndarray                |
| --------------------- | ------ | ---------------------- |
| GPU Support           | ✅ Yes  | ❌ No                   |
| Automatic Gradients   | ✅ Yes  | ❌ No                   |
| Neural Networks       | ✅ Yes  | ❌ No                   |
| CUDA                  | ✅ Yes  | ❌ No                   |
| Backpropagation       | ✅ Yes  | ❌ No                   |
| Used in Deep Learning | ✅ Yes  | ❌ Mostly preprocessing |

---

## GPU Example

```python
device = torch.device("cuda")

x = torch.tensor([1,2,3]).to(device)

print(x.device)
```

Output

```
cuda:0
```

NumPy always stays on CPU.

---

## Shared Memory

PyTorch CPU tensors and NumPy arrays can share memory.

```python
import torch

x = torch.tensor([1,2,3])

a = x.numpy()

a[0] = 100

print(x)
```

Output

```
tensor([100,2,3])
```

Changing NumPy also changed Tensor because they share the same memory buffer.

---

## Convert NumPy → Tensor

```python
import numpy as np
import torch

a = np.array([1,2,3])

t = torch.from_numpy(a)

print(t)
```

---

## Tensor → NumPy

```python
a = t.numpy()
```

If the tensor is on GPU:

```python
a = t.cpu().numpy()
```

---

## Interview Tip

Many interviewers ask:

> Why can't we call `.numpy()` on a CUDA tensor?

Because NumPy only understands CPU memory.

You must first move the tensor to CPU:

```python
tensor.cpu().numpy()
```

---

# 2. Autograd

## Interview Questions

* What is Autograd?
* How does backward() work?
* What is a computation graph?
* What is requires_grad?

---

## What is Autograd?

Autograd is PyTorch's automatic differentiation engine.

Instead of manually calculating derivatives, PyTorch records operations performed on tensors and computes gradients automatically.

---

## Example

```python
import torch

x = torch.tensor(2.0, requires_grad=True)

y = x**2 + 3*x

print(y)
```

Internally, PyTorch builds a computation graph:

```
x
│
├── square
│
├── multiply
│
└── add
     │
     y
```

---

## Compute Gradient

```python
y.backward()

print(x.grad)
```

Output

```
7
```

Because

```
y = x² + 3x

dy/dx = 2x + 3

= 7
```

---

## What does requires_grad=True mean?

```python
x = torch.tensor(5.0, requires_grad=True)
```

PyTorch now tracks every operation performed on x.

---

## Dynamic Computation Graph

Unlike TensorFlow 1.x, PyTorch builds the graph during execution.

```
Forward pass

↓

Graph built dynamically

↓

Backward pass

↓

Graph freed
```

Benefits:

* Easier debugging
* Variable-length inputs
* Python control flow
* Dynamic models

---

## Why optimizer.zero_grad()?

Gradients accumulate by default.

Example

```python
loss.backward()

loss.backward()

print(model.weight.grad)
```

Gradient doubles.

Correct way

```python
optimizer.zero_grad()

loss.backward()
```

---

# 3. Dataset

## Interview Questions

* Why Dataset?
* Why not simply load all data into memory?

---

Dataset represents one sample at a time.

```python
from torch.utils.data import Dataset

class MyDataset(Dataset):

    def __init__(self):
        self.data = [1,2,3,4]

    def __len__(self):
        return len(self.data)

    def __getitem__(self,index):
        return self.data[index]
```

---

Real example

```python
class ImageDataset(Dataset):

    def __getitem__(self,index):

        image = read_image(index)

        label = read_label(index)

        return image,label
```

Advantages

* Lazy loading
* Memory efficient
* Custom preprocessing
* Easy augmentation

---

# 4. DataLoader

## Interview Questions

* Why DataLoader?
* What is num_workers?
* What is pin_memory?
* Why shuffle?

---

DataLoader batches data efficiently.

```python
from torch.utils.data import DataLoader

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True
)
```

---

Internally

```
Dataset

↓

Sampler

↓

Worker Processes

↓

Batch

↓

Training Loop
```

---

Multiple Workers

```python
loader = DataLoader(
    dataset,
    batch_size=64,
    num_workers=8
)
```

Workers load data in parallel while the GPU trains on the previous batch.

---

pin_memory

```python
DataLoader(
    dataset,
    pin_memory=True
)
```

Pinned (page-locked) memory speeds up CPU → GPU transfers.

---

# 5. torch.no_grad()

## Interview Questions

* Why use no_grad()?
* What happens internally?

---

Normally PyTorch stores every operation for backpropagation.

```
Forward

↓

Build Graph

↓

Save Activations

↓

Backward
```

During inference we don't need gradients.

```python
with torch.no_grad():

    output = model(images)
```

Benefits

* Less memory
* Faster inference
* No graph construction

---

# 6. model.train()

## Interview Questions

* Why call train()?

---

Training mode affects specific layers.

```python
model.train()
```

Affected layers:

### Dropout

Randomly disables neurons.

```
Input

↓

Dropout

↓

Output
```

### BatchNorm

Updates running mean and variance.

---

# 7. model.eval()

## Interview Questions

* Why eval()?
* What happens if we forget it?

---

```python
model.eval()
```

Effects

Dropout

```
Disabled
```

BatchNorm

```
Uses stored statistics
```

---

Typical inference

```python
model.eval()

with torch.no_grad():

    output = model(images)
```

---

If you forget eval()

* Dropout stays active.
* BatchNorm keeps updating running statistics.
* Predictions become inconsistent and validation metrics may be incorrect.

---

# train() vs eval()

| Feature        | train()                                   | eval()                                             |
| -------------- | ----------------------------------------- | -------------------------------------------------- |
| Dropout        | Active                                    | Disabled                                           |
| BatchNorm      | Update running stats                      | Use stored stats                                   |
| Weight Updates | Possible (with backward + optimizer.step) | No updates unless you explicitly compute gradients |
| Typical Use    | Training                                  | Validation / Inference                             |

---

# 8. state_dict()

## Interview Questions

* What is state_dict()?
* Why save state_dict instead of the whole model?

---

A `state_dict` is a Python dictionary containing all learnable parameters and registered buffers.

```python
model.state_dict()
```

Example keys

```
conv1.weight

conv1.bias

fc.weight

fc.bias
```

---

Save

```python
torch.save(model.state_dict(), "model.pth")
```

Load

```python
model.load_state_dict(torch.load("model.pth"))

model.eval()
```

---

Why not save the whole model?

Saving only `state_dict()` is the recommended approach because it is:

* More portable.
* Less dependent on the exact Python class implementation.
* Easier to maintain across code changes and PyTorch versions.

---

# 9. Checkpoint

## Interview Questions

* What is a checkpoint?
* How do you resume interrupted training?

---

A checkpoint stores the complete training state.

```python
checkpoint = {
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "loss": loss
}

torch.save(checkpoint, "checkpoint.pth")
```

---

Resume Training

```python
checkpoint = torch.load("checkpoint.pth")

model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

start_epoch = checkpoint["epoch"] + 1
loss = checkpoint["loss"]
```

---

## What should a production checkpoint contain?

```python
checkpoint = {
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "scheduler_state_dict": scheduler.state_dict(),
    "scaler_state_dict": scaler.state_dict(),  # for mixed precision
    "best_accuracy": best_accuracy,
    "random_seed": seed
}
```

This allows training to resume as closely as possible to the original state.

---

# End-to-End Training and Evaluation Loop

```python
model.train()

for epoch in range(num_epochs):
    for images, labels in train_loader:
        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

# Validation
model.eval()

with torch.no_grad():
    for images, labels in val_loader:
        outputs = model(images)
        predictions = outputs.argmax(dim=1)
```

---

# Senior AI Engineer Interview Tips

A senior engineer should be able to discuss not only the APIs but also their implementation details and production considerations:

* **Tensor vs ndarray:** GPU memory, shared memory, views vs copies, contiguous tensors, and interoperability with NumPy.
* **Autograd:** Dynamic computation graphs, leaf tensors, gradient accumulation, in-place operation pitfalls, and `detach()`.
* **DataLoader:** Efficient batching, `num_workers`, `pin_memory`, `persistent_workers`, `prefetch_factor`, custom `collate_fn`, and handling variable-length sequences.
* **Training vs inference:** Correct use of `model.train()`, `model.eval()`, `torch.no_grad()`, and when `torch.inference_mode()` can provide additional performance benefits.
* **Checkpointing:** Saving model, optimizer, scheduler, mixed-precision scaler, epoch, and random number generator (RNG) states for reproducibility.
* **Distributed training:** Saving and loading checkpoints with `DistributedDataParallel` (DDP), synchronizing BatchNorm, and handling multi-GPU model states.

These are among the highest-frequency PyTorch topics in senior-level interviews at companies building production AI and LLM systems.

