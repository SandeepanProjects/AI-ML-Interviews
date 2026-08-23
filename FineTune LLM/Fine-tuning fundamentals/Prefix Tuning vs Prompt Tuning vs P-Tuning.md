# Prefix Tuning vs Prompt Tuning vs P-Tuning

These are all **Parameter-Efficient Fine-Tuning (PEFT)** techniques.

The main idea is:

> Instead of updating billions of LLM parameters, keep the LLM frozen and learn a small number of trainable vectors.

The difference is **where those trainable vectors are inserted** and **how they interact with the model**.

---

# 1. The big picture

```text
Full Fine-Tuning
│
├── Update entire LLM
│
PEFT
│
├── LoRA
│     └── Learn low-rank weight updates
│
├── Adapter Tuning
│     └── Add small neural modules
│
├── Prefix Tuning
│     └── Learn virtual prefixes for Transformer layers
│
├── Prompt Tuning
│     └── Learn virtual tokens at the input
│
└── P-Tuning
      └── Learn continuous prompts, often generated/reparameterized
```

---

# 2. First understand: what is a "soft prompt"?

Normally, you give an LLM text:

```text
Translate English to French:
Hello, how are you?
```

The tokenizer converts this into token IDs:

```text
[Translate] [English] [to] [French] [:] [Hello] [,] ...
```

Then tokens become embeddings:

```text
Token IDs
    ↓
Embedding Layer
    ↓
Token Embeddings
```

A soft prompt adds **trainable vectors**.

```text
Trainable Prompt Vectors
        │
        ▼
[v1] [v2] [v3] [v4]
        +
Actual Input Tokens
        │
        ▼
LLM
```

These vectors are sometimes called:

* soft prompts
* virtual tokens
* virtual embeddings
* continuous prompts

They usually **do not correspond to actual vocabulary words**.

---

# 3. Prompt Tuning

## Definition

**Prompt tuning freezes the entire pre-trained model and learns a small set of trainable prompt embeddings that are added before the input embeddings.**

```text
Frozen LLM ❄️

Input:

"What is RAG?"

Prompt tuning adds:

[V1] [V2] [V3] [V4]
          +
"What is RAG?"
          │
          ▼
         LLM
```

Only:

```text
V1
V2
V3
V4
```

are trained.

---

# 4. Prompt tuning architecture

```text
                    Trainable
                Soft Prompt Vectors
                      ✓
                       │
                       ▼

[V1] [V2] [V3] [V4] [Actual Input Tokens]
                       │
                       ▼
                 Embedding Sequence
                       │
                       ▼
                  Frozen LLM ❄️
                       │
                       ▼
                     Output
```

---

# 5. Prompt tuning mathematics

Suppose input embeddings are:

[
X = [x_1, x_2, ..., x_n]
]

Learn `m` virtual prompt embeddings:

[
P = [p_1, p_2, ..., p_m]
]

The model receives:

[
X' = [P; X]
]

Meaning:

```text
[p1, p2, p3, ..., x1, x2, x3]
```

The LLM is frozen:

[
\theta_{LLM} = frozen
]

Only:

[
P
]

is updated.

---

# 6. Prompt tuning from scratch with PyTorch

Let's implement the core concept.

```python
import torch
import torch.nn as nn
```

## Step 1: Create a soft prompt

```python
class SoftPrompt(nn.Module):

    def __init__(
        self,
        num_virtual_tokens: int,
        hidden_size: int
    ):
        super().__init__()

        self.prompt_embeddings = nn.Parameter(
            torch.randn(
                num_virtual_tokens,
                hidden_size
            )
        )

    def forward(
        self,
        batch_size: int
    ):

        # Same prompt for every item in batch
        return self.prompt_embeddings.unsqueeze(0).expand(
            batch_size,
            -1,
            -1
        )
```

Create it:

```python
soft_prompt = SoftPrompt(
    num_virtual_tokens=20,
    hidden_size=768
)
```

The parameter shape is:

```text
20 × 768
```

Only:

```text
15,360 parameters
```

are trained.

Compare that with a model containing:

```text
7,000,000,000 parameters
```

---

# 7. Add the prompt to input embeddings

```python
batch_size = 2
sequence_length = 10
hidden_size = 768

input_embeddings = torch.randn(
    batch_size,
    sequence_length,
    hidden_size
)

prompt_embeddings = soft_prompt(
    batch_size
)
```

Now:

```python
print(input_embeddings.shape)
```

Output:

```text
[2, 10, 768]
```

Prompt:

```python
print(prompt_embeddings.shape)
```

Output:

```text
[2, 20, 768]
```

Concatenate:

```python
combined_embeddings = torch.cat(
    [
        prompt_embeddings,
        input_embeddings
    ],
    dim=1
)

print(combined_embeddings.shape)
```

Output:

```text
[2, 30, 768]
```

The model sees:

```text
[V1][V2]...[V20][INPUT TOKEN 1][INPUT TOKEN 2]...
```

---

# 8. Prompt tuning with Hugging Face PEFT

Conceptually, you can use PEFT like this:

```bash
pip install transformers peft datasets accelerate
```

Load the model:

```python
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

MODEL_NAME = "gpt2"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME
)

tokenizer.pad_token = tokenizer.eos_token
```

Create prompt tuning configuration:

```python
from peft import (
    PromptTuningConfig,
    PromptTuningInit,
    TaskType,
    get_peft_model
)
```

```python
prompt_config = PromptTuningConfig(
    task_type=TaskType.CAUSAL_LM,

    num_virtual_tokens=20,

    prompt_tuning_init=PromptTuningInit.RANDOM,

    tokenizer_name_or_path=MODEL_NAME
)
```

Apply:

```python
prompt_model = get_peft_model(
    model,
    prompt_config
)
```

Check trainable parameters:

```python
prompt_model.print_trainable_parameters()
```

Conceptually:

```text
Base Model Parameters:
Frozen ❄️

Prompt Embeddings:
Trainable ✓
```

---

# 9. Training prompt tuning

The training loop is almost identical to normal fine-tuning.

```python
from transformers import (
    TrainingArguments,
    Trainer
)

training_args = TrainingArguments(
    output_dir="./prompt_tuned_model",

    num_train_epochs=3,

    per_device_train_batch_size=2,

    learning_rate=1e-3,

    logging_steps=10,

    report_to="none"
)
```

```python
trainer = Trainer(
    model=prompt_model,
    args=training_args,
    train_dataset=tokenized_dataset
)

trainer.train()
```

During training:

```text
Loss
 ↓
Backpropagation
 ↓

Base LLM             ❄️ Frozen
Prompt Embeddings     ✓ Updated
```

---

# 10. What is Prefix Tuning?

Now we move to **prefix tuning**.

## Definition

**Prefix tuning freezes the LLM and learns a set of continuous prefix vectors that influence the Transformer at multiple layers.**

Unlike simple prompt tuning, prefix tuning is not limited to adding embeddings only at the model input.

Conceptually:

```text
PROMPT TUNING

Trainable Prompt
       │
       ▼
Input Layer
       │
       ▼
Layer 1
       │
       ▼
Layer 2
       │
       ▼
Layer 3
```

Prefix tuning:

```text
PREFIX TUNING

        Trainable Prefix
              │
              ├────► Layer 1
              │
              ├────► Layer 2
              │
              └────► Layer 3
```

---

# 11. Why prefix tuning works

Transformers use:

```text
Query (Q)
Key (K)
Value (V)
```

Attention is:

[
Attention(Q, K, V)
==================

softmax
\left(
\frac{QK^T}{\sqrt{d}}
\right)V
]

Prefix tuning learns additional prefix representations that can act like additional context in the attention mechanism.

Conceptually:

```text
Original:

K = [K1, K2, K3]
V = [V1, V2, V3]
```

Prefix tuning adds:

```text
K = [PrefixK1, PrefixK2, K1, K2, K3]

V = [PrefixV1, PrefixV2, V1, V2, V3]
```

Then attention becomes:

[
Attention(Q, [K_p;K], [V_p;V])
]

The prefix can influence how every layer attends to information.

---

# 12. Prefix tuning architecture

```text
                 TRAINABLE PREFIX

                   Prefix K
                      │
                      ▼

Input ───► Frozen Transformer Layer
                 ▲
                 │
              Prefix V


                 │
                 ▼

          Frozen Layer 2 ❄️
                 ▲
                 │
              Prefix K/V
```

The base model remains:

```text
Frozen ❄️
```

The prefixes are:

```text
Trainable ✓
```

---

# 13. Prefix tuning mathematics

For a Transformer layer:

[
H_l
]

Normally:

[
K_l = H_lW_K
]

[
V_l = H_lW_V
]

Prefix tuning introduces:

[
P_K^l
]

and:

[
P_V^l
]

Then:

[
K'_l = [P_K^l;K_l]
]

[
V'_l = [P_V^l;V_l]
]

Attention uses:

[
Attention(Q_l,K'_l,V'_l)
]

The model learns:

```text
Prefix K/V ✓
```

while:

```text
WQ ❄️
WK ❄️
WV ❄️
```

remain frozen.

---

# 14. Simplified prefix tuning code

A simplified educational implementation:

```python
import torch
import torch.nn as nn
```

```python
class PrefixTuning(nn.Module):

    def __init__(
        self,
        prefix_length,
        hidden_size
    ):
        super().__init__()

        self.prefix_embeddings = nn.Parameter(
            torch.randn(
                prefix_length,
                hidden_size
            )
        )

    def forward(self, batch_size):

        return self.prefix_embeddings.unsqueeze(0).expand(
            batch_size,
            -1,
            -1
        )
```

This looks similar to prompt tuning at first.

The key difference is how the prefix is injected into Transformer attention.

---

# 15. Simplified attention with prefix

```python
import math
```

```python
class PrefixAttention(nn.Module):

    def __init__(
        self,
        hidden_size,
        prefix_length
    ):
        super().__init__()

        self.hidden_size = hidden_size

        self.q = nn.Linear(
            hidden_size,
            hidden_size
        )

        self.k = nn.Linear(
            hidden_size,
            hidden_size
        )

        self.v = nn.Linear(
            hidden_size,
            hidden_size
        )

        self.prefix = nn.Parameter(
            torch.randn(
                prefix_length,
                hidden_size
            )
        )
```

Forward pass:

```python
    def forward(self, x):

        batch_size = x.size(0)

        # Q, K, V from input
        Q = self.q(x)

        K = self.k(x)

        V = self.v(x)


        # Create prefix
        prefix = self.prefix.unsqueeze(0).expand(
            batch_size,
            -1,
            -1
        )


        # Prefix is transformed into K and V
        prefix_K = self.k(prefix)

        prefix_V = self.v(prefix)


        # Add prefix to K and V
        K = torch.cat(
            [prefix_K, K],
            dim=1
        )

        V = torch.cat(
            [prefix_V, V],
            dim=1
        )


        # Attention
        scores = (
            Q @ K.transpose(-2, -1)
        ) / math.sqrt(self.hidden_size)

        weights = torch.softmax(
            scores,
            dim=-1
        )

        output = weights @ V

        return output
```

Conceptually:

```text
Q comes from actual input

K = Prefix + Input Keys

V = Prefix + Input Values
```

---

# 16. Prefix tuning with PEFT

```python
from peft import (
    PrefixTuningConfig,
    TaskType,
    get_peft_model
)
```

Configuration:

```python
prefix_config = PrefixTuningConfig(

    task_type=TaskType.CAUSAL_LM,

    num_virtual_tokens=20
)
```

Apply:

```python
prefix_model = get_peft_model(
    model,
    prefix_config
)
```

Check:

```python
prefix_model.print_trainable_parameters()
```

Conceptually:

```text
Base LLM        Frozen ❄️

Prefix Params   Trainable ✓
```

Training:

```python
trainer = Trainer(
    model=prefix_model,
    args=training_args,
    train_dataset=tokenized_dataset
)

trainer.train()
```

---

# 17. Prompt tuning vs Prefix tuning

| Feature                  | Prompt Tuning           | Prefix Tuning                              |
| ------------------------ | ----------------------- | ------------------------------------------ |
| Base model               | Frozen                  | Frozen                                     |
| Trainable component      | Input prompt embeddings | Prefix representations                     |
| Main injection           | Input embedding layer   | Transformer attention/layers               |
| Model parameters updated | No                      | No                                         |
| Complexity               | Lower                   | Higher                                     |
| Layer-level influence    | Indirect                | More direct                                |
| Trainable parameters     | Very few                | More than prompt tuning, often still small |

The easiest memory trick:

```text
Prompt Tuning
= Learn soft tokens BEFORE the input

Prefix Tuning
= Learn prefixes that influence Transformer layers/attention
```

---

# 18. What is P-Tuning?

This terminology is often confusing.

There are multiple related ideas, especially **P-Tuning v1** and **P-Tuning v2**.

The broad concept:

> **P-Tuning learns continuous, trainable prompt representations instead of manually writing fixed text prompts.**

Instead of:

```text
"You are an expert financial advisor."
```

you learn something like:

```text
[V1][V2][V3][V4][V5]
```

where each virtual token is a learned vector.

---

# 19. P-Tuning v1

P-Tuning v1 introduces **trainable continuous prompts**, often with a reparameterization network.

Conceptually:

```text
Learnable Prompt Embeddings
          │
          ▼
    Prompt Encoder
   (e.g. MLP / LSTM)
          │
          ▼
Continuous Prompt Vectors
          │
          ▼
       Frozen LLM
```

Instead of directly optimizing:

```text
P
```

you may optimize:

```text
P
  ↓
Prompt Encoder
  ↓
Generated Prompt Representations
```

This can make optimization more stable.

---

# 20. P-Tuning v1 code

## Prompt encoder

```python
import torch
import torch.nn as nn
```

```python
class PromptEncoder(nn.Module):

    def __init__(
        self,
        num_virtual_tokens,
        hidden_size
    ):
        super().__init__()

        # Learnable embeddings
        self.prompt_embeddings = nn.Parameter(
            torch.randn(
                num_virtual_tokens,
                hidden_size
            )
        )

        # Reparameterization network
        self.encoder = nn.Sequential(

            nn.Linear(
                hidden_size,
                hidden_size
            ),

            nn.Tanh(),

            nn.Linear(
                hidden_size,
                hidden_size
            )
        )


    def forward(self, batch_size):

        prompt = self.prompt_embeddings

        # Transform prompt representations
        prompt = self.encoder(
            prompt
        )

        # Add batch dimension
        prompt = prompt.unsqueeze(0)

        # Repeat for each batch item
        prompt = prompt.expand(
            batch_size,
            -1,
            -1
        )

        return prompt
```

Usage:

```python
prompt_encoder = PromptEncoder(
    num_virtual_tokens=20,
    hidden_size=768
)

prompt_vectors = prompt_encoder(
    batch_size=4
)

print(
    prompt_vectors.shape
)
```

Output:

```text
torch.Size([4, 20, 768])
```

Then combine with actual input embeddings:

```python
combined = torch.cat(
    [
        prompt_vectors,
        input_embeddings
    ],
    dim=1
)
```

---

# 21. P-Tuning v2

P-Tuning v2 is closer in spirit to **deep prompt tuning/prefix-style tuning**.

Instead of adding trainable prompts only at the input:

```text
Input
  +
Prompt
  ↓
Layer 1
  ↓
Layer 2
```

It can inject trainable prompt representations at multiple layers:

```text
Layer 1
  ▲
  │
Prompt ✓

Layer 2
  ▲
  │
Prompt ✓

Layer 3
  ▲
  │
Prompt ✓
```

This gives the learned prompts more influence throughout the model.

---

# 22. Simplified P-Tuning v2 concept

```python
class DeepPrompt(nn.Module):

    def __init__(
        self,
        num_layers,
        num_virtual_tokens,
        hidden_size
    ):
        super().__init__()

        self.prompts = nn.ParameterList(
            [
                nn.Parameter(
                    torch.randn(
                        num_virtual_tokens,
                        hidden_size
                    )
                )

                for _ in range(num_layers)
            ]
        )


    def get_prompt(
        self,
        layer_index,
        batch_size
    ):

        prompt = self.prompts[
            layer_index
        ]

        return prompt.unsqueeze(0).expand(
            batch_size,
            -1,
            -1
        )
```

Usage:

```python
deep_prompt = DeepPrompt(

    num_layers=12,

    num_virtual_tokens=20,

    hidden_size=768
)
```

Get a prompt for Layer 5:

```python
layer_prompt = deep_prompt.get_prompt(
    layer_index=5,
    batch_size=4
)
```

Conceptually:

```text
Layer 1 → Prompt 1
Layer 2 → Prompt 2
Layer 3 → Prompt 3
...
Layer N → Prompt N
```

---

# 23. Complete comparison

## Prompt Tuning

```text
Trainable:

[V1][V2][V3]

          +
Input:

[Hello][World]

          ↓

Frozen LLM
```

---

## Prefix Tuning

```text
                  Prefix K/V
                       │
                       ▼

Input ───► Transformer Layer 1
                       │
                       ▼

                  Prefix K/V
                       │
                       ▼

          Transformer Layer 2
```

---

## P-Tuning

```text
Virtual Tokens
       │
       ▼

Prompt Encoder
MLP / LSTM
       │
       ▼

Continuous Prompts
       │
       ▼

Frozen LLM
```

P-Tuning v2:

```text
Prompt Layer 1 → Transformer Layer 1

Prompt Layer 2 → Transformer Layer 2

Prompt Layer 3 → Transformer Layer 3
```

---

# 24. A single training example

Suppose the task is sentiment classification.

Input:

```text
The movie was amazing.
```

Target:

```text
positive
```

With prompt tuning:

```text
[Virtual Prompt 1]
[Virtual Prompt 2]
[Virtual Prompt 3]
+
The movie was amazing.
```

The LLM predicts:

```text
positive
```

Loss:

```python
loss = cross_entropy(
    prediction,
    target
)
```

Backward:

```python
loss.backward()
```

Updates:

```text
Virtual Prompt 1 ✓
Virtual Prompt 2 ✓
Virtual Prompt 3 ✓

LLM Weights ❄️
```

---

# 25. Which should you use?

## Prompt Tuning

Use when:

```text
✓ Very limited trainable parameters
✓ Large base model
✓ Task adaptation
✓ Many tasks with separate prompts
```

---

## Prefix Tuning

Use when:

```text
✓ Need stronger control over generation
✓ Want adaptation through attention
✓ Generative tasks
✓ Want layer-level conditioning
```

---

## P-Tuning

Use when:

```text
✓ Prompt optimization is important
✓ Manual prompts are insufficient
✓ You want learned continuous prompts
✓ You want deep/layer-wise prompts (P-Tuning v2)
```

---

# 26. Production comparison

| Technique     | What is trained?                  | Base LLM | Where added?             |
| ------------- | --------------------------------- | -------- | ------------------------ |
| Full FT       | All weights                       | Updated  | Entire model             |
| LoRA          | Low-rank matrices                 | Frozen   | Selected linear layers   |
| Adapter       | Small neural modules              | Frozen   | Between/around layers    |
| Prompt Tuning | Prompt embeddings                 | Frozen   | Input                    |
| Prefix Tuning | Prefix representations            | Frozen   | Attention/layers         |
| P-Tuning      | Continuous prompt representations | Frozen   | Input or multiple layers |

---

# 27. Important practical difference

Suppose you have:

```text
Base Model
```

And three tasks:

```text
Finance
Legal
Support
```

You could store:

```text
Base Model

finance_prompt.bin
legal_prompt.bin
support_prompt.bin
```

Then dynamically select:

```python
def select_prompt(task):

    prompts = {

        "finance":
            "finance_prompt",

        "legal":
            "legal_prompt",

        "support":
            "support_prompt"
    }

    return prompts.get(
        task,
        "general_prompt"
    )
```

This is very storage efficient because:

```text
One Large Base Model
        +
Many Tiny Task-Specific Prompts
```

---

# 28. Important interview answer

> **Prompt tuning freezes the entire language model and learns a small number of continuous prompt embeddings that are prepended to the input embeddings.**

> **Prefix tuning also freezes the base model but learns prefix representations that are injected into the Transformer's attention computation, typically as additional key-value context, allowing the learned parameters to influence generation across layers.**

> **P-Tuning is a family of methods that learns continuous prompt representations instead of manually written prompts. P-Tuning v1 commonly uses a prompt encoder to transform learnable embeddings, while P-Tuning v2 extends prompt-based adaptation deeper across Transformer layers and can work well for more challenging tasks.**

# Final memory trick

```text
Prompt Tuning
= Train virtual tokens at INPUT

Prefix Tuning
= Train virtual prefix context for ATTENTION

P-Tuning
= Learn continuous prompts, often using a prompt encoder

P-Tuning v2
= Deep prompts across MULTIPLE layers
```

For interviews, also remember:

```text
Manual Prompt
     ↓
Prompt Engineering

Learned Soft Prompt
     ↓
Prompt Tuning / P-Tuning

Learned Attention Prefix
     ↓
Prefix Tuning

Learned Weight Updates
     ↓
LoRA
```
