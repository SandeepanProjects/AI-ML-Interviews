# How do you handle long-context training examples?

Long-context handling is an important part of LLM fine-tuning.

The main problem is simple:

```text
Training example
      ↓
Very long prompt + answer
      ↓
Too many tokens
      ↓
Cannot fit into GPU memory
```

For example:

```text
Context/document     = 20,000 tokens
Question             = 100 tokens
Answer               = 500 tokens
--------------------------------
Total                = 20,600 tokens
```

But suppose your model/training configuration supports:

```text
max_seq_length = 4,096
```

You cannot simply send all 20,600 tokens into a 4,096-token sequence.

So we need a strategy.

---

# 1. First understand what "sequence length" means

For causal LLM fine-tuning, the model sees something like:

```text
<System Prompt>
+
<User Input>
+
<Assistant Response>
```

Example:

```text
<system>
You are an AI assistant.

<user>
Explain LoRA.

<assistant>
LoRA is a parameter-efficient fine-tuning technique...
```

After tokenization:

```text
Token 1
Token 2
Token 3
...
Token N
```

The sequence length is:

[
L = \text{number of tokens}
]

Example:

```text
System prompt      50 tokens
User question     200 tokens
Context         3,000 tokens
Answer            500 tokens
-----------------------------
Total           3,750 tokens
```

If:

```python
MAX_SEQ_LENGTH = 4096
```

then:

```text
3750 <= 4096
```

So it fits.

---

# 2. Why long context is expensive

The attention mechanism compares tokens with other tokens.

The cost is roughly related to:

[
O(n^2)
]

where (n) is sequence length.

Example:

```text
Sequence length = 2,000

Attention relationships:

2,000 × 2,000
= 4 million
```

Now:

```text
Sequence length = 8,000

8,000 × 8,000
= 64 million
```

Increasing sequence length by 4× can dramatically increase attention computation and memory.

This is why:

```text
2048 tokens  → relatively manageable
4096 tokens  → significantly more expensive
8192 tokens  → much more expensive
32768 tokens → expensive
```

Actual memory also depends on architecture, attention implementation, batch size, activations, optimizer state, precision, and whether checkpointing is used.

---

# 3. The first step: measure your dataset

Do not randomly choose:

```python
max_seq_length = 4096
```

First measure the actual token lengths.

## Example with Hugging Face tokenizer

```python
from transformers import AutoTokenizer
```

Load a tokenizer:

```python
model_name = "meta-llama/Llama-3.1-8B"

tokenizer = AutoTokenizer.from_pretrained(
    model_name
)
```

Create a formatting function:

```python
def format_example(example):

    return f"""
### Instruction:
{example["instruction"]}

### Response:
{example["response"]}
"""
```

Now count tokens:

```python
def count_tokens(example):

    text = format_example(example)

    tokens = tokenizer(
        text,
        add_special_tokens=True,
        truncation=False
    )

    return len(tokens["input_ids"])
```

Apply:

```python
lengths = [
    count_tokens(example)
    for example in dataset
]
```

Check:

```python
print(lengths)
```

Example:

```text
[120, 340, 500, 1000, 4200, 8100]
```

Now calculate statistics.

```python
import numpy as np


print("Minimum:", np.min(lengths))
print("Maximum:", np.max(lengths))
print("Mean:", np.mean(lengths))

print(
    "P50:",
    np.percentile(lengths, 50)
)

print(
    "P90:",
    np.percentile(lengths, 90)
)

print(
    "P95:",
    np.percentile(lengths, 95)
)

print(
    "P99:",
    np.percentile(lengths, 99)
)
```

Example:

```text
Minimum: 120
Maximum: 8100
Mean: 780

P50: 400
P90: 1200
P95: 3000
P99: 7500
```

This is much more useful than choosing sequence length blindly.

---

# 4. How do you determine maximum sequence length?

There are three constraints:

```text
Maximum sequence length
          │
          ├── Model capability
          │
          ├── GPU memory
          │
          └── Dataset requirements
```

Let's examine each.

---

# 5. Constraint 1: Model context window

Every model architecture/checkpoint has a supported context window.

Conceptually:

```text
Model A → 4K tokens
Model B → 8K tokens
Model C → 32K tokens
Model D → 128K tokens
```

Suppose your model supports:

```text
Maximum context = 32,768 tokens
```

That does **not automatically mean** you should train with:

```python
max_seq_length = 32768
```

Why?

Because your GPU may not support the required memory.

Also, your dataset may mostly contain examples under 2,000 tokens.

Training everything at 32K would waste resources.

---

# 6. Constraint 2: Dataset distribution

Suppose token analysis gives:

```text
P50 = 500
P90 = 1,500
P95 = 2,500
P99 = 7,000
```

Possible choices:

```text
max_seq_length = 2,048
```

Advantages:

```text
✓ Cheap
✓ Fast
✓ Covers most examples
```

Disadvantage:

```text
Some long examples are truncated
```

Or:

```text
max_seq_length = 4,096
```

Advantages:

```text
✓ Covers more examples
✓ Still manageable
```

Or:

```text
max_seq_length = 8,192
```

Advantages:

```text
✓ Covers almost everything
```

Disadvantages:

```text
✗ More GPU memory
✗ Slower
```

A practical decision often looks like:

```text
P50/P90/P95 analysis
       ↓
Choose length covering most useful examples
       ↓
Verify GPU memory
       ↓
Benchmark throughput
```

---

# 7. Constraint 3: GPU memory

Suppose you have:

```text
GPU: 24 GB
Model: 8B parameters
```

You might try:

```python
max_seq_length = 8192
batch_size = 4
```

and get:

```text
CUDA Out of Memory
```

Then you reduce:

```python
max_seq_length = 4096
batch_size = 2
```

Or use:

```text
QLoRA
+
Gradient checkpointing
+
Flash Attention
+
Gradient accumulation
```

---

# 8. The best way: test memory and throughput

You can benchmark candidate sequence lengths.

```python
candidate_lengths = [
    1024,
    2048,
    4096,
    8192
]
```

For each:

```text
1. Run a few training steps
2. Measure GPU memory
3. Measure training throughput
4. Check stability
```

Conceptually:

```text
Sequence    GPU Memory    Tokens/sec
-------------------------------------
1024        12 GB         5000
2048        16 GB         3500
4096        24 GB         1800
8192        OOM           -
```

Then:

```text
2048 → probably a good choice
```

Because:

```text
Good coverage
+
Fits GPU
+
Good throughput
```

---

# 9. Code: analyze dataset and recommend a sequence length

Let's build a utility.

```python
import numpy as np


def analyze_sequence_lengths(
    dataset,
    tokenizer,
    formatter
):

    lengths = []

    for example in dataset:

        text = formatter(example)

        tokenized = tokenizer(
            text,
            add_special_tokens=True,
            truncation=False
        )

        length = len(
            tokenized["input_ids"]
        )

        lengths.append(length)

    stats = {
        "count": len(lengths),
        "min": int(np.min(lengths)),
        "mean": float(np.mean(lengths)),
        "median": float(np.median(lengths)),
        "p90": int(np.percentile(lengths, 90)),
        "p95": int(np.percentile(lengths, 95)),
        "p99": int(np.percentile(lengths, 99)),
        "max": int(np.max(lengths))
    }

    return lengths, stats
```

Formatter:

```python
def formatter(example):

    return (
        f"### Instruction:\n"
        f"{example['instruction']}\n\n"
        f"### Response:\n"
        f"{example['response']}"
    )
```

Run:

```python
lengths, stats = analyze_sequence_lengths(
    dataset=dataset,
    tokenizer=tokenizer,
    formatter=formatter
)

print(stats)
```

Example:

```text
{
    'count': 100000,
    'min': 20,
    'mean': 780,
    'median': 450,
    'p90': 1500,
    'p95': 2400,
    'p99': 6000,
    'max': 30000
}
```

A reasonable initial choice might be:

```python
MAX_SEQ_LENGTH = 4096
```

Because:

```text
P95 = 2400
4096 covers most examples
```

But you should still inspect what is being truncated.

---

# 10. Calculate truncation rate

This is critical.

```python
def calculate_truncation_rate(
    lengths,
    max_seq_length
):

    truncated = sum(
        length > max_seq_length
        for length in lengths
    )

    total = len(lengths)

    return {
        "total": total,
        "truncated": truncated,
        "truncation_rate": (
            truncated / total
        )
    }
```

Test:

```python
for max_length in [
    1024,
    2048,
    4096,
    8192
]:

    result = calculate_truncation_rate(
        lengths,
        max_length
    )

    print(
        max_length,
        result
    )
```

Example:

```text
1024:
{
    "truncation_rate": 0.25
}

2048:
{
    "truncation_rate": 0.08
}

4096:
{
    "truncation_rate": 0.02
}

8192:
{
    "truncation_rate": 0.005
}
```

Now you can make a data-driven decision.

---

# 11. How do you handle examples that exceed the maximum length?

There are several strategies.

```text
Long Example
     │
     ├── Truncate
     │
     ├── Split / Chunk
     │
     ├── Summarize
     │
     ├── Select important context
     │
     ├── Use retrieval
     │
     └── Train with a longer context
```

Let's look at each.

---

# 12. Strategy 1: Truncation

The simplest method:

```python
tokenized = tokenizer(
    text,
    max_length=4096,
    truncation=True
)
```

But truncation direction matters.

Suppose:

```text
User context: 10,000 tokens
Question: 50 tokens
Answer: 500 tokens
```

If you blindly truncate:

```text
Beginning ---------------------------------> End
[keep 4096 tokens] [remove rest]
```

You might remove:

```text
Question ❌
Answer ❌
```

For instruction tuning, this is dangerous.

---

# 13. Preserve the response

For SFT, the assistant response is usually the supervised target.

Suppose:

```text
Long context
+
Question
+
Answer
```

You often want:

```text
Trim context
+
Keep question
+
Keep complete answer
```

Instead of:

```text
Keep full context
+
Lose answer ❌
```

---

# 14. Code: truncate context but preserve answer

Let's create a practical function.

```python
def tokenize_with_response_priority(
    instruction,
    context,
    response,
    tokenizer,
    max_length
):

    prompt = (
        f"### Instruction:\n"
        f"{instruction}\n\n"
        f"### Context:\n"
        f"{context}\n\n"
        f"### Response:\n"
    )

    prompt_tokens = tokenizer(
        prompt,
        add_special_tokens=False
    )["input_ids"]

    response_tokens = tokenizer(
        response,
        add_special_tokens=False
    )["input_ids"]

    # Reserve space for response
    available_for_prompt = (
        max_length
        - len(response_tokens)
        - 1
    )

    if available_for_prompt <= 0:
        raise ValueError(
            "Response alone exceeds max sequence length"
        )

    # Keep only allowed prompt tokens
    prompt_tokens = prompt_tokens[
        -available_for_prompt:
    ]

    input_ids = (
        prompt_tokens
        + response_tokens
    )

    return input_ids
```

However, this keeps the **end of the prompt**. Whether that's correct depends on your prompt structure. If important instructions are at the beginning, you may need a smarter policy that preserves both the system/instruction prefix and the most relevant context.

A better approach:

```text
System instruction     → always keep
User question          → always keep
Most relevant context  → keep
Assistant answer       → always keep
```

---

# 15. Strategy 2: Chunk long documents

Suppose you have:

```text
Document = 20,000 tokens
```

Split it:

```text
Document

Chunk 1 → 2,000 tokens
Chunk 2 → 2,000 tokens
Chunk 3 → 2,000 tokens
...
```

Code:

```python
def chunk_tokens(
    token_ids,
    chunk_size,
    overlap
):

    chunks = []

    start = 0

    while start < len(token_ids):

        end = start + chunk_size

        chunks.append(
            token_ids[start:end]
        )

        start += (
            chunk_size - overlap
        )

    return chunks
```

Usage:

```python
tokens = tokenizer(
    long_document,
    add_special_tokens=False
)["input_ids"]


chunks = chunk_tokens(
    token_ids=tokens,
    chunk_size=2048,
    overlap=200
)
```

But there is an important warning:

> **Do not blindly split one instruction-response example into independent chunks.**

For example:

```text
Question
+
Context part 1
+
Context part 2
+
Answer
```

You cannot simply create:

```text
Example 1:
Context part 1 → same answer

Example 2:
Context part 2 → same answer
```

That can create poor training examples.

---

# 16. Better: create meaningful training units

Suppose you have a long enterprise document.

```text
Company Policy Document
        │
        ├── Leave Policy
        ├── Security Policy
        ├── Travel Policy
        └── Expense Policy
```

Instead of training:

```text
Entire document → one example
```

Create:

```text
Question:
How many annual leaves are allowed?

Relevant policy section
        ↓
Answer
```

Another:

```text
Question:
What expenses require approval?

Relevant policy section
        ↓
Answer
```

This produces shorter, higher-quality examples.

---

# 17. Strategy 3: Retrieval-based context selection

For very large context:

```text
20,000-token document
       ↓
User question
       ↓
Retrieve relevant sections
       ↓
Top 3 chunks
       ↓
3,000 tokens
       ↓
LLM
```

This is often better than:

```text
Entire 20,000-token document
       ↓
LLM
```

For training, you can create examples using relevant context.

Example:

```python
def create_training_example(
    question,
    retrieved_chunks,
    answer
):

    context = "\n\n".join(
        retrieved_chunks
    )

    return {
        "instruction": question,
        "context": context,
        "response": answer
    }
```

---

# 18. Strategy 4: Sliding window

Useful when every part of a long sequence matters.

Example:

```text
Tokens:

1 ----------------------------------- 10000

Window 1:
1 ---------------------- 4096

Window 2:
3000 ------------------- 7096

Window 3:
6000 ------------------- 10000
```

Code:

```python
def sliding_window(
    token_ids,
    window_size,
    stride
):

    windows = []

    for start in range(
        0,
        len(token_ids),
        stride
    ):

        end = (
            start
            + window_size
        )

        window = token_ids[
            start:end
        ]

        if len(window) < 2:
            break

        windows.append(window)

        if end >= len(token_ids):
            break

    return windows
```

Example:

```python
windows = sliding_window(
    token_ids=tokens,
    window_size=4096,
    stride=3072
)
```

This gives overlap:

```text
Overlap = 4096 - 3072
        = 1024 tokens
```

Use sliding windows carefully for instruction tuning because splitting a coherent response across windows can change the learning objective.

---

# 19. Strategy 5: Pack multiple short examples

The opposite problem is also common.

Suppose:

```text
Example 1 = 300 tokens
Example 2 = 400 tokens
Example 3 = 200 tokens
```

If:

```python
max_seq_length = 4096
```

Training each example independently wastes space.

Packing:

```text
Example 1 ─┐
Example 2 ─┼──► One 4096-token sequence
Example 3 ─┘
```

Code concept:

```python
def pack_examples(
    tokenized_examples,
    max_length
):

    packed = []
    current = []

    for example in tokenized_examples:

        if (
            len(current)
            + len(example)
            <= max_length
        ):

            current.extend(
                example
            )

        else:

            packed.append(
                current
            )

            current = list(
                example
            )

    if current:
        packed.append(
            current
        )

    return packed
```

This improves:

```text
GPU utilization
Training throughput
```

Production trainers often support packing, but you must ensure separator/EOS tokens and label masking are handled correctly so examples do not accidentally train across boundaries.

---

# 20. Dynamic padding

Avoid padding every example to the global maximum.

Bad:

```text
Example 1 = 200 tokens
Example 2 = 300 tokens
Example 3 = 4,000 tokens

All padded to 4,096
```

Wasted computation.

Better:

```text
Batch 1:
200, 220, 250
       ↓
Pad to 250

Batch 2:
1500, 1700, 1800
       ↓
Pad to 1800
```

With Hugging Face:

```python
from transformers import DataCollatorForLanguageModeling


data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)
```

Or a task-specific collator that dynamically pads `input_ids`, `attention_mask`, and labels.

This is why **length bucketing** is also useful.

---

# 21. Length bucketing

Group examples of similar lengths.

```text
Short:
100 - 500 tokens

Medium:
500 - 1500 tokens

Long:
1500 - 4096 tokens
```

Then batch:

```text
Batch 1:
300
350
400
380

Batch 2:
1200
1300
1250
1400
```

Instead of:

```text
Batch:
300
350
400
4000
```

The second case wastes padding.

Conceptually:

```python
dataset = sorted(
    dataset,
    key=lambda x: x["token_length"]
)
```

Many modern training frameworks handle grouping by length automatically.

---

# 22. Maximum sequence length for QLoRA

Suppose:

```text
Model = 8B
GPU = 24 GB
```

You may use:

```python
MAX_SEQ_LENGTH = 2048
```

with:

```text
4-bit base model
+
LoRA adapters
+
bf16 compute
+
gradient checkpointing
+
gradient accumulation
```

Example configuration:

```python
MAX_SEQ_LENGTH = 2048

per_device_train_batch_size = 1

gradient_accumulation_steps = 16
```

Effective batch size is approximately:

[
1 \times 16 = 16
]

for one device, ignoring distributed training details.

Example:

```python
training_args = {
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 16,
    "gradient_checkpointing": True
}
```

---

# 23. Full Hugging Face + QLoRA-style setup

A simplified example:

```python
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)

from peft import (
    LoraConfig,
    get_peft_model
)
```

Configuration:

```python
MODEL_NAME = "your-model"

MAX_SEQ_LENGTH = 4096
```

Tokenizer:

```python
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
```

4-bit configuration:

```python
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="bfloat16",
    bnb_4bit_use_double_quant=True
)
```

Model:

```python
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=quantization_config,
    device_map="auto"
)
```

Enable memory saving:

```python
model.gradient_checkpointing_enable()
```

LoRA configuration:

```python
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj"
    ],
    task_type="CAUSAL_LM"
)
```

Create PEFT model:

```python
model = get_peft_model(
    model,
    lora_config
)
```

Check trainable parameters:

```python
model.print_trainable_parameters()
```

This gives:

```text
Large frozen quantized base model
          +
Small trainable LoRA adapters
```

---

# 24. A production approach for selecting sequence length

I would build a report like this.

```python
import numpy as np


def sequence_length_report(
    lengths,
    candidates=(1024, 2048, 4096, 8192)
):

    print("Dataset statistics")

    print(
        "Examples:",
        len(lengths)
    )

    print(
        "P50:",
        np.percentile(lengths, 50)
    )

    print(
        "P90:",
        np.percentile(lengths, 90)
    )

    print(
        "P95:",
        np.percentile(lengths, 95)
    )

    print(
        "P99:",
        np.percentile(lengths, 99)
    )

    print(
        "\nCandidate lengths:"
    )

    for candidate in candidates:

        truncated = sum(
            length > candidate
            for length in lengths
        )

        rate = (
            truncated
            / len(lengths)
        )

        print(
            f"{candidate}: "
            f"{rate:.2%} truncated"
        )
```

Run:

```python
sequence_length_report(
    lengths
)
```

Example output:

```text
Dataset statistics

Examples: 100000
P50: 450
P90: 1400
P95: 2500
P99: 7000

Candidate lengths:

1024: 22.00% truncated
2048: 7.00% truncated
4096: 1.80% truncated
8192: 0.30% truncated
```

Then benchmark GPU:

```text
Length    Truncation    GPU Memory    Tokens/sec
------------------------------------------------
1024      22%           10 GB         5000
2048       7%           14 GB         3500
4096     1.8%           21 GB         2100
8192     0.3%           OOM           -
```

A good engineering decision might be:

```python
MAX_SEQ_LENGTH = 4096
```

Because it balances:

```text
Coverage
+
Memory
+
Training speed
```

---

# 25. What if important examples are longer than your max length?

Do **not** automatically truncate them.

First inspect them.

```python
long_examples = [
    example
    for example, length in zip(
        dataset,
        lengths
    )
    if length > 4096
]
```

Then ask:

```text
Are these examples valuable?
        │
        ├── No → Remove/filter
        │
        ├── Yes, but extra context → Truncate context
        │
        ├── Long document → Retrieve relevant chunks
        │
        ├── Long structured content → Chunk intelligently
        │
        └── Genuine long-context task → Train longer
```

This is much better than blindly truncating everything.

---

# 26. Important concept: model context window vs training sequence length

These are not always the same.

Example:

```text
Model supports:
128K context
```

But you train with:

```text
4096 tokens
```

This is possible.

However, if your goal is to improve the model's ability to reliably process 100K-token inputs, training only on 4K examples may not adequately exercise the long-context behavior you care about.

```text
Model capability = 128K

Your fine-tuning examples = 4K
```

The model can technically accept 128K, but your fine-tuning dataset does not teach the desired task behavior across the full long-context range.

---

# 27. Long-context fine-tuning example

Suppose your task is legal document QA.

A poor example:

```text
Entire 100-page contract
        +
Question
        ↓
Answer
```

Better:

```text
Question:
What is the termination notice period?

        ↓

Retrieve relevant contract section

        ↓

Termination clause

        ↓

Answer:
30 days
```

Why?

```text
Shorter sequence
+
Less noise
+
Lower GPU cost
+
Better signal
```

But if the task genuinely requires:

```text
Compare clause 3 on page 2
with clause 47 on page 90
```

then retrieval/chunking may not be enough. You need training and evaluation examples designed for long-range reasoning, potentially with a genuinely long-context configuration.

---

# 28. Interview answer

### How do you handle long-context training examples?

> I first tokenize the fully formatted examples and analyze the token-length distribution using percentiles such as P50, P90, P95, and P99. I then identify which examples exceed the training sequence length. I don't blindly truncate them. For instruction tuning, I preserve critical instructions and the target response, and truncate less important context. For long documents, I prefer creating meaningful task-specific examples, selecting relevant context, or using retrieval. I may use sliding windows when the full sequence genuinely needs local coverage. I also use dynamic padding, length bucketing, packing, gradient checkpointing, and gradient accumulation to improve efficiency.

### How do you determine maximum sequence length?

> I choose it based on three factors: the model's supported context window, the token-length distribution of the actual training data, and available hardware. I calculate truncation rates for candidate lengths such as 1024, 2048, 4096, and 8192 tokens, then benchmark memory usage and throughput. I select the smallest length that preserves the important training signal while keeping GPU memory and training cost reasonable.

# Final rule

```text
Do not choose max_seq_length randomly.

Dataset token distribution
          +
Model context capability
          +
GPU memory
          +
Production use case
          ↓
Choose sequence length
```

And for long examples:

```text
Never ask only:

"How do I fit this into the context window?"

Also ask:

"Which parts of this example are actually useful for teaching the model?"
```
