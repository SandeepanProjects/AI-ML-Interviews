# What is catastrophic forgetting in LLM fine-tuning?

**Catastrophic forgetting** is when a pretrained LLM loses some of its previously learned capabilities or knowledge while being fine-tuned on a new, narrow task or dataset.

In simple terms:

```text
Pretrained model
     │
     │ Knows many things
     │
     ▼
Fine-tune heavily on one narrow task
     │
     ▼
Becomes good at new task
     │
     ├── May retain old knowledge ✓
     │
     └── May lose some old capabilities ✗
```

Example:

A general LLM originally understands:

* English
* Python
* history
* mathematics
* general reasoning
* summarization

You fine-tune it only on **customer-support conversations**.

After aggressive fine-tuning:

```text
Customer support → Excellent
General coding    → Worse
Math             → Worse
General QA       → Worse
```

That degradation is catastrophic forgetting.

---

# 1. Simple analogy

Imagine a person who knows:

```text
Python
Java
Swift
Machine Learning
Databases
```

Then for six months they study only:

```text
Customer support scripts
```

They might become excellent at support responses but become rusty in other areas.

The model has limited capacity to preserve behavior while parameters are repeatedly optimized toward a new training distribution.

---

# 2. What happens mathematically?

A pretrained model has parameters:

[
\theta_{base}
]

It was trained to minimize loss over a broad pretraining distribution:

[
L_{pretrain}(\theta)
]

Then we fine-tune on a new domain:

[
L_{new}(\theta)
]

Standard fine-tuning updates parameters:

[
\theta_{t+1}
============

## \theta_t

\eta
\nabla_\theta L_{new}(\theta_t)
]

Where:

* (\theta) = model parameters
* (\eta) = learning rate
* (\nabla L) = gradient
* (L_{new}) = loss on new fine-tuning data

The problem is:

> The optimizer usually optimizes the new objective. It does not automatically preserve every old capability.

So:

```text
Old capability
      │
      │ parameter update
      ▼
Parameters change
      │
      ├── New task improves
      │
      └── Old task can degrade
```

---

# 3. A simple catastrophic forgetting example

Suppose a model originally performs well on two tasks:

```text
Task A → General Python questions
Task B → Customer support
```

Before fine-tuning:

```text
Python accuracy           = 90%
Customer support accuracy = 65%
```

Now you fine-tune only on customer-support data.

After fine-tuning:

```text
Python accuracy           = 72%
Customer support accuracy = 95%
```

The model gained:

```text
Customer support:
65% → 95%
```

But lost:

```text
Python:
90% → 72%
```

This is evidence of catastrophic forgetting.

---

# 4. Why does fine-tuning cause catastrophic forgetting?

There are several important reasons.

---

## Reason 1: All model weights are updated

In **full fine-tuning**, every parameter is trainable.

```text
Before:

θ₁ θ₂ θ₃ θ₄ θ₅ θ₆ θ₇
│  │  │  │  │  │  │
All contain pretrained knowledge
```

During full fine-tuning:

```text
θ₁ → updated
θ₂ → updated
θ₃ → updated
θ₄ → updated
θ₅ → updated
θ₆ → updated
θ₇ → updated
```

Every update can potentially change previously learned behavior.

---

## Code: full fine-tuning

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "gpt2"
)

for name, parameter in model.named_parameters():
    parameter.requires_grad = True
```

Check:

```python
total = 0
trainable = 0

for parameter in model.parameters():

    total += parameter.numel()

    if parameter.requires_grad:
        trainable += parameter.numel()


print("Total parameters:", total)
print("Trainable parameters:", trainable)
```

For full fine-tuning:

```text
Trainable parameters ≈ Total parameters
```

Therefore:

```text
Large part of pretrained representation
           ↓
Can change during optimization
```

---

# 5. Reason 2: Narrow fine-tuning dataset

Suppose your original model was trained on:

```text
Internet
Books
Code
Documentation
Math
Science
Conversation
```

Now your fine-tuning dataset contains only:

```text
Insurance customer support
```

The new optimization objective becomes strongly biased toward:

```text
Insurance
Claims
Policies
Premiums
Support responses
```

If the dataset is narrow and training is aggressive:

```text
General distribution
        ↓
Narrow domain distribution
        ↓
Model adapts strongly
        ↓
Old behavior may degrade
```

---

# 6. Reason 3: Too many training steps or epochs

Imagine:

```text
Epoch 1
Small adaptation

Epoch 2
More adaptation

Epoch 3
Good domain performance

Epoch 10
Heavy specialization

Epoch 20
Potential degradation of general capabilities
```

Code:

```python
training_args = TrainingArguments(
    output_dir="./model",
    num_train_epochs=20
)
```

This is not always wrong, but with a small or narrow dataset it increases the risk of over-specialization.

Better:

```python
training_args = TrainingArguments(
    output_dir="./model",
    num_train_epochs=5,
    eval_strategy="epoch"
)
```

Then select the best checkpoint based on evaluation.

---

# 7. Reason 4: Learning rate is too high

Gradient descent:

[
\theta_{new}
============

## \theta_{old}

\eta \nabla L
]

If:

```text
η = small
```

Then:

```text
Small changes
```

If:

```text
η = large
```

Then:

```text
Large changes to pretrained weights
```

Example:

```python
# Potentially aggressive for many full fine-tuning setups
learning_rate = 1e-3
```

Compared with:

```python
learning_rate = 2e-5
```

Conceptually:

```text
Low learning rate

Pretrained knowledge
████████████████████

Small adaptation
██


High learning rate

Pretrained knowledge
████████████████████

Large changes
████████████
```

---

# 8. Reason 5: Distribution shift

Pretraining distribution:

[
P_{pretrain}(x)
]

Fine-tuning distribution:

[
P_{fine-tune}(x)
]

If:

[
P_{pretrain} \neq P_{fine-tune}
]

especially if the difference is large:

```text
General language
      ↓
Very narrow specialized domain
```

the model receives repeated gradient signals toward the new distribution.

This can shift its behavior.

Example:

```text
Pretraining:

General conversations
Programming
Books
News
Science


Fine-tuning:

"Hello, thank you for contacting XYZ insurance."
"Please provide your policy number."
"Your claim has been processed."
```

Repeated exposure to only one style can make the model overly specialized.

---

# 9. Demonstrating catastrophic forgetting with code

A real experiment needs:

1. Base evaluation dataset
2. New domain fine-tuning dataset
3. Evaluation before fine-tuning
4. Fine-tune
5. Evaluate again

Architecture:

```text
                    BASE MODEL
                        │
              ┌─────────┴─────────┐
              ▼                   ▼
        Evaluate Task A      Fine-tune Task B
              │                   │
              │                   ▼
              │              NEW MODEL
              │                   │
              └───────────┬───────┘
                          ▼
                 Evaluate Task A again
                          │
                          ▼
                Did performance drop?
```

---

## Step 1: Load model

```python
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

model_name = "gpt2"

tokenizer = AutoTokenizer.from_pretrained(
    model_name
)

tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name
)
```

---

## Step 2: Create a general evaluation dataset

For demonstration:

```python
general_eval = [
    {
        "prompt": "Write a Python function that adds two numbers.",
        "expected_topic": "python"
    },
    {
        "prompt": "What is the capital of France?",
        "expected_topic": "geography"
    },
    {
        "prompt": "Explain photosynthesis.",
        "expected_topic": "science"
    }
]
```

---

## Step 3: Generate answers before fine-tuning

```python
import torch


def generate_answer(
    model,
    tokenizer,
    prompt
):

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(model.device)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False
        )

    return tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )
```

Evaluate:

```python
for example in general_eval:

    answer = generate_answer(
        model,
        tokenizer,
        example["prompt"]
    )

    print(
        "\nPROMPT:",
        example["prompt"]
    )

    print(
        "ANSWER:",
        answer
    )
```

Save these results:

```text
BASE_MODEL_PERFORMANCE
```

---

# 10. Fine-tune on a narrow dataset

Example dataset:

```python
insurance_data = [
    {
        "text": (
            "Customer: How do I file a claim?\n"
            "Agent: You can file a claim through "
            "the claims portal."
        )
    },
    {
        "text": (
            "Customer: What is my deductible?\n"
            "Agent: Your deductible is listed "
            "in your policy details."
        )
    },
    {
        "text": (
            "Customer: How do I update my policy?\n"
            "Agent: You can update your policy "
            "through the customer portal."
        )
    }
]
```

Create a dataset:

```python
from datasets import Dataset

dataset = Dataset.from_list(
    insurance_data
)
```

Tokenize:

```python
def tokenize_function(
    examples
):

    result = tokenizer(

        examples["text"],

        truncation=True,

        max_length=256,

        padding="max_length"
    )

    result["labels"] = result[
        "input_ids"
    ].copy()

    return result
```

```python
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"]
)
```

---

## Step 4: Full fine-tune

```python
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

training_args = TrainingArguments(

    output_dir="./insurance_model",

    num_train_epochs=10,

    learning_rate=5e-5,

    per_device_train_batch_size=2,

    logging_steps=1,

    save_strategy="no",

    report_to="none"
)
```

Data collator:

```python
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)
```

Create trainer:

```python
trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=tokenized_dataset,

    data_collator=data_collator
)
```

Train:

```python
trainer.train()
```

Now:

```text
General pretrained model
        ↓
Repeated full-weight updates
        ↓
Insurance-specialized model
```

---

# 11. Evaluate the old capability again

After fine-tuning:

```python
for example in general_eval:

    answer = generate_answer(
        model,
        tokenizer,
        example["prompt"]
    )

    print(
        "\nPROMPT:",
        example["prompt"]
    )

    print(
        "ANSWER:",
        answer
    )
```

Compare:

```text
                Before          After

Python          Good            ?
Geography       Good            ?
Science         Good            ?
Insurance       Moderate        Better
```

A rigorous experiment would use proper benchmark datasets and metrics, rather than judging a few generated examples manually.

The key measurement is:

[
Forgetting =
Performance_{before}
--------------------

Performance_{after}
]

Example:

```python
before_score = 0.90
after_score = 0.72

forgetting = before_score - after_score

print(forgetting)
```

Output:

```text
0.18
```

Meaning:

```text
18 percentage point drop
```

on the old evaluation task.

---

# 12. How do you prevent catastrophic forgetting?

Several techniques can help.

---

# Method 1: Use PEFT / LoRA

Instead of changing all base-model parameters:

```text
FULL FINE-TUNING

Base weights
████████████████████

All updated
████████████████████
```

With LoRA:

```text
BASE MODEL
████████████████████

Frozen

        +
        ↓

LoRA ADAPTERS
██

Only adapters trained
```

Code:

```python
from peft import (
    LoraConfig,
    get_peft_model
)
```

Configure:

```python
lora_config = LoraConfig(

    r=8,

    lora_alpha=16,

    lora_dropout=0.05,

    target_modules=[
        "c_attn"
    ],

    task_type="CAUSAL_LM"
)
```

Apply:

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

Typical idea:

```text
Total parameters:
124M

Trainable:
Small percentage
```

Why can this help?

```text
Base model knowledge
        ↓
Frozen
        ↓
Preserved more directly

New knowledge
        ↓
Stored in adapter updates
```

Important:

> LoRA reduces the risk of modifying the base weights, but the adapted model's behavior can still change substantially when the adapter is active.

---

# Method 2: Freeze some layers

You don't always need to update every layer.

Example:

```python
for name, parameter in model.named_parameters():

    if "transformer.h.10" in name:

        parameter.requires_grad = True

    elif "transformer.h.11" in name:

        parameter.requires_grad = True

    else:

        parameter.requires_grad = False
```

This is just an illustrative example.

Conceptually:

```text
Embedding layer     Frozen
Transformer 1       Frozen
Transformer 2       Frozen
...
Transformer 10      Trainable
Transformer 11      Trainable
Output layer        Depends on design
```

Fewer parameters change, which can reduce destructive updates.

---

# Method 3: Use a smaller learning rate

Instead of:

```python
learning_rate = 1e-3
```

Use a more conservative learning rate appropriate to the model and method, for example:

```python
learning_rate = 2e-5
```

Concept:

```text
Small learning rate

θ
│
└── small movement


Large learning rate

θ
│
└────────── large movement
```

Small updates generally preserve more of the pretrained solution.

---

# Method 4: Use fewer epochs and early stopping

Example:

```python
from transformers import EarlyStoppingCallback


trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=validation_dataset,

    callbacks=[
        EarlyStoppingCallback(
            early_stopping_patience=2
        )
    ]
)
```

Monitor both:

```text
New task performance
        +
Old capability performance
```

Do not optimize only:

```text
Fine-tuning validation loss
```

because the new-task loss can improve while an important old capability degrades.

---

# Method 5: Replay / mixed training

One powerful approach is to mix new-domain data with representative old/general data.

Instead of:

```text
100% new domain data
```

use something like:

```text
80% new domain
20% general/replay data
```

Concept:

```text
New dataset
      │
      ├── New task examples
      │
      └── General examples
                │
                ▼
           Mixed training
```

Example:

```python
new_domain_data = [
    "How do I file an insurance claim?",
    "What is my policy deductible?"
]

general_data = [
    "Explain Python decorators.",
    "What is photosynthesis?",
    "Solve 2 + 2."
]
```

Mix:

```python
mixed_data = (
    new_domain_data * 4
    +
    general_data
)
```

Then shuffle:

```python
import random

random.shuffle(
    mixed_data
)
```

In production, replay data should be representative and properly governed. Simply using random public text may not preserve the exact capabilities you care about.

---

# Method 6: Regularization toward the original model

You can penalize the model for moving too far from the pretrained parameters.

Conceptually:

[
L_{total}
=========

L_{new}
+
\lambda ||\theta - \theta_{original}||^2
]

Where:

* (L_{new}) = new task loss
* second term = penalty for moving too far from the original model

Implementation idea:

```python
import torch


def regularized_loss(
    task_loss,
    model,
    original_weights,
    lambda_value=0.01
):

    regularization_loss = 0

    for name, parameter in model.named_parameters():

        original = original_weights[name].to(
            parameter.device
        )

        regularization_loss += (
            (parameter - original)
            .pow(2)
            .sum()
        )

    return (
        task_loss
        +
        lambda_value * regularization_loss
    )
```

Before training, save a copy:

```python
original_weights = {

    name: parameter.detach().clone()

    for name, parameter
    in model.named_parameters()
}
```

This basic example can be extremely memory-expensive for large LLMs because it keeps another copy of parameters. Production implementations need careful memory management and often use more sophisticated continual-learning techniques.

---

# Method 7: Knowledge distillation

Use the original model as a teacher.

```text
Original Model
     │
     │ Teacher output
     ▼

Fine-tuned Model
     │
     │ Student
     ▼
Learn new task
+
Preserve original behavior
```

The loss can combine:

[
L =
L_{task}
+
\lambda L_{distillation}
]

Example concept:

```python
import torch.nn.functional as F


def distillation_loss(
    student_logits,
    teacher_logits,
    labels,
    temperature=2.0,
    alpha=0.5
):

    task_loss = F.cross_entropy(
        student_logits.view(
            -1,
            student_logits.size(-1)
        ),
        labels.view(-1)
    )

    student_log_probs = F.log_softmax(
        student_logits / temperature,
        dim=-1
    )

    teacher_probs = F.softmax(
        teacher_logits / temperature,
        dim=-1
    )

    distill_loss = F.kl_div(
        student_log_probs,
        teacher_probs,
        reduction="batchmean"
    ) * (temperature ** 2)

    return (
        alpha * task_loss
        +
        (1 - alpha) * distill_loss
    )
```

The student learns:

```text
New domain behavior
        +
Some behavior of the original model
```

For autoregressive LLMs, the implementation needs masking and careful alignment of teacher/student token positions; the example above is conceptual.

---

# Method 8: Evaluate multiple capabilities continuously

A production evaluation pipeline could look like:

```text
Fine-tuned model
       │
       ├── New domain evaluation
       │
       ├── General knowledge evaluation
       │
       ├── Reasoning evaluation
       │
       ├── Code evaluation
       │
       └── Safety evaluation
```

Example:

```python
evaluation_results = {

    "customer_support": {
        "before": 0.70,
        "after": 0.94
    },

    "general_qa": {
        "before": 0.90,
        "after": 0.86
    },

    "python": {
        "before": 0.88,
        "after": 0.87
    }
}
```

Calculate changes:

```python
for task, scores in evaluation_results.items():

    difference = (
        scores["after"]
        -
        scores["before"]
    )

    print(
        f"{task}: {difference:+.2f}"
    )
```

Output:

```text
customer_support: +0.24
general_qa: -0.04
python: +0.01
```

This tells you whether the new model improved **without unacceptable regressions**.

---

# 13. Production strategy: before-and-after evaluation

A robust workflow is:

```text
                 BASE MODEL
                      │
                      ▼
          Baseline Evaluation Suite
                      │
                      ▼
              Record Metrics
                      │
                      ▼
                 Fine-tuning
                      │
                      ▼
             New Evaluation Suite
                      │
          ┌───────────┴───────────┐
          ▼                       ▼

    New Task Metrics       Old Capability Metrics
          │                       │
          └───────────┬───────────┘
                      ▼
              Regression Analysis
                      │
                      ▼
           Accept / Reject Model
```

Code concept:

```python
baseline_scores = {
    "new_domain": 0.65,
    "general_qa": 0.91,
    "coding": 0.87
}


fine_tuned_scores = {
    "new_domain": 0.94,
    "general_qa": 0.89,
    "coding": 0.85
}
```

Compare:

```python
for metric in baseline_scores:

    change = (
        fine_tuned_scores[metric]
        -
        baseline_scores[metric]
    )

    print(
        metric,
        f"{change:+.3f}"
    )
```

Possible output:

```text
new_domain +0.290
general_qa -0.020
coding +0.020
```

This may be acceptable.

But:

```text
new_domain +0.30
general_qa -0.20
coding -0.25
```

is a serious regression.

---

# Catastrophic forgetting vs overfitting

These are related but different.

| Overfitting                                          | Catastrophic Forgetting                     |
| ---------------------------------------------------- | ------------------------------------------- |
| Model memorizes or over-specializes to training data | Model loses previously learned capabilities |
| Training performance is high                         | New task performance may be high            |
| Validation/generalization often becomes worse        | Old-task performance becomes worse          |
| Main concern: new task generalization                | Main concern: preserving previous knowledge |

Example:

```text
Overfitting:

Train customer support = 99%
New customer questions = 60%
```

Catastrophic forgetting:

```text
New customer support = 95%
Old Python ability = dropped from 90% to 60%
```

You can have:

```text
Overfitting only
Catastrophic forgetting only
Both together
```

---

# Strong interview answer

> **Catastrophic forgetting is the degradation of previously learned capabilities when a pretrained model is fine-tuned on a new task or narrow dataset. It happens because gradient updates optimize the new objective and can overwrite or interfere with parameter configurations that supported earlier capabilities. The risk increases with full fine-tuning, narrow datasets, distribution shift, high learning rates, and excessive training steps.**
>
> **I detect it by establishing baseline evaluations before fine-tuning and then comparing the fine-tuned model on both the new task and representative old capabilities. To reduce it, I use conservative learning rates, appropriate training duration, PEFT methods such as LoRA, selective layer freezing, replay or mixed training, regularization toward the original model, and regression evaluation suites.**

## Final mental model

```text
PRETRAINED MODEL
     │
     │ General knowledge
     ▼
Fine-tuning on narrow task
     │
     ├── Controlled adaptation
     │        │
     │        ▼
     │   New skill + old knowledge
     │
     └── Aggressive adaptation
              │
              ▼
       New skill improves
              +
       Old skills degrade
              │
              ▼
    CATASTROPHIC FORGETTING
```

The key difference from overfitting is:

> **Overfitting asks: "Does the model generalize to new examples?"**
> **Catastrophic forgetting asks: "Did the model lose capabilities it already had?"**
