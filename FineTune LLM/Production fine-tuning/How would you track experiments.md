# How would you track experiments in ML/LLM production?

Experiment tracking means recording **everything needed to understand, compare, reproduce, and select a training run**.

When you run 100 fine-tuning experiments, you should be able to answer:

> Which model performed best, which dataset was used, what hyperparameters were used, what code produced it, and why did we deploy it?

A training experiment is:

```text
Experiment
    │
    ├── Model
    ├── Dataset
    ├── Hyperparameters
    ├── Training Code
    ├── Metrics
    ├── Artifacts
    └── Environment
```

---

# 1. What should you track?

For an LLM fine-tuning experiment, I would track:

## A. Model information

```text
base_model = meta-llama/Llama-3.1-8B
base_model_revision = abc123
```

## B. Dataset information

```text
dataset = customer-support
dataset_version = v5
dataset_fingerprint = a84bcf...
```

## C. Training configuration

```text
learning_rate = 2e-4
epochs = 3
batch_size = 4
gradient_accumulation = 8
max_seq_length = 2048
```

## D. LoRA configuration

```text
lora_r = 16
lora_alpha = 32
lora_dropout = 0.05
```

## E. Metrics

```text
training_loss
validation_loss
perplexity
BLEU
ROUGE
BERTScore
instruction_following_score
hallucination_rate
latency
```

## F. Artifacts

```text
model checkpoints
LoRA adapters
tokenizer
evaluation reports
training config
```

---

# 2. The experiment tracking architecture

```text
                Git Repository
                     │
                     │
                 Code SHA
                     │
                     ▼
Dataset ───────► Training Run ◄────── Base Model
                     │
                     │
          ┌──────────┼──────────┐
          ▼          ▼          ▼
       Params      Metrics    Artifacts
          │          │          │
          └──────────┼──────────┘
                     │
                     ▼
             Experiment Tracker
                     │
                     ▼
              Compare Runs
                     │
                     ▼
              Best Candidate
                     │
                     ▼
              Model Registry
```

---

# 3. MLflow example

A common production approach is to use [MLflow](https://mlflow.org/?utm_source=chatgpt.com) for experiment tracking.

Install:

```bash
pip install mlflow
```

---

# 4. Basic experiment tracking

```python
import mlflow


mlflow.set_experiment(
    "customer-support-finetuning"
)


with mlflow.start_run():

    # Parameters
    mlflow.log_param(
        "learning_rate",
        2e-4
    )

    mlflow.log_param(
        "epochs",
        3
    )

    mlflow.log_param(
        "batch_size",
        4
    )

    # Metrics
    mlflow.log_metric(
        "training_loss",
        1.42
    )

    mlflow.log_metric(
        "validation_loss",
        1.21
    )
```

Each call creates a run:

```text
Experiment:
customer-support-finetuning

    Run 1
       ├── learning_rate = 2e-4
       ├── epochs = 3
       ├── training_loss = 1.42
       └── validation_loss = 1.21
```

---

# 5. Track hyperparameters

Instead of manually logging each parameter:

```python
training_config = {

    "learning_rate": 2e-4,

    "epochs": 3,

    "batch_size": 4,

    "gradient_accumulation_steps": 8,

    "max_seq_length": 2048,

    "warmup_ratio": 0.03,

    "weight_decay": 0.01
}
```

Log all of them:

```python
mlflow.log_params(
    training_config
)
```

For LoRA:

```python
lora_config = {

    "lora_r": 16,

    "lora_alpha": 32,

    "lora_dropout": 0.05,

    "target_modules":
        "q_proj,k_proj,v_proj,o_proj"
}
```

```python
mlflow.log_params(
    lora_config
)
```

---

# 6. Track the dataset version

This is critical.

```python
dataset_info = {

    "dataset_name":
        "customer-support",

    "dataset_version":
        "v5",

    "dataset_fingerprint":
        "a84bcf72"
}
```

Log:

```python
mlflow.log_params(
    dataset_info
)
```

Now you can see:

```text
Run 42
│
├── Dataset: customer-support
├── Version: v5
└── Fingerprint: a84bcf72
```

This connects:

```text
Experiment
    │
    ▼
Dataset Version
```

---

# 7. Track model information

```python
model_info = {

    "base_model":
        "llama-3.1-8b",

    "base_model_revision":
        "abc123"
}
```

```python
mlflow.log_params(
    model_info
)
```

This is important because:

```text
Same Dataset
+
Different Base Model
=
Different Experiment
```

---

# 8. Track training loss during training

With Hugging Face `Trainer`, you can create a callback.

```python
from transformers import TrainerCallback


class MLflowCallback(
    TrainerCallback):

    def on_log(
        self,
        args,
        state,
        control,
        logs=None,
        **kwargs
    ):

        if logs:

            metrics = {}

            for key, value in logs.items():

                if isinstance(
                    value,
                    (int, float)
                ):

                    metrics[key] = value

            mlflow.log_metrics(
                metrics,
                step=state.global_step
            )
```

Use it:

```python
trainer.add_callback(
    MLflowCallback()
)
```

Now:

```text
Step 100
Loss = 2.1

Step 200
Loss = 1.7

Step 300
Loss = 1.3

Step 400
Loss = 1.1
```

MLflow can plot:

```text
Loss
  │
2.0│ *
   │  *
1.5│    *
   │      *
1.0│         *
   └────────────────
      100 200 300 400
             Steps
```

---

# 9. Track validation metrics

After training:

```python
evaluation = trainer.evaluate()

print(evaluation)
```

Example:

```python
{
    "eval_loss": 1.12,
    "eval_runtime": 120.5
}
```

Log:

```python
mlflow.log_metrics({

    "eval_loss":
        evaluation["eval_loss"],

    "eval_runtime":
        evaluation["eval_runtime"]
})
```

You can also calculate perplexity:

```python
import math


eval_loss = evaluation[
    "eval_loss"
]

perplexity = math.exp(
    eval_loss
)

mlflow.log_metric(
    "perplexity",
    perplexity
)
```

---

# 10. Track custom LLM metrics

For example:

```python
evaluation_metrics = {

    "instruction_following":
        0.91,

    "hallucination_rate":
        0.03,

    "bert_score":
        0.89,

    "rouge_l":
        0.72
}
```

Log:

```python
mlflow.log_metrics(
    evaluation_metrics
)
```

Now experiments can be compared:

| Run   | Eval Loss | Instruction | Hallucination |
| ----- | --------: | ----------: | ------------: |
| Run 1 |      1.30 |        0.82 |            8% |
| Run 2 |      1.15 |        0.89 |            5% |
| Run 3 |      1.12 |        0.91 |            3% |

---

# 11. Track artifacts

Metrics are not enough.

You should save:

```text
Artifacts
│
├── training_config.json
├── evaluation_report.json
├── confusion_matrix.png
├── LoRA adapter
└── tokenizer
```

Example:

```python
mlflow.log_artifact(
    "configs/training_config.json"
)
```

Log an entire directory:

```python
mlflow.log_artifacts(
    "artifacts/"
)
```

For example:

```text
artifacts/
├── adapter/
│   ├── adapter_model.safetensors
│   └── adapter_config.json
│
├── tokenizer/
│
└── evaluation.json
```

---

# 12. Track Git commit

Your code also changes.

Run 1:

```text
Commit: abc123
```

Run 2:

```text
Commit: xyz789
```

Even with identical parameters, code differences can change results.

```python
import subprocess


def get_git_commit():

    return (
        subprocess
        .check_output(
            ["git", "rev-parse", "HEAD"]
        )
        .decode()
        .strip()
    )
```

Log it:

```python
git_commit = get_git_commit()

mlflow.log_param(
    "git_commit",
    git_commit
)
```

Now:

```text
Experiment
│
├── Dataset: v5
├── Code: abc123
├── Base Model: llama-3.1-8b
└── LoRA: r=16
```

This makes the run reproducible.

---

# 13. Track environment information

Another engineer may ask:

> Why can't I reproduce this model?

Because they are using:

```text
PyTorch 2.3
CUDA 12.4
```

while the original used:

```text
PyTorch 2.5
CUDA 12.6
```

Track:

```python
import torch
import platform


environment = {

    "python_version":
        platform.python_version(),

    "pytorch_version":
        torch.__version__,

    "cuda_version":
        torch.version.cuda
}
```

Log:

```python
mlflow.log_params(
    environment
)
```

Also save:

```text
requirements.txt
Dockerfile
```

as artifacts.

---

# 14. Complete fine-tuning experiment example

Here is a simplified but realistic structure.

```python
import math
import mlflow
import torch
import platform
import subprocess

from transformers import (
    TrainingArguments,
    Trainer
)


def get_git_commit():

    try:

        return (
            subprocess
            .check_output(
                [
                    "git",
                    "rev-parse",
                    "HEAD"
                ]
            )
            .decode()
            .strip()
        )

    except Exception:

        return "unknown"


def run_training_experiment(
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    dataset_version
):

    mlflow.set_experiment(
        "llm-finetuning"
    )


    with mlflow.start_run() as run:

        print(
            "Run ID:",
            run.info.run_id
        )


        # -------------------------
        # Dataset information
        # -------------------------

        mlflow.log_params({

            "dataset_version":
                dataset_version,

            "train_examples":
                len(train_dataset),

            "eval_examples":
                len(eval_dataset)
        })


        # -------------------------
        # Training configuration
        # -------------------------

        config = {

            "learning_rate":
                2e-4,

            "num_train_epochs":
                3,

            "batch_size":
                4,

            "gradient_accumulation":
                8,

            "max_sequence_length":
                2048,

            "weight_decay":
                0.01
        }

        mlflow.log_params(
            config
        )


        # -------------------------
        # Environment
        # -------------------------

        mlflow.log_params({

            "python_version":
                platform.python_version(),

            "pytorch_version":
                torch.__version__,

            "cuda_version":
                str(
                    torch.version.cuda
                ),

            "git_commit":
                get_git_commit()
        })


        # -------------------------
        # Training
        # -------------------------

        training_args = (
            TrainingArguments(

                output_dir=
                    "./checkpoints",

                learning_rate=
                    config[
                        "learning_rate"
                    ],

                num_train_epochs=
                    config[
                        "num_train_epochs"
                    ],

                per_device_train_batch_size=
                    config[
                        "batch_size"
                    ],

                gradient_accumulation_steps=
                    config[
                        "gradient_accumulation"
                    ],

                eval_strategy=
                    "steps",

                logging_steps=
                    10
            )
        )


        trainer = Trainer(

            model=model,

            args=training_args,

            train_dataset=
                train_dataset,

            eval_dataset=
                eval_dataset
        )


        trainer.train()


        # -------------------------
        # Evaluation
        # -------------------------

        results = (
            trainer.evaluate()
        )


        eval_loss = results[
            "eval_loss"
        ]

        perplexity = math.exp(
            eval_loss
        )


        mlflow.log_metrics({

            "eval_loss":
                eval_loss,

            "perplexity":
                perplexity
        })


        # -------------------------
        # Save model
        # -------------------------

        output_dir = (
            "./artifacts/model"
        )

        trainer.save_model(
            output_dir
        )

        tokenizer.save_pretrained(
            output_dir
        )


        # -------------------------
        # Log artifact
        # -------------------------

        mlflow.log_artifacts(
            output_dir
        )


        return run.info.run_id
```

---

# 15. Compare experiments

Suppose you run:

```text
Experiment: customer-support
```

### Run 1

```text
Learning Rate: 1e-4
LoRA Rank: 8

Eval Loss: 1.30
Instruction Score: 0.85
```

### Run 2

```text
Learning Rate: 2e-4
LoRA Rank: 16

Eval Loss: 1.12
Instruction Score: 0.91
```

### Run 3

```text
Learning Rate: 5e-4
LoRA Rank: 32

Eval Loss: 1.45
Instruction Score: 0.84
```

MLflow lets you compare:

```text
                     Run 1   Run 2   Run 3

Learning Rate         1e-4    2e-4    5e-4
LoRA Rank               8       16      32
Eval Loss              1.30    1.12    1.45
Instruction Score      0.85    0.91    0.84
```

Clearly:

```text
Run 2 → Best candidate
```

---

# 16. Organize experiments hierarchically

A good structure is:

```text
Experiment
│
├── Baseline Runs
│
├── LoRA Experiments
│   ├── Run 1
│   ├── Run 2
│   └── Run 3
│
├── QLoRA Experiments
│
├── Dataset v5 Experiments
│
└── Dataset v6 Experiments
```

Naming runs helps:

```python
mlflow.start_run(
    run_name=
    "llama-v5-lora-r16-lr2e4"
)
```

Then you can immediately understand:

```text
llama-v5-lora-r16-lr2e4
│
├── Dataset v5
├── LoRA r=16
└── Learning rate 2e-4
```

---

# 17. Parent-child runs for hyperparameter experiments

For a hyperparameter search:

```text
Parent Run
│
├── LR=1e-4
├── LR=2e-4
└── LR=5e-4
```

Example:

```python
learning_rates = [
    1e-4,
    2e-4,
    5e-4
]


with mlflow.start_run(
    run_name="lr-experiment"
):

    for lr in learning_rates:

        with mlflow.start_run(
            nested=True
        ):

            mlflow.log_param(
                "learning_rate",
                lr
            )

            # train model
            # evaluate model

            mlflow.log_metric(
                "eval_loss",
                eval_loss
            )
```

This makes hyperparameter experiments easier to organize.

---

# 18. Automatic model selection

You can automatically select the best experiment.

Example:

```python
def calculate_model_score(
    metrics: dict
):

    return (

        0.5
        *
        metrics[
            "instruction_score"
        ]

        -

        0.3
        *
        metrics[
            "hallucination_rate"
        ]

        -

        0.2
        *
        metrics[
            "normalized_latency"
        ]
    )
```

This is useful because the model with the lowest loss is not always the best production model.

For example:

```text
Model A
Loss: Low
Latency: Very High

Model B
Loss: Slightly Higher
Latency: Much Lower
```

Model B may be better for production.

---

# 19. Complete experiment lineage

A good experiment record looks like:

```text
Experiment: LLM Support Fine-tuning
│
├── Run ID: 84729
│
├── Model
│   ├── Base Model
│   └── Revision
│
├── Dataset
│   ├── Version
│   └── Fingerprint
│
├── Code
│   └── Git Commit
│
├── Configuration
│   ├── Learning Rate
│   ├── Batch Size
│   ├── Epochs
│   └── LoRA Config
│
├── Metrics
│   ├── Training Loss
│   ├── Validation Loss
│   ├── Perplexity
│   └── Hallucination Rate
│
└── Artifacts
    ├── Model
    ├── Tokenizer
    └── Evaluation Report
```

This gives complete reproducibility.

---

# 20. Interview-ready answer

> **I track experiments by recording every training run as a reproducible unit. For each run, I capture the base model and revision, dataset version and fingerprint, Git commit, hyperparameters, LoRA or QLoRA configuration, training and evaluation metrics, environment details, and generated artifacts such as checkpoints and evaluation reports.**
>
> **I typically use an experiment tracking system such as MLflow. Each training run logs parameters and metrics over time, allowing me to compare experiments and identify the best candidate. I use nested runs for hyperparameter tuning and automated quality gates before promoting a model to the model registry.**
>
> **The key goal is reproducibility: given a model, I should be able to trace it back to the exact dataset, code version, configuration, environment, and training run that produced it.**

## Simple distinction

```text
Dataset Versioning
        ↓
Which data changed?

Experiment Tracking
        ↓
Which training configuration produced the best result?

Model Versioning
        ↓
Which model artifact was produced and deployed?
```

Together:

```text
Dataset v5
    │
    ▼
Experiment Run 42
    │
    ▼
Model v1.2
    │
    ▼
Production
```

That is how I would implement **experiment tracking for a production ML/LLM system**.
