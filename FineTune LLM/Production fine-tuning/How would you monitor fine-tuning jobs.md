# How would you monitor fine-tuning jobs?

In production, monitoring a fine-tuning job means monitoring **more than training loss**.

You want to know:

1. **Is the job running?**
2. **Is training making progress?**
3. **Is the model learning?**
4. **Is it overfitting?**
5. **Is the GPU healthy and utilized?**
6. **Is memory about to run out?**
7. **Did the job fail or hang?**
8. **How much is training costing?**
9. **Can the job resume from a checkpoint?**

A production architecture looks like this:

```text
                    Fine-Tuning Job
                          │
         ┌────────────────┼─────────────────┐
         ▼                ▼                 ▼
   Training Metrics    System Metrics     Logs
         │                │                 │
         ├── loss          ├── GPU usage     ├── errors
         ├── eval loss     ├── GPU memory    ├── warnings
         ├── learning rate ├── CPU           └── checkpoints
         ├── grad norm     └── disk
         └── throughput
                │
                ▼
        Experiment Tracker
             (MLflow)
                │
                ▼
        Metrics Monitoring
       Prometheus / Grafana
                │
                ▼
             Alerts
```

---

# 1. What metrics should you monitor?

## A. Training metrics

```text
train_loss
eval_loss
learning_rate
gradient_norm
throughput
tokens_per_second
steps_per_second
```

Example:

```text
Step     Train Loss     Eval Loss

100         2.10          -
500         1.60          1.70
1000        1.20          1.30
1500        0.90          1.10
```

This looks healthy because both losses decrease.

---

## B. Detect overfitting

A dangerous pattern:

```text
Step     Train Loss     Eval Loss

100        2.0            2.1
500        1.2            1.3
1000       0.7            1.0
1500       0.3            1.8  ❌
```

Training loss decreases, but validation loss increases.

This can indicate:

```text
Overfitting
```

Monitor:

```text
train_loss ↓
eval_loss  ↑
```

---

# 2. Build a custom training monitoring callback

Hugging Face provides callbacks through `TrainerCallback`.

```python
from transformers import TrainerCallback
import logging


logger = logging.getLogger(__name__)


class TrainingMonitoringCallback(
    TrainerCallback
):

    def on_log(
        self,
        args,
        state,
        control,
        logs=None,
        **kwargs
    ):

        if not logs:
            return

        logger.info(
            {
                "event": "training_metrics",
                "step": state.global_step,
                "epoch": state.epoch,
                "metrics": logs
            }
        )

        return control
```

Now add it to the trainer:

```python
trainer.add_callback(
    TrainingMonitoringCallback()
)
```

Every time Hugging Face logs metrics:

```text
Step 100
```

you might see:

```json
{
  "event": "training_metrics",
  "step": 100,
  "epoch": 0.25,
  "metrics": {
    "loss": 1.82,
    "learning_rate": 0.00018,
    "grad_norm": 1.25
  }
}
```

---

# 3. Monitor training loss and learning rate

A more focused callback:

```python
from transformers import TrainerCallback


class MetricsCallback(
    TrainerCallback):

    def on_log(
        self,
        args,
        state,
        control,
        logs=None,
        **kwargs
    ):

        logs = logs or {}

        step = state.global_step

        train_loss = logs.get(
            "loss"
        )

        learning_rate = logs.get(
            "learning_rate"
        )

        grad_norm = logs.get(
            "grad_norm"
        )

        print(
            f"""
            Step: {step}
            Loss: {train_loss}
            Learning Rate: {learning_rate}
            Gradient Norm: {grad_norm}
            """
        )

        return control
```

Example output:

```text
Step: 100
Loss: 1.84
Learning Rate: 0.0002
Gradient Norm: 1.3

Step: 200
Loss: 1.51
Learning Rate: 0.00018
Gradient Norm: 1.1
```

---

# 4. Monitor evaluation loss

`Trainer` automatically performs evaluation if configured.

```python
from transformers import TrainingArguments


training_args = TrainingArguments(

    output_dir="./checkpoints",

    num_train_epochs=3,

    per_device_train_batch_size=4,

    per_device_eval_batch_size=4,

    eval_strategy="steps",

    eval_steps=500,

    logging_steps=50,

    save_strategy="steps",

    save_steps=500,

    load_best_model_at_end=True,

    metric_for_best_model="eval_loss",

    greater_is_better=False
)
```

Now training behaves like:

```text
Step 500
    │
    ├── Train
    │
    ├── Evaluate
    │
    ├── Save checkpoint
    │
    └── Continue
```

Metrics:

```text
Step 500
Train Loss: 1.40
Eval Loss: 1.30

Step 1000
Train Loss: 0.90
Eval Loss: 1.10

Step 1500
Train Loss: 0.50
Eval Loss: 1.50 ⚠️
```

---

# 5. Automatically detect overfitting

Create a callback.

```python
from transformers import TrainerCallback


class OverfittingCallback(
    TrainerCallback
):

    def __init__(
        self,
        patience=2
    ):

        self.best_eval_loss = float(
            "inf"
        )

        self.bad_epochs = 0

        self.patience = patience


    def on_evaluate(
        self,
        args,
        state,
        control,
        metrics=None,
        **kwargs
    ):

        metrics = metrics or {}

        eval_loss = metrics.get(
            "eval_loss"
        )

        if eval_loss is None:
            return control


        if eval_loss < self.best_eval_loss:

            self.best_eval_loss = eval_loss

            self.bad_epochs = 0

        else:

            self.bad_epochs += 1


        print(
            f"""
            Eval Loss: {eval_loss}
            Best Eval Loss:
            {self.best_eval_loss}
            """
        )


        if (
            self.bad_epochs
            >=
            self.patience
        ):

            print(
                "Overfitting detected. "
                "Stopping training."
            )

            control.should_training_stop = True


        return control
```

Add it:

```python
trainer.add_callback(
    OverfittingCallback(
        patience=3
    )
)
```

This is essentially early stopping logic.

You can also use Hugging Face's built-in:

```python
from transformers import (
    EarlyStoppingCallback
)


trainer.add_callback(
    EarlyStoppingCallback(
        early_stopping_patience=3
    )
)
```

---

# 6. Monitor gradient norms

Gradient norm is important.

Healthy:

```text
Step 100 → grad_norm = 1.2
Step 200 → grad_norm = 1.4
Step 300 → grad_norm = 1.1
```

Potential gradient explosion:

```text
Step 100 → grad_norm = 2
Step 200 → grad_norm = 50
Step 300 → grad_norm = 10,000 ❌
```

Monitor it:

```python
class GradientMonitoringCallback(
    TrainerCallback
):

    def on_log(
        self,
        args,
        state,
        control,
        logs=None,
        **kwargs
    ):

        logs = logs or {}

        grad_norm = logs.get(
            "grad_norm"
        )

        if grad_norm is not None:

            if grad_norm > 100:

                print(
                    f"WARNING: "
                    f"High gradient norm: "
                    f"{grad_norm}"
                )

        return control
```

Use gradient clipping:

```python
training_args = TrainingArguments(

    output_dir="./output",

    max_grad_norm=1.0
)
```

Conceptually:

```text
Original Gradient:

100.0

        ↓

Gradient Clipping

        ↓

1.0
```

---

# 7. Monitor GPU memory

You should monitor:

```text
GPU Memory Used
GPU Utilization
GPU Temperature
GPU Power
```

With PyTorch:

```python
import torch


def get_gpu_metrics():

    if not torch.cuda.is_available():

        return {}

    device = torch.cuda.current_device()

    allocated = (
        torch.cuda.memory_allocated(
            device
        )
        /
        1024**3
    )

    reserved = (
        torch.cuda.memory_reserved(
            device
        )
        /
        1024**3
    )

    return {

        "gpu_memory_allocated_gb":
            round(
                allocated,
                2
            ),

        "gpu_memory_reserved_gb":
            round(
                reserved,
                2
            )
    }
```

Usage:

```python
print(
    get_gpu_metrics()
)
```

Output:

```text
{
    "gpu_memory_allocated_gb": 18.4,
    "gpu_memory_reserved_gb": 20.1
}
```

---

# 8. Monitor GPU utilization

You can use `nvidia-smi`.

Example:

```bash
nvidia-smi
```

Output:

```text
GPU 0

Utilization: 95%
Memory: 22GB / 24GB
Temperature: 72C
```

Programmatically:

```python
import subprocess


def get_gpu_utilization():

    command = [

        "nvidia-smi",

        "--query-gpu="
        "utilization.gpu,"
        "memory.used,"
        "memory.total,"
        "temperature.gpu",

        "--format=csv,noheader,nounits"
    ]

    result = subprocess.check_output(
        command
    )

    return result.decode()
```

Example:

```text
95, 22000, 24576, 72
```

---

# 9. Detect GPU problems

Create thresholds.

```python
def check_gpu_health(
    metrics
):

    if (
        metrics[
            "gpu_utilization"
        ]
        <
        20
    ):

        print(
            "WARNING: Low GPU utilization"
        )


    if (
        metrics[
            "gpu_memory_percent"
        ]
        >
        95
    ):

        print(
            "WARNING: GPU OOM risk"
        )


    if (
        metrics[
            "temperature"
        ]
        >
        85
    ):

        print(
            "WARNING: GPU temperature high"
        )
```

---

# 10. Monitor throughput

GPU utilization alone isn't enough.

Monitor:

```text
examples/sec
tokens/sec
steps/sec
```

Example:

```python
import time


class ThroughputMonitor:

    def __init__(self):

        self.start_time = (
            time.time()
        )

        self.start_step = 0


    def calculate(
        self,
        current_step
    ):

        elapsed = (
            time.time()
            -
            self.start_time
        )

        steps = (
            current_step
            -
            self.start_step
        )

        if elapsed == 0:

            return 0

        return steps / elapsed
```

Example:

```text
Training:

100 steps / 10 seconds

Throughput:
10 steps/sec
```

If suddenly:

```text
10 steps/sec
      ↓
0.5 steps/sec
```

something is wrong:

```text
DataLoader bottleneck
CPU bottleneck
Network storage problem
GPU issue
```

---

# 11. Monitor tokens per second

For LLM training, this is particularly useful.

```python
def calculate_tokens_per_second(

    batch_size,
    sequence_length,
    elapsed_time
):

    total_tokens = (
        batch_size
        *
        sequence_length
    )

    return (
        total_tokens
        /
        elapsed_time
    )
```

Example:

```python
tokens_per_second = (
    calculate_tokens_per_second(

        batch_size=32,

        sequence_length=2048,

        elapsed_time=2
    )
)

print(
    tokens_per_second
)
```

Output:

```text
32768 tokens/sec
```

For distributed training, include:

```text
global_batch_size
```

not just one GPU's batch.

---

# 12. Monitor checkpoints

Fine-tuning jobs can fail.

A production system should continuously save checkpoints.

```python
training_args = TrainingArguments(

    output_dir="./checkpoints",

    save_strategy="steps",

    save_steps=500,

    save_total_limit=3
)
```

You might have:

```text
checkpoints/

checkpoint-500
checkpoint-1000
checkpoint-1500
```

If the job crashes at:

```text
step = 1700
```

resume:

```python
trainer.train(
    resume_from_checkpoint=
        "./checkpoints/checkpoint-1500"
)
```

Monitor checkpoint freshness.

```python
from pathlib import Path
import time


def checkpoint_age_seconds(
    checkpoint_path
):

    modified = (
        Path(
            checkpoint_path
        )
        .stat()
        .st_mtime
    )

    return (
        time.time()
        -
        modified
    )
```

If the checkpoint is too old:

```text
No checkpoint for 2 hours
```

possible problems:

```text
Training stuck
Job crashed
Checkpoint writing failed
```

---

# 13. Detect NaN loss

One of the most important alerts:

```text
Loss = 1.2
Loss = 0.9
Loss = NaN ❌
```

Create:

```python
import math


def is_invalid_loss(
    loss
):

    return (

        math.isnan(loss)

        or

        math.isinf(loss)
    )
```

Callback:

```python
class NaNDetectionCallback(
    TrainerCallback
):

    def on_log(
        self,
        args,
        state,
        control,
        logs=None,
        **kwargs
    ):

        logs = logs or {}

        loss = logs.get(
            "loss"
        )

        if loss is not None:

            if is_invalid_loss(
                loss
            ):

                print(
                    "CRITICAL: "
                    "Invalid training loss"
                )

                control.should_training_stop = True

        return control
```

---

# 14. Send metrics to Prometheus

For infrastructure monitoring, Prometheus is useful.

Install:

```bash
pip install prometheus-client
```

Define metrics:

```python
from prometheus_client import Gauge


TRAIN_LOSS = Gauge(
    "llm_training_loss",
    "Current training loss"
)

EVAL_LOSS = Gauge(
    "llm_eval_loss",
    "Current evaluation loss"
)

GPU_MEMORY = Gauge(
    "gpu_memory_gb",
    "GPU memory usage"
)

TRAINING_STEP = Gauge(
    "llm_training_step",
    "Current training step"
)
```

Update them:

```python
TRAIN_LOSS.set(
    1.24
)

EVAL_LOSS.set(
    1.10
)

GPU_MEMORY.set(
    18.5
)

TRAINING_STEP.set(
    1000
)
```

A callback can export these automatically:

```python
class PrometheusCallback(
    TrainerCallback
):

    def on_log(
        self,
        args,
        state,
        control,
        logs=None,
        **kwargs
    ):

        logs = logs or {}

        TRAINING_STEP.set(
            state.global_step
        )

        if "loss" in logs:

            TRAIN_LOSS.set(
                logs["loss"]
            )

        if "eval_loss" in logs:

            EVAL_LOSS.set(
                logs["eval_loss"]
            )

        return control
```

---

# 15. Expose Prometheus metrics

For a standalone training process:

```python
from prometheus_client import (
    start_http_server
)


start_http_server(
    8000
)
```

Prometheus can scrape:

```text
http://training-job:8000/metrics
```

Metrics:

```text
llm_training_loss 1.24

llm_eval_loss 1.10

gpu_memory_gb 18.5

llm_training_step 1000
```

---

# 16. Grafana dashboard

A production dashboard could contain:

```text
┌──────────────────────────────────┐
│ Fine-Tuning Job: RUNNING         │
├──────────────────────────────────┤
│                                  │
│ Training Loss        1.24        │
│ Evaluation Loss      1.10        │
│ Learning Rate        0.00018     │
│                                  │
├──────────────────────────────────┤
│ GPU Utilization      95%         │
│ GPU Memory           20 / 24 GB  │
│ GPU Temperature      72°C         │
│                                  │
├──────────────────────────────────┤
│ Throughput           28k tokens/s│
│ Current Step         1000        │
│                                  │
└──────────────────────────────────┘
```

---

# 17. Configure alerts

You need automatic alerts.

## High GPU memory

```text
gpu_memory_percent > 95%
```

## Training NaN

```text
training_loss == NaN
```

## Job stuck

```text
No step increase for 10 minutes
```

## Low GPU utilization

```text
GPU utilization < 20%
```

## Overfitting

```text
eval_loss increases
for 3 evaluations
```

Conceptually:

```python
def should_alert(
    metrics
):

    alerts = []

    if (
        metrics["gpu_memory_percent"]
        > 95
    ):

        alerts.append(
            "GPU memory critical"
        )


    if (
        metrics["gpu_utilization"]
        < 20
    ):

        alerts.append(
            "GPU underutilized"
        )


    if (
        metrics["training_stuck"]
    ):

        alerts.append(
            "Training job stuck"
        )

    return alerts
```

---

# 18. Monitor job status

A training job should have explicit states.

```python
from enum import Enum


class TrainingJobStatus(
    str,
    Enum
):

    QUEUED = "queued"

    RUNNING = "running"

    COMPLETED = "completed"

    FAILED = "failed"

    CANCELLED = "cancelled"
```

Store job information:

```python
job = {

    "job_id":
        "train-123",

    "status":
        "RUNNING",

    "current_step":
        1000,

    "total_steps":
        5000,

    "last_heartbeat":
        "2026-08-28T10:00:00"
}
```

Progress:

```python
def calculate_progress(
    current_step,
    total_steps
):

    return (
        current_step
        /
        total_steps
        *
        100
    )
```

Example:

```text
1000 / 5000

= 20%
```

---

# 19. Implement heartbeat monitoring

A process might still exist but be stuck.

Use a heartbeat.

```python
import time


class JobHeartbeat:

    def __init__(self):

        self.last_heartbeat = (
            time.time()
        )


    def beat(self):

        self.last_heartbeat = (
            time.time()
        )


    def is_stale(
        self,
        timeout_seconds=600
    ):

        return (

            time.time()
            -
            self.last_heartbeat

            >

            timeout_seconds
        )
```

During training:

```python
heartbeat = JobHeartbeat()


for batch in train_loader:

    # Train
    loss = train_step(
        batch
    )

    # Update heartbeat
    heartbeat.beat()
```

If:

```text
No heartbeat for 10 minutes
```

then:

```text
ALERT: Training may be stuck
```

In distributed production systems, store the heartbeat in:

```text
Redis
PostgreSQL
Kubernetes Job status
```

rather than only process memory.

---

# 20. Estimate training cost

For cloud GPU training:

```text
GPU cost/hour × training hours
```

Example:

```python
def calculate_training_cost(

    gpu_hourly_cost,
    elapsed_hours,
    gpu_count
):

    return (

        gpu_hourly_cost
        *
        elapsed_hours
        *
        gpu_count
    )
```

Example:

```python
cost = calculate_training_cost(

    gpu_hourly_cost=3.0,

    elapsed_hours=10,

    gpu_count=8
)

print(cost)
```

Output:

```text
$240
```

Monitor:

```text
Estimated Cost
Budget Remaining
Cost per 1M Tokens
```

---

# 21. Production callback combining everything

Here is a simplified monitoring callback:

```python
import math
import torch

from transformers import TrainerCallback


class ProductionMonitoringCallback(
    TrainerCallback
):

    def __init__(self):

        self.best_eval_loss = float(
            "inf"
        )


    def on_log(
        self,
        args,
        state,
        control,
        logs=None,
        **kwargs
    ):

        logs = logs or {}


        # ---------------------
        # Training progress
        # ---------------------

        print(
            f"Step: {state.global_step}"
        )


        # ---------------------
        # Loss monitoring
        # ---------------------

        loss = logs.get(
            "loss"
        )

        if loss is not None:

            print(
                f"Loss: {loss}"
            )

            if (
                math.isnan(loss)
                or
                math.isinf(loss)
            ):

                print(
                    "CRITICAL: "
                    "Invalid loss"
                )

                control.should_training_stop = True


        # ---------------------
        # Learning rate
        # ---------------------

        lr = logs.get(
            "learning_rate"
        )

        if lr:

            print(
                f"Learning Rate: {lr}"
            )


        # ---------------------
        # Gradient norm
        # ---------------------

        grad_norm = logs.get(
            "grad_norm"
        )

        if grad_norm:

            print(
                f"Gradient Norm: "
                f"{grad_norm}"
            )


        # ---------------------
        # GPU metrics
        # ---------------------

        if torch.cuda.is_available():

            memory_gb = (

                torch.cuda.memory_allocated()

                /

                1024**3
            )

            print(
                f"GPU Memory: "
                f"{memory_gb:.2f} GB"
            )


        return control


    def on_evaluate(

        self,

        args,

        state,

        control,

        metrics=None,

        **kwargs
    ):

        metrics = metrics or {}

        eval_loss = metrics.get(
            "eval_loss"
        )


        if eval_loss:

            print(
                f"Eval Loss: "
                f"{eval_loss}"
            )


            if (

                eval_loss

                >

                self.best_eval_loss

            ):

                print(
                    "WARNING: "
                    "Evaluation loss increased"
                )

            else:

                self.best_eval_loss = (
                    eval_loss
                )


        return control
```

Use:

```python
trainer.add_callback(
    ProductionMonitoringCallback()
)
```

---

# 22. Production monitoring stack

For a serious fine-tuning platform, I would use:

```text
┌────────────────────────────────────┐
│          Training Job              │
│     Hugging Face / PyTorch         │
└─────────────────┬──────────────────┘
                  │
       ┌──────────┼───────────┐
       ▼          ▼           ▼
   MLflow     Prometheus    Logs
       │          │           │
       │          ▼           ▼
       │       Grafana   Loki / ELK
       │
       ▼
Experiment Tracking
       │
       ▼
Model Registry
```

### Responsibilities

| Tool                 | Purpose                            |
| -------------------- | ---------------------------------- |
| Hugging Face Trainer | Training loop and callbacks        |
| MLflow               | Experiments, parameters, artifacts |
| Prometheus           | Time-series infrastructure metrics |
| Grafana              | Dashboards and visualization       |
| Loki/ELK             | Logs and errors                    |
| Kubernetes           | Job lifecycle and GPU scheduling   |

---

# 23. What I would monitor in a real production job

```text
TRAINING
├── train_loss
├── eval_loss
├── learning_rate
├── gradient_norm
├── epoch
└── global_step

PERFORMANCE
├── tokens/sec
├── steps/sec
├── dataloader latency
└── checkpoint time

GPU
├── utilization
├── memory
├── temperature
└── power

SYSTEM
├── CPU
├── RAM
├── disk
└── network

RELIABILITY
├── job status
├── heartbeat
├── errors
├── retries
└── checkpoint freshness

QUALITY
├── validation metrics
├── hallucination rate
├── instruction following
└── safety metrics

COST
├── GPU hours
├── estimated cost
└── cost per token
```

---

# Interview-ready answer

> **I monitor fine-tuning jobs at three levels: training quality, infrastructure health, and job reliability.**
>
> **For training quality, I track training loss, validation loss, learning rate, gradient norm, throughput, and evaluation metrics. I specifically watch for NaN losses, exploding gradients, and divergence between training and validation loss, which can indicate instability or overfitting.**
>
> **For infrastructure, I monitor GPU utilization, GPU memory, temperature, CPU, RAM, disk, and tokens per second. For reliability, I track job state, progress, heartbeats, logs, errors, checkpoint freshness, and retries.**
>
> **I use Hugging Face callbacks to emit training metrics, MLflow for experiment tracking, Prometheus and Grafana for infrastructure dashboards and alerting, and Kubernetes job status for lifecycle monitoring. I also configure alerts for NaN loss, high GPU memory, stuck jobs, low GPU utilization, failed checkpoints, and excessive training cost.**

## The most important production principle

```text
Don't just ask:

"Is training loss going down?"
```

Monitor:

```text
Is the model learning?
Is validation improving?
Is the job healthy?
Is the GPU being used efficiently?
Can the job recover?
Is training within budget?
```

That is what makes fine-tuning monitoring **production-grade**.
