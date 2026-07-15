# Write a Batch Inference Pipeline (Senior AI Engineer Interview)

This is one of the most common **Senior AI Engineer**, **ML Engineer**, and **LLM Engineer** coding interview questions.

> **"Design and implement a batch inference pipeline."**

The interviewer wants to evaluate your understanding of:

* Model loading
* Batch processing
* Data pipelines
* Memory efficiency
* GPU utilization
* Error handling
* Production deployment
* Parallel processing

---

# What is Batch Inference?

Batch inference processes **many inputs together** instead of one request at a time.

For example:

```
100,000 Images
        │
        ▼
Batch Size = 256
        │
        ▼
391 batches
        │
        ▼
Model Predictions
        │
        ▼
CSV / Database / Data Lake
```

---

# When is Batch Inference Used?

Examples include:

* Predicting customer churn overnight
* Fraud detection on yesterday's transactions
* Product recommendations
* Embedding millions of documents
* Sentiment analysis of reviews
* Image classification at scale
* LLM document summarization

Unlike online inference, latency per individual request is less important than overall throughput.

---

# Batch Inference Architecture

```
               Input Files
             (CSV / Parquet)
                     │
                     ▼
              Data Loader
                     │
                     ▼
             Batch Generator
                     │
                     ▼
               ML Model
                     │
                     ▼
             Predictions
                     │
                     ▼
          CSV / Database / S3
```

---

# Step 1 — Load the Model

```python
import torch

class SentimentModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 2)

    def forward(self, x):
        return self.linear(x)


model = SentimentModel()

model.load_state_dict(torch.load("model.pt"))

model.eval()
```

Notice:

```python
model.eval()
```

This disables:

* Dropout
* BatchNorm updates

which is required for deterministic inference.

---

# Step 2 — Create a Dataset

```python
from torch.utils.data import Dataset


class CustomerDataset(Dataset):

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
```

---

# Step 3 — DataLoader

```python
from torch.utils.data import DataLoader

loader = DataLoader(
    dataset,
    batch_size=128,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)
```

### Why these settings?

* `shuffle=False` preserves input order.
* `num_workers=4` loads data in parallel.
* `pin_memory=True` speeds CPU → GPU transfers.

---

# Step 4 — Inference Loop

```python
import torch

predictions = []

device = torch.device("cuda")

model.to(device)

with torch.no_grad():

    for batch in loader:

        batch = batch.to(device)

        outputs = model(batch)

        pred = outputs.argmax(dim=1)

        predictions.extend(pred.cpu().tolist())
```

### Why `torch.no_grad()`?

It disables gradient tracking:

* Lower memory usage
* Faster execution
* No computation graph

---

# Step 5 — Save Predictions

```python
import pandas as pd

df = pd.DataFrame({
    "prediction": predictions
})

df.to_csv("predictions.csv", index=False)
```

---

# End-to-End Pipeline

```python
def batch_predict(model, loader, device):

    model.eval()

    results = []

    with torch.no_grad():

        for batch in loader:

            batch = batch.to(device)

            output = model(batch)

            pred = output.argmax(dim=1)

            results.extend(pred.cpu().numpy())

    return results
```

Usage:

```python
predictions = batch_predict(model, loader, "cuda")
```

---

# Pipeline Flow

```
CSV
 │
 ▼
Dataset
 │
 ▼
DataLoader
 │
 ▼
Batch
 │
 ▼
GPU
 │
 ▼
Prediction
 │
 ▼
Write CSV
```

---

# Large Dataset Pipeline

When the dataset is too large to fit into memory:

```
S3 / HDFS
     │
     ▼
Read Chunk
     │
     ▼
Batch
     │
     ▼
Inference
     │
     ▼
Write Output
     │
     ▼
Next Chunk
```

This chunked approach keeps memory usage bounded.

---

# Production Batch Pipeline

```
Apache Airflow
       │
       ▼
Read from S3
       │
       ▼
Preprocessing
       │
       ▼
PyTorch Model
       │
       ▼
Batch Prediction
       │
       ▼
Post-processing
       │
       ▼
Write to Snowflake / BigQuery / S3
```

---

# Error Handling

Rather than stopping the entire job because of one bad record:

```python
results = []

for batch in loader:
    try:
        with torch.no_grad():
            output = model(batch)
            results.extend(output.argmax(1).cpu().tolist())
    except Exception as exc:
        print(f"Skipping batch: {exc}")
```

In production, log structured error details and optionally retry transient failures.

---

# Performance Optimizations

| Optimization                       | Benefit                                                    |
| ---------------------------------- | ---------------------------------------------------------- |
| `model.eval()`                     | Correct inference behavior                                 |
| `torch.no_grad()`                  | Reduces memory and improves speed                          |
| GPU inference                      | Higher throughput                                          |
| `pin_memory=True`                  | Faster host-to-device copies                               |
| `num_workers>0`                    | Parallel data loading                                      |
| Batching                           | Better GPU utilization                                     |
| Mixed precision (`torch.autocast`) | Faster inference on supported GPUs with lower memory usage |
| Chunked reads                      | Handles datasets larger than RAM                           |
| Multiple GPUs                      | Horizontal scaling for large workloads                     |

---

# Batch vs Real-Time Inference

| Feature    | Batch                   | Real-Time            |
| ---------- | ----------------------- | -------------------- |
| Latency    | Seconds to hours        | Milliseconds         |
| Throughput | Very high               | Moderate             |
| Input      | Large datasets          | Single requests      |
| Scheduling | Periodic (cron/Airflow) | On demand            |
| Example    | Overnight churn scoring | Live fraud detection |

---

# Production Enhancements

For a senior-level system, add:

* Configuration management (batch size, device, input/output paths).
* Structured logging with request/job IDs.
* Metrics (throughput, latency, GPU utilization, error rate).
* Checkpointing to resume long-running jobs.
* Idempotent writes to avoid duplicate outputs.
* Input schema validation.
* Model version tracking.
* Retry with exponential backoff for transient storage/network failures.
* Distributed execution using Spark, Ray, or Kubernetes Jobs for very large datasets.
* Data quality checks before and after inference.

---

# Senior AI Engineer Interview Answer

A concise answer that demonstrates production thinking is:

> "I design batch inference as a streaming data pipeline. Data is read in chunks through a `Dataset` and `DataLoader`, processed in batches with the model in `eval()` mode under `torch.no_grad()`, and predictions are written incrementally to persistent storage. I optimize throughput using batching, parallel data loading, pinned memory, and GPU acceleration. In production, I add structured logging, metrics, retries, checkpointing, model versioning, and distributed execution so the pipeline can reliably process millions of records without exhausting memory or losing progress."
