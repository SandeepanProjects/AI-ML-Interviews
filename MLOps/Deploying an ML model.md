Deploying an ML model means making it available so other applications or users can send data to it and receive predictions reliably, securely, and at scale.

A production deployment involves much more than simply saving a trained model and loading it in Python.

## End-to-End ML Deployment Pipeline

```text
                 Training Data
                       │
                       ▼
               Model Training
                       │
                       ▼
              Model Evaluation
                       │
                       ▼
               Model Registry
                       │
                       ▼
             Build Docker Image
                       │
                       ▼
          Deploy to Kubernetes/Cloud
                       │
                       ▼
                REST/gRPC API
                       │
                       ▼
                Client Applications
                       │
                       ▼
      Monitoring + Logging + Retraining
```

---

# Step 1: Train the Model

Example using Scikit-learn:

```python
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

X, y = load_iris(return_X_y=True)

model = RandomForestClassifier()
model.fit(X, y)
```

---

# Step 2: Save the Model

Using `joblib`:

```python
import joblib

joblib.dump(model, "iris_model.pkl")
```

Later:

```python
model = joblib.load("iris_model.pkl")
```

For deep learning:

```python
torch.save(model.state_dict(), "model.pt")
```

or

```python
model.save("model.keras")
```

---

# Step 3: Create an Inference Service

A FastAPI application exposes the model through an HTTP API.

```python
from fastapi import FastAPI
import joblib

app = FastAPI()

model = joblib.load("iris_model.pkl")

@app.post("/predict")
def predict(features: list[float]):
    prediction = model.predict([features])
    return {"prediction": int(prediction[0])}
```

Run:

```bash
uvicorn app:app --reload
```

Request:

```http
POST /predict

{
    "features":[5.1,3.5,1.4,0.2]
}
```

Response:

```json
{
    "prediction":0
}
```

---

# Step 4: Containerize with Docker

**Dockerfile**

```dockerfile
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

CMD ["uvicorn","app:app","--host","0.0.0.0","--port","8000"]
```

Build:

```bash
docker build -t iris-api .
```

Run:

```bash
docker run -p 8000:8000 iris-api
```

---

# Step 5: Deploy to Kubernetes

Deployment:

```yaml
apiVersion: apps/v1
kind: Deployment

metadata:
  name: iris-api

spec:
  replicas: 3

  selector:
    matchLabels:
      app: iris

  template:
    metadata:
      labels:
        app: iris

    spec:
      containers:
      - name: api
        image: iris-api:latest

        ports:
        - containerPort: 8000
```

Service:

```yaml
apiVersion: v1
kind: Service

metadata:
  name: iris-service

spec:
  selector:
    app: iris

  ports:
  - port: 80
    targetPort: 8000
```

Traffic flows like this:

```text
Client
   │
   ▼
Load Balancer
   │
   ▼
Kubernetes Service
   │
   ▼
Pod 1
Pod 2
Pod 3
```

---

# Step 6: Model Registry

Instead of copying model files manually, store versioned models in a registry.

Example:

```text
MLflow Registry

Model

├── Version 1
├── Version 2
└── Version 3
```

The inference service loads a specific version.

```python
model = load_model(version="3")
```

Benefits:

* Versioning
* Rollback
* Audit trail
* Approval workflow

---

# Step 7: Monitoring

Track operational and model metrics.

Operational:

* Request rate
* Latency (P50/P95/P99)
* Error rate
* CPU
* Memory

Model:

* Prediction distribution
* Confidence
* Accuracy (when labels become available)
* Data drift
* Model drift

Example:

```python
import time

start = time.time()

prediction = model.predict(data)

latency = time.time() - start

logger.info({
    "latency_ms": latency * 1000,
    "prediction": int(prediction[0])
})
```

---

# Step 8: Handle Model Versions

Never overwrite models.

```text
Registry

Version 1

↓

Version 2

↓

Version 3
```

Deployment options:

* Blue-Green deployment
* Canary deployment
* Shadow deployment
* A/B testing

Example:

```text
Users

      ├── 90% → Model V2
      │
      └── 10% → Model V3
```

---

# Step 9: CI/CD Pipeline

A typical pipeline looks like this:

```text
Git Push
   │
   ▼
Run Unit Tests
   │
   ▼
Train Model
   │
   ▼
Evaluate Metrics
   │
   ▼
Register Model
   │
   ▼
Build Docker Image
   │
   ▼
Push Image to Registry
   │
   ▼
Deploy to Kubernetes
   │
   ▼
Smoke Tests
```

Tools commonly used:

| Stage            | Common Tools                       |
| ---------------- | ---------------------------------- |
| Source Control   | Git, GitHub, GitLab                |
| CI/CD            | GitHub Actions, GitLab CI, Jenkins |
| Model Registry   | MLflow, SageMaker Model Registry   |
| Containerization | Docker                             |
| Orchestration    | Kubernetes                         |
| Monitoring       | Prometheus, Grafana                |
| Logging          | ELK Stack, Loki                    |
| Tracing          | OpenTelemetry                      |

---

# Step 10: Production Architecture

```text
                 Client
                    │
                    ▼
            API Gateway / Ingress
                    │
                    ▼
             Load Balancer
                    │
                    ▼
          FastAPI Inference Service
                    │
        ┌───────────┼───────────┐
        ▼           ▼           ▼
    Model Registry  Redis    Feature Store
        │
        ▼
   Model Artifacts
        │
        ▼
 Prediction Response
        │
        ▼
 Monitoring & Logging
```

---

# Real-World Example: Fraud Detection

A bank deploys a fraud detection model as follows:

1. A customer makes a credit card transaction.
2. The payment service calls the prediction API.
3. The API loads the latest approved model.
4. Features (amount, merchant, location, time) are processed.
5. The model returns a fraud probability in under 100 ms.
6. If the score is above a threshold, the transaction is flagged for review.
7. Every request, prediction, latency, and outcome is logged for monitoring and future retraining.

---

# Common Deployment Patterns

| Pattern                                            | Best For                              | Advantages                                   | Trade-offs                                          |
| -------------------------------------------------- | ------------------------------------- | -------------------------------------------- | --------------------------------------------------- |
| REST API (FastAPI/Flask)                           | Most ML applications                  | Simple, language-agnostic                    | HTTP overhead                                       |
| gRPC                                               | High-throughput microservices         | Low latency, efficient serialization         | More complex clients                                |
| Batch Inference                                    | Offline scoring of large datasets     | Cost-effective                               | Not real-time                                       |
| Streaming Inference                                | Event-driven systems (Kafka, Kinesis) | Continuous processing                        | More infrastructure                                 |
| Serverless (AWS Lambda, Cloud Functions)           | Low or unpredictable traffic          | No server management                         | Cold starts, execution limits                       |
| Kubernetes                                         | Enterprise production workloads       | Autoscaling, resilience, rolling updates     | Higher operational complexity                       |
| Managed Platforms (SageMaker, Vertex AI, Azure ML) | Teams wanting managed infrastructure  | Built-in scaling, monitoring, model registry | Cloud provider lock-in and potentially higher costs |

For senior ML engineering roles, interviewers often expect you to understand not just serving the model, but the complete lifecycle: **training, versioning, containerization, deployment, scaling, monitoring, rollback, and continuous retraining**.
