# Blue-Green Deployment

**Blue-Green Deployment** is a deployment strategy where you maintain **two identical production environments**:

* **Blue** → Current production environment (live)
* **Green** → New version waiting to be released

At any given time, **only one environment serves user traffic**. Once the Green environment is verified, traffic is switched from Blue to Green. If anything goes wrong, traffic can immediately be switched back.

This approach minimizes downtime and makes rollbacks very fast.

---

# Why Blue-Green Deployment?

Suppose your fraud detection model (V1) is serving customers.

```
Current Production

Fraud Model V1
```

You train a new model (V2).

Without Blue-Green deployment:

```
Deploy V2

↓

Users experience errors

↓

Need emergency rollback
```

This can cause downtime and affect users.

With Blue-Green deployment:

```
Blue Environment

Fraud Model V1

↓

Green Environment

Fraud Model V2

↓

Switch Traffic

↓

Rollback if needed
```

Users continue using the Blue environment until Green is fully validated.

---

# Architecture

```
                    Users
                      │
                      ▼
              Load Balancer
                      │
          ┌───────────┴───────────┐
          ▼                       ▼
   Blue Environment         Green Environment
   API + Model V1           API + Model V2
```

Initially:

```
Users

↓

Blue (100%)

Green (0%)
```

After validation:

```
Users

↓

Green (100%)

Blue (Idle)
```

---

# Step-by-Step Deployment

## Step 1: Current Production

```
Users

↓

Load Balancer

↓

Blue

↓

Fraud Model V1
```

Everything is working normally.

---

## Step 2: Deploy New Version

Deploy the new model to the Green environment.

```
Blue

Model V1

Running
```

```
Green

Model V2

Deploying
```

No production traffic reaches Green yet.

---

## Step 3: Test Green

Run:

* Health checks
* Integration tests
* API tests
* Load tests
* Smoke tests

Example:

```python
response = requests.post(
    "http://green-api/predict",
    json={"amount": 2500}
)

assert response.status_code == 200
```

You can also compare predictions against Blue:

```python
blue_prediction = blue_model.predict(transaction)
green_prediction = green_model.predict(transaction)

print(blue_prediction, green_prediction)
```

---

## Step 4: Switch Traffic

If Green passes all checks:

```
Before

Users

↓

Blue
```

↓

```
After

Users

↓

Green
```

In Kubernetes or a cloud load balancer, this is usually done by updating the Service or Load Balancer configuration.

---

## Step 5: Monitor

After the switch, monitor:

* API latency
* Error rate
* CPU
* Memory
* Prediction distribution
* Business metrics

Example:

```
Latency

Blue 120 ms

Green 118 ms
```

```
Accuracy

Blue 92%

Green 95%
```

---

## Step 6: Rollback

Suppose Green starts producing incorrect predictions.

```
Users

↓

Green

↓

Errors
```

Simply redirect traffic:

```
Users

↓

Blue
```

Rollback usually takes only seconds because Blue is still running.

---

# Kubernetes Example

Suppose both deployments exist.

```
Blue Deployment

fraud-api-v1
```

```
Green Deployment

fraud-api-v2
```

Initially, the Kubernetes Service points to Blue:

```yaml
apiVersion: v1
kind: Service

spec:
  selector:
    version: blue
```

To switch traffic:

```yaml
spec:
  selector:
    version: green
```

No application code changes are required.

---

# Real-World Example

Imagine an e-commerce recommendation system.

### Blue

```
Recommendation Model V1

CTR = 8.2%
```

### Green

```
Recommendation Model V2

CTR = Unknown
```

Deployment process:

```
Deploy V2

↓

Run Tests

↓

Switch Traffic

↓

Monitor CTR

↓

Keep or Rollback
```

If CTR drops or errors increase:

```
Green

↓

Rollback

↓

Blue
```

---

# Blue-Green vs Canary Deployment

| Feature        | Blue-Green                 | Canary                                           |
| -------------- | -------------------------- | ------------------------------------------------ |
| Traffic switch | 100% at once               | Gradual (e.g., 5%, 10%, 50%, 100%)               |
| Rollback       | Immediate                  | Gradual or immediate depending on rollout        |
| Downtime       | Nearly zero                | Nearly zero                                      |
| Infrastructure | Two full environments      | Shared infrastructure with controlled routing    |
| Risk           | Higher at switch time      | Lower due to incremental rollout                 |
| Best for       | Simple production releases | High-risk changes and continuous experimentation |

---

# Advantages

* Zero or near-zero downtime.
* Very fast rollback.
* Easy to validate the new environment before users see it.
* Predictable deployment process.
* Works well with containers and Kubernetes.

---

# Disadvantages

* Requires two complete environments, increasing infrastructure costs.
* Databases need careful handling because schema changes must be compatible with both environments.
* Instant traffic switch means issues can affect all users if testing missed a problem.
* Stateful applications require additional planning for session management.

---

# Production Architecture

```
                    Users
                      │
                      ▼
                Load Balancer
                      │
        ┌─────────────┴─────────────┐
        ▼                           ▼
 Blue Environment             Green Environment
 ├── FastAPI                  ├── FastAPI
 ├── Model V1                 ├── Model V2
 ├── Redis                    ├── Redis
 └── Monitoring               └── Monitoring
        │                           │
        └─────────────┬─────────────┘
                      ▼
            Prometheus + Grafana
```

---

# Interview Question

**Q: How would you deploy a new fraud detection model without downtime?**

A strong answer:

1. Deploy the new model to a separate **Green** environment.
2. Keep the **Blue** environment serving all production traffic.
3. Run health checks, integration tests, and performance validation on Green.
4. Switch the load balancer or Kubernetes Service to route traffic to Green.
5. Monitor latency, error rates, prediction quality, and business metrics.
6. If any issue is detected, redirect traffic back to Blue immediately.

This approach provides **near-zero downtime**, **rapid rollback**, and a **safe deployment process**, which is why it is widely used for production ML services and AI inference APIs.
