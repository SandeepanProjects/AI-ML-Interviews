# Rollback Strategy in ML and AI Systems

A **rollback strategy** is a predefined plan to **quickly restore the previous stable version** of an application or ML model when a new deployment causes problems.

For AI systems, rollback is even more important because failures may not be crashes—they can include **poor predictions, hallucinations, increased latency, or higher business losses**.

---

# Why Do We Need Rollback?

Imagine you deploy a new fraud detection model.

Before deployment:

```text
Fraud Model V1

Accuracy = 95%

Latency = 120 ms
```

After deployment:

```text
Fraud Model V2

Accuracy = 88%

Latency = 350 ms

False Positives ↑
```

Without rollback:

```text
Deploy V2

↓

Users complain

↓

Revenue loss

↓

Engineers investigate

↓

Hours of downtime
```

With rollback:

```text
Deploy V2

↓

Problem detected

↓

Rollback to V1

↓

System stable
```

---

# Rollback Workflow

```text
              Deploy New Version
                      │
                      ▼
               Monitor Metrics
                      │
        ┌─────────────┴─────────────┐
        │                           │
    Metrics OK                 Metrics Bad
        │                           │
        ▼                           ▼
 Continue Rollout          Rollback to Previous Version
```

---

# What Can Trigger a Rollback?

### 1. High Error Rate

```text
Normal

0.2%
```

↓

```text
After Deployment

5%
```

Rollback immediately.

---

### 2. Increased Latency

```text
V1

120 ms
```

↓

```text
V2

900 ms
```

Users experience slow responses.

---

### 3. Accuracy Drops

Example:

```text
V1

95%
```

↓

```text
V2

88%
```

Offline metrics looked good, but production data differs.

---

### 4. Business Metrics Decrease

Example:

```text
Recommendation CTR

8.5%

↓

6.8%
```

or

```text
Checkout Conversion

3.5%

↓

2.9%
```

Business KPI degradation is often more important than model accuracy.

---

### 5. Infrastructure Problems

Examples:

* Pods crash repeatedly
* GPU memory exhausted
* Database unavailable
* Network failures

---

### 6. Data Drift

```text
Training Data

↓

Age = 20–60
```

Production:

```text
Age = 5–95
```

The model may become unreliable.

---

# Rollback Strategies

## Strategy 1: Blue-Green Rollback

```text
          Users
             │
     Load Balancer
      │         │
      ▼         ▼
 Blue (V1)   Green (V2)
```

Initially:

```text
100% → Blue
```

Switch:

```text
100% → Green
```

If Green fails:

```text
100% → Blue
```

Rollback time is typically just a few seconds.

---

## Strategy 2: Canary Rollback

Traffic progression:

```text
95% → V1

5% → V2
```

↓

```text
75% → V1

25% → V2
```

If problems occur:

```text
100% → V1
```

Only a small percentage of users are affected.

---

## Strategy 3: Model Registry Rollback

Suppose the registry contains:

```text
Fraud Model

Version 6

Version 7

Version 8
```

Current production:

```text
Version 8
```

Rollback:

```text
Production

↓

Version 7
```

The inference service loads the previous approved version.

---

## Strategy 4: Kubernetes Rollback

Deployment history:

```text
Revision 1

↓

Revision 2

↓

Revision 3
```

Rollback:

```bash
kubectl rollout undo deployment/fraud-api
```

Rollback to a specific revision:

```bash
kubectl rollout undo deployment/fraud-api --to-revision=2
```

---

## Strategy 5: Docker Image Rollback

Current:

```text
fraud-api:v3
```

Rollback:

```text
fraud-api:v2
```

This works because Docker images are immutable.

---

# Automatic Rollback

Many production systems automatically roll back if health checks fail.

Example:

```text
Deploy V2

↓

Health Check

↓

High Error Rate

↓

Automatic Rollback
```

Pseudo-code:

```python
if error_rate > 2:
    rollback()

elif latency > 500:
    rollback()

elif accuracy_drop > 3:
    rollback()
```

---

# Monitoring During Deployment

Monitor continuously:

```text
Deployment

↓

Latency

↓

Error Rate

↓

CPU

↓

Memory

↓

Prediction Accuracy

↓

Business KPIs
```

If any metric crosses a threshold, trigger rollback.

---

# Real-World Example

An e-commerce company deploys a new recommendation model.

Traffic:

```text
90% → V1

10% → V2
```

Metrics after 30 minutes:

| Metric     | V1     | V2     |
| ---------- | ------ | ------ |
| Latency    | 120 ms | 125 ms |
| Error Rate | 0.1%   | 0.1%   |
| CTR        | 8.5%   | 6.9%   |

Although latency and errors are acceptable, **CTR** drops significantly.

Decision:

```text
Rollback

↓

100% → V1
```

This prevents a larger revenue impact.

---

# Production Architecture

```text
                    Users
                      │
               API Gateway
                      │
              Load Balancer
                      │
        ┌─────────────┴─────────────┐
        ▼                           ▼
   Model V1 (Stable)          Model V2 (New)
        │                           │
        └─────────────┬─────────────┘
                      ▼
             Metrics Collection
                      │
      ┌───────────────┼────────────────┐
      ▼               ▼                ▼
  Prometheus      OpenTelemetry     Business KPIs
                      │
                      ▼
              Rollback Controller
                      │
        ┌─────────────┴─────────────┐
        │                           │
   Healthy Metrics           Bad Metrics
        │                           │
        ▼                           ▼
 Continue Rollout        Restore Previous Version
```

---

# Best Practices

1. **Never overwrite production models.** Store every version in a model registry.
2. **Keep previous Docker images** so deployments can revert quickly.
3. **Use canary or blue-green deployments** instead of replacing production directly.
4. **Define clear rollback thresholds** for latency, error rate, and business KPIs before deployment.
5. **Automate rollback** for objective failures (e.g., repeated health check failures), but consider **manual approval** for business-metric degradation that may need more observation.
6. **Monitor both technical and business metrics.** A model with low latency can still hurt revenue if prediction quality declines.
7. **Test rollback procedures regularly.** A rollback plan that has never been exercised may fail when it's needed most.

---

# Comparison of Rollback Strategies

| Strategy                      | Rollback Time      | Risk     | Best For                                         |
| ----------------------------- | ------------------ | -------- | ------------------------------------------------ |
| Blue-Green                    | Seconds            | Low      | Web apps and ML APIs with duplicate environments |
| Canary                        | Seconds to minutes | Very Low | High-risk ML model releases                      |
| Kubernetes Rollout            | Seconds            | Low      | Containerized microservices                      |
| Model Registry Version Switch | Seconds            | Low      | ML model serving platforms                       |
| Docker Image Rollback         | Seconds            | Low      | Container-based deployments                      |
| Database Restore              | Minutes to hours   | High     | Data corruption or schema failures               |

---

# Interview Question

**Q: How would you safely roll back a new fraud detection model?**

A strong answer would be:

1. Deploy the new model using a **canary rollout** (for example, 5–10% of traffic).
2. Monitor latency, error rate, prediction quality, false positives, and business KPIs.
3. If any predefined threshold is exceeded, stop the rollout.
4. Route all traffic back to the previous stable model (or switch the load balancer back in a blue-green setup).
5. Record the incident, investigate the root cause, fix the issue, and only redeploy after validation.

This approach minimizes user impact while ensuring the production system can recover quickly from unexpected model or infrastructure issues.
