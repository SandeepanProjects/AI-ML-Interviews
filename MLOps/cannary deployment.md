# Canary Deployment

**Canary Deployment** is a deployment strategy where a **small percentage of users are gradually routed to the new version** while the remaining users continue using the old version.

Instead of switching all traffic at once (as in Blue-Green deployment), you increase traffic to the new version in stages.

The name comes from the **"canary in a coal mine"**—miners used canaries as an early warning system for dangerous gases. Similarly, a canary deployment exposes a small number of users to the new version first to detect problems before a full rollout.

---

# Why Canary Deployment?

Suppose your fraud detection model (V1) is serving all requests.

```text
Current Production

Fraud Model V1
```

You train a new model (V2).

If you immediately switch all users:

```text
Users

↓

Model V2

↓

Bug

↓

100% of users affected
```

With a canary deployment:

```text
Users

↓

95% → Model V1

5%  → Model V2
```

If V2 performs well, increase traffic gradually.

---

# Traffic Flow

```text
                 Users
                    │
            Load Balancer
                    │
         ┌──────────┴──────────┐
         ▼                     ▼
    Model V1               Model V2
     (95%)                  (5%)
```

After validation:

```text
Users

↓

90% → V1

10% → V2
```

↓

```text
Users

↓

50% → V1

50% → V2
```

↓

```text
Users

↓

100% → V2
```

---

# Step-by-Step Deployment

## Step 1: Current State

```text
Users

↓

Model V1

100%
```

Everything is stable.

---

## Step 2: Deploy the New Model

Deploy V2 alongside V1.

```text
Model V1

Running
```

```text
Model V2

Running
```

At this point, V2 receives **no production traffic**.

---

## Step 3: Route 5% of Traffic

Configure the load balancer or service mesh.

```text
Users

↓

95%

↓

Model V1
```

```text
Users

↓

5%

↓

Model V2
```

---

## Step 4: Monitor Metrics

Watch:

* API latency
* Error rate
* CPU and memory
* Prediction accuracy
* Business metrics (conversion rate, fraud detection rate, CTR, etc.)

Example dashboard:

```text
Model V1

Latency 120 ms

Error Rate 0.1%
```

```text
Model V2

Latency 118 ms

Error Rate 0.2%
```

If the metrics remain healthy, continue.

---

## Step 5: Increase Traffic

Gradually move more users.

```text
5%

↓

10%

↓

25%

↓

50%

↓

75%

↓

100%
```

At each stage, verify:

* No increase in failures.
* Prediction quality is acceptable.
* Business KPIs remain stable or improve.

---

## Step 6: Complete the Rollout

Once V2 proves stable:

```text
Users

↓

Model V2

100%
```

V1 can then be retired.

---

# Rollback

Suppose at 25% traffic, V2 starts returning incorrect predictions.

Current routing:

```text
75% → V1

25% → V2
```

Rollback:

```text
100% → V1
```

Only a subset of users was exposed to the issue, reducing risk.

---

# Kubernetes Example

Create two deployments.

```text
fraud-api-v1

fraud-api-v2
```

With a service mesh (such as Istio), you can define traffic weights.

Example:

```yaml
apiVersion: networking.istio.io/v1beta1
kind: VirtualService

spec:
  http:
  - route:
    - destination:
        host: fraud-api
        subset: v1
      weight: 90

    - destination:
        host: fraud-api
        subset: v2
      weight: 10
```

Increase the weights over time:

```text
90/10

↓

75/25

↓

50/50

↓

0/100
```

---

# Real-World Example

An e-commerce company deploys a new recommendation model.

Current production:

```text
Recommendation Model V1

CTR = 8.2%
```

Deploy V2.

Traffic distribution:

```text
95% → V1

5% → V2
```

After one hour:

| Metric     | V1     | V2     |
| ---------- | ------ | ------ |
| Latency    | 120 ms | 115 ms |
| Error Rate | 0.1%   | 0.1%   |
| CTR        | 8.2%   | 8.9%   |

Increase traffic:

```text
20%

↓

50%

↓

100%
```

If V2's CTR had dropped to **6.5%**, traffic would be shifted back to V1 immediately.

---

# Canary vs Blue-Green

| Feature         | Canary                                      | Blue-Green                                       |
| --------------- | ------------------------------------------- | ------------------------------------------------ |
| Traffic rollout | Gradual                                     | All at once                                      |
| User exposure   | Small subset initially                      | Entire user base after switch                    |
| Risk            | Lower                                       | Higher at switch time                            |
| Rollback        | Shift traffic back gradually or immediately | Switch back to Blue instantly                    |
| Infrastructure  | Two versions running simultaneously         | Two complete environments                        |
| Best for        | ML models, AI systems, continuous releases  | Traditional applications, low-frequency releases |

---

# Canary for Machine Learning

Canary deployments are especially useful for ML models because you can compare **online performance** before fully replacing the existing model.

```text
                   Prediction Request
                           │
                ┌──────────┴──────────┐
                ▼                     ▼
         Fraud Model V1         Fraud Model V2
             (90%)                 (10%)
                │                     │
                └──────────┬──────────┘
                           ▼
                 Metrics Collection
                           │
       ┌───────────────────┼───────────────────┐
       ▼                   ▼                   ▼
    Latency          Fraud Detection      False Positives
                           │
                           ▼
                  Increase or Roll Back
```

Instead of relying only on offline metrics like accuracy or F1-score, monitor **online metrics** such as:

* Prediction latency
* Request failure rate
* Conversion rate
* Fraud detection rate
* False positive rate
* User satisfaction
* Revenue impact

---

# Canary Rollout Timeline

```text
Time 0
100% → V1

↓

5 minutes
95% → V1
5%  → V2

↓

30 minutes
75% → V1
25% → V2

↓

2 hours
50% → V1
50% → V2

↓

6 hours
0%  → V1
100% → V2
```

If any stage shows degraded metrics, stop the rollout and route traffic back to V1.

---

# Interview Question

**Q: How would you safely deploy a new recommendation model to production?**

A strong answer:

1. Deploy the new model alongside the existing one.
2. Route **5–10%** of production traffic to the new model.
3. Monitor latency, error rate, prediction quality, and business KPIs.
4. Increase traffic incrementally (for example, **5% → 10% → 25% → 50% → 100%**) if the metrics remain healthy.
5. If performance degrades, immediately route traffic back to the previous model.

This strategy minimizes user impact while validating the new model under real production traffic, making it one of the most widely used deployment approaches for production AI and ML systems.
