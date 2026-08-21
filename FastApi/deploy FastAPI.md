For a production AI/RAG system, I would deploy FastAPI as a **stateless Kubernetes Deployment**, expose it through a **Service + Ingress/Gateway**, and scale it horizontally using **HPA** based on CPU/memory and, for AI workloads, preferably **custom metrics such as request rate, latency, or in-flight requests**.

The architecture I'd explain in an interview is:

```text
                         Internet
                            │
                            ▼
                  ┌──────────────────┐
                  │ Ingress / Gateway│
                  └────────┬─────────┘
                           │
                           ▼
                  ┌──────────────────┐
                  │ Kubernetes       │
                  │ Service          │
                  └────────┬─────────┘
                           │
             ┌─────────────┼─────────────┐
             ▼             ▼             ▼
        FastAPI Pod   FastAPI Pod   FastAPI Pod
             │             │             │
             └─────────────┼─────────────┘
                           │
             ┌─────────────┼─────────────┐
             ▼             ▼             ▼
        PostgreSQL       Redis         Qdrant
                           │
                           ▼
                       LLM APIs
```

## 1. Containerize FastAPI

I would first create a production Docker image.

```dockerfile
FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

For a high-throughput deployment, I might use multiple Uvicorn workers, but on Kubernetes I generally prefer **one main process per container and scale pods horizontally**.

So instead of:

```text
1 Pod
 ├── worker
 ├── worker
 ├── worker
 └── worker
```

I generally prefer:

```text
Pod 1 → FastAPI process
Pod 2 → FastAPI process
Pod 3 → FastAPI process
Pod 4 → FastAPI process
```

Kubernetes then handles distribution and scaling.

---

# 2. Kubernetes Deployment

I'd define a Deployment:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rag-api
spec:
  replicas: 3

  selector:
    matchLabels:
      app: rag-api

  template:
    metadata:
      labels:
        app: rag-api

    spec:
      containers:
        - name: rag-api
          image: myregistry/rag-api:v1.0.0

          ports:
            - containerPort: 8000

          resources:
            requests:
              cpu: "500m"
              memory: "512Mi"

            limits:
              cpu: "2"
              memory: "2Gi"

          envFrom:
            - secretRef:
                name: rag-api-secrets

          readinessProbe:
            httpGet:
              path: /health/ready
              port: 8000
            initialDelaySeconds: 5
            periodSeconds: 10

          livenessProbe:
            httpGet:
              path: /health/live
              port: 8000
            initialDelaySeconds: 15
            periodSeconds: 20
```

The important pieces are:

* replicas
* resource requests
* resource limits
* readiness probe
* liveness probe
* secrets
* immutable/versioned image

---

# 3. Readiness vs liveness

This is a common Kubernetes interview question.

### Liveness

Answers:

> "Is this container still alive?"

```text
/liveness
```

If it fails repeatedly, Kubernetes restarts the container.

### Readiness

Answers:

> "Can this pod receive traffic?"

```text
/readiness
```

If it fails, Kubernetes removes the pod from the Service endpoints.

That's extremely important during deployments.

For example:

```text
New Pod
   │
   ▼
Starting
   │
   ▼
Not Ready
   │
   ▼
Load dependencies
   │
   ▼
Ready
   │
   ▼
Receive traffic
```

---

# 4. FastAPI health endpoints

I would expose separate endpoints:

```python
@app.get("/health/live")
async def liveness():
    return {"status": "alive"}


@app.get("/health/ready")
async def readiness():
    # Check critical dependencies if appropriate
    return {"status": "ready"}
```

I would be careful about making readiness dependent on every external service.

For example, if Redis is temporarily unavailable but Redis is only used for caching, I might still consider the API ready.

But if PostgreSQL is mandatory for every request, database readiness should affect the readiness decision.

---

# 5. Kubernetes Service

The Deployment isn't directly exposed.

I'd create a Service:

```yaml
apiVersion: v1
kind: Service
metadata:
  name: rag-api
spec:
  selector:
    app: rag-api

  ports:
    - port: 80
      targetPort: 8000

  type: ClusterIP
```

Now:

```text
Ingress
   ↓
Service
   ↓
Pod 1
Pod 2
Pod 3
```

Kubernetes automatically distributes requests among healthy pods.

---

# 6. Ingress

For external traffic I'd use an Ingress or Gateway.

Conceptually:

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: rag-api
spec:
  rules:
    - host: api.example.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: rag-api
                port:
                  number: 80
```

In a modern production setup, I'd also consider Kubernetes Gateway API depending on the platform.

---

# 7. Horizontal Pod Autoscaler

Now we can scale FastAPI.

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: rag-api
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: rag-api

  minReplicas: 3
  maxReplicas: 20

  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70

    - type: Resource
      resource:
        name: memory
      target:
        type: Utilization
        averageUtilization: 75
```

So:

```text
Low traffic
   ↓
3 pods

High traffic
   ↓
6 pods

Very high traffic
   ↓
12 pods

Maximum
   ↓
20 pods
```

---

# 8. But CPU isn't always the best metric for AI APIs

This is an important senior-level point.

For a normal CRUD API:

```text
CPU → HPA
```

can work well.

But an AI API may spend most of its time waiting for:

* LLM API
* Qdrant
* PostgreSQL
* external tools

So CPU might remain low while request latency becomes terrible.

I'd consider custom metrics such as:

```text
requests_per_second
in_flight_requests
p95_latency
LLM_queue_depth
tokens_per_second
```

For example:

```text
                    ┌── CPU
                    │
                    ├── Memory
HPA metrics ────────┼── Requests/sec
                    │
                    ├── In-flight requests
                    │
                    └── Queue depth
```

For AI workloads, **in-flight requests or queue depth** can be especially useful.

---

# 9. Stateless FastAPI

This is critical for horizontal scaling.

I would avoid storing application state inside the pod:

```python
users = {}
conversation_memory = {}
cache = {}
```

because:

```text
Pod 1
  └── memory

Pod 2
  └── different memory
```

Instead:

```text
FastAPI Pods
     │
     ├── PostgreSQL
     ├── Redis
     └── Object Storage
```

For example:

```text
Conversation state → Redis/PostgreSQL
Cache             → Redis
Persistent data   → PostgreSQL
Files             → Object Storage
```

Then any pod can serve any request.

---

# 10. Redis and PostgreSQL scaling

FastAPI pods aren't the only scalability concern.

Suppose:

```text
FastAPI
  ↓
100 pods
  ↓
PostgreSQL
```

You can overwhelm PostgreSQL.

I'd use:

* connection pooling
* PgBouncer where appropriate
* sensible connection limits
* read replicas for read-heavy workloads
* indexes
* query optimization

For async SQLAlchemy:

```text
FastAPI Pod
   │
   ▼
SQLAlchemy pool
   │
   ▼
PostgreSQL
```

Don't allow every pod to create an unlimited number of DB connections.

---

# 11. Qdrant scaling

For RAG:

```text
FastAPI
   │
   ▼
Qdrant
```

I'd scale Qdrant independently from FastAPI.

This is another reason to keep services separate.

```text
FastAPI Deployment
       │
       ▼
Qdrant Cluster
```

FastAPI should remain stateless while Qdrant handles vector storage/retrieval.

---

# 12. Separate ingestion workers

I would **not run heavy document ingestion inside the FastAPI pods**.

Instead:

```text
POST /documents
       │
       ▼
FastAPI
       │
       ▼
Queue
       │
       ├── Worker
       ├── Worker
       ├── Worker
       └── Worker
```

Workers handle:

```text
PDF parsing
   ↓
Chunking
   ↓
Embedding
   ↓
Qdrant indexing
```

Then workers can scale independently.

For example:

```text
API workload
    ↓
10 FastAPI pods

Ingestion workload
    ↓
30 worker pods
```

This is much better than scaling everything together.

---

# 13. Graceful shutdown

This matters especially for streaming LLM responses.

Suppose Kubernetes wants to terminate:

```text
FastAPI Pod
   │
   └── 50 active LLM streams
```

I don't want to immediately kill them.

I'd implement:

```text
SIGTERM
   ↓
Stop accepting new requests
   ↓
Readiness = false
   ↓
Wait for active requests
   ↓
Close resources
   ↓
Pod terminates
```

FastAPI/Uvicorn supports graceful shutdown behavior, but your application should also clean up:

* DB connections
* Redis clients
* HTTP clients
* background resources
* streaming tasks

---

# 14. Rolling deployment

Kubernetes Deployments support rolling updates.

Suppose:

```text
v1
v1
v1
```

Deploy:

```text
v2
```

Kubernetes gradually moves:

```text
v1 v1 v1
 ↓
v2 v1 v1
 ↓
v2 v2 v1
 ↓
v2 v2 v2
```

while maintaining availability according to the deployment strategy.

I'd configure:

```yaml
strategy:
  type: RollingUpdate
  rollingUpdate:
    maxUnavailable: 0
    maxSurge: 1
```

This is particularly useful for APIs where downtime isn't acceptable.

---

# 15. PodDisruptionBudget

I'd also protect availability during node maintenance.

```yaml
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: rag-api
spec:
  minAvailable: 2

  selector:
    matchLabels:
      app: rag-api
```

If I have:

```text
3 replicas
```

I don't want Kubernetes maintenance to voluntarily take down all three.

---

# 16. Security

I would not put secrets directly in the Deployment YAML.

Instead:

```text
Kubernetes Secrets
       │
       ▼
FastAPI
```

For larger environments, I'd consider an external secrets manager.

Secrets might include:

```text
DATABASE_URL
REDIS_URL
QDRANT_URL
LLM_API_KEY
JWT_SECRET
```

I'd also use:

* RBAC
* NetworkPolicies
* non-root containers
* read-only filesystem where practical
* minimal container permissions
* image vulnerability scanning

For an enterprise AI platform, I'd also ensure network access to PostgreSQL/Qdrant is restricted.

---

# 17. Resource requests and limits

This:

```yaml
resources:
  requests:
    cpu: "500m"
    memory: "512Mi"

  limits:
    cpu: "2"
    memory: "2Gi"
```

means Kubernetes can make better scheduling decisions.

I'd determine actual values from production metrics rather than blindly copying numbers.

For AI workloads, memory sizing is particularly important because:

```text
large prompts
+
document processing
+
JSON serialization
+
concurrent requests
```

can cause significant memory usage.

---

# 18. Observability

I'd instrument the API with:

```text
Prometheus
     ↓
Grafana
```

and distributed tracing:

```text
OpenTelemetry
```

For one RAG request:

```text
Request
 │
 ├── FastAPI
 │
 ├── PostgreSQL
 │
 ├── Redis
 │
 ├── Embedding
 │
 ├── Qdrant
 │
 ├── Reranker
 │
 └── LLM
```

I want to see where the latency is actually coming from.

For example:

```text
p95 latency = 4.2 seconds

FastAPI       = 20ms
Redis         = 3ms
Qdrant        = 80ms
Reranker      = 200ms
LLM           = 3.8s
```

Then I know scaling FastAPI isn't going to solve the problem.

---

# 19. Deployment pipeline

I'd use CI/CD:

```text
Developer
   │
   ▼
Git
   │
   ▼
CI
 ├── pytest
 ├── mypy
 ├── ruff
 ├── security scan
 └── Docker build
   │
   ▼
Container Registry
   │
   ▼
Kubernetes
   │
   ▼
Deployment
```

For production, I'd prefer immutable image tags:

```text
rag-api:git-a81f23
```

rather than repeatedly deploying:

```text
rag-api:latest
```

---

# 20. Blue-green / canary deployment

For higher-risk releases:

```text
Production
   │
   ├── v1 → 90%
   │
   └── v2 → 10%
```

Monitor:

```text
error rate
latency
LLM cost
RAG quality
```

If v2 looks good:

```text
10% → 25% → 50% → 100%
```

If it fails:

```text
rollback → v1
```

For AI systems, I'd go beyond normal API metrics and also monitor **answer quality and model behavior**.

---

# 21. One important AI-specific issue: autoscaling isn't enough

Suppose traffic increases:

```text
100 → 1,000 requests/sec
```

HPA creates more FastAPI pods.

But then:

```text
10 FastAPI pods
      ↓
LLM provider
      ↓
429
```

Now we've simply scaled our application into a rate-limit problem.

So I'd combine:

```text
Kubernetes HPA
+
application rate limiter
+
LLM concurrency limiter
+
LLM retry/backoff
+
provider quota management
```

This is a very good senior-level point.

---

# Interview answer

If asked **"How would you deploy and scale FastAPI on Kubernetes?"**, I'd answer:

> **"I'd containerize FastAPI and deploy it as a stateless Kubernetes Deployment. I'd expose it through a ClusterIP Service and Ingress or Gateway, configure readiness and liveness probes, resource requests and limits, and use HPA for horizontal scaling. For a normal API, CPU and memory can drive HPA, but for an AI/RAG API I'd also consider custom metrics such as request rate, p95 latency, in-flight requests or queue depth because FastAPI may spend most of its time waiting on LLM, Qdrant or database calls rather than consuming CPU. I'd keep persistent state in PostgreSQL, Redis and Qdrant rather than pod memory, and run document ingestion as separate worker deployments so API and ingestion workloads scale independently. I'd use rolling or canary deployments, PodDisruptionBudgets, graceful shutdown, secrets management, NetworkPolicies and observability through Prometheus, Grafana and OpenTelemetry. Finally, I'd make sure scaling FastAPI doesn't overwhelm downstream systems by applying database connection pooling, Redis limits, Qdrant capacity planning and LLM rate/concurrency limits."**

### The architecture to memorize

```text
                 Ingress / Gateway
                        │
                        ▼
                  K8s Service
                        │
          ┌─────────────┼─────────────┐
          ▼             ▼             ▼
      FastAPI         FastAPI       FastAPI
        Pod             Pod           Pod
          │             │             │
          └─────────────┼─────────────┘
                        │
          ┌─────────────┼─────────────┐
          ▼             ▼             ▼
     PostgreSQL       Redis         Qdrant
                                      │
                                      ▼
                                     LLM

                  HPA
                   │
                   ▼
          scale FastAPI pods

                  Queue
                   │
          ┌────────┼────────┐
          ▼        ▼        ▼
       Worker   Worker   Worker
          │        │        │
          └────────┼────────┘
                   ▼
                 Qdrant
```

**Senior-level takeaway:** Kubernetes scales your **API layer**, but production AI scaling requires you to manage the entire dependency chain—**FastAPI → PostgreSQL/Redis/Qdrant → LLM provider**—otherwise HPA can simply move the bottleneck downstream.
