# Deploying a production LLM application on AWS using Docker and Kubernetes

As a Senior/Staff AI Engineer, I would deploy an LLM application using Docker for containerization, Amazon ECR for image storage, Amazon EKS for Kubernetes orchestration, and AWS managed services for databases, caching, storage, networking, and security.

For your Enterprise Multi-Agent Financial Advisor Copilot, I would use this architecture:

- FastAPI for the application/API layer.
- LangGraph for multi-agent orchestration.
- PostgreSQL for transactional data and workflow metadata.
- Qdrant for vector search.
- Redis for caching and short-lived state.
- Celery workers for asynchronous ingestion and background processing.
- OpenAI or another hosted LLM provider for inference.
- AWS EKS for application and worker deployment.

I'll explain the architecture first, then the Docker configuration, Kubernetes manifests, AWS infrastructure, CI/CD pipeline, and production considerations.

## 1. High-level AWS architecture

Users / Web / Mobile applications

Amazon Route 53 + AWS WAF

DNS, TLS entry and web application protection

Application Load Balancer

HTTPS routing into EKS

Amazon EKS cluster

Kubernetes

FastAPI pods

Authentication, API, request validation

LangGraph pods

Supervisor, agents, workflow execution

Celery workers

Document ingestion, embeddings

Scheduler / consumers

Async jobs and event processing

AWS managed services and external inference

RDS PostgreSQL

ElastiCache Redis

S3

Qdrant

Secrets Manager / KMS

Hosted LLM API

Observability

CloudWatch · OpenTelemetry · Prometheus · Grafana

The key design decision is that Kubernetes should run stateless application workloads, while databases, caching, and object storage should generally be managed separately.

For a financial application, I would also use private subnets for application and data workloads, encrypted connections, workload-specific IAM permissions, and tenant-aware authorization.

## 2. Understand the deployment components

| Component                       | Responsibility                               |
| ------------------------------- | -------------------------------------------- |
| Docker                          | Packages Python application and dependencies |
| Amazon ECR                      | Stores Docker images                         |
| Amazon EKS                      | Runs Kubernetes workloads                    |
| Kubernetes Deployment           | Maintains the desired number of pods         |
| Kubernetes Service              | Provides internal service discovery          |
| AWS Load Balancer Controller    | Creates and manages ALBs                     |
| HPA                             | Scales pods based on resource metrics        |
| Cluster Autoscaler or Karpenter | Adds/removes worker nodes                    |
| AWS Secrets Manager             | Stores API keys and credentials              |
| CloudWatch                      | AWS logs, metrics and alarms                 |

Docker and Kubernetes solve different problems.

- Docker packages your application so it runs consistently.
- Kubernetes schedules and manages containers across machines.
- EKS provides the managed Kubernetes control plane on AWS.

Amazon EKS does not automatically deploy your application simply because you created a cluster. You still need images, Kubernetes manifests, networking, IAM permissions, and a deployment pipeline.

## 3. Prepare the application for deployment

First, I would separate the application into deployable services.

A possible repository structure:

```
financial-advisor-copilot/
│
├── app/
│   ├── main.py
│   ├── api/
│   ├── agents/
│   ├── workflows/
│   ├── services/
│   ├── repositories/
│   ├── security/
│   ├── schemas/
│   └── config/
│
├── workers/
│   ├── celery_app.py
│   └── tasks/
│
├── tests/
│
├── Dockerfile
├── requirements.txt
├── .dockerignore
│
├── k8s/
│   ├── namespace.yaml
│   ├── api-deployment.yaml
│   ├── api-service.yaml
│   ├── worker-deployment.yaml
│   ├── hpa.yaml
│   └── ingress.yaml
│
└── .github/
    └── workflows/
        └── deploy.yml
```

The API and background workers should be independently deployable. I would build separate Docker images if their dependencies or runtime requirements differ; alternatively, one image can have different entry points.

### Step 1: Create the FastAPI entry point

`app/main.py`

```
from contextlib import asynccontextmanagerfrom fastapi import FastAPIfrom fastapi.middleware.cors import CORSMiddlewarefrom app.api.routes import routerfrom app.config.settings import settings@asynccontextmanagerasync def lifespan(app: FastAPI):    # Initialize shared resources here.    # For example: database pools, Redis clients,    # HTTP clients and workflow dependencies.    yield    # Close resources gracefully here.app = FastAPI(    title="Financial Advisor Copilot",    version="1.0.0",    lifespan=lifespan,)app.add_middleware(    CORSMiddleware,    allow_origins=settings.allowed_origins,    allow_methods=["GET", "POST"],    allow_headers=["Authorization", "Content-Type"],)app.include_router(router, prefix="/api/v1")
```

The readiness endpoint above is a placeholder. Before production, it should check that the application has initialized and can reach the dependencies required to serve requests.

I keep liveness and readiness separate:

- Liveness: Is the process alive? If not, Kubernetes may restart it.
- Readiness: Can this pod safely receive traffic? If not, Kubernetes removes it from service endpoints.

Do not make liveness depend on PostgreSQL, Redis, or the LLM provider. A temporary dependency outage should not cause Kubernetes to continuously restart otherwise healthy application processes.

### Step 2: Create the Dockerfile

`Dockerfile`

```
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install only required system dependencies.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies separately for better caching.
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy application source.
COPY app/ ./app/
COPY workers/ ./workers/

# Run as a non-root user.
RUN useradd --create-home appuser \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

CMD [
    "uvicorn",
    "app.main:app",
    "--host", "0.0.0.0",
    "--port", "8000"
]
```

A few production considerations:

1. Pin dependencies and use a lock file.
2. Use a trusted, patched base image and scan it.
3. Avoid putting API keys inside the Docker image.
4. Do not run the container as root.
5. Use multi-stage builds if compilation tools are needed.
6. Pin the image by digest in production deployments.
7. Run multiple Uvicorn workers only after benchmarking CPU and memory use. Kubernetes replicas may be a better scaling unit.

Create `.dockerignore`:

```
.git
.github
.venv
venv
__pycache__
.pytest_cache
.env
.env.*
*.pyc
*.log
tests/
```

Never include `.env` files, private keys, credentials, or customer documents in the image.

### Step 3: Build and run locally

```
docker build -t financial-advisor-api:1.0 .

docker run --rm \
  -p 8000:8000 \
  -e ENVIRONMENT=local \
  financial-advisor-api:1.0
```

Test:

```
curl http://localhost:8000/health/live
```

Expected response:

```
{
  "status": "alive"
}
```

At this stage, I would also run integration tests against test instances of PostgreSQL, Redis, and Qdrant before pushing the image.

## 4. Create the AWS infrastructure

I would use Terraform for repeatable infrastructure provisioning.

### Step 1: Create a VPC

Create a VPC with:

- Public subnets across at least two Availability Zones for the internet-facing load balancer.
- Private subnets across at least two Availability Zones for EKS worker nodes and pods.
- Isolated private subnets for databases where appropriate.
- NAT gateways or controlled egress for workloads that need outbound internet.
- VPC endpoints for services such as S3, ECR, CloudWatch, and Secrets Manager where appropriate.

Example architecture:

```
AWS Region
   |
   +---- VPC
           |
           +---- Public Subnet AZ-A
           |       ALB
           |
           +---- Public Subnet AZ-B
           |       ALB
           |
           +---- Private Subnet AZ-A
           |       EKS Worker Nodes
           |
           +---- Private Subnet AZ-B
           |       EKS Worker Nodes
           |
           +---- Isolated Data Subnets
                   RDS PostgreSQL
```

The EKS control plane is AWS-managed. The worker nodes and pods run in your VPC.

I would plan IP capacity carefully: EKS pods use VPC networking through the Amazon VPC CNI, and subnet IP exhaustion can prevent new pods from starting.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://docs.aws.amazon.com\&sz=32)

Amazon EKS+1



### Step 2: Create the EKS cluster

For a learning environment, `eksctl` provides a straightforward way to create a cluster.

```
eksctl create cluster \
  --name financial-ai-cluster \
  --region ap-south-1 \
  --version 1.34 \
  --nodegroup-name ai-workers \
  --node-type m7i.large \
  --nodes 2 \
  --nodes-min 2 \
  --nodes-max 6 \
  --managed
```

The Kubernetes version and instance type must be supported in the selected AWS region when you actually create the cluster. Treat these as example configuration, not a guaranteed current availability combination.

For production, I would provision the VPC, EKS cluster, managed node groups, IAM, security groups, logging, and add-ons using version-controlled Terraform modules rather than relying on an interactive cluster creation command.

Connect to the cluster:

```
aws eks update-kubeconfig \
  --region ap-south-1 \
  --name financial-ai-cluster

kubectl get nodes
```

You should see your worker nodes in `Ready` status.

### Step 3: Create an ECR repository

Amazon Elastic Container Registry stores the Docker images.

```
aws ecr create-repository \
  --repository-name financial-advisor-api \
  --region ap-south-1
```

Authenticate Docker:

```
aws ecr get-login-password \
  --region ap-south-1 \
  | docker login \
      --username AWS \
      --password-stdin \
      YOUR_ACCOUNT_ID.dkr.ecr.ap-south-1.amazonaws.com
```

Build and tag:

```
docker build \
  -t financial-advisor-api:1.0 .

docker tag \
  financial-advisor-api:1.0 \
  YOUR_ACCOUNT_ID.dkr.ecr.ap-south-1.amazonaws.com/financial-advisor-api:1.0
```

Push:

```
docker push \
  YOUR_ACCOUNT_ID.dkr.ecr.ap-south-1.amazonaws.com/financial-advisor-api:1.0
```

Now the image is available to your EKS cluster.

In production, I would tag images with an immutable Git commit SHA, enable ECR image scanning and lifecycle policies, and deploy by digest rather than relying on a mutable `latest` tag.

## 5. Deploy the Docker container to Kubernetes

Now we move to the Kubernetes layer.

I would create a namespace to isolate the application:

`k8s/namespace.yaml`

```
apiVersion: v1
kind: Namespace
metadata:
  name: financial-ai
```

Apply it:

```
kubectl apply -f k8s/namespace.yaml
```

### Step 1: Create a Kubernetes Deployment

A Deployment defines how many replicas of your application should run.

`k8s/api-deployment.yaml`

```
apiVersion: apps/v1
kind: Deployment

metadata:
  name: financial-advisor-api
  namespace: financial-ai

spec:
  replicas: 3

  selector:
    matchLabels:
      app: financial-advisor-api

  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0

  template:
    metadata:
      labels:
        app: financial-advisor-api

    spec:
      serviceAccountName: financial-advisor-api

      terminationGracePeriodSeconds: 45

      containers:
        - name: api

          image: YOUR_ACCOUNT_ID.dkr.ecr.ap-south-1.amazonaws.com/financial-advisor-api:1.0

          ports:
            - containerPort: 8000

          env:
            - name: ENVIRONMENT
              value: production

            - name: LOG_LEVEL
              value: INFO

          resources:
            requests:
              cpu: "500m"
              memory: "1Gi"

            limits:
              cpu: "2"
              memory: "3Gi"

          readinessProbe:
            httpGet:
              path: /health/ready
              port: 8000
            initialDelaySeconds: 10
            periodSeconds: 10
            timeoutSeconds: 3
            failureThreshold: 3

          livenessProbe:
            httpGet:
              path: /health/live
              port: 8000
            initialDelaySeconds: 20
            periodSeconds: 15
            timeoutSeconds: 3
            failureThreshold: 3

          lifecycle:
            preStop:
              exec:
                command: ["/bin/sh", "-c", "sleep 10"]

          securityContext:
            allowPrivilegeEscalation: false
            readOnlyRootFilesystem: true
            runAsNonRoot: true
            runAsUser: 1000
            capabilities:
              drop:
                - ALL
```

A few important points:

Replicas = 3

Kubernetes tries to maintain three running API pods.

If one pod crashes, the Deployment controller creates a replacement.

Resource requests

```
requests:
  cpu: "500m"
  memory: "1Gi"
```

These are the resources Kubernetes uses when scheduling the pod.

Resource limits

```
limits:
  cpu: "2"
  memory: "3Gi"
```

These cap container resource consumption. CPU can be throttled; exceeding the memory limit can result in an OOM kill.

Rolling updates

```
maxSurge: 1
maxUnavailable: 0
```

Kubernetes can start an additional pod before removing an existing one, provided the cluster has capacity and the readiness checks succeed.

The `readOnlyRootFilesystem` setting requires the application not to write to its root filesystem. If temporary files are needed, mount a writable `emptyDir` at a specific path such as `/tmp`.

### Step 2: Create a Kubernetes Service

The Deployment manages pods, but we need a stable internal endpoint to access them.

`k8s/api-service.yaml`

```
apiVersion: v1
kind: Service

metadata:
  name: financial-advisor-service
  namespace: financial-ai

spec:
  type: ClusterIP

  selector:
    app: financial-advisor-api

  ports:
    - name: http
      port: 80
      targetPort: 8000
      protocol: TCP
```

Now other services inside Kubernetes can call:

```
http://financial-advisor-service.financial-ai.svc.cluster.local
```

### Step 3: Deploy the API

```
kubectl apply -f k8s/api-deployment.yaml

kubectl apply -f k8s/api-service.yaml
```

Verify:

```
kubectl get pods -n financial-ai

kubectl get deployments -n financial-ai

kubectl get services -n financial-ai
```

Check logs:

```
kubectl logs \
  deployment/financial-advisor-api \
  -n financial-ai
```

For a specific pod:

```
kubectl logs POD_NAME -n financial-ai
```

### Step 4: Create an Ingress and ALB

I would use the AWS Load Balancer Controller to provision an Application Load Balancer from a Kubernetes Ingress resource.

It requires an appropriately configured IAM role and Kubernetes service account. AWS recommends workload-specific permissions rather than granting broad permissions to worker-node roles.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub+1



Example:

`k8s/ingress.yaml`

```
apiVersion: networking.k8s.io/v1
kind: Ingress

metadata:
  name: financial-advisor-ingress
  namespace: financial-ai

  annotations:
    alb.ingress.kubernetes.io/scheme: internet-facing
    alb.ingress.kubernetes.io/target-type: ip
    alb.ingress.kubernetes.io/listen-ports: '[{"HTTPS":443}]'
    alb.ingress.kubernetes.io/certificate-arn: YOUR_ACM_CERTIFICATE_ARN
    alb.ingress.kubernetes.io/healthcheck-path: /health/ready

spec:
  ingressClassName: alb

  rules:
    - host: api.example.com

      http:
        paths:
          - path: /
            pathType: Prefix

            backend:
              service:
                name: financial-advisor-service
                port:
                  number: 80
```

Apply:

```
kubectl apply -f k8s/ingress.yaml

kubectl get ingress -n financial-ai
```

The controller provisions the ALB and connects it to the service's pod targets.

The resulting traffic flow is:

```
Client
  |
  v
Route 53
  |
  v
Application Load Balancer
  |
  v
Kubernetes Service
  |
  v
FastAPI Pod
```

For production, I would configure WAF, TLS, access logging, appropriate ALB security groups, rate limiting, and network restrictions. An internet-facing ALB does not mean the EKS worker nodes or databases should be publicly accessible.

## 6. Configure secrets and AWS IAM correctly

This is one of the most important parts of deploying financial AI applications.

Never put credentials directly into:

- Dockerfiles.
- Git repositories.
- Kubernetes YAML files.
- Docker image environment variables during image build.
- Application logs.

For example, you might need:

```
OPENAI_API_KEY
DATABASE_URL
REDIS_URL
QDRANT_API_KEY
JWT_SECRET
```

I would store these in AWS Secrets Manager and provide the application with access through EKS Pod Identity or IRSA.

### Example: Kubernetes service account

```
apiVersion: v1
kind: ServiceAccount

metadata:
  name: financial-advisor-api
  namespace: financial-ai
```

Associate this service account with a dedicated IAM role using EKS Pod Identity or IRSA.

For example, the API might receive permission to read only its required secrets and access specific S3 buckets. The document ingestion worker should have a separate role with only the S3 permissions it needs.

AWS documents both workload identity options and recommends giving each application its own appropriately scoped IAM role.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://docs.aws.amazon.com\&sz=32)

Amazon EKS+1



The important principle:

The API pod should not automatically inherit all permissions of the EKS worker node.

For secret injection, use the AWS Secrets and Configuration Provider for the Secrets Store CSI Driver or a controlled external-secrets integration. Configure encryption, rotation, and access policies. Do not assume that a mounted secret automatically updates an application's already-loaded environment variable; applications must handle rotation appropriately.

## 7. Deploy Celery workers separately

Your document ingestion and embedding jobs should not run inside the FastAPI request-serving process.

For example:

```
FastAPI API
    |
    v
Redis / Celery broker
    |
    v
Celery workers
    |
    +---- Parse PDF
    |
    +---- Chunk documents
    |
    +---- Generate embeddings
    |
    +---- Upsert Qdrant
    |
    +---- Update PostgreSQL
```

I would deploy workers using a separate Kubernetes Deployment with independent resource limits and scaling.

`k8s/worker-deployment.yaml`

```
apiVersion: apps/v1
kind: Deployment

metadata:
  name: ingestion-worker
  namespace: financial-ai

spec:
  replicas: 2

  selector:
    matchLabels:
      app: ingestion-worker

  template:
    metadata:
      labels:
        app: ingestion-worker

    spec:
      serviceAccountName: financial-ingestion-worker

      containers:
        - name: worker

          image: YOUR_ACCOUNT_ID.dkr.ecr.ap-south-1.amazonaws.com/financial-advisor-api:1.0

          command:
            - celery
            - -A
            - workers.celery_app
            - worker
            - --loglevel=INFO
            - --concurrency=4

          resources:
            requests:
              cpu: "1"
              memory: "2Gi"

            limits:
              cpu: "2"
              memory: "4Gi"
```

Here, `workers.celery_app` must expose the Celery application object, and the image must contain the worker code and its dependencies.

The API and worker use the same image in this example, but they are independently deployed and scaled. In a larger system, I would likely build separate images to avoid shipping unnecessary dependencies.

Staff-level consideration: A Celery task must be idempotent and resumable. If a worker crashes after writing vectors but before updating PostgreSQL, a retry should not create duplicate chunks or corrupt document state. Use stable IDs, durable task state, and reconciliation.

## 8. Configure Kubernetes autoscaling

There are two different scaling problems.

1. Scaling application pods.
2. Scaling the underlying EC2 worker nodes.

### A. Horizontal Pod Autoscaler (HPA)

Suppose API traffic increases from 100 requests per second to 1,000 requests per second.

Instead of manually increasing replicas, Kubernetes can scale the API based on resource utilization.

`k8s/hpa.yaml`

```
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler

metadata:
  name: financial-advisor-api-hpa
  namespace: financial-ai

spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: financial-advisor-api

  minReplicas: 3
  maxReplicas: 15

  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 65

  behavior:
    scaleUp:
      stabilizationWindowSeconds: 30
    scaleDown:
      stabilizationWindowSeconds: 180
```

Apply:

```
kubectl apply -f k8s/hpa.yaml

kubectl get hpa -n financial-ai
```

HPA needs Kubernetes metrics infrastructure, typically Metrics Server for CPU and memory metrics.

For LLM applications, CPU utilization alone is not sufficient. I would also consider:

- Requests per second.
- Active requests.
- Request queue depth.
- Concurrent LLM calls.
- Worker queue backlog.
- GPU utilization for self-hosted inference.

These can be exposed through custom or external metrics adapters.

### B. Node autoscaling

What happens if HPA requests 15 pods but the cluster only has enough capacity for 5?

The additional pods remain pending until Kubernetes has capacity.

I would use Karpenter or Cluster Autoscaler to provision additional worker nodes.

```
Incoming traffic increases
          |
          v
HPA increases pod replicas
          |
          v
Insufficient node capacity?
          |
          v
Karpenter provisions EC2 nodes
          |
          v
Pods are scheduled
```

For production, I would evaluate Karpenter against Cluster Autoscaler based on workload patterns, instance flexibility, availability requirements, and operational maturity. AWS maintains EKS guidance for cluster autoscaling and reliability.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://docs.aws.amazon.com\&sz=32)

Amazon EKS



## 9. Deploy PostgreSQL, Redis, Qdrant and S3

I would not put every stateful dependency inside the same Kubernetes cluster just because EKS is available.

For your architecture, my preferred deployment choices would be:

| Component            | AWS deployment choice                       | Reason                                       |
| -------------------- | ------------------------------------------- | -------------------------------------------- |
| PostgreSQL           | Amazon RDS PostgreSQL                       | Managed backups, replication and maintenance |
| Redis                | Amazon ElastiCache                          | Managed caching and broker infrastructure    |
| Documents            | Amazon S3                                   | Durable object storage                       |
| Vector database      | Qdrant Cloud or dedicated Qdrant deployment | Vector retrieval                             |
| Workflow checkpoints | PostgreSQL-backed durable checkpoint store  | Workflow recovery                            |
| Secrets              | AWS Secrets Manager                         | Credential management                        |

### Why not deploy PostgreSQL and Redis directly inside Kubernetes?

You can deploy them using StatefulSets and persistent volumes, but then you are responsible for:

- Persistent storage.
- Backups and restoration.
- Replication.
- Failover.
- Database upgrades.
- Storage performance.
- Recovery testing.

For a financial application, I would generally prefer managed database services.

Qdrant is a separate decision. You can use Qdrant Cloud or self-host Qdrant with persistent EBS volumes and carefully designed backups, replication, and availability. Do not deploy a single Qdrant pod with ephemeral storage and assume your vector index is durable.

### Document ingestion flow

```
User uploads PDF
       |
       v
FastAPI API
       |
       v
Store original file in S3
       |
       v
Create ingestion job in PostgreSQL
       |
       v
Publish job to queue
       |
       v
Celery worker
       |
       v
Parse PDF
       |
       v
Chunk document
       |
       v
Generate embeddings
       |
       v
Upsert vectors into Qdrant
       |
       v
Update PostgreSQL ingestion status
```

The ingestion pipeline should use document versioning, idempotent chunk IDs, retry handling, and reconciliation between PostgreSQL and Qdrant.

## 10. Implement CI/CD using GitHub Actions

I would automate Docker image creation, testing, pushing to ECR, and deploying to EKS.

The pipeline would look like this:

```
Developer pushes code
        |
        v
GitHub Actions
        |
        v
Run unit tests
        |
        v
Run integration tests
        |
        v
Build Docker image
        |
        v
Scan image for vulnerabilities
        |
        v
Push image to ECR
        |
        v
Deploy to EKS
        |
        v
Wait for rollout
        |
        v
Run smoke tests
        |
        v
Deployment complete
```

### Example GitHub Actions workflow

`.github/workflows/deploy.yml`

```
name: Deploy Financial AI Application

on:
  push:
    branches:
      - main

permissions:
  id-token: write
  contents: read

env:
  AWS_REGION: ap-south-1
  ECR_REPOSITORY: financial-advisor-api
  EKS_CLUSTER: financial-ai-cluster

jobs:

  deploy:

    runs-on: ubuntu-latest

    steps:

      - name: Checkout code
        uses: actions/checkout@v4

      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: ${{ secrets.AWS_DEPLOY_ROLE_ARN }}
          aws-region: ${{ env.AWS_REGION }}

      - name: Login to ECR
        id: login-ecr
        uses: aws-actions/amazon-ecr-login@v2

      - name: Build Docker image
        run: |
          docker build \
            -t ${{ steps.login-ecr.outputs.registry }}/${{ env.ECR_REPOSITORY }}:${{ github.sha }} .

      - name: Push image
        run: |
          docker push \
            ${{ steps.login-ecr.outputs.registry }}/${{ env.ECR_REPOSITORY }}:${{ github.sha }}

      - name: Configure kubectl
        run: |
          aws eks update-kubeconfig \
            --name $EKS_CLUSTER \
            --region $AWS_REGION

      - name: Deploy application
        run: |
          kubectl set image deployment/financial-advisor-api \
            api=${{ steps.login-ecr.outputs.registry }}/${{ env.ECR_REPOSITORY }}:${{ github.sha }} \
            -n financial-ai

      - name: Verify deployment
        run: |
          kubectl rollout status \
            deployment/financial-advisor-api \
            -n financial-ai \
            --timeout=300s
```

This is a starting template, not a complete hardened pipeline. I would pin GitHub Actions to reviewed commit SHAs, add tests and image scanning, configure an EKS deployment role with least privilege, and use Helm or Kustomize for environment-specific configuration.

For production, I would also introduce a manual approval gate before production deployment, smoke tests, and an automated rollback strategy. Database migrations should be deployed as controlled, backward-compatible steps rather than being run independently by every API replica.

## 11. Monitoring and observability

Once the application is running in EKS, I need visibility into both infrastructure and AI-specific behavior.

I would use:

| Layer      | Monitoring                                          |
| ---------- | --------------------------------------------------- |
| Kubernetes | Pod restarts, CPU, memory, pending pods             |
| FastAPI    | Request rate, error rate, P95/P99 latency           |
| LangGraph  | Node execution time, retries, checkpoints           |
| RAG        | Retrieval latency, Recall\@K evaluation samples     |
| LLM        | Tokens, provider latency, timeout and error rates   |
| Redis      | Memory, connections, cache hit rate                 |
| PostgreSQL | Connections, slow queries, CPU                      |
| Qdrant     | Search latency, resource usage, failures            |
| Celery     | Queue depth, task failures, retry count             |
| Security   | Unauthorized tool attempts, tenant isolation alerts |

For example:

```
FastAPI
   |
   v
OpenTelemetry instrumentation
   |
   v
Distributed trace
   |
   +---- PostgreSQL
   |
   +---- Redis
   |
   +---- Qdrant
   |
   +---- LangGraph nodes
   |
   +---- LLM provider
   |
   v
Tracing backend / CloudWatch
```

I would propagate a trace ID across API requests, LangGraph nodes, and background jobs.

For an LLM request, I would want to investigate:

```
Request ID: req-123

API validation:       10 ms
Authentication:       15 ms
Vector retrieval:    150 ms
Reranking:            90 ms
Supervisor agent:    300 ms
Research agent:      800 ms
LLM generation:     1800 ms
Final response:       50 ms
--------------------------------
Total:              3215 ms
```

This is an illustrative trace, not a performance benchmark.

It helps me identify whether the delay is caused by retrieval, agent execution, external inference, or application overhead.

I would avoid putting raw financial documents, customer PII, full prompts, or secrets into CloudWatch logs and tracing attributes. Use redacted telemetry, access-controlled trace storage, and retention policies.

## 12. How do I handle LLM inference on AWS?

There are two deployment approaches, and this is an important architectural decision.

My preferred starting point

Option A: Use an external hosted LLM API

[Multi-tenant generative AI platform scenario - Generative AI Lens](https://images.openai.com/static-rsc-4/nQeZbRzXyeq9E6YJ804amA9eWMGRuuitmdQtOWl2Pv8mA9ZD0RSzFcsYPyL804HgzSrLUbIDhZQs-IWE7hVg7AdRmq0FOrUrDV8BD_iuOIW-kK15OzLixPjmgLWAGZwSKrfRr_lIVGn5Lbom5p4TL9obNoIbxUUt11nN5DthbL4?purpose=inline)

[docs.aws.amazon.com](https://docs.aws.amazon.com/wellarchitected/latest/generative-ai-lens/multi-tenant-generative-ai-platform-scenario.html?utm_source=chatgpt.com)

- EKS runs FastAPI and LangGraph.
- Requests go to OpenAI, Anthropic, or another provider.
- No GPU cluster is needed for model inference.
- Easier initial deployment and model upgrades.
- Requires provider availability, network egress, and data governance controls.

Option B: Host your own LLM on AWS

[NVIDIA NIM LLM on Amazon EKS | AI on EKS](https://images.openai.com/static-rsc-4/q5fqDGSFis_Y8xRsC_hEQS3TadNVfNskY4wqbx70DyQhmknHLB_zfv2YK-I8zfvLjU3FosobVWh8A7EpSqVoPVXbcDxQ_yKb9LjCzURVOmcT45atMHL25F1U2CnKg18PvwB_ZtlAFojT-oiYa0omKELmUVCcriTmcOdaonKD_w4?purpose=inline)

[awslabs.github.io](https://awslabs.github.io/ai-on-eks/docs/blueprints/inference/framework-guides/GPUs/nvidia-nim-llama3?utm_source=chatgpt.com)

- Deploy models using vLLM or another inference server.
- Run inference pods on GPU-backed EC2 nodes.
- Scale GPU workloads separately from API pods.
- Greater control over model hosting and data residency.
- Higher infrastructure, model-serving and operations complexity.

For your current financial advisor copilot, I would begin with hosted inference and use EKS for the application and agent workflows.

If inference volume, data residency, cost, or model customization justify self-hosting, I would introduce a separate GPU node group and inference service.

Do not mix API scaling and GPU inference scaling. They have different resource requirements, scaling behavior, and cost profiles.

## 13. Production security architecture

For your financial application, I would enforce these controls:

1. Network security: EKS workloads and databases in private subnets, with restricted ingress and egress.
2. IAM: Dedicated service accounts and least-privilege IAM roles.
3. Secrets: Secrets Manager, KMS encryption, and controlled secret rotation.
4. Image security: ECR scanning, patched base images, non-root containers.
5. API security: JWT validation, RBAC, rate limiting, WAF and TLS.
6. Data security: Tenant-scoped PostgreSQL queries and Qdrant filters, plus authorization checks.
7. Agent security: Tool gateway, allowlisted tools, argument validation and durable HITL approval.
8. Audit: Immutable or tamper-resistant audit records for sensitive operations.

For tenant isolation, I would not assume that Kubernetes namespaces alone provide sufficient data isolation. Tenant authorization must be enforced at the application, database, vector retrieval, cache, and object-storage layers.

## 14. Complete deployment sequence

Here is the practical order I would follow when deploying this project.

| Step | Action                           | AWS / technology                   |
| ---- | -------------------------------- | ---------------------------------- |
| 1    | Create VPC and subnets           | VPC, Terraform                     |
| 2    | Create cluster                   | EKS                                |
| 3    | Create worker nodes              | EC2 managed node groups            |
| 4    | Configure workload IAM           | EKS Pod Identity / IRSA            |
| 5    | Create image repository          | ECR                                |
| 6    | Build and push API image         | Docker                             |
| 7    | Provision database               | RDS PostgreSQL                     |
| 8    | Provision cache                  | ElastiCache                        |
| 9    | Configure document storage       | S3                                 |
| 10   | Configure vector database        | Qdrant                             |
| 11   | Install load balancer controller | AWS Load Balancer Controller       |
| 12   | Deploy API and workers           | Kubernetes                         |
| 13   | Configure external traffic       | ALB, Route 53, ACM                 |
| 14   | Configure scaling                | HPA, Karpenter                     |
| 15   | Configure observability          | CloudWatch, OpenTelemetry          |
| 16   | Configure CI/CD                  | GitHub Actions                     |
| 17   | Run release tests                | Pytest, RAG evaluation, load tests |
| 18   | Canary deploy                    | Production rollout                 |

Useful official references:

- [Amazon EKS Best Practices Guide](https://docs.aws.amazon.com/eks/latest/best-practices/introduction.html?utm_source=chatgpt.com)
- [EKS Networking Best Practices](https://docs.aws.amazon.com/eks/latest/best-practices/networking.html?utm_source=chatgpt.com)
- [EKS workload IAM permissions](https://docs.aws.amazon.com/eks/latest/userguide/service-accounts.html?utm_source=chatgpt.com)
- [AWS Load Balancer Controller installation](https://github.com/kubernetes-sigs/aws-load-balancer-controller/blob/main/docs/deploy/installation.md?utm_source=chatgpt.com)

## 15. How I would explain this in a Senior/Staff AI Engineer interview

Interviewer: How would you deploy an LLM application on AWS using Docker and Kubernetes?

"I would first containerize the application using Docker. I would create a lightweight Python image for the FastAPI and LangGraph application, use a non-root user, pin dependencies, and ensure that secrets are never included in the image.

I would push the versioned image to Amazon ECR and deploy it to Amazon EKS.

At the infrastructure level, I would create a VPC with public and private subnets across multiple Availability Zones. The Application Load Balancer would receive HTTPS traffic and route it to Kubernetes services, while the application pods run in private subnets.

I would deploy FastAPI, agent workflows, and Celery workers as separate Kubernetes workloads. I would use Deployments, Services, readiness and liveness probes, resource requests and limits, and Horizontal Pod Autoscalers.

For stateful services, I would generally use Amazon RDS for PostgreSQL, ElastiCache for Redis, S3 for documents, and a managed or appropriately operated Qdrant deployment for vector search.

For security, I would use IAM roles associated with Kubernetes service accounts, Secrets Manager, KMS, private networking, and tenant-level authorization.

For scalability, I would use HPA for pods and Karpenter or Cluster Autoscaler for nodes. I would scale API pods, ingestion workers, and GPU inference workloads independently.

Finally, I would implement CI/CD using GitHub Actions, ECR, and Kubernetes deployments. I would integrate automated tests, image scanning, LLM evaluations, canary releases, CloudWatch, and OpenTelemetry.

My goal would be to build a secure, highly available, observable, cost-efficient, and independently scalable AI platform."

## 16. What I would prioritize as a Staff Engineer

I would not start by creating a massive Kubernetes cluster. I would build incrementally.

| Phase   | Deliverable                                      |
| ------- | ------------------------------------------------ |
| Phase 1 | Dockerize FastAPI and run locally                |
| Phase 2 | Push image to ECR                                |
| Phase 3 | Deploy API to EKS                                |
| Phase 4 | Connect RDS, Redis, S3 and Qdrant                |
| Phase 5 | Deploy Celery and LangGraph workloads            |
| Phase 6 | Add ALB, IAM, secrets and network security       |
| Phase 7 | Add autoscaling and observability                |
| Phase 8 | Automate CI/CD, evaluation and canary deployment |

My recommendation: Build a complete deployment for one FastAPI service first. Then introduce the LangGraph workers, ingestion workers, and independent scaling. This makes troubleshooting substantially easier and gives you a strong end-to-end AWS deployment project to discuss in Senior AI Engineer interviews.