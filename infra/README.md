# AWS Deployment Plan

This repository intentionally does not include executable Terraform.
The case study deliverable is a runnable model/API repository, and an
untested Terraform stack would add review surface without proving the
service can be operated correctly. The plan below describes the AWS
resources I would provision for production and the boundaries where IaC
would be introduced.

## Target Architecture

```text
Client
  -> Route 53 / ACM
  -> Application Load Balancer
  -> ECS Fargate service running the FastAPI container
  -> CloudWatch Logs
  -> Secrets Manager / SSM Parameter Store
  -> S3 model artifact bucket
```

For million-document backfills:

```text
Client
  -> API submits job metadata
  -> SQS queue
  -> ECS/Fargate worker service
  -> S3 input/output artifacts
  -> DynamoDB or Postgres job status table
```

## Core Resources

| Area | AWS resource | Purpose |
|---|---|---|
| Container registry | ECR | Stores the Docker image built from this repo |
| Runtime | ECS Fargate | Runs stateless API tasks without managing EC2 hosts |
| Traffic | Application Load Balancer | TLS termination, health checks, path routing |
| Networking | VPC, public/private subnets, security groups | Isolates the service and controls ingress/egress |
| Secrets | Secrets Manager or SSM Parameter Store | Stores `SECURITY__API_KEY` and other sensitive runtime config |
| Logs | CloudWatch Logs | Receives structured JSON application logs |
| Model artifact | S3 | Stores versioned `document_classifier.joblib` artifacts |
| Autoscaling | ECS Service Auto Scaling | Scales tasks on CPU, memory, or ALB request count |

## Runtime Configuration

The container should receive configuration through environment variables:

```bash
ENV_FOR_DYNACONF=production
MODEL_PATH=/app/models/document_classifier.joblib
API__MAX_BATCH_SIZE=100
LOGGING__JSON=true
SECURITY__API_KEY_ENABLED=true
SECURITY__API_KEY=<from Secrets Manager>
```

The model artifact can be handled in one of two ways:

1. Bake a specific model artifact into an image tag for immutable
   releases.
2. Download a versioned S3 artifact into `/app/models/` during task
   startup.

For this project I would start with the mounted/copied artifact approach
used by the Docker instructions, then move to S3 versioned artifacts
once retraining cadence is established.

## Health Checks

The load balancer should call:

```text
GET /health
```

Expected healthy response:

```json
{"status": "ok", "model_loaded": true}
```

The process can be alive while `model_loaded` is false. In production I
would configure an application-level alarm for `model_loaded=false` so a
bad deployment does not silently serve 503 classification responses.

## Observability

The application already emits JSON logs with:

- request ID
- path and status code
- latency
- label, confidence, and decision
- document length and SHA-256 hash
- batch size and label/decision distributions for batch requests

CloudWatch Logs can ingest those records directly. Next production
steps would be:

- CloudWatch metric filters or embedded metric format for request
  counts, latency, error rate, fallback rate, and batch size.
- OpenTelemetry traces if this service becomes part of a larger
  workflow.
- Alarms on 5xx rate, p95 latency, task restarts, and fallback-rate
  drift.

Raw document text should never be logged or stored for observability.

## Security

Current application-level protection is optional static API-key auth via
`X-API-Key`. In AWS I would also add:

- TLS via ACM on the ALB.
- Secrets Manager or SSM for the API key.
- IAM roles scoped to only the needed S3 model artifact path.
- Private ECS tasks with public ingress only through the ALB.
- AWS WAF or gateway-level rate limiting if the API is internet-facing.

If this became a multi-tenant product, API keys should move from a
single secret to hashed keys in a database with revocation, scopes,
created/last-used timestamps, and per-client rate limits.

## Scaling Path

For regular API traffic, scale ECS tasks horizontally. The service is
stateless and the model is read-only in each worker, so additional tasks
can be added behind the load balancer.

For large backfills, use asynchronous processing:

1. Client uploads input documents or document references to S3.
2. API creates a job and enqueues batch pointers to SQS.
3. Worker tasks classify batches with `predict_batch`.
4. Workers write results to S3 and job status to DynamoDB or Postgres.
5. Client polls job status or receives a completion notification.

This avoids long synchronous HTTP requests and makes retries,
backpressure, and partial failures explicit.

## Terraform Boundary

If Terraform were added, I would split modules roughly as:

```text
infra/terraform/
  modules/
    network/
    ecr/
    ecs_service/
    alb/
    secrets/
    observability/
  envs/
    staging/
    production/
```

Required inputs would include image tag, model artifact version, desired
task count, CPU/memory, environment name, and domain/TLS settings.

I did not include this Terraform stack here because it cannot be safely
validated without AWS account context, networking decisions, domain
ownership, and deployment credentials. The Dockerfile plus this plan are
the portable production boundary for the case-study repository.
