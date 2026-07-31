# Telco Customer Churn Prediction — Deployment Guide

---

## 1. Deployment Overview

This guide covers deploying the Telco Customer Churn prediction model as a containerized FastAPI service. The deployment target is a production-ready REST API that returns churn probability and a binary risk label for one or many customers.

| Component | Technology | Purpose |
|---|---|---|
| API Framework | FastAPI + Uvicorn | Low-latency, OpenAPI-documented inference service |
| Containerization | Docker | Consistent, portable runtime environment |
| Model Registry | MLflow | Versioned model artifacts and metadata |
| Logging | Python `logging` + structlog | Structured, queryable logs |
| Monitoring | Prometheus metrics + optional health endpoint | Observability and alerting |
| Orchestration | Docker Compose / Kubernetes | Local and production-scale deployment |

---

## 2. Requirements

### 2.1 Infrastructure
- Linux/Windows host with Docker Engine 24.0+ or Kubernetes 1.27+.
- At least 2 vCPU, 4 GB RAM, and 5 GB disk per container instance.
- Outbound network access only if MLflow tracking or external model store is remote.

### 2.2 Software
| Tool | Minimum Version | Purpose |
|---|---|---|
| Python | 3.11 | Runtime for local development |
| Poetry | 1.7 | Dependency and environment management |
| Docker | 24.0 | Container build and runtime |
| Docker Compose | 2.20+ | Multi-service local deployment |
| (Optional) kubectl | 1.27+ | Kubernetes deployment |

### 2.3 Model Artifact
A trained model must be available in one of the following forms:
- Local path: `models/churn_model/` (MLflow model directory with `MLmodel` file).
- MLflow tracking URI: `mlflow-artifacts:/...` or remote S3/Azure blob storage.
- Direct path to a serialized model + preprocessing pipeline.

---

## 3. Environment Variables

All runtime configuration is supplied through environment variables.

| Variable | Required | Default | Description |
|---|---|---|---|
| `MODEL_PATH` | Yes | — | Path to the trained model artifact or MLflow run URI |
| `APP_HOST` | No | `0.0.0.0` | Host interface the server binds to |
| `APP_PORT` | No | `8000` | Port the server listens on |
| `LOG_LEVEL` | No | `INFO` | Logging level: `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `ENVIRONMENT` | No | `production` | Deployment environment name (used in logs and headers) |
| `MLFLOW_TRACKING_URI` | No | `http://localhost:5000` | MLflow server for model registry |
| `MODEL_VERSION` | No | — | Specific model version or stage (`Production`, `Staging`) |
| `PROMETHEUS_MULTIPROC_DIR` | No | `/tmp/prometheus` | Directory for Prometheus multiprocess metrics |
| `REQUEST_TIMEOUT` | No | `30` | Maximum request processing time in seconds |
| `MAX_BATCH_SIZE` | No | `1000` | Maximum number of records per batch prediction |

Create a local `.env` file:

```bash
MODEL_PATH=models/churn_model/
APP_HOST=0.0.0.0
APP_PORT=8000
LOG_LEVEL=INFO
ENVIRONMENT=development
MLFLOW_TRACKING_URI=http://localhost:5000
MAX_BATCH_SIZE=500
```

---

## 4. Docker Deployment

### 4.1 Build the Image

```bash
docker build -t telco-churn-api:latest .
```

### 4.2 Run the Container

```bash
docker run -d \
  --name telco-churn-api \
  -p 8000:8000 \
  -e MODEL_PATH=/app/models/churn_model \
  -e LOG_LEVEL=INFO \
  -e ENVIRONMENT=production \
  -v $(pwd)/models:/app/models:ro \
  telco-churn-api:latest
```

### 4.3 Verify the Container

```bash
curl http://localhost:8000/health
```

Expected response:

```json
{
  "status": "healthy",
  "model_loaded": true,
  "environment": "production",
  "version": "1.0.0"
}
```

---

## 5. FastAPI Deployment

### 5.1 Local Development Server

```bash
poetry shell
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```

### 5.2 Production ASGI Server

Use Uvicorn with Gunicorn for production concurrency:

```bash
gunicorn src.api.app:app \
  -k uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --workers 4 \
  --worker-connections 100 \
  --timeout 60 \
  --access-logfile - \
  --error-logfile -
```

### 5.3 Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health and model load status |
| `GET` | `/ready` | Readiness probe for load balancers |
| `GET` | `/metrics` | Prometheus-compatible metrics |
| `POST` | `/predict` | Single-record churn prediction |
| `POST` | `/predict/batch` | Batch churn predictions |

### 5.4 Example Prediction Request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 85.0,
    "TotalCharges": 1020.0
  }'
```

Expected response:

```json
{
  "churn_probability": 0.82,
  "churn_prediction": true,
  "threshold": 0.5,
  "top_features": [
    {"feature": "Contract", "contribution": 0.21},
    {"feature": "tenure", "contribution": -0.15},
    {"feature": "InternetService", "contribution": 0.12}
  ]
}
```

---

## 6. Running Locally

### 6.1 Install Dependencies

```bash
poetry install --with dev,test,lint,format,type,docs,notebook
poetry shell
```

### 6.2 Train and Save the Model

```bash
poetry run python src/models/train.py
```

### 6.3 Start the API

```bash
export MODEL_PATH=models/churn_model
poetry run uvicorn src.api.app:app --reload
```

### 6.4 Run Tests

```bash
poetry run pytest tests/ -v
```

---

## 7. Production Deployment

### 7.1 Docker Compose (Recommended for Single-Host Production)

```yaml
version: "3.8"

services:
  api:
    build: .
    image: telco-churn-api:latest
    container_name: telco-churn-api
    ports:
      - "8000:8000"
    environment:
      - MODEL_PATH=/app/models/churn_model
      - LOG_LEVEL=INFO
      - ENVIRONMENT=production
      - MAX_BATCH_SIZE=1000
    volumes:
      - ./models:/app/models:ro
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 5s
      retries: 3
      start_period: 10s
    restart: unless-stopped
    deploy:
      resources:
        limits:
          cpus: "2.0"
          memory: 4G
        reservations:
          cpus: "1.0"
          memory: 2G
```

Start:

```bash
docker compose up -d
```

### 7.2 Kubernetes (Recommended for Multi-Instance Production)

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: telco-churn-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: telco-churn-api
  template:
    metadata:
      labels:
        app: telco-churn-api
    spec:
      containers:
        - name: api
          image: telco-churn-api:latest
          ports:
            - containerPort: 8000
          env:
            - name: MODEL_PATH
              value: "/app/models/churn_model"
            - name: LOG_LEVEL
              value: "INFO"
            - name: ENVIRONMENT
              value: "production"
          resources:
            requests:
              cpu: "1"
              memory: "2Gi"
            limits:
              cpu: "2"
              memory: "4Gi"
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 10
            periodSeconds: 15
          readinessProbe:
            httpGet:
              path: /ready
              port: 8000
            initialDelaySeconds: 5
            periodSeconds: 10
```

Apply:

```bash
kubectl apply -f k8s/
```

### 7.3 Scaling Considerations
- The model is stateless and CPU-bound; scale horizontally behind a load balancer.
- Preload the model in each worker to avoid cold-start latency.
- Use `MAX_BATCH_SIZE` and request timeouts to protect against abuse.
- Consider ONNX export if sub-50ms p95 latency is required.

---

## 8. Health Checks

### 8.1 Liveness Probe
`GET /health` returns the service status and whether the model is loaded.

### 8.2 Readiness Probe
`GET /ready` returns `200 OK` only when the model and dependencies are ready to serve traffic.

### 8.3 Recommended Probe Configuration
| Probe | Path | Interval | Timeout | Failure Threshold |
|---|---|---|---|---|
| Liveness | `/health` | 15s | 5s | 3 |
| Readiness | `/ready` | 10s | 5s | 3 |

---

## 9. Logging

### 9.1 Log Format
Logs are emitted as structured JSON by default:

```json
{
  "timestamp": "2026-07-31T12:00:00Z",
  "level": "INFO",
  "environment": "production",
  "request_id": "abc-123",
  "method": "POST",
  "path": "/predict",
  "status_code": 200,
  "latency_ms": 12.4,
  "message": "prediction completed"
}
```

### 9.2 Important Log Events
- Model load success/failure.
- Prediction requests (latency, batch size, outcome).
- Validation errors and malformed payloads.
- Health check failures.

### 9.3 Log Aggregation
Forward container logs to your centralized logging platform:
- ELK Stack (Elasticsearch, Logstash, Kibana)
- Grafana Loki
- Datadog / Splunk / CloudWatch Logs

---

## 10. Monitoring

### 10.1 Metrics Endpoint
`GET /metrics` exposes Prometheus-compatible metrics:

```text
# HELP prediction_requests_total Total prediction requests
# TYPE prediction_requests_total counter
prediction_requests_total{endpoint="/predict",status="200"} 1024

# HELP prediction_latency_seconds Prediction request latency
# TYPE prediction_latency_seconds histogram
prediction_latency_seconds_bucket{endpoint="/predict",le="0.05"} 980

# HELP model_loaded Whether the model is loaded
# TYPE model_loaded gauge
model_loaded 1.0
```

### 10.2 Key Metrics to Alert On
| Metric | Warning | Critical |
|---|---|---|
| p95 latency | > 150 ms | > 300 ms |
| Error rate (5xx) | > 0.1% | > 1% |
| Model load status | — | `model_loaded == 0` |
| Memory usage | > 80% | > 95% |
| CPU usage | > 70% sustained | > 90% sustained |

### 10.3 Dashboards
Recommended Grafana dashboard panels:
- Request rate and error rate by endpoint.
- Latency percentiles (p50, p95, p99).
- Prediction outcome distribution.
- Model load status and version.
- Container resource utilization.

---

## 11. Troubleshooting

### 11.1 Model Fails to Load

**Symptom**: `/health` returns `model_loaded: false`.

**Steps**:
1. Verify `MODEL_PATH` is set correctly.
2. Ensure the model artifact contains `model.pkl` or `MLmodel`.
3. Check file permissions and volume mounts.
4. Review startup logs for serialization or dependency errors.

### 11.2 High Latency

**Symptom**: p95 latency exceeds target.

**Steps**:
1. Preload the model once per worker, not per request.
2. Reduce `MAX_BATCH_SIZE` or increase workers.
3. Profile batch prediction logic for pandas/NumPy inefficiencies.
4. Consider model quantization or ONNX conversion.

### 11.3 Out-of-Memory Errors

**Symptom**: Container restarts with OOMKilled.

**Steps**:
1. Reduce the number of Gunicorn/Uvicorn workers.
2. Lower `MAX_BATCH_SIZE`.
3. Increase container memory limits.
4. Check for memory leaks in SHAP or feature preprocessing.

### 11.4 Poor Predictions After Deployment

**Symptom**: Prediction distribution shifts or accuracy drops.

**Steps**:
1. Validate incoming feature schema matches training schema.
2. Check for data drift in input distributions.
3. Confirm the correct model version is loaded.
4. Trigger retraining and A/B test the new model.

### 11.5 Container Won't Start

**Symptom**: `docker run` exits immediately.

**Steps**:
1. Check `docker logs telco-churn-api` for traceback.
2. Verify all required environment variables are set.
3. Confirm the image was built successfully.
4. Ensure port `8000` is not already in use.

---

## 12. Security Checklist

- [ ] Run containers as a non-root user.
- [ ] Mount model volumes read-only (`:ro`).
- [ ] Use secrets management for MLflow credentials.
- [ ] Enable TLS termination at the load balancer or ingress.
- [ ] Validate and sanitize all incoming prediction payloads.
- [ ] Keep base images and dependencies updated.

---

## 13. Document Control

| Property | Value |
|---|---|
| Version | 1.0 |
| Author | Richard Obeng |
| Last Updated | 2026-07-31 |
| Review Cycle | Per release or quarterly |
| Related Documents | `project_overview.md`, `model_documentation.md` |
