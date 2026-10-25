# Diabetes Risk Detection - MLOps Project

An end-to-end machine-learning service for diabetes-risk estimation. The project
contains a reproducible scikit-learn training pipeline, a FastAPI service with a
small browser UI, automated tests and quality gates, a Docker image, GitHub
Actions CI/CD, and Prometheus/Grafana monitoring.

> This is an educational project. Its output is not a medical diagnosis.

## Architecture

```text
data/diabetes.csv
        |
        v
src/data.py -> src/train.py -> models/model.joblib + models/metrics.json
                                      |
                         Dockerfile trains at image build time
                                      |
                app/main.py -> GHCR -> Render (optional deployment)
                    |   \
                    |    -> /metrics -> Prometheus -> Grafana
                    -> / (HTML UI), /health, /predict, /model-info
```

The preprocessing pipeline treats impossible zero measurements as missing,
median-imputes them, scales features, and trains a `RandomForestClassifier`.
Training writes the model metadata and stops with a non-zero exit code when
ROC-AUC is below `MIN_ROC_AUC` (default `0.70`).

## Project layout

| Path | Purpose |
| --- | --- |
| `src/data.py` | Pima dataset download, synthetic fallback, and preprocessing helper |
| `src/train.py` | Training, evaluation, metadata, and ROC-AUC quality gate |
| `app/main.py` | FastAPI endpoints and Prometheus metrics |
| `app/static/` | Browser UI |
| `tests/` | Unit, integration, and API tests |
| `Dockerfile` | Non-root production image that trains during the build |
| `.github/workflows/ci-cd.yml` | Test, container, GHCR, and optional Render deployment jobs |
| `monitoring/` | Prometheus and pre-provisioned Grafana configuration |
| `scripts/load_test.py` | Demo traffic generator, including drift simulation |

## Local setup

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
```

### macOS/Linux

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
```

Download the real Pima Indians Diabetes CSV and train:

```bash
python -m src.data
python -m src.train
pytest
ruff check .
```

The downloader stores the dataset at `data/diabetes.csv`. If downloading is
not possible, training uses deterministic synthetic data and logs a warning.
For a real-data run, download the Pima CSV from a trusted source and save it as
`data/diabetes.csv`; the file may be committed for this small demo dataset.
Larger projects should use DVC or object storage instead of Git.

Start the API:

```bash
uvicorn app.main:app --reload
```

Open [http://localhost:8000](http://localhost:8000). Interactive API
documentation is available at [http://localhost:8000/docs](http://localhost:8000/docs).
The API exposes:

* `GET /health` - readiness check; returns HTTP 503 if the model is unavailable.
* `POST /predict` - validates patient data and returns probability and risk level.
* `GET /model-info` - model version, data hash, training time, and metrics.
* `GET /metrics` - Prometheus exposition format.

Run all commands from the repository root because the serialized pipeline
imports `src.data.zeros_to_nan`.

## Docker and monitoring demo

Docker Desktop is required:

```bash
docker compose up --build
python scripts/load_test.py http://localhost:8000 120
```

Open:

* [http://localhost:8000](http://localhost:8000) - API UI
* [http://localhost:9090/targets](http://localhost:9090/targets) - Prometheus target status
* [http://localhost:3000](http://localhost:3000) - Grafana (`admin` / `admin`)

Grafana automatically provisions the dashboard **Diabetes Risk API - MLOps
Monitoring**. Generate shifted inputs and compare the glucose and probability
panels:

```bash
python scripts/load_test.py http://localhost:8000 120 --drift
```

Stop the stack with `docker compose down`.

## CI/CD and release

The workflow in `.github/workflows/ci-cd.yml` runs on pull requests and pushes
to `main` or `v*` tags:

1. **Test** installs dependencies, runs Ruff, trains the model, and runs tests.
2. **Docker** builds the image, smoke-tests `/health`, `/predict`, and `/metrics`,
   and publishes to GHCR for push events.
3. **Deploy** optionally triggers Render on pushes to `main`.

To use it in your own repository:

```bash
git init
git add .
git commit -m "Initial MLOps pipeline"
git branch -M main
git remote add origin https://github.com/<you>/diabetes-mlops.git
git push -u origin main
```

Open the **Actions** tab and verify the green run. To demonstrate the quality
gate, create a branch, set `MIN_ROC_AUC=0.99` or intentionally break a test,
and open a pull request; the run should fail.

Create a versioned image and model metadata with:

```bash
git tag v1.0.0
git push --tags
```

The image is published as
`ghcr.io/<you>/diabetes-mlops:1.0.0`. After the first push, change the GHCR
package visibility to public if the demo requires unauthenticated pulls.

### Optional Render deployment

1. In Render, create a **Web Service** connected to the repository, choose
   **Docker**, and set the health-check path to `/health`.
2. Set **Auto-Deploy** to **Off** so GitHub Actions controls deployments.
3. Create a Render **Deploy Hook** and copy its URL.
4. Add that URL as the GitHub Actions secret
   `RENDER_DEPLOY_HOOK_URL`.
5. Add the deployed URL as the repository variable `APP_URL`.
6. Push to `main`. The deploy job triggers the hook and checks `APP_URL/health`.

If either Render value is absent, the deploy job is skipped or only triggers
the hook without a post-deploy health check. The free tier may sleep, so the
first request can take approximately 50 seconds.

To run the published image locally instead:

```bash
docker run --rm -p 8000:8000 ghcr.io/<you>/diabetes-mlops:latest
```

## Future improvements

Useful next steps include MLflow or DVC for experiment/data tracking, alert
rules for latency and drift, protected production environments, and automated
retraining after a validated drift signal.

Innovative Assignment by<br>
-23bce001<br>
-23bce012<br>
-23bce056<br>
-23bce120
