FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /code

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY app ./app
COPY data ./data

# Train at build time -> every image contains a model that passed the quality gate
ARG GIT_SHA=unknown
ARG MODEL_VERSION=dev
ENV GIT_SHA=${GIT_SHA} MODEL_VERSION=${MODEL_VERSION}
RUN python -m src.train

RUN useradd -m appuser && chown -R appuser /code
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
