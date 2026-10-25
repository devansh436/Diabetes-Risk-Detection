from tests.conftest import VALID


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok" and r.json()["model_loaded"] is True


def test_predict_valid(client):
    r = client.post("/predict", json=VALID)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["probability"] <= 1
    assert body["risk_level"] in {"low", "moderate", "high"}
    assert body["prediction"] in (0, 1)


def test_high_glucose_scores_higher_than_low_glucose(client):
    low = client.post("/predict", json={**VALID, "Glucose": 80, "BMI": 21, "Age": 22}).json()
    high = client.post("/predict", json={**VALID, "Glucose": 190, "BMI": 42, "Age": 60}).json()
    assert high["probability"] > low["probability"]


def test_predict_rejects_invalid_input(client):
    assert client.post("/predict", json={**VALID, "Glucose": -5}).status_code == 422
    bad = {k: v for k, v in VALID.items() if k != "Age"}
    assert client.post("/predict", json=bad).status_code == 422


def test_model_info(client):
    r = client.get("/model-info")
    assert r.status_code == 200 and "metrics" in r.json()


def test_metrics_endpoint_exposes_custom_metrics(client):
    client.post("/predict", json=VALID)
    text = client.get("/metrics").text
    assert "predictions_total" in text
    assert "http_requests_total" in text
    assert "http_request_duration_seconds_bucket" in text


def test_homepage_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "Diabetes risk check" in r.text
