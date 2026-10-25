"""Send fake traffic so the Grafana dashboard has something to show.
Usage: python scripts/load_test.py [base_url] [seconds]
(stdlib only - no extra installs). Pass --drift to simulate data drift."""
import json
import random
import sys
import time
import urllib.error
import urllib.request

args = [a for a in sys.argv[1:] if not a.startswith("--")]
base = args[0] if args else "http://localhost:8000"
duration = int(args[1]) if len(args) > 1 else 120
drift = "--drift" in sys.argv


def post(payload):
    req = urllib.request.Request(f"{base}/predict", json.dumps(payload).encode(),
                                 {"Content-Type": "application/json"})
    try:
        return urllib.request.urlopen(req, timeout=5).status
    except urllib.error.HTTPError as e:
        return e.code


end, n = time.time() + duration, 0
while time.time() < end:
    g = random.gauss(175 if drift else 120, 25)  # drifted glucose = much higher
    payload = {"Pregnancies": random.randint(0, 10), "Glucose": max(50, g),
               "BloodPressure": random.gauss(70, 10), "SkinThickness": random.gauss(25, 8),
               "Insulin": random.gauss(100, 50), "BMI": max(16, random.gauss(38 if drift else 32, 5)),
               "DiabetesPedigreeFunction": random.uniform(0.1, 1.5), "Age": random.randint(21, 70)}
    if random.random() < 0.05:
        payload["Glucose"] = -1  # invalid -> 422, shows up as error rate
    post(payload)
    if random.random() < 0.2:
        urllib.request.urlopen(f"{base}/health", timeout=5).read()
    n += 1
    time.sleep(random.uniform(0.05, 0.3))
print(f"sent {n} requests")
