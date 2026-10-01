"""Use: python scripts/infer.py --url http://localhost:8000"""

import argparse
import json
import urllib.request
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--url", default="http://localhost:8000")
a = p.parse_args()
payload = (Path(__file__).parent / "sample.json").read_bytes()
request = urllib.request.Request(
    a.url.rstrip("/") + "/predict", data=payload, headers={"Content-Type": "application/json"}
)
with urllib.request.urlopen(request, timeout=15) as response:
    result = json.load(response)
assert result["prediction"] in [0, 1]
assert 0 <= result["confidence"] <= 1
print(json.dumps(result, indent=2))
