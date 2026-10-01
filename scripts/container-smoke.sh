#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
IMAGE="${IMAGE:-heart-api:local}"
docker build -t "$IMAGE" .
CID=$(docker run -d --read-only --cap-drop ALL --security-opt no-new-privileges -p 127.0.0.1:8000:8000 "$IMAGE")
trap 'docker logs "$CID"; docker rm -f "$CID"' EXIT
for attempt in $(seq 1 60); do
  if curl --fail --silent http://127.0.0.1:8000/ready; then break; fi
  sleep 2
done
python scripts/infer.py --url http://127.0.0.1:8000
curl --fail --silent http://127.0.0.1:8000/metrics
