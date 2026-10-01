#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
IMAGE="${IMAGE:-heart-api:local}"
CLUSTER="${CLUSTER:-heart-mlops}"
if ! kind get clusters | grep -qx "$CLUSTER"; then
  kind create cluster --name "$CLUSTER" --config k8s/kind.yaml --image kindest/node:v1.32.2 --wait 180s
fi
kind load docker-image "$IMAGE" --name "$CLUSTER"
kubectl --context "kind-$CLUSTER" apply -f k8s/app.yaml
kubectl --context "kind-$CLUSTER" -n heart-mlops set image deployment/heart-api api="$IMAGE"
kubectl --context "kind-$CLUSTER" apply -f k8s/ingress-controller.yaml
kubectl --context "kind-$CLUSTER" -n heart-mlops rollout status deployment/heart-api --timeout=180s
kubectl --context "kind-$CLUSTER" -n heart-mlops rollout status deployment/traefik --timeout=180s
for attempt in $(seq 1 30); do
  if curl --fail --silent http://127.0.0.1:8080/ready; then break; fi
  sleep 2
done
curl --fail --silent -H 'Content-Type: application/json' --data @scripts/sample.json http://127.0.0.1:8080/predict
kubectl --context "kind-$CLUSTER" -n heart-mlops get deploy,pods,svc,ingress
