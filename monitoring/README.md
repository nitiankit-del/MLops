# Operational monitoring

`GET /metrics` exposes Prometheus text format. `GET /monitor` is a working, live API metrics dashboard (the assignment allows this alternative to Prometheus + Grafana).

Structured logs intentionally exclude request payloads and raw patient attributes. Route labels are bounded by declared route templates; all unknown paths share `unmatched`, preventing user-controlled metric cardinality.

For a production Prometheus server that discovers every pod, useful PromQL expressions are:

```promql
sum(rate(heart_requests_total{route="/predict"}[5m]))
sum(rate(heart_requests_total{status=~"5.."}[5m])) / clamp_min(sum(rate(heart_requests_total[5m])), 0.001)
histogram_quantile(0.95, sum by (le) (rate(heart_request_seconds_bucket{route="/predict"}[5m])))
sum by (label) (rate(heart_predictions_total[5m]))
```

Suggested alerts: prediction 5xx ratio >1% for 5 minutes; prediction p95 >250ms for 5 minutes; no ready replicas. These are proposals, not deployed alerts. Alert limits need a workload baseline. Production additions include authenticated TLS ingress, a model registry, image scanning/signing, drift baselines, delayed label joins, and rollout approval. This assignment demonstrates a local cloud-ready deployment, not a clinical production certification.
