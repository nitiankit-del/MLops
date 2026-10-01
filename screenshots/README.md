# Screenshot provenance

All images were captured from actual local execution on 1 October 2026. They are not mockups.

| Image | Source |
|---|---|
| `api-docs.png` | FastAPI OpenAPI UI reached through the running Kubernetes ingress at localhost:8080 |
| `monitoring.png` | Live dashboard after a real `/predict` call through ingress |
| `input-validation.png` | Live dashboard displaying actual HTTP 422 from an invalid category |
| `mlflow.png` | MLflow experiment list; two development/release comparisons, 34 runs total |
| `mlflow-run.png` | Final selected release run, showing actual parameters and holdout metrics |
| `jenkins.png` | Actual successful Jenkins build #3 status page |
| `jenkins-tests.png`, `jenkins-report.png` | Jenkins build #3 test report: 26 tests passed |
| `deployment.png` | **Rendered transcript**, not a Kubernetes product UI: browser view of saved actual `kubectl get` output and real API JSON from `evidence/` |

Original execution logs are under `evidence/`. The walkthrough records the browser as it visits EDA/model evidence, MLflow, Jenkins, deployment output and the live API. It contains live prediction and input-validation requests. The report's charts come from the accompanying Python scripts and measured data. Deployment captures can precede the final immutable-tag rolling update; `evidence/kubernetes-final.log` documents that later update.
