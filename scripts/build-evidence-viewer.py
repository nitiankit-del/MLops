"""Create a labelled browser view of genuine saved command output."""

import html
import json

from heart.data import ROOT

style = """body{font:19px system-ui;background:#edf3f5;color:#163445;margin:0}main{max-width:1320px;margin:35px auto}h1{font-size:36px;margin:12px 0}small{color:#55717b}pre{font:16px/1.6 monospace;background:#102b39;color:#d5eef0;padding:24px;border-radius:12px;white-space:pre-wrap;word-break:break-word}img{max-width:100%;max-height:670px;object-fit:contain;background:white;border-radius:12px}nav{margin:20px 0}a{color:#087e8b;margin-right:24px}table{border-collapse:collapse;background:white;width:100%;margin:22px 0}td,th{padding:16px;text-align:left;border-bottom:1px solid #d2e1e5}"""
nav = '<nav><a href="overview.html">Overview</a><a href="eda.html">EDA</a><a href="results.html">Models</a><a href="deployment.html">Deployment evidence</a></nav>'


def write(name, title, body):
    text = f'<!doctype html><html lang="en"><meta charset="utf-8"><title>{title}</title><style>{style}</style><main><small>AIMLCZG523 / MLOPS ASSIGNMENT 01 / MEASURED EVIDENCE</small><h1>{title}</h1>{nav}{body}</main></html>'
    (ROOT / "reports" / name).write_text(text)


m = json.loads((ROOT / "models/metadata.json").read_text())
write(
    "overview.html",
    "Cleveland heart disease: end-to-end MLOps",
    '<p>303 records · 13 predictors · binary target · seed 523</p><img src="figures/architecture.png"><p>Reproducible acquisition → leakage-aware training → tracked models → CI → container → Kubernetes → live monitoring.</p><p>Review the measured results, then demonstrate the deployed API at <a href="http://127.0.0.1:8080/monitor">localhost:8080/monitor</a>.</p>',
)
write(
    "eda.html",
    "Data quality and training-only EDA",
    '<p>242 training / 61 untouched holdout records. Nulls are imputed inside CV folds.</p><img src="figures/histograms.png"><p><img src="figures/balance_missing.png"></p>',
)
rows = "<tr><th>Model</th><th>Nested CV AUC ± SD</th><th>Nested CV recall</th></tr>"
for name in ["logistic", "forest"]:
    r = m["models"][name]["nested_cv"]
    rows += f"<tr><td>{name}</td><td>{r['roc_auc']['mean']:.4f} ± {r['roc_auc']['std']:.4f}</td><td>{r['recall']['mean']:.4f}</td></tr>"
write(
    "results.html",
    "Model selection and final holdout",
    f'<table>{rows}</table><p>Selected logistic regression. Holdout ROC-AUC 0.9232; accuracy 0.8033; recall 0.8214.</p><img src="figures/evaluation.png"><p>Threshold 0.5 was fixed before holdout evaluation. Five false negatives and seven false positives; no clinical validation claimed.</p>',
)
status = (ROOT / "evidence/deployment-status.txt").read_text()
prediction = json.loads((ROOT / "evidence/prediction.json").read_text())
write(
    "deployment.html",
    "Kubernetes deployment verification",
    "<p>Rendered transcript of actual saved command output. This is an evidence viewer, not a Kubernetes UI.</p><small>kubectl --context kind-heart-mlops -n heart-mlops get deploy,pods,svc,ingress -o wide</small><pre>"
    + html.escape(status)
    + "</pre><small>POST http://127.0.0.1:8080/predict · actual API response</small><pre>"
    + html.escape(json.dumps(prediction, indent=2))
    + "</pre><p>Request path: localhost:8080 → Traefik ingress → Service → 2 API replicas.</p>",
)
print("Evidence viewer generated under reports/")
