"""Build the 10-page report from measured results and captured evidence."""

import json
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from heart.data import ROOT

m = json.loads((ROOT / "models/metadata.json").read_text())
pred = json.loads((ROOT / "evidence/prediction.json").read_text())
handoff = (
    json.loads((ROOT / "submission-details.json").read_text())
    if (ROOT / "submission-details.json").exists()
    else {}
)
styles = getSampleStyleSheet()
styles.add(
    ParagraphStyle(
        name="TitleX",
        fontName="Helvetica-Bold",
        fontSize=27,
        leading=31,
        textColor=colors.HexColor("#123747"),
        spaceAfter=13,
    )
)
styles.add(
    ParagraphStyle(
        name="SubX",
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#087e8b"),
        spaceBefore=10,
        spaceAfter=6,
    )
)
styles.add(
    ParagraphStyle(
        name="BodyX",
        fontName="Helvetica",
        fontSize=9.6,
        leading=14,
        spaceAfter=8,
        textColor=colors.HexColor("#223b46"),
    )
)
styles.add(
    ParagraphStyle(
        name="SmallX",
        fontName="Helvetica",
        fontSize=8,
        leading=11,
        spaceAfter=6,
        textColor=colors.HexColor("#49626a"),
    )
)
styles.add(
    ParagraphStyle(
        name="CodeX",
        fontName="Courier",
        fontSize=7.4,
        leading=10,
        spaceAfter=9,
        backColor=colors.HexColor("#eef4f5"),
        borderPadding=8,
    )
)
story = []


def P(t, style="BodyX"):
    story.append(Paragraph(t, styles[style]))


def H(t):
    P(t, "SubX")


def C(t):
    story.append(Preformatted(t, styles["CodeX"]))


def FIG(path, width=495, maxheight=310):
    path = ROOT / path
    if not path.exists():
        raise FileNotFoundError(path)
    iw, ih = ImageReader(str(path)).getSize()
    scale = min(width / iw, maxheight / ih)
    im = Image(str(path), width=iw * scale, height=ih * scale)
    im.hAlign = "CENTER"
    story.append(im)
    story.append(Spacer(1, 7))


def T(rows, widths=None):
    cells = [[Paragraph(escape(str(c)), styles["SmallX"]) for c in row] for row in rows]
    t = Table(cells, colWidths=widths, repeatRows=1, hAlign="LEFT")
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e0eef0")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LINEBELOW", (0, 0), (-1, 0), 0.7, colors.HexColor("#087e8b")),
                ("LINEBELOW", (0, 1), (-1, -1), 0.3, colors.HexColor("#d5e0e4")),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(t)
    story.append(Spacer(1, 8))


def page(n, title, kicker):
    if n > 1:
        story.append(PageBreak())
    P(f"AIMLCZG523  /  ASSIGNMENT 01  /  {kicker.upper()}", "SmallX")
    P(f"{n:02d}  {title}", "TitleX")


page(1, "Heart disease prediction", "Overview and architecture")
P(
    "An end-to-end MLOps implementation using the UCI Cleveland dataset: reproducible data preparation, leakage-aware experiments, a packaged prediction API, automated CI, container execution and a monitored local Kubernetes deployment."
)
if handoff.get("name"):
    P(escape(handoff["name"]) + " | " + escape(handoff.get("student_id", "")))
T(
    [
        ["Release outcome", "Measured evidence"],
        ["Selected model", "Logistic regression; nested CV AUC 0.9151 +/- 0.0318"],
        ["Untouched holdout", "61 rows; accuracy 0.8033; recall 0.8214; ROC-AUC 0.9232"],
        ["Validation", "26 pytest cases passed; independent release verification passed"],
        ["Serving", "Non-root Docker API; 2 ready Kubernetes replicas; Traefik ingress"],
    ],
    [130, 365],
)
FIG(Path("reports/figures/architecture.png"), maxheight=230)
H("Scope and submission location")
P(
    "This is an educational classifier for the recorded presence of heart disease. It is not a prospectively validated clinical risk tool. The delivered artifact is the exact training-partition model evaluated on holdout; the holdout is not reused for refitting."
)
if handoff.get("repository_url"):
    P(
        'Code repository: <link href="'
        + escape(handoff["repository_url"])
        + '" color="#087e8b">'
        + escape(handoff["repository_url"])
        + "</link>"
    )
else:
    P(
        "<b>Repository publication pending:</b> GitHub account authentication and the repository destination were not supplied. All source is in the accompanying mlops-heart project. Add the real repository URL before submission; a local path is not a GitHub link."
    )
P(
    'Local deployed access: <link href="http://127.0.0.1:8080/docs" color="#087e8b">http://127.0.0.1:8080/docs</link>. Reproduction instructions and the video are included. Screenshots document actual execution, with rendered transcripts explicitly labelled.',
    "SmallX",
)

page(2, "Data acquisition & quality", "Task 1 / 5 marks")
P(
    'Source: Janosi, Steinbrunn, Pfisterer and Detrano (1989), UCI Heart Disease, DOI 10.24432/C52P4X. The official processed Cleveland file has 303 records and 13 predictors plus the target. This explains the assignment\'s shorthand "14+ features": the canonical table has 14 columns, not 14 independent predictors. Original UCI category codes are preserved.'
)
T(
    [
        ["Group", "Variables and original coding"],
        [
            "Continuous",
            "age (years), trestbps (mm Hg), chol (mg/dL), thalach (bpm), oldpeak (ST depression)",
        ],
        [
            "Categorical",
            "sex 0/1; cp 1-4; fbs 0/1 (>120 mg/dL); restecg 0-2; exang 0/1; slope 1-3; ca 0-3; thal 3/6/7",
        ],
        ["Target", "num=0 -> absent (0); num=1,2,3,4 -> present (1)"],
        [
            "Quality checks",
            "303 retained rows; no exact duplicates; 4 ca and 2 thal values missing; 164 negative / 139 positive",
        ],
    ],
    [94, 401],
)
FIG(Path("reports/figures/balance_missing.png"), maxheight=210)
H("Repeatable cleaning, without leakage")
P(
    "The acquisition script fetches the official archive or uses the included raw file, verifies its SHA-256, validates columns, target values and category domains, normalizes missing markers and removes exact duplicate rows. The cleaned CSV retains nulls because imputation is learned separately within each CV training fold. The manifest records data origin, hash, schema and quality counts."
)
C("python -m heart.data\n# outputs: data/processed/heart.csv and manifest.json")
P(
    "Raw SHA-256: a74b7efa387bc9d108d7d0115d831fe9b414b29ae7124f331b622b4efa0427c8. The downloader fails if the raw content changes, rather than silently changing the experiment. UCI identifies the dataset license as CC BY 4.0.",
    "SmallX",
)

page(3, "Exploratory analysis", "Task 1 / training partition only")
P(
    "A stratified 80/20 split (seed 523) is made before exploratory plotting. The training partition contains 242 records: 131 negative and 111 positive. The test partition remains separate. Mild class imbalance supports stratified folds and explicit recall/precision reporting rather than synthetic resampling."
)
FIG(Path("reports/figures/histograms.png"), maxheight=275)
FIG(Path("reports/figures/correlation.png"), maxheight=240)
P(
    "Histograms show continuous distributions by class, including overlapping groups and right-skewed oldpeak. The Spearman heatmap is restricted to continuous features and the binary target: numeric codes for nominal categories are not interpreted as distances. Observational associations are not causal effects. Extreme values are retained because this small dataset offers no justified automatic clinical outlier-removal rule.",
    "SmallX",
)

page(4, "Features, tuning & selection", "Task 2 / 8 marks")
P(
    "One sklearn Pipeline includes a ColumnTransformer and classifier. Numeric inputs receive median imputation and StandardScaler; categorical inputs receive most-frequent imputation and OneHotEncoder with explicit original UCI categories. Encoding is nominal, including ca and slope, to avoid assuming equal spacing. This specification is shared by training and inference."
)
T(
    [
        ["Model", "Prespecified search space"],
        ["Logistic regression", "C = 0.1, 1, 10; class_weight = None or balanced; max_iter=2000"],
        [
            "Random forest",
            "200 trees; max_depth = 3 or None; min_samples_leaf = 2 or 5; class_weight = None or balanced",
        ],
    ],
    [116, 379],
)
P(
    "Each family is evaluated using five stratified outer folds; a four-fold inner GridSearchCV tunes ROC-AUC. All learned preprocessing is refitted inside the inner folds. The final family is selected by mean outer AUC, then its grid search is refitted on all 242 training rows. Threshold 0.5 is specified in advance. The holdout never selects hyperparameters or threshold."
)
rows = [["Outer-CV mean +/- sample SD", "Logistic regression", "Random forest"]]
for metric in ["accuracy", "precision", "recall", "roc_auc"]:
    row = [metric.replace("_", " ").upper()]
    for family in ["logistic", "forest"]:
        v = m["models"][family]["nested_cv"][metric]
        row.append(f"{v['mean']:.4f} +/- {v['std']:.4f}")
    rows.append(row)
T(rows, [191, 152, 152])
FIG(Path("reports/figures/logistic_tuning.png"), maxheight=155)
P(
    "Logistic regression wins by a small mean-AUC margin (0.0033); fold variability is much larger, so this is a deterministic selection rule, not proof of statistical superiority. The simpler model is also easier to inspect. Selected logistic parameters: "
    + escape(str(m["models"]["logistic"]["best_params"]))
    + ".",
    "SmallX",
)

page(5, "Final evaluation & limitations", "Held-out results")
P(
    "The selected model was evaluated on the 61 held-out records only after model-family selection. The persisted split and row-level predictions support independent recomputation. The saved serving model is the same fitted object used here."
)
T(
    [
        ["Metric", "Holdout value", "Interpretation"],
        ["Accuracy", f"{m['test_metrics']['accuracy']:.4f}", "49 of 61 correctly classified"],
        [
            "Precision",
            f"{m['test_metrics']['precision']:.4f}",
            "23 of 30 predicted-positive cases were positive",
        ],
        ["Recall", f"{m['test_metrics']['recall']:.4f}", "23 of 28 positive cases detected"],
        ["F1", f"{m['test_metrics']['f1']:.4f}", "Balance of precision and recall"],
        ["ROC-AUC", f"{m['test_metrics']['roc_auc']:.4f}", "Ranking quality across thresholds"],
    ],
    [100, 91, 304],
)
FIG(Path("reports/figures/evaluation.png"), maxheight=230)
H("Uncertainty and operational meaning")
P(
    f"A 2,000-resample bootstrap on the held-out predictions gives a descriptive 95% AUC interval of [{m['test_auc_bootstrap_95_ci'][0]:.3f}, {m['test_auc_bootstrap_95_ci'][1]:.3f}]. This reflects resampling of this small holdout, not all model-selection uncertainty. Confusion counts are TN=26, FP=7, FN=5, TP=23. False negatives matter in screening, but changing the threshold requires training-only validation and a documented cost trade-off."
)
P(
    "The small historical cohort may not represent current or diverse patient populations. There is no external validation, calibration study, subgroup guarantee or longitudinal outcome validation. Confidence is the model's returned-class probability, not a guaranteed probability of disease in a new population. Clinical use would require independent validation and governance.",
    "SmallX",
)

page(6, "Experiment tracking", "Task 3 / 5 marks")
P(
    "MLflow captures a release parent, two family runs and fourteen candidate runs (six logistic, eight forest): 17 completed runs per release. Candidate records log parameters, fold AUC values, metric means and standard deviations, a JSON artifact and tuning plot. Family runs retain complete CV tables, selected hyperparameters and nested-CV summaries."
)
FIG(Path("screenshots/mlflow-run.png"), maxheight=325)
H("What can be inspected")
P(
    "The parent logs dataset hash, random seed, partition sizes, fold counts and threshold; holdout metrics; EDA and evaluation plots; exact split and predictions; and an MLflow sklearn model with signature and example. The experiment export in reports/experiments.csv and .json contains the final release run tree. Earlier development runs are distinguishable by run ID."
)
C(
    "python scripts/relocate-mlflow.py  # only after moving this directory\nmlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5001\n# Open :5001, then select heart-disease-cleveland"
)
P(
    "Selected release run ID: "
    + m["mlflow_run_id"]
    + ". Artifact paths in MLflow's local file store are absolute, so relocation explicitly updates them. The portable JSON/CSV exports remain readable without a tracking server.",
    "SmallX",
)

page(7, "Packaging & API contract", "Tasks 4 and 6 / 12 marks")
P(
    "The reusable artifact includes all fitted transforms plus the classifier in models/heart_pipeline.joblib. Metadata stores features, threshold, model/data hashes, split sizes and run ID. API startup verifies SHA-256 before loading. It fails closed on missing or mismatched artifacts; no serving-time training or data download occurs."
)
C(
    "python3 -m venv .venv\nsource .venv/bin/activate\npip install -r requirements.txt -e .\npython -m heart.data && python -m heart.eda && python -m heart.train\npytest -q\npython scripts/verify-release.py\nbash scripts/container-smoke.sh"
)
P(
    "requirements.txt pins training/testing dependencies; requirements-serve.txt is a smaller serving set. The Docker image uses Python 3.13.3, runs as non-root UID 10001 and includes the model. The isolated smoke test uses a read-only filesystem, no Linux capabilities and no host code/model mount. Build and real response proof are saved in evidence/docker.log."
)
H("Prediction request and measured response")
C("POST /predict\n" + (ROOT / "scripts/sample.json").read_text().strip().replace(',"', ',\n "'))
T(
    [
        ["Response field", "Observed value"],
        ["prediction / threshold", f"{pred['prediction']} / {pred['threshold']}"],
        [
            "disease_probability / confidence",
            f"{pred['disease_probability']:.6f} / {pred['confidence']:.6f}",
        ],
        ["model_version", pred["model_version"]],
    ],
    [185, 310],
)
P(
    "Pydantic rejects unexpected fields, invalid categories, out-of-range values and missing required fields with HTTP 422. Declared nullable fields use the fitted imputers. For a positive prediction confidence=p; for a negative prediction confidence=1-p. /health is liveness, /ready confirms loaded model, /docs supplies interactive OpenAPI documentation.",
    "SmallX",
)

page(8, "Automated CI & testing", "Task 5 / 8 marks")
P(
    "The repository contains both GitHub Actions YAML and the allowed Jenkins alternative. The actual local Jenkins run uses a clean Linux virtual environment to install pinned dependencies, lint, validate data, run unit tests, train tracked models, test the API and verify model provenance. Tests and artifacts are retained. Jenkins has no host Docker socket access."
)
FIG(Path("screenshots/jenkins-report.png"), maxheight=280)
T(
    [
        ["Test coverage", "Assertions"],
        [
            "Data processing (13)",
            "Target mapping, missing values, deduplication, invalid categories/targets/ranges, schema drift",
        ],
        [
            "Model pipeline (3)",
            "Both families impute, probability sums, serialization round-trip and fit-only imputation",
        ],
        [
            "API (10)",
            "Prediction semantics, metrics/health, extra/invalid/missing fields, nullable values, checksum rejection",
        ],
    ],
    [138, 357],
)
P(
    "Local pytest result: 26 passed. The workflow fails on lint or test failures; bash pipefail preserves failures through logging pipes. GitHub Actions additionally builds/tests the container and creates an ephemeral kind cluster to verify ingress deployment. Its artifacts are uploaded even on failure. A GitHub Actions run is not claimed without repository authentication; the screenshot above is Jenkins evidence."
)
P(
    "Docker and Kubernetes were also verified directly on this machine, independently of Jenkins. CI retraining may serialize a different model checksum across Python/OS versions; each job verifies its own newly generated artifact and reports the resulting metrics.",
    "SmallX",
)

page(9, "Kubernetes deployment", "Task 7 / 7 marks")
P(
    "The actual deployment uses a dedicated kind cluster on Colima. k8s/app.yaml defines the namespace, two-replica Deployment, Service and Ingress; k8s/ingress-controller.yaml defines Traefik. Host port 8080 maps to NodePort 30080, then the Ingress routes through the Service to ready API pods. No public cloud URL is claimed."
)
FIG(Path("screenshots/deployment.png"), maxheight=290)
C(
    "docker build -t heart-api:local .\nbash scripts/deploy-local.sh\ncurl -fsS http://127.0.0.1:8080/ready\npython scripts/infer.py --url http://127.0.0.1:8080\nkubectl --context kind-heart-mlops -n heart-mlops get pods,ingress"
)
H("Availability and release control")
P(
    "Startup/readiness/liveness probes prevent premature traffic and detect failures. Each API pod requests 100m CPU / 256Mi RAM and is limited to 1 CPU / 768Mi RAM. Pods run non-root with read-only root filesystems and dropped capabilities. Rolling updates set maxUnavailable=0 and maxSurge=1. Logs confirm real requests and the reported model version."
)
P(
    "For repeat releases use unique image tags, load the image into kind and update the Deployment. Reusing a local tag requires an explicit rollout restart. Rollback: kubectl --context kind-heart-mlops -n heart-mlops rollout undo deployment/heart-api. Verify readiness and sample inference after changes. These controls demonstrate cloud-ready packaging; TLS, authentication, persistent monitoring and a managed registry remain production extensions.",
    "SmallX",
)

page(10, "Monitoring & handoff", "Tasks 8 and 9 / 5 marks")
P(
    "Request middleware emits JSON request ID, method, bounded route label, status and duration without patient payloads. Prometheus counters and histograms back the live /monitor dashboard. Successful inference and HTTP 422 rejection are demonstrated; screenshots and raw logs are included."
)
FIG(Path("screenshots/monitoring.png"), width=355, maxheight=300)
P(
    "The dashboard displays one process at a time; with two replicas its counters are not a cluster-wide total and reset on restart. Production Prometheus should scrape every pod and aggregate. monitoring/README.md supplies example throughput, error-rate and p95 expressions. Accuracy/drift monitoring requires ground-truth labels and is not claimed by request metrics alone.",
    "SmallX",
)
H("Deliverables and final submission check")
P(
    "Included: source, cleaned/raw data and downloader, EDA/training/inference scripts, pinned dependencies, serialized model, 26 tests, Dockerfile, GitHub Actions YAML, Jenkinsfile, manifests, execution logs, screenshots, MLflow runs/exports, this 10-page report and reports/pipeline-walkthrough.mp4. README.md gives complete setup, API access, restart and cleanup instructions. The supplied repository link and student details are included. Confirm the source files are uploaded and the instructor has repository access before submission.",
    "SmallX",
)
H("Sources and implementation references")
P(
    'UCI Heart Disease: <link href="https://doi.org/10.24432/C52P4X" color="#087e8b">doi.org/10.24432/C52P4X</link>. scikit-learn nested CV: <link href="https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html" color="#087e8b">official nested cross-validation example</link>. MLflow: <link href="https://mlflow.org/docs/latest/ml/tracking/" color="#087e8b">tracking documentation</link>. kind: <link href="https://kind.sigs.k8s.io/docs/user/configuration/" color="#087e8b">cluster and port-mapping configuration</link>. Assignment specification: supplied MLOps Assignment 1 Instructions.pdf. All numeric results and operational evidence in this report come from the accompanying executions.',
    "SmallX",
)


def footer(canvas, doc):
    canvas.setStrokeColor(colors.HexColor("#d5e0e4"))
    canvas.line(48, 42, 547, 42)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#49626a"))
    canvas.drawString(48, 28, "CLEVELAND HEART DISEASE  |  MLOPS EXPERIMENTAL LEARNING")
    canvas.drawRightString(547, 28, f"{doc.page} / 10")


out = ROOT / "output/pdf/MLOps_Assignment_1_Report.pdf"
out.parent.mkdir(parents=True, exist_ok=True)
doc = SimpleDocTemplate(
    str(out),
    pagesize=(595.28, 841.89),
    rightMargin=50,
    leftMargin=50,
    topMargin=43,
    bottomMargin=55,
    title="MLOps Assignment 01 - Cleveland Heart Disease",
    author=handoff.get("name", ""),
)
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print(out)
