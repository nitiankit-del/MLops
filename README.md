# Cleveland Heart Disease — MLOps Assignment 01

Student: **Ankit Kumar Agrawal** | ID: **2025AE05620**

AIMLCZG523 · reproducible classification, tracked experiments, tested API, container delivery and local Kubernetes ingress.

## Start here

- `output/pdf/MLOps_Assignment_1_Report.pdf`: exactly 10 pages, measured results and deployment evidence.
- `reports/pipeline-walkthrough-narrated.mp4`: detailed 13-minute 19-second walkthrough with your recorded narration, synchronized evidence and ten chapters.
- `reports/pipeline-walkthrough.mp4`: original 97-second browser demonstration, including real API requests.
- `SUBMISSION_CHECKLIST.md`: rubric-to-file mapping and any outstanding handoff items.
- `reports/results.json`: authoritative measured results.
- `evidence/`: raw execution logs; `screenshots/`: genuine UI captures and rendered command transcripts.

**Submission status:** the implementation and local execution evidence are provided. The designated repository is [nitiankit-del/MLops](https://github.com/nitiankit-del/MLops). Upload is pending GitHub authentication. Read the checklist before submitting. Review the implementation and be prepared to explain your own decisions in accordance with the assignment's independent-work requirement.

## 1. Clean setup

Use Python 3.11–3.13 (3.13 recommended), Git, Docker, kubectl and kind. On macOS, Docker Desktop or Colima is suitable. This work was executed with a dedicated Colima VM and kind v1.32.2 node image. Allocate 4 CPUs, 6 GB RAM and 30 GB disk for the full demonstration (including Jenkins).

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt -e .
python -m heart.data
python -m heart.eda
python -m heart.train
ruff check .
pytest -q --junitxml=evidence/tests.xml
python scripts/verify-release.py
python scripts/export-experiments.py
```

On Windows, activate with `.venv\Scripts\activate`; run the bash deployment scripts through WSL or Git Bash with Docker configured. Commands run from the repository root. Training uses deterministic seeds and single-process CV; allow several minutes. The API never downloads or trains a model at startup.

The raw UCI data is included and SHA-256 verified, so acquisition works offline when the raw file is present. Remove `data/raw/processed.cleveland.data` to exercise the official download path. Missing markers are normalized to empty CSV cells and filled **inside each fitted pipeline**. They are deliberately not imputed across the complete dataset.

## 2. Data and modelling design

Source: [UCI Heart Disease](https://archive.ics.uci.edu/dataset/45/heart+disease), DOI [10.24432/C52P4X](https://doi.org/10.24432/C52P4X), Janosi, Steinbrunn, Pfisterer and Detrano (1989), CC BY 4.0 as stated by UCI. Cleveland has 303 rows, 13 predictors and a target: 14 columns, rather than 14 independent predictors. The assignment's “14+ features” description is interpreted as this canonical UCI table; no artificial feature is added.

Target: `num=0 -> 0` and `num in {1,2,3,4} -> 1`. Raw category codes are retained (not the re-encoded Kaggle variant). Four missing `ca` values and two missing `thal` values; no exact duplicate rows. Class counts: 164 absent, 139 present. No oversampling is needed for this mild imbalance.

An 80/20 stratified split (seed 523) creates 242 training and 61 holdout rows. EDA is restricted to training rows. Five outer stratified folds compare model families; four inner folds tune hyperparameters using ROC-AUC. Numeric columns use median imputation and standardization. Discrete clinical codes use most-frequent imputation and one-hot encoding with explicit original UCI categories. All learned transforms stay inside the sklearn Pipeline during CV.

- Logistic regression: C ∈ {0.1, 1, 10}; class_weight ∈ {None, balanced}.
- Random forest: 200 trees; max_depth ∈ {3, None}; min_samples_leaf ∈ {2, 5}; class_weight ∈ {None, balanced}.
- Select the family with highest mean outer-CV AUC; refit its tuned model on the 242 training rows, then evaluate once on holdout. Threshold is prespecified at 0.5; no holdout tuning or all-data refit.

Measured selected model: logistic regression. Holdout accuracy 0.8033, precision 0.7667, recall 0.8214, F1 0.7931, ROC-AUC 0.9232. The bootstrap AUC interval is descriptive and conditional on this one split; it is not external clinical validation.

## 3. Experiments and model provenance

```bash
# Only needed if the supplied mlruns directory has moved:
python scripts/relocate-mlflow.py
mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5001
```

Open http://127.0.0.1:5001 and select `heart-disease-cleveland`. Each release comparison has 17 runs: parent + 2 model families + 14 candidates. Parameters, mean/std metrics, candidate fold AUCs, tuning plots and artifacts are logged. The parent includes the model with signature and example, EDA, holdout plots, exact split, and report artifacts. `reports/experiments.csv` and `.json` are portable exports of the final release run tree. Extra earlier exploratory runs may remain in the file store.

`models/heart_pipeline.joblib` contains preprocessing and classifier. `models/metadata.json` records feature order, hashes, seed, threshold, evaluation and MLflow run ID. Startup verifies the model checksum before loading. Load only trusted joblib artifacts. Direct dependencies are pinned; `evidence/environment-freeze.txt` records the full executed environment (including optional reporting tools). Cross-platform retraining can produce tiny numerical or serialization differences; prediction metrics are independently checked rather than assuming identical binary bytes across platforms.

## 4. Run API directly

```bash
uvicorn heart.api:app --host 127.0.0.1 --port 8000 --no-access-log
# another terminal:
python scripts/infer.py --url http://127.0.0.1:8000
```

Routes: `/predict`, `/health`, `/ready`, `/docs`, `/metrics`, `/monitor`. Prediction accepts one flat JSON patient record; a working request is in `scripts/sample.json`.

```bash
curl -fsS -H 'Content-Type: application/json' \
  --data @scripts/sample.json http://127.0.0.1:8000/predict
```

Response: prediction (0/1), disease_probability, confidence of the returned class, model_version, threshold. Confidence is `p` for predicted disease and `1-p` otherwise. It is an uncalibrated model score, not a clinically validated risk probability. Invalid/extra fields produce HTTP 422; explicitly allowed nulls use the training-fitted imputers. A missing required field is rejected. `/ready` is available only after model loading succeeds.

## 5. Docker build/run proof

```bash
bash scripts/container-smoke.sh
# Keep the API running instead:
docker build -t heart-api:local .
docker run -d --name heart-api-demo --read-only --cap-drop ALL \
  --security-opt no-new-privileges -p 127.0.0.1:8000:8000 heart-api:local
```

No host model or source mounts are needed. The image runs as UID 10001 and uses a readiness healthcheck. `evidence/docker.log` contains the executed build and successful isolated prediction. Stop the optional container with `docker rm -f heart-api-demo`.

## 6. Local Kubernetes deployment (actual assignment deployment)

```bash
# macOS with Colima, if Docker is not already running:
colima start --profile mlops-assignment --cpu 4 --memory 6 --disk 30 --vm-type vz --runtime docker
# Docker image must already have been built:
bash scripts/deploy-local.sh
curl -fsS http://127.0.0.1:8080/ready
python scripts/infer.py --url http://127.0.0.1:8080
kubectl --context kind-heart-mlops -n heart-mlops get deploy,pods,svc,ingress
```

Access [API docs](http://127.0.0.1:8080/docs) and [monitoring](http://127.0.0.1:8080/monitor). Path: host 127.0.0.1:8080 -> kind NodePort 30080 -> Traefik -> Ingress -> ClusterIP Service -> two API replicas. A local address is expected; no public URL is claimed. Manifests specify probes, resources, non-root user, read-only filesystem, restricted capabilities and rolling updates. The controller and cluster are dedicated to this demonstration.

For a new release use a unique image tag (the GitHub workflow uses the commit SHA), load it into kind and update the Deployment image. Reusing `heart-api:local` does not guarantee existing pods restart; explicitly rebuild, reload and `kubectl ... rollout restart deployment/heart-api` when doing local iterative development. Roll back with `kubectl --context kind-heart-mlops -n heart-mlops rollout undo deployment/heart-api` and verify `/ready` and predictions. Remove the cluster only when finished: `kind delete cluster --name heart-mlops`.

## 7. CI/CD

`.github/workflows/ci.yml` runs install -> lint -> data checks -> unit tests -> EDA/training -> API tests -> container smoke -> kind ingress deployment. Every failing command fails the job; `pipefail` preserves training and build failures through `tee`. Artifacts are uploaded even after failure. A GitHub runner's Kubernetes cluster is temporary and is not a persistent public hosting service.

`Jenkinsfile` supplies the allowed Jenkins alternative: clean Linux environment, install/lint, unit tests, tracked training, API/provenance checks, with test reports and artifacts. Docker and Kubernetes verification run separately on the local host. The supplied local Jenkins evidence documents an actual run; it must not be represented as a GitHub Actions run.

To reproduce the optional dedicated local Jenkins controller:

```bash
docker build -t heart-jenkins:local -f ci/Dockerfile.jenkins .
docker run -d --name heart-jenkins -p 127.0.0.1:8081:8080 heart-jenkins:local
```

Open http://127.0.0.1:8081/job/heart-mlops/ and choose Build Now. This disposable demonstration controller runs as the jenkins user, without host Docker socket access, and with no interactive setup; it is bound to localhost only. Do not expose it publicly or treat it as hardened production CI. For shared CI use authenticated Jenkins agents with restricted permissions or GitHub-hosted runners. Remove with `docker rm -f heart-jenkins` after assessment.

## 8. Logging and monitoring

Every request produces a JSON log with generated request ID, route template, status, method and duration; no patient fields are logged. Prometheus metrics expose request count, latency histogram and class counters. `/monitor` displays live request/error counts, mean latency, prediction count and raw exposition, with buttons demonstrating successful inference and input rejection.

This lightweight dashboard is process-local. With two replicas, requests can land on different counters; refreshes are not a cluster-wide total. For production, scrape each pod and aggregate using Prometheus (example expressions in `monitoring/README.md`). Probe/metrics requests themselves contribute to request counts. No model-accuracy drift claim is made without delayed ground-truth labels.

## 9. Publish for submission

Sign into your own GitHub account with `gh auth login` (never put credentials in files). Push this project to the supplied repository: `git push -u origin HEAD:main`. Recommended initial visibility is private, with instructor access granted as required. Run the GitHub Actions workflow and add its real URL/screenshots if using GitHub CI. Put the repository URL in the report and checklist before final submission. No fabricated repository URL or workflow success is supplied.

## 10. Rebuild the report / repeat the video

Reporting tools are optional for training/serving. Install them with `pip install -r requirements-report.txt -e .`. The report builder reads measured result JSON and genuine screenshots, and produces `output/pdf/MLOps_Assignment_1_Report.pdf`. Your name and student ID are recorded in `submission-details.json`; the supplied repository URL is also recorded there. Check that the final PDF remains exactly 10 pages.

To record the browser walkthrough, start the local API ingress, MLflow UI, Jenkins and `python -m http.server 8765 --bind 127.0.0.1` from this directory. Install Google Chrome and run `python -m playwright install ffmpeg`, then `python scripts/record-walkthrough.py`. It records real browser interactions and converts the recording to MP4. The included 97-second video is captioned and has no audio narration. `screenshots/README.md` identifies the origin of each evidence image.

The final verified deployment also uses the immutable local image tag `heart-api:release-074c8b3372d9`; its build, isolated prediction and rolling-update logs are `evidence/docker-final.log` and `evidence/kubernetes-final.log`. The model version remains `074c8b3372d9`.

## Narrated walkthrough

`reports/pipeline-walkthrough-narrated.mp4` uses the provided Zoom recording as its narration. It preserves the natural speaking pace and all spoken content, trims the initial/final silence, and applies gentle high-pass filtering and loudness normalization. The output is 1920×1080 at 24 fps with H.264 video and AAC audio. Thirty-six evidence-based scenes follow the recorded explanation; the API demonstration includes footage from the original verified browser recording.

Ten embedded chapter markers are also listed in `reports/narrated-walkthrough-chapters.txt`. `evidence/narrated-video-verification.json` records full decoding and audio-alignment checks. The raw personal Zoom recording and working transcript are excluded from the repository and submission archive.
