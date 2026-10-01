# Submission checklist — 50-mark rubric

This maps requirements to actual files. It is not a promise of a grade.

| Requirement | Marks | Implementation and evidence |
|---|---:|---|
| Acquisition, cleaning, EDA | 5 | `src/heart/data.py`, `src/heart/eda.py`, official raw file, cleaned CSV/manifest, training-only histograms/heatmap/class balance |
| Feature engineering and models | 8 | `src/heart/model.py`, `train.py`; leakage-aware transforms; logistic and forest; nested CV; `reports/results.json`, candidate tables and holdout predictions |
| Experiment tracking | 5 | MLflow parent/family/candidate runs; `reports/experiments.csv/json`; `mlruns/`; UI screenshot |
| Packaging and reproducibility | 7 | complete joblib Pipeline, metadata/checksums, pinned requirements, saved split, independent verification script |
| CI/CD and automated tests | 8 | `test/` (26 cases); `.github/workflows/ci.yml`; executed Jenkins alternative (`Jenkinsfile`), raw console/run records, test reports and screenshots |
| Docker API | 5 | Dockerfile; `/predict`; validated JSON; prediction/confidence; actual build and isolated smoke evidence |
| Deployment | 7 | kind cluster; Traefik Ingress; 2 API replicas; manifests, real kubectl/HTTP evidence, screenshot, local access instructions |
| Monitoring and logging | 3 | structured request logs without patient payloads; Prometheus metrics; live `/monitor`; success and 422 captures |
| Documentation/report | 2 | README, architecture, exactly 10-page PDF, CI/deployment captures, video; repository publication pending below |
| Total | 50 | See measured results; final grading belongs to the instructor |

## Required handoff items before submitting

- [ ] Provide/authenticate the GitHub account, publish the project and add the **real repository URL** to `submission-details.json`, README and report. An authenticated account and repository destination were not supplied; no hosted repository is claimed.
- [ ] Add your name/student ID if required by the course submission system. These details were not supplied and have not been guessed.
- [ ] Review and understand the work; follow course rules for independent work and any required assistance disclosure. Explain data splitting, fold-local imputation, nested CV, confidence semantics, CI failure handling, and Kubernetes routing yourself.
- [ ] If using GitHub Actions as the assessed CI instead of the included executed Jenkins alternative, push and retain its actual successful workflow URL/screenshots. The GitHub workflow is provided but no hosted execution is claimed.

## Included submission files

- `output/pdf/MLOps_Assignment_1_Report.pdf` — exactly 10 pages.
- `reports/pipeline-walkthrough-narrated.mp4` — detailed narrated walkthrough, 13:19, with synchronized project evidence and recorded API interactions.
- `reports/pipeline-walkthrough.mp4` — original 97-second recorded browser demonstration.
- `screenshots/` — API, monitoring, validation, MLflow, Jenkins and deployment evidence.
- `evidence/` — raw container, Kubernetes, training, test and CI evidence.
- `README.md` — clean setup, training, inference, Docker, local Kubernetes, MLflow, CI and cleanup.

## Add personal details and rebuild report

Create `submission-details.json` with real values, for example keys `name`, `student_id` and `repository_url`. Do not submit invented values. Then:

```bash
pip install -r requirements-report.txt -e .
python scripts/build-report.py
```

Keep exactly 10 PDF pages after adding details. The supplied model/metrics are actual measured results; do not replace them with invented scores. The local deployment has no public API URL; localhost access instructions satisfy the local-testing alternative.
