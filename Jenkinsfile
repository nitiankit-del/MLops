pipeline {
  agent any
  options { timeout(time: 30, unit: 'MINUTES') }
  environment { MPLBACKEND = 'Agg'; PYTHONUNBUFFERED = '1' }
  stages {
    stage('Source') {
      steps { deleteDir(); sh 'cp -R /submission/. .; mkdir -p evidence; python3 -m venv .venv' }
    }
    stage('Install and lint') {
      steps { sh '.venv/bin/pip install -r requirements.txt -e .; .venv/bin/ruff check .' }
    }
    stage('Data and unit tests') {
      steps { sh '.venv/bin/python -m heart.data; .venv/bin/pytest test/test_data.py test/test_model.py --junitxml=evidence/unit.xml' }
    }
    stage('EDA and tracked training') {
      steps { sh '.venv/bin/python -m heart.eda; .venv/bin/python -m heart.train' }
    }
    stage('API tests and provenance') {
      steps { sh '.venv/bin/pytest test/test_api.py --junitxml=evidence/api.xml; .venv/bin/python scripts/verify-release.py' }
    }
  }

  post {
    always {
      junit allowEmptyResults: true, testResults: 'evidence/*.xml'
      archiveArtifacts allowEmptyArchive: true, artifacts: 'models/**,reports/**,evidence/**,mlruns/**'
    }
  }
}
