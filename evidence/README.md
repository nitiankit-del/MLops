# Execution evidence

- `tests.log/xml`: final local pytest run, 26 passing tests.
- `lint.log`: final lint check.
- `training.log`, `release-verification.json`: actual selected release training and independently recomputed predictions/metrics.
- `jenkins-run.json`, `jenkins-console.log`, `jenkins-unit.xml`, `jenkins-api.xml`: actual successful Jenkins build #3; 16 data/model + 10 API tests, tracked training and archived artifacts.
- `jenkins-image-final.log`: successful build of the final non-root controller image. No host Docker socket is mounted in the controller.
- `docker.log`, `kubernetes.log`: original successful container and deployment demonstrations, associated with screenshots/video.
- `docker-final.log`, `kubernetes-final.log`, `deployment-final.txt`, `docker-image-final.json`: later successful build/smoke and rolling update to the immutable release image tag.
- `prediction.json`, `ready.json`, `api-requests.log`, `deployment-status.txt`: actual first deployment responses and status, used by the labelled evidence viewer.
- `environment-freeze.txt`: environment versions from execution; includes extra PDF/browser/video tooling. Use the supplied requirements files to install, not the recorded absolute editable-install path.
- `video-verification.txt`: successful full MP4 decode with ffmpeg, including duration and codec.
- `delivery-validation.json`: final document, model, test and publication status.

CI controller builds encountered setup problems during development; the submitted CI evidence is the corrected successful run, not a claimed first-attempt success. GitHub publication and hosted Actions execution remain pending authentication. No generated screenshot is presented as a hosted GitHub workflow.
