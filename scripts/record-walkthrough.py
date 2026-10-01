"""Record a real browser walkthrough of running local services (no fabricated frames)."""

import json
import subprocess
import urllib.request

import imageio_ffmpeg
from playwright.sync_api import sync_playwright

from heart.data import ROOT

out = ROOT / "reports/video-source"
out.mkdir(parents=True, exist_ok=True)
with urllib.request.urlopen(
    "http://127.0.0.1:8081/job/heart-mlops/lastSuccessfulBuild/api/json"
) as response:
    run = json.load(response)
assert run["result"] == "SUCCESS"
with sync_playwright() as p:
    browser = p.chromium.launch(channel="chrome", headless=True)
    context = browser.new_context(
        viewport={"width": 1440, "height": 1000},
        record_video_dir=str(out),
        record_video_size={"width": 1440, "height": 1000},
    )
    page = context.new_page()

    def stage(url, caption, seconds=10):
        page.goto(url, wait_until="networkidle")
        page.evaluate(
            """caption => {
          const e=document.createElement('div');e.textContent=caption;
          Object.assign(e.style,{position:'fixed',bottom:'0',left:'0',right:'0',padding:'18px 28px',
          background:'#0b2635',color:'white',font:'21px system-ui',zIndex:'2147483647',boxShadow:'0 -2px 10px #0003'});
          document.body.appendChild(e);
        }""",
            caption,
        )
        page.wait_for_timeout(seconds * 1000)

    stage(
        "http://127.0.0.1:8765/reports/overview.html",
        "1 / 7  Architecture: reproducible data, tracked models, CI, containers and monitored Kubernetes.",
    )
    stage(
        "http://127.0.0.1:8765/reports/eda.html",
        "2 / 7  EDA uses training rows only. Missing values are imputed within each cross-validation fold.",
    )
    stage(
        "http://127.0.0.1:8765/reports/results.html",
        "3 / 7  Nested CV selects logistic regression. Untouched holdout ROC-AUC: 0.9232; accuracy: 0.8033.",
    )
    page.goto("http://127.0.0.1:5001")
    page.get_by_text("heart-disease-cleveland", exact=True).click()
    page.wait_for_timeout(9000)
    stage(
        f"http://127.0.0.1:8081/job/heart-mlops/{run['number']}/",
        "4 / 7  Actual successful Jenkins CI: dependency install, lint, data/model tests, training, API tests and artifacts.",
    )
    stage(
        "http://127.0.0.1:8765/reports/deployment.html",
        "5 / 7  Actual saved kubectl evidence: two ready API replicas behind a Traefik Ingress.",
    )
    stage(
        "http://127.0.0.1:8080/monitor",
        "6 / 7  Live Kubernetes API and process-local Prometheus metrics. Send a real prediction now.",
        3,
    )
    page.get_by_role("button", name="Send sample prediction").click()
    page.wait_for_timeout(10000)
    page.get_by_role("button", name="Demonstrate validation (422)").click()
    page.wait_for_timeout(7000)
    page.get_by_role("button", name="Send sample prediction").click()
    page.wait_for_timeout(5000)
    stage(
        "http://127.0.0.1:8080/docs",
        "7 / 7  Validated JSON contract and OpenAPI docs. Code, 10-page report, logs and access instructions accompany this recording.",
        8,
    )
    video = page.video
    context.close()
    video.save_as(str(ROOT / "reports/pipeline-walkthrough.webm"))
    browser.close()
    print(ROOT / "reports/pipeline-walkthrough.webm")

# Produce a widely playable MP4 from the actual recorded WebM.


subprocess.run(
    [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-y",
        "-i",
        str(ROOT / "reports/pipeline-walkthrough.webm"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(ROOT / "reports/pipeline-walkthrough.mp4"),
    ],
    check=True,
)
