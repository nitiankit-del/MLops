FROM python:3.13.3-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app/src
WORKDIR /app
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt && useradd --uid 10001 --create-home appuser
COPY src ./src
COPY models ./models
COPY monitoring ./monitoring
USER 10001
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=20s --retries=3 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/ready',timeout=2)"
CMD ["uvicorn","heart.api:app","--host","0.0.0.0","--port","8000","--no-access-log"]
