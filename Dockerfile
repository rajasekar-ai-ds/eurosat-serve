FROM python:3.12-slim-bookworm

RUN useradd --create-home appuser

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY app.py predict.py ./
COPY models/ ./models/

USER appuser

HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=2)"]

CMD ["uvicorn", "app:app" , "--host", "0.0.0.0", "--port", "8000"]



