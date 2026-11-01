FROM python:3.12-slim-bookworm

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY app.py predict.py ./
COPY models/ ./models/

CMD ["uvicorn", "app:app" , "--host", "0.0.0.0", "--port", "8000"]



