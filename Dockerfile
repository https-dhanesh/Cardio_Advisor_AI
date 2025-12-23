FROM python:3.9-slim

WORKDIR /app

RUN mkdir -p /app/cache
ENV HF_HOME="/app/cache"

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p models

EXPOSE 7860

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]