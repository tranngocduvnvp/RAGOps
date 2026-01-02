FROM python:3.12.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["bentoml", "serve", "bentomlRanking.py:RerankService", "--host", "0.0.0.0", "--port", "8888"]
