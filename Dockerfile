# Пульс маршрута: один образ для ML-сервиса, backend и скриптов (обучение, сабмит, replay)
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt requirements-docs.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-docs.txt
COPY . .
# документация кода (Sphinx) — отдаётся backend'ом по /code-docs/
RUN python -m sphinx -q -b html docs/source docs/html || echo "sphinx: сборка документации пропущена"
ENV PYTHONPATH=/app PYTHONUNBUFFERED=1 HOME=/tmp
