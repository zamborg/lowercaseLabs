FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN pip install --no-cache-dir poetry==2.1.2

COPY backend/pyproject.toml /app/pyproject.toml
COPY backend/poetry.lock /app/poetry.lock
RUN poetry config virtualenvs.create false \
    && poetry install --only main --no-interaction --no-ansi

COPY backend/app /app/app
COPY backend/alembic /app/alembic
COPY backend/alembic.ini /app/alembic.ini
COPY backend/model_assets /app/model_assets

EXPOSE 8080

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
