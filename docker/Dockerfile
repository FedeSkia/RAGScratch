FROM python:3.11-slim

ARG APP_ENV
ENV APP_ENV=$APP_ENV

WORKDIR /app

# ---- Install Poetry ----
RUN pip install --no-cache-dir "poetry==1.8.3" && poetry --version

# ---- Copy entrypoint + env file ----
COPY docker/entrypoint_app.sh .
RUN chmod +x /app/entrypoint_app.sh
COPY ${APP_ENV} ./

# ---- Install Python deps with Poetry (cache-friendly layer) ----
COPY poetry.lock* ./
COPY pyproject.toml ./
COPY src/ /app/src
COPY README.md ./

RUN poetry install --no-ansi --without dev

# ---- Default entrypoint ----
ENTRYPOINT ["/app/entrypoint_app.sh"]
