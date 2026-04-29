FROM python:3.12-slim

# System deps: ffmpeg (for pydub / audio processing) + build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        curl \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
ENV POETRY_HOME=/opt/poetry \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1 \
    PATH="/opt/poetry/bin:/app/.venv/bin:$PATH"
RUN curl -sSL https://install.python-poetry.org | python3 -

WORKDIR /app

# Install dependencies first for better layer caching
COPY pyproject.toml poetry.lock* /app/
RUN poetry install --no-root --without dev || poetry install --no-root

# Copy app code
COPY app /app/app
COPY README.md /app/README.md

EXPOSE 8000
CMD ["fastapi", "run", "app/main.py", "--host", "0.0.0.0", "--port", "8000"]
