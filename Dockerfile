# Single image used for all three service roles (api / worker / dashboard).
# The role is chosen by the compose command, not by separate images.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install deps first for layer caching.
COPY requirements.txt requirements-service.txt ./
RUN pip install -r requirements-service.txt

# App code.
COPY . .

EXPOSE 8000 8501

# Default role: API. Override `command:` in compose for worker / dashboard.
CMD ["uvicorn", "service.api:app", "--host", "0.0.0.0", "--port", "8000"]
