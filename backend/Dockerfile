FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (needed for some python packages like psycopg2 sometimes, though binary is used)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy project definition
COPY pyproject.toml /app/

ENV PIP_ROOT_USER_ACTION=ignore
# Install python dependencies
# Using pip to install directly from pyproject.toml (requires modern pip)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

COPY src /app/src

# Run the application
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
