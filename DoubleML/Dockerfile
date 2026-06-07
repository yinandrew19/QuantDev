FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System libraries needed by compiled wheels.
# LightGBM links against libgomp (OpenMP) at import time, and it is not
# included in python:3.12-slim — without this the container fails with
# "OSError: libgomp.so.1: cannot open shared object file".
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies first (better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code last
COPY . .

CMD ["python", "mbsSpread.py"]