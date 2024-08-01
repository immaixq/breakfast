FROM python:3.9-slim

WORKDIR /app

# Opencv dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt

RUN python -m ensurepip --upgrade
RUN pip install --no-cache-dir -r requirements.txt

COPY src /app/src
COPY tests /app/tests

WORKDIR /app

ENTRYPOINT [ "python", "src/image_deduplicator.py" ]