FROM pytorch/pytorch:2.2.2-cuda11.8-cudnn8-runtime AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN /opt/conda/bin/pip install --no-cache-dir -r requirements.txt


FROM nvidia/cuda:11.8.0-base-ubuntu22.04

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PATH /opt/conda/bin:$PATH

WORKDIR /app

COPY --from=builder /opt/conda /opt/conda

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

COPY . .

CMD ["sleep", "infinity"]

