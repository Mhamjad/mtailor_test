# Dockerfile
FROM pytorch/pytorch:2.6.0-cuda11.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN apt update && apt install -y \
    wget \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libgdbm-dev \
    libdb5.3-dev \
    libbz2-dev \
    libexpat1-dev \
    liblzma-dev \
    tk-dev \
    curl \
    git \
    libffi-dev \
    make \
    && rm -rf /var/lib/apt/lists/*

# Download and install Python 3.10
RUN wget https://www.python.org/ftp/python/3.10.14/Python-3.10.14.tgz && \
    tar -xvf Python-3.10.14.tgz && \
    cd Python-3.10.14 && \
    ./configure --enable-optimizations && \
    make -j$(nproc) && \
    make altinstall && \
    cd .. && \
    rm -rf Python-3.10.14.tgz Python-3.10.14
 
# COPY model.onnx .
# COPY app.py model.py preprocess.py convert_to_onnx.py .
 
# expose port for Cerebrium
# EXPOSE 8080
 
# CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]