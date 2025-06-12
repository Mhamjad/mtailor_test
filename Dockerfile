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
    curl \
    git \
    libffi-dev \
    make \
    && rm -rf /var/lib/apt/lists/*

# Download and install Python 3.10
RUN wget https://www.python.org/ftp/python/3.10.0/Python-3.10.0.tgz && \
    tar -xvf Python-3.10.0.tgz && \
    cd Python-3.10.0 && \
    ./configure --enable-optimizations && \
    make -j$(nproc) && \
    make altinstall && \
    cd .. && \
    rm -rf Python-3.10.0.tgz Python-3.10.0 &&\
    ln -sf /usr/local/bin/python3.10 /usr/local/bin/python && \
    ln -sf /usr/local/bin/python3.10 /usr/local/bin/python3 && \
    ln -sf /usr/local/bin/pip3.10 /usr/local/bin/pip
 
COPY . /app/
 
# expose port for Cerebrium
EXPOSE 8080
 
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]