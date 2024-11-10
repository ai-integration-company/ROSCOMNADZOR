FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu20.04 AS builder

WORKDIR /app
COPY ./requirements.txt .

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y python3.9 python3.9-distutils git wget curl unzip && \ 
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3.9 get-pip.py && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip install kaggle gdown

RUN kaggle datasets download -d stachyslanata/sekrrno && \
    unzip sekrrno.zip -d /app/ && \
    rm sekrrno.zip

FROM builder AS stage

WORKDIR /app

RUN pip install --no-cache-dir -r requirements.txt 

RUN gdown "https://drive.google.com/uc?export=download&id=1gqA-cp3CLLsX9a9URK5GXhwrYTUs47Vy"

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]