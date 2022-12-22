FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    graphviz \
 && apt-get -y clean \
 && rm -rf /var/lib/apt/lists/*

RUN pip install torchviz