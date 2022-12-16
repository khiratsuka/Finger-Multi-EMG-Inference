FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

RUN conda install matplotlib
RUN conda install tqdm
RUN pip3 install opencv-python
RUN pip3 install opencv-contrib-python
RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y libgl1-mesa-dev
RUN pip3 install scipy
RUN pip3 install xmltodict
