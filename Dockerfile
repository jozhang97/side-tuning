# This Docker was originally set up for Habitat Challenge

#FROM nvidia/cudagl:9.0-base-ubuntu16.04
FROM fairembodied/habitat-challenge:latest

RUN apt-get update && apt-get install -y --no-install-recommends \
        cuda-samples-$CUDA_PKG_VERSION && \
    rm -rf /var/lib/apt/lists/*
WORKDIR /usr/local/cuda/samples/5_Simulations/nbody
RUN make

#CMD ./nbody

RUN apt-get update && apt-get install -y curl  && apt-get install -y apt-utils && apt-get install -y ffmpeg
RUN conda update -y conda

RUN . activate habitat
ENV PATH /opt/conda/envs/habitat/bin:$PATH


###############################
#  set up habitat
###############################
WORKDIR /root/side-tuning
RUN git clone https://github.com/facebookresearch/habitat-sim.git
RUN git clone https://github.com/facebookresearch/habitat-api.git

WORKDIR /root/side-tuning/habitat-sim
RUN conda install -y cmake
RUN pip install numpy
RUN python setup.py install --headless

WORKDIR /root/side-tuning/habitat-api
RUN git checkout 05dbf7220e8386eb2337502c4d4851fc8dce30cd
RUN pip install --upgrade -e .
ADD habitat_data /root/side-tuning/habitat-api/data
RUN rm -r /root/side-tuning/habitat-api/configs
ADD habitat_configs /root/side-tuning/habitat-api/configs
RUN rm -r baselines


###############################
#  set up side-tuning
###############################
ADD requirements.txt /root/side-tuning/requirements.txt
ADD __init__.py      /root/side-tuning/__init__.py
ADD assets           /root/side-tuning/assets
ADD configs          /root/side-tuning/configs
ADD evkit            /root/side-tuning/evkit
ADD feature_selector /root/side-tuning/feature_selector
ADD scripts          /root/side-tuning/scripts
ADD tlkit            /root/side-tuning/tlkit
WORKDIR /root/side-tuning
RUN pip install -r requirements.txt
RUN ln -s habitat-api/data .


###############################
#  set up baselines
###############################
WORKDIR /root
RUN apt-get update && apt-get install -y cmake libopenmpi-dev python3-dev zlib1g-dev
RUN git clone https://github.com/openai/baselines.git; cd baselines; pip install -e .


######################################
# install tnt
######################################
ADD tnt /root/side-tuning/tnt
WORKDIR /root/side-tuning/tnt
RUN  pip install -e .


######################################
# and... we are ready!
######################################
WORKDIR /root/side-tuning
RUN pip install gym==0.10.9