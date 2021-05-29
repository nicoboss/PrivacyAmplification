FROM nvidia/cuda:11.3.0-devel-ubuntu20.04
MAINTAINER Nico Bosshard
LABEL version="1.0"
LABEL description="Fast Privacy Amplification implementation using CUDA"
EXPOSE 22
WORKDIR /root
ENV TERM xterm-256color
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y \
    rsyslog \
    openssh-server \
    iputils-ping \
    iproute2 \
    libzmq3-dev \
    wget \
    git \
    tmux \
    htop \
    nano \
    rhash \
    gdb
RUN echo "alias tmux='tmux -u'" >> ~/.bashrc && \
    echo 'set -g default-terminal "screen-256color"' >> ~/.tmux.conf
RUN mkdir .ssh && chmod 700 .ssh
COPY PrivacyAmplificationDocker.pub .ssh/authorized_keys
RUN chmod 755 .ssh/authorized_keys
RUN git clone --recursive https://oauth2:3_wtn2uTreSXt9q1ihXV@gitlab.enterpriselab.ch/qkd/gpu/privacyamplification.git \
    && chmod +x privacyamplification/run.sh
WORKDIR /root/privacyamplification
RUN make -j 4
CMD /bin/bash --init-file ./run.sh
