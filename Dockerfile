FROM ubuntu:xenial

MAINTAINER amelkonyan <sasha.melkonyan@gmail.com>

# Install system libraries
RUN apt-get update

# Install system libraries for TORCS
RUN apt-get install -y \
    libglib2.0-dev \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    libpng12-dev \
    git \
    wget \
    freeglut3-dev \
    libplib-dev \
    libopenal-dev \
    libpng12-dev \
    zlib1g-dev \
    libogg-dev \
    libvorbis-dev \
    g++ \
    libalut-dev \
    libxi-dev \
    libxmu-dev \
    libxrandr-dev \
    make \
    patch \
    xautomation  \
    libopenblas-dev \
    cmake \
    zlib1g-dev \
    libjpeg-dev \
    xvfb \
    libav-tools \
    xorg-dev \
    python-opengl \
    python3 \
    libboost-all-dev \
    libsdl2-dev \
    swig \
    vim 

# Install system libraries for deep learning
RUN apt-get install -y \
    python3-pip


WORKDIR "/root"

RUN git clone https://github.com/fmirus/torcs-1.3.7.git && \
    cd torcs-1.3.7 && \
    ./configure && \
    make && \
    make install && \
    make datainstall && \
    cd /root

WORKDIR "/root/vaperc"

ADD requirements.txt requirements.txt

RUN pip3 install -r requirements.txt

CMD ["/bin/bash"]


