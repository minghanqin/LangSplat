FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# Do not check for keyboard type and answer with default
ENV DEBIAN_FRONTEND=noninteractive

# Arguments passed along with build command
ARG user
ARG uid

# Add user with the same user name, id as the host
RUN useradd -m -G root -u ${uid} ${user}

# Set up CUDA environment variables
ENV PATH="/usr/local/cuda-12.1/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda-12.1/lib64:${LD_LIBRARY_PATH}"
ENV CUDA_PATH="/usr/local/cuda-12.1"
ENV CUDA_HOME="/usr/local/cuda-12.1"
# To match your GPU architecture for PyTorch build refer to the page below
# (https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/)
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6 8.9"

# Install dependencies
RUN apt-get update \
    && apt-get install -y \
    && apt-get install -y \
    git gcc-12 g++-12 \
    ninja-build build-essential x11-apps \
    wget bzip2 tar unzip imagemagick \
    libglew-dev libglm-dev libassimp-dev \
    mesa-utils libglu1-mesa-dev freeglut3-dev \
    libglx-dev libglvnd-dev \
    libboost-all-dev libgtk-3-dev \
    libopencv-dev libglfw3-dev libatlas-base-dev \
    libavdevice-dev libavcodec-dev \
    libeigen3-dev libxxf86vm-dev libembree-dev \
    libsuitesparse-dev libfreeimage-dev \
    libmetis-dev libgoogle-glog-dev \
    libgflags-dev libpcl-dev \
    qtbase5-dev libqt5opengl5-dev \
    libcgal-dev libcgal-qt5-dev

# Install cmake for building packages and programs
RUN wget https://github.com/Kitware/CMake/releases/download/v3.24.4/cmake-3.24.4.tar.gz \
    && tar -xzf cmake-3.24.4.tar.gz \
    && cd cmake-3.24.4 \
    && ./bootstrap && make && make install

RUN rm -rf cmake-3.24.4 && rm cmake-3.24.4.tar.gz

# Install Ceres Solver which COLMAP depends on
RUN wget http://ceres-solver.org/ceres-solver-2.1.0.tar.gz \
    && tar zxf ceres-solver-2.1.0.tar.gz \
    && mkdir ceres-bin \
    && cd ceres-bin \
    && cmake ../ceres-solver-2.1.0 -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF -DCMAKE_CUDA_ARCHITECTURES=native \
    && make -j3 \
    && make test \
    && make install

RUN rm -rf ceres-solver-2.1.0 && rm -rf ceres-bin && rm ceres-solver-2.1.0.tar.gz

# Install COLMAP
RUN wget https://github.com/colmap/colmap/archive/refs/tags/3.9.1.zip \
    && unzip 3.9.1.zip \
    && cd colmap-3.9.1 \
    && mkdir build \
    && cd build \
    && cmake .. -GNinja -DCMAKE_CUDA_ARCHITECTURES=all-major \
    && ninja \
    && ninja install

RUN rm -rf colmap-3.9.1 3.9.1.zip

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py311_23.11.0-2-Linux-x86_64.sh \
    && bash Miniconda3-py311_23.11.0-2-Linux-x86_64.sh -b -p /home/${user}/miniconda \
    && rm Miniconda3-py311_23.11.0-2-Linux-x86_64.sh

ENV PATH="/home/${user}/miniconda/bin:${PATH}"

# Change ownership of Miniconda to the new user so that conda init works for the new user shell
RUN chown -R ${uid}:${uid} /home/${user}/miniconda

USER ${user}

RUN conda init bash

WORKDIR /home/${user}

# Clone my Gaussian-splatting repository
RUN git clone https://github.com/notingcode/gaussian-splatting.git --recursive

WORKDIR /home/${user}/gaussian-splatting

# Copy package requirements from Docker directory to container
COPY requirements_conda.txt .

# Install all package dependencies
RUN conda install -y --file requirements_conda.txt -c pytorch -c nvidia -c defaults -c conda-forge
RUN pip install ./submodules/diff-gaussian-rasterization
RUN pip install ./submodules/simple-knn

# Clean up
USER root

RUN apt-get clean && rm -rf /var/lib/apt/lists/* && rm -rf /home/${user}/gaussian-splatting
RUN pip cache purge
RUN conda clean -y -a

# Working directory set to Gaussain-splatting when container runs
WORKDIR /home/${user}/gaussian-splatting
USER ${user}

# Reset DEBIAN_FRONTEND to default value
ENV DEBIAN_FRONTEND=dialog

# Set bash to default shell
SHELL [ "/bin/bash", "-c" ]