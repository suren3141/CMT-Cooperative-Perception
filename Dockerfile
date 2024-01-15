ARG PYTORCH="1.9.1"
ARG CUDA="11.1"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

## To fix GPG key error when running apt-get update
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        ccache \
        cmake \
        curl \
        git \
        libjpeg-dev \
        libpng-dev && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 wget nano unzip \
    && apt-get clean

# Install MMCV, MMDetection and MMSegmentation
# ARG PYTORCH
# ARG CUDA
# ARG MMCV
# ARG MMDET
# ARG MMSEG
# RUN ["/bin/bash", "-c", "pip install --no-cache-dir mmcv-full==1.6.2"]
#-f https://download.openmmlab.com/mmcv/dist/cu${CUDA//./}/torch${PYTORCH}/index.html"]
# RUN pip install --no-cache-dir mmdet==${MMDET} mmsegmentation==${MMSEG}

# Install MMDetection3D
# RUN conda clean --all
# COPY . /mmdetection3d
# WORKDIR /mmdetection3d
# ENV FORCE_CUDA="1"
# RUN pip install -r requirements/build.txt
# RUN pip install --no-cache-dir -e .

ENV MMCV="==1.6.2"
ENV MMDET="==2.28.2"
ENV MMSEG="==0.30.0"
# ARG MMCV="1.6.0"
# ARG MMDET="2.24.0"
# ARG MMSEG="0.29.1"

RUN pip install openmim
RUN mim install "mmcv-full${MMCV}" "mmdet${MMDET}" "mmsegmentation${MMSEG}"

WORKDIR /workspace
RUN wget https://github.com/open-mmlab/mmdetection3d/archive/refs/tags/v1.0.0rc6.zip
RUN unzip v1.0.0rc6.zip && mv mmdetection3d-1.0.0rc6 mmdetection3d

WORKDIR /workspace/mmdetection3d
# RUN git clone https://github.com/open-mmlab/mmdetection3d.git
ENV FORCE_CUDA="1"
RUN pip install -r requirements/build.txt
RUN pip install --no-cache-dir -e .


## CMT --------------------------------------
RUN pip install spconv-cu111==2.1.21
RUN pip install flash-attn==0.2.2  
# RUN pip install flash-attn==1.0.4

RUN pip install open3d && \
    pip install --upgrade git+https://github.com/klintan/pypcd.git

COPY ./ /workspace/CMTCoop/
#WORKDIR /workspace
#RUN git clone https://github.com/junjie18/CMT.git && cd CMT
RUN ln -s /workspace/mmdetection3d /workspace/CMTCoop/


## add dataset ------------------------------------
WORKDIR /workspace/data
RUN ln -s /mnt/datasets/00_a9_dataset /dataset
RUN ln -s /mnt/datasets/00_a9_dataset /workspace/data/ && \
    ln -s /mnt/mydataset/a9_nusc /workspace/data/a9_r02_dataset_processed && \
    ln -s /mnt/mydataset/a9_nusc_new /workspace/data/a9_r02_dataset_new_processed

# RUN ln -s /mnt/datasets/09_nuscenes_mini_new /workspace/data/nuscenes && \
#     ln -s /mnt/datasets/00_a9_dataset/a9_r02_dataset /workspace/data/a9 && \
#     ln -s /mnt/mydataset/a9_nusc /workspace/data/a9_nusc && \
#     ln -s /mnt/datasets/00_a9_dataset/a9_r02_dataset_new /workspace/data/a9_new && \
#     ln -s /mnt/mydataset/a9_nusc_new /workspace/data/a9_nusc_new

## softlink data to repo --------------------------------
RUN ln -s /workspace/data /workspace/CMTCoop/data
RUN rm -rf /workspace/mmdetection3d/data && \
    ln -s /workspace/data /workspace/mmdetection3d/data


## Copy files to mmdetection3d --------------------------------
# RUN cp -r /workspace/CMT/files/configs/* /workspace/mmdetection3d/configs/ && \
#     cp -r /workspace/CMT/files/mmdet3d/* /workspace/mmdetection3d/mmdet3d/ && \
#     cp -r /workspace/CMT/files/tools/* /workspace/mmdetection3d/tools/


## Copy weights
#ADD ./files /workspace/files
#RUN ln -s /workspace/files/ckpts /workspace/CMT/ckpts

# RUN rm -rf /workspace/CMT/projects/configs && \
#    ln -s /workspace/mmdetection3d/configs/cmt /workspace/CMT/projects/configs


# #RUN export PYTHONPATH="/workspace/CMT:$PYTHONPATH"
ENV PYTHONPATH "$PYTHONPATH:/workspace/CMT"

WORKDIR /workspace/CMTCoop
