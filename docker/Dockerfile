FROM pytorch/pytorch:1.5.1-cuda10.1-cudnn7-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub \
    && apt-get update \
    && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6

# Update torch and torchvision
RUN pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html

# Install MMCV and MMDetection
RUN pip install mmcv-full==1.2.7 -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.6.0/index.html
RUN conda install cython
RUN pip install mmdet==2.10.0

# Install MMDetection3D (ImVoxelNet)
RUN conda clean --all
RUN git clone https://github.com/samsunglabs/imvoxelnet /mmdetection3d
WORKDIR /mmdetection3d
ENV FORCE_CUDA="1"
RUN pip install -r requirements/build.txt
RUN pip install --no-cache-dir -e .

# Uninstall pycocotools installed by nuscenes-devkit and reinstall mmpycocotools
RUN pip uninstall pycocotools --no-cache-dir -y
RUN pip install mmpycocotools==12.0.3 --no-cache-dir --force --no-deps

# Install differentiable IoU
RUN git clone https://github.com/lilanxiao/Rotated_IoU /rotated_iou
RUN cp -r /rotated_iou/cuda_op /mmdetection3d/mmdet3d/ops/rotated_iou
WORKDIR /mmdetection3d/mmdet3d/ops/rotated_iou/cuda_op
RUN python setup.py install
