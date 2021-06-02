FROM pytorch/pytorch:1.5.1-cuda10.1-cudnn7-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Update torch and torchvision
RUN pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html

# Install MMCV
RUN pip install mmcv-full==1.2.7+torch1.6.0+cu101 -f https://openmmlab.oss-accelerate.aliyuncs.com/mmcv/dist/index.html
RUN pip install mmdet==2.10.0

# Install MMDetection
RUN conda clean --all
RUN git clone https://github.com/saic-vul/imvoxelnet.git /mmdetection3d
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
