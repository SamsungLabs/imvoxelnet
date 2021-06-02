# ImVoxelNet: Image to Voxels Projection for Monocular and Multi-View General-Purpose 3D Object Detection

This repository contains implementation of the monocular/multi-view 3D object detector ImVoxelNet, introduced in our paper:

> **ImVoxelNet: Image to Voxels Projection for Monocular and Multi-View General-Purpose 3D Object Detection**<br>
> [Danila Rukhovich](https://github.com/filaPro),
> [Anna Vorontsova](https://github.com/highrut),
> [Anton Konushin](https://scholar.google.com/citations?user=ZT_k-wMAAAAJ)
> <br>
> Samsung AI Center Moscow <br>
> https://arxiv.org/abs/210?.?????

<p align="center"><img src="./resources/scheme.png" alt="drawing" width="90%"/></p>

### Installation
For convenience, we provide a [Dockerfile](docker/Dockerfile). Alternatively, you can install all required packages manually.

This implementation is based on [mmdetection3d](https://github.com/open-mmlab/mmdetection3d) framework. Please refer to the original installation guide [install.md](docs/install.md).
Also, [rotated_iou](https://github.com/lilanxiao/Rotated_IoU) should be installed.

### Datasets

We support three benchmarks based on the **SUN RGB-D** dataset. 
 * The [VoteNet] (https://github.com/facebookresearch/votenet) benchmark 
Use [sunrgbd](data/sunrgbd) to generate annotations with official 10 object categories.
 * The [PerspectiveNet] (https://papers.nips.cc/paper/2019/hash/b87517992f7dce71b674976b280257d2-Abstract.html) benchmark
To use 30 object categories as proposed in the PerspectiveNet paper, set `--dataset sunrgbd_monocular`.
 * The [Total3DUnderstanding](https://github.com/yinyunie/Total3DUnderstanding) benchmark
There are 37 object categories in this benchmark. It also includes camera pose and room layout estimation.
First of all, you should download preprocessed data as 
   [train.json](https://github.com/saic-vul/imvoxelnet/releases/download/v1.0/sunrgbd_total_infos_train.json) and 
   [val.json](https://github.com/saic-vul/imvoxelnet/releases/download/v1.0/sunrgbd_total_infos_val.json) 
   and move the downloaded JSON files into the `./data/sunrgbd` directory. Then, you should convert data to a proper format by running:
   ```shell
   python tools/data_converter/sunrgbd_total.py
   ```

**ScanNet.** Please follow instructions in [scannet](data/scannet).
Note that this script works with point clouds and does not accept RGB images.
You may obtain point clouds by running a script from the official [SensReader](https://github.com/ScanNet/ScanNet/tree/master/SensReader/python).
Then, put the extracted camera poses and JPG images along with other `scannet` data. The dataset should be organized as follows:
```
scannet
├── sens_reader
│   ├── scans
│   │   ├── scene0000_00
│   │   │   ├── out
│   │   │   │   ├── frame-000001.color.jpg
│   │   │   │   ├── frame-000001.pose.txt
│   │   │   │   ├── frame-000002.color.jpg
│   │   │   │   ├── ....
│   │   ├── ...
```
Now, run `create_data.py`

For **KITTI** and **nuScenes**, please follow instructions in [getting_started.md](docs/getting_started.md).
For `nuScenes`, set `--dataset nuscenes_monocular`.

### Getting Started

Please see [getting_started.md](docs/getting_started.md) for basic usage examples.
We provide configs for ImVoxelNet [configs](configs/imvoxelnet), and publish scripts for [training](tools/dist_train.sh) and 
[testing](tools/dist_test.sh):
```shell
bash tools/dist_train.sh configs/imvoxelnet/imvoxelnet_kitti.py 8
bash tools/dist_test.sh configs/imvoxelnet/imvoxelnet_kitti.py \
    work_dirs/imvoxelnet_kitti/latest.pth 8 --eval mAP
```
Visualizations can be created with [testing](tools/test.py) script. 
For better visualizations, you may set `score_thr` in configs to `0.15` or more.
```shell
python tools/test.py configs/imvoxelnet/imvoxelnet_kitti.py \
    work_dirs/imvoxelnet_kitti/latest.pth --show
```

### Models

| Dataset   | Object Classes | Download Link | Log |
|:---------:|:--------------:|:-------------:|:---:|
| SUN RGB-D | 37 from Total3dUnderstanding | [total_sunrgbd.pth](https://github.com/saic-vul/imvoxelnet/releases/download/v1.0/20210525_091810.pth) | [total_sunrgbd.log](https://github.com/saic-vul/imvoxelnet/releases/download/v1.0/20210525_091810_atlas_total_sunrgbd.log) |
| SUN RGB-D | 30 from PerspectiveNet | [perspective_sunrgbd.pth](https://github.com/saic-vul/imvoxelnet/releases/download/v1.0/20210526_072029.pth) | [perspective_sunrgbd.log](https://github.com/saic-vul/imvoxelnet/releases/download/v1.0/20210526_072029_atlas_perspective_sunrgbd.log) |
| SUN RGB-D | 10 from VoteNet | [sunrgbd.pth](https://github.com/saic-vul/imvoxelnet/releases/download/v1.0/20210428_124351.pth) | [sunrgbd.log](https://github.com/saic-vul/imvoxelnet/releases/download/v1.0/20210428_124351_atlas_sunrgbd.log) |
| ScanNet   | 18 from VoteNet | [scannet.pth](https://github.com/saic-vul/imvoxelnet/releases/download/v1.0/20210520_223109.pth) | [scannet.log](https://github.com/saic-vul/imvoxelnet/releases/download/v1.0/20210520_223109_atlas_scannet.log) |
| KITTI     | Car | [kitti.pth](https://github.com/saic-vul/imvoxelnet/releases/download/v1.0/20210503_214214.pth) | [kitti.log](https://github.com/saic-vul/imvoxelnet/releases/download/v1.0/20210503_214214_atlas_kitti.log) |
| nuScenes  | Car | [nuscenes.pth](https://github.com/saic-vul/imvoxelnet/releases/download/v1.0/20210505_131108.pth) | [nuscenes.log](https://github.com/saic-vul/imvoxelnet/releases/download/v1.0/20210505_131108_atlas_nuscenes.log) |

### Example Detections

<p align="center"><img src="./resources/github.png" alt="drawing" width="90%"/></p>

### Citation

If you find this work useful for your research, please cite our paper:
```
@article{rukhovich2021imvoxelnet,
  title={ImVoxelNet: Image to Voxels Projection for Monocular and Multi-View General-Purpose 3D Object Detection},
  author={Danila Rukhovich, Anna Vorontsova, Anton Konushin},
  journal={arXiv preprint arXiv:210?.?????},
  year={2021}
}
```
