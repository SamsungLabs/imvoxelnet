# ImVoxelNet: Image to Voxels Projection for Monocular and Multi-View General-Purpose 3D Object Detection

This repository contains implementation of the monocular/multi-view 3D object detector ImVoxelNet, introduced in our paper:

> **ImVoxelNet: Image to Voxels Projection for Monocular and Multi-View General-Purpose 3D Object Detection**<br>
> [Danila Rukhovich](https://github.com/filaPro),
> [Anna Vorontsova](https://github.com/highrut),
> [Anton Konushin](https://scholar.google.com/citations?user=ZT_k-wMAAAAJ)
> <br>
> Samsung AI Center Moscow <br>
> https://arxiv.org/abs/2106.01178

<p align="center"><img src="./resources/scheme.png" alt="drawing" width="90%"/></p>

### Installation
For convenience, we provide a [Dockerfile](docker/Dockerfile). Alternatively, you can install all required packages manually.

This implementation is based on [mmdetection3d](https://github.com/open-mmlab/mmdetection3d) framework.
Please refer to the original installation guide [install.md](docs/install.md).
Also, [rotated_iou](https://github.com/lilanxiao/Rotated_IoU) should be installed.

Most of the `ImVoxelNet`-related code locates in these files: 
[detectors/imvoxelnet.py](mmdet3d/models/detectors/imvoxelnet.py),
[necks/imvoxelnet.py](mmdet3d/models/necks/imvoxelnet.py),
[dense_heads/imvoxel_head.py](mmdet3d/models/dense_heads/imvoxel_head.py),
[pipelines/multi_view.py](mmdet3d/datasets/pipelines/multi_view.py).

### Datasets

We support three benchmarks based on the **SUN RGB-D** dataset.
 * Default [sunrgbd](data/sunrgbd) instruction follows the 
   [VoteNet](https://github.com/facebookresearch/votenet) benchmark with 10 object classes.
 * With minor modification the same instruction can be applied for the
   [PerspectiveNet](https://papers.nips.cc/paper/2019/hash/b87517992f7dce71b674976b280257d2-Abstract.html)
   benchmark with 30 object classes. Just run `create_data.py` with `--dataset sunrgbd_monocular`.
 * The [Total3DUnderstanding](https://github.com/yinyunie/Total3DUnderstanding)
   benchmark deals with 37 object classes along with camera pose and room layout estimation.
   Download their preprocessed data as 
   [train.json](https://github.com/saic-vul/imvoxelnet/releases/download/v1.0/sunrgbd_total_infos_train.json) and 
   [val.json](https://github.com/saic-vul/imvoxelnet/releases/download/v1.0/sunrgbd_total_infos_val.json) 
   and put it to `./data/sunrgbd`. Then run:
   ```shell
   python tools/data_converter/sunrgbd_total.py
   ```

**ScanNet.** Please follow instructions in [scannet](data/scannet).
Note that this script works with point clouds and does not accept RGB images.
RGB images may be obtained by running a script from the official [SensReader](https://github.com/ScanNet/ScanNet/tree/master/SensReader/python).
Then, put the extracted camera poses and JPG images along with other `ScanNet` data:
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
Now, run `create_data.py` with `--dataset scannet_monocular`.

For **KITTI** and **nuScenes**, please follow instructions in [getting_started.md](docs/getting_started.md).
For `nuScenes`, set `--dataset nuscenes_monocular`.

### Getting Started

Please see [getting_started.md](docs/getting_started.md) for basic usage examples.
Original [dist_train](tools/dist_train.sh) and [dist_test](tools/dist_test.sh) scripts can be run
with `ImVoxelNet` [configs](configs/imvoxelnet):
```shell
bash tools/dist_train.sh configs/imvoxelnet/imvoxelnet_kitti.py 8
bash tools/dist_test.sh configs/imvoxelnet/imvoxelnet_kitti.py \
    work_dirs/imvoxelnet_kitti/latest.pth 8 --eval mAP
```
Visualizations can be created with [test](tools/test.py) script. 
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
  journal={arXiv preprint arXiv:2106.01178},
  year={2021}
}
```
