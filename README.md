# ImVoxelNet: Image to Voxels Projection for Monocular and Multi-View General-Purpose 3D Object Detection

This project hosts the code for implementing monocular and multi-view 3D object detector ImVoxelNet,
as presented in our paper:

> **ImVoxelNet: Image to Voxels Projection for Monocular and Multi-View General-Purpose 3D Object Detection**<br>
> [Danila Rukhovich](https://github.com/filaPro),
> [Anna Vorontsova](https://github.com/highrut),
> [Anton Konushin](https://scholar.google.com/citations?user=ZT_k-wMAAAAJ)
> <br>
> Samsung AI Center Moscow <br>
> https://arxiv.org/abs/210?.?????

<p align="center"><img src="./resources/scheme.png" alt="drawing" width="90%"/></p>

### Installation

This implementation is based on [mmdetection3d](https://github.com/open-mmlab/mmdetection3d) framework.
Please refer to original [install.md](docs/install.md) for installation. 
Also [rotated_iou](https://github.com/lilanxiao/Rotated_IoU) should be installed.
We recommend to follow our [Dockerfile](docker/Dockerfile).

### Datasets

We support 3 benchmarks for the **SUN RGB-D** dataset. 
 * Please follow [sunrgbd](data/sunrgbd). 
This will generate annotations with official 10 object classes from 
[VoteNet](https://github.com/facebookresearch/votenet).
 * To use 30 object classes from 
[PerspectiveNet](https://papers.nips.cc/paper/2019/hash/b87517992f7dce71b674976b280257d2-Abstract.html) 
set `--dataset sunrgbd_monocular`.
 * [Total3DUnderstanding](https://github.com/yinyunie/Total3DUnderstanding) benchmark deals with
37 object classes along with camera pose and room layout estimation.
   Download their preprocessed data for [train]() and [val]() and put it to `./data/sunrgbd`. Then run:
   ```shell
   python tools/data_converter/sunrgbd_total.py
   ```

**ScanNet.** Please follow [scannet](data/scannet) not forgetting to set 
`--dataset scannet_monocular`.
Note that this script deals with point clouds and do not accept RGB images.
To get them from raw data please run official 
[SensReader](https://github.com/ScanNet/ScanNet/tree/master/SensReader/python).
Then before running `create_data.py` put extracted camera poses and `.jpg` images 
along with other `scannet` data:
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


For **KITTI** and **nuScenes** please follow [getting_started.md](docs/getting_started.md).
Just set `--dataset nuscenes_monocular` for `nuScenes`.

### Getting Started

Please see original [getting_started.md](docs/getting_started.md) for basic usage examples.
ImVoxelNet [configs](configs/imvoxelnet) can be used for [train](tools/dist_train.sh) and 
[test](tools/dist_test.sh) scripts:
```shell
bash tools/dist_train.sh configs/imvoxelnet/imvoxelnet_kitti.py 8
bash tools/dist_test.sh configs/imvoxelnet/imvoxelnet_kitti.py \
    work_dirs/imvoxelnet_kitti/latest.pth 8 --eval mAP
```
For visualization [test](tools/test.py) script can be used. 
Also, for better visualizations set `score_thr` in configs to `0.15` or more.
```shell
python tools/test.py configs/imvoxelnet/imvoxelnet_kitti.py \
    work_dirs/imvoxelnet_kitti/latest.pth --show
```

### Models

| Dataset   | Object Classes | Download Link | Log |
|:---------:|:--------------:|:-------------:|:---:|
| SUN RGB-D | 37 from Total3dUnderstanding | [total_sunrgbd.pth]() | [total_sunrgbd.log]() |
| SUN RGB-D | 30 from PerspectiveNet | [perspective_sunrgbd.pth]() | [perspective_sunrgbd.log]() |
| SUN RGB-D | 10 from VoteNet | [sunrgbd.pth]() | [sunrgbd.log]() |
| ScanNet   | 18 from VoteNet | [scannet.pth]() | [scannet.log]() |
| KITTI     | Car | [kitti.pth]() | [kitti.log]() |
| nuScenes  | Car | [nuscenes.pth]() | [nuscenes.log]() |

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
