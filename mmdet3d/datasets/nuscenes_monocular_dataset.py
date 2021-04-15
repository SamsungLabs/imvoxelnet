import numpy as np
from mmdet.datasets import DATASETS
from .nuscenes_dataset import NuScenesDataset


@DATASETS.register_module()
class NuScenesMultiViewDataset(NuScenesDataset):
    def get_data_info(self, index):
        data_info = super().get_data_info(index)
        n_cameras = len(data_info['img_filename'])
        assert n_cameras == 6
        new_info = dict(
            sample_idx=data_info['sample_idx'],
            img_prefix=[None] * n_cameras,
            img_info=[dict(filename=x) for x in data_info['img_filename']],
            lidar2img=dict(
                extrinsic=[x.astype(np.float32) for x in data_info['lidar2img']],
                intrinsic=np.eye(4, dtype=np.float32)
            )
        )
        if 'ann_info' in data_info:
            new_info['ann_info'] = data_info['ann_info']
        return new_info
