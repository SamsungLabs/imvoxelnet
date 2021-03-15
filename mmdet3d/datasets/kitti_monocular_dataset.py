import os
import numpy as np

from mmdet.datasets import DATASETS
from .kitti_dataset import KittiDataset


@DATASETS.register_module()
class KittiMultiViewDataset(KittiDataset):
    def get_data_info(self, index):
        info = self.data_infos[index]
        sample_idx = info['image']['image_idx']
        img_filename = os.path.join(self.data_root, info['image']['image_path'])

        rect = info['calib']['R0_rect'].astype(np.float32)
        Trv2c = info['calib']['Tr_velo_to_cam'].astype(np.float32)
        P2 = info['calib']['P2'].astype(np.float32)
        extrinsic = rect @ Trv2c
        extrinsic[:3, 3] += np.linalg.inv(P2[:3, :3]) @ P2[:3, 3]
        intrinsic = np.copy(P2)
        intrinsic[:3, 3] = 0

        input_dict = dict(
            sample_idx=sample_idx,
            img_prefix=[None],
            img_info=[dict(filename=img_filename)],
            lidar2img=dict(
                extrinsic=[extrinsic],
                intrinsic=intrinsic
            )
        )

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos
        return input_dict
