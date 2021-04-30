import os
import numpy as np

from mmdet.datasets import DATASETS
from .kitti_dataset import KittiDataset
from .dataset_wrappers import MultiViewMixin


@DATASETS.register_module()
class KittiMultiViewDataset(MultiViewMixin, KittiDataset):
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


@DATASETS.register_module()
class KittiStereoDataset(KittiDataset):
    def get_data_info(self, index):
        info = self.data_infos[index]
        sample_idx = info['image']['image_idx']
        img_2_filename = os.path.join(self.data_root, info['image']['image_path'])
        img_3_filename = img_2_filename.replace('image_2', 'image_3')
        assert img_2_filename != img_3_filename

        rect = info['calib']['R0_rect'].astype(np.float32)
        Trv2c = info['calib']['Tr_velo_to_cam'].astype(np.float32)
        P2 = info['calib']['P2'].astype(np.float32)
        P3 = info['calib']['P3'].astype(np.float32)
        extrinsic = rect @ Trv2c
        extrinsic_2 = np.copy(extrinsic)
        extrinsic_2[:3, 3] += np.linalg.inv(P2[:3, :3]) @ P2[:3, 3]
        extrinsic_3 = np.copy(extrinsic)
        extrinsic_3[:3, 3] += np.linalg.inv(P3[:3, :3]) @ P3[:3, 3]
        intrinsic_2 = np.copy(P2)
        intrinsic_2[:3, 3] = 0
        intrinsic_3 = np.copy(P3)
        intrinsic_3[:3, 3] = 0
        assert np.allclose(intrinsic_2, intrinsic_3)

        input_dict = dict(
            sample_idx=sample_idx,
            img_prefix=[None, None],
            img_info=[dict(filename=img_2_filename), dict(filename=img_3_filename)],
            lidar2img=dict(
                extrinsic=[extrinsic_2, extrinsic_3],
                intrinsic=intrinsic_2
            )
        )

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos
        return input_dict
