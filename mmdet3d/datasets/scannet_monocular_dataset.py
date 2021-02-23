import numpy as np
import os.path as osp
from collections import defaultdict
from mmdet.datasets import DATASETS

from .custom_3d import Custom3DDataset
from .scannet_dataset import ScanNetDataset
from mmdet3d.core.bbox import DepthInstance3DBoxes


@DATASETS.register_module()
class ScanNetMultiViewDataset(Custom3DDataset):
    CLASSES = ScanNetDataset.CLASSES

    def get_data_info(self, index):
        info = self.data_infos[index]
        input_dict = defaultdict(list)
        for i in range(len(info['image_paths'])):
            img_filename = osp.join(self.data_root, info['image_paths'][i])
            input_dict['img_prefix'].append(None)
            input_dict['img_info'].append(dict(filename=img_filename))
            input_dict['lidar2img'].append(info['extrinsic'][i].astype(np.float32))
        input_dict = dict(input_dict)
        origin = np.mean(info['annos']['gt_boxes_upright_depth'][:, :3], axis=0)
        input_dict['lidar2img'] = dict(
            extrinsic=input_dict['lidar2img'],
            intrinsic=info['intrinsic'].astype(np.float32),
            origin=origin.astype(np.float32)
        )

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos
            if self.filter_empty_gt and len(annos['gt_bboxes_3d']) == 0:
                return None
        return input_dict

    def get_ann_info(self, index):
        info = self.data_infos[index]
        if info['annos']['gt_num'] != 0:
            gt_bboxes_3d = info['annos']['gt_boxes_upright_depth'].astype(
                np.float32)  # k, 6
            gt_labels_3d = info['annos']['class'].astype(np.long)
        else:
            gt_bboxes_3d = np.zeros((0, 6), dtype=np.float32)
            gt_labels_3d = np.zeros((0,), dtype=np.long)

        # to target box structure
        gt_bboxes_3d = DepthInstance3DBoxes(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1],
            with_yaw=False,
            origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d)
        return anns_results
