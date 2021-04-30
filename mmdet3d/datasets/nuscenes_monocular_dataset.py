import torch
import numpy as np

from mmdet.datasets import DATASETS
from .nuscenes_dataset import NuScenesDataset
from .dataset_wrappers import MultiViewMixin


@DATASETS.register_module()
class NuScenesMultiViewDataset(MultiViewMixin, NuScenesDataset):
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
            # remove gt velocity
            gt_bboxes_3d = data_info['ann_info']['gt_bboxes_3d'].tensor
            gt_bboxes_3d = gt_bboxes_3d[:, :-2]
            gt_bboxes_3d = self.box_type_3d(gt_bboxes_3d)
            # keep only car class
            gt_labels_3d = data_info['ann_info']['gt_labels_3d'].copy()
            gt_labels_3d[gt_labels_3d > 0] = -1
            mask = gt_labels_3d >= 0
            gt_bboxes_3d = gt_bboxes_3d[mask]
            gt_names = data_info['ann_info']['gt_names'][mask]
            gt_labels_3d = gt_labels_3d[mask]
            new_info['ann_info'] = dict(
                gt_bboxes_3d=gt_bboxes_3d,
                gt_names=gt_names,
                gt_labels_3d=gt_labels_3d
            )
        return new_info

    def evaluate(self, results, *args, **kwargs):
        # update boxes with zero velocity
        new_results = []
        for i in range(len(results)):
            box_type = type(results[i]['boxes_3d'])
            boxes_3d = results[i]['boxes_3d'].tensor
            boxes_3d = box_type(torch.cat((
                boxes_3d, boxes_3d.new_zeros((boxes_3d.shape[0], 2))
            ), dim=-1), box_dim=9)
            new_results.append(dict(
                boxes_3d=boxes_3d,
                scores_3d=results[i]['scores_3d'],
                labels_3d=results[i]['labels_3d']
            ))
        result_dict = super().evaluate(new_results, *args, **kwargs)
        print(result_dict)
        return result_dict
