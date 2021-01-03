import numpy as np
from os import path as osp

from mmdet3d.core.bbox import DepthInstance3DBoxes
from mmdet.datasets import DATASETS
from .sunrgbd_dataset import SUNRGBDDataset


@DATASETS.register_module()
class SUNRGBDMonocularDataset(SUNRGBDDataset):
    def __init__(self,
                 data_root,
                 ann_file,
                 pipeline=None,
                 classes=None,
                 modality=None,
                 box_type_3d='Depth',
                 filter_empty_gt=True,
                 test_mode=False):
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            classes=classes,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode)

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - file_name (str): Filename of point clouds.
                - ann_info (dict): Annotation info.
        """
        info = self.data_infos[index]
        sample_idx = info['point_cloud']['lidar_idx']
        pts_filename = osp.join(self.data_root, info['pts_path'])
        # TODO: [9:] + '.jpg' here is a temporary hack due to the bug in sunrgbd_data_utils.py
        img_filename = osp.join(self.data_root, info['image']['image_path'][53:] + '.jpg')
        input_dict = dict(
            pts_filename=pts_filename,
            sample_idx=sample_idx,
            file_name=pts_filename,
            img_prefix=None,
            img_info=dict(filename=img_filename),
            lidar2img=self._get_matrices(index)
        )

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos
            if self.filter_empty_gt and len(annos['gt_bboxes_3d']) == 0:
                return None
        return input_dict

    def get_ann_info(self, index):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`DepthInstance3DBoxes`): \
                    3D ground truth bboxes
                - gt_labels_3d (np.ndarray): Labels of ground truths.
                - pts_instance_mask_path (str): Path of instance masks.
                - pts_semantic_mask_path (str): Path of semantic masks.
        """
        # Use index to get the annos, thus the evalhook could also use this api
        info = self.data_infos[index]
        if info['annos']['gt_num'] != 0:
            gt_bboxes_3d = info['annos']['gt_boxes_upright_depth'].astype(
                np.float32)  # k, 6
            gt_labels_3d = info['annos']['class'].astype(np.long)
        else:
            gt_bboxes_3d = np.zeros((0, 7), dtype=np.float32)
            gt_labels_3d = np.zeros((0, ), dtype=np.long)

        # to target box structure
        gt_bboxes_3d = DepthInstance3DBoxes(
            gt_bboxes_3d, origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)

        return {
            'gt_bboxes_3d': gt_bboxes_3d,
            'gt_labels_3d': gt_labels_3d
        }

    def _get_matrices(self, index):
        info = self.data_infos[index]

        intrinsic = info['calib']['K'].copy().reshape(3, 3).T
        extrinsic = info['calib']['Rt'].copy()
        extrinsic[:, [1, 2]] = extrinsic[:, [2, 1]]
        extrinsic[:, 1] = -1 * extrinsic[:, 1]

        return dict(intrinsic=intrinsic, extrinsic=extrinsic)

    def get_cat_ids(self, idx):
        """Get category ids by index.

        Args:
            idx (int): Index of data.

        Returns:
            list[int]: All categories in the image of specified index.
        """
        if self.data_infos[idx]['annos']['gt_num'] != 0:
            return self.data_infos[idx]['annos']['class'].astype(np.int).tolist()
        else:
            return []

    def evaluate(self,
                 results,
                 metric=None,
                 iou_thr=(0.15,),
                 logger=None,
                 show=False,
                 out_dir=None):
        return super().evaluate(results, metric, iou_thr, logger, show, out_dir)
