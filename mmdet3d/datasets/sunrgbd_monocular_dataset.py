import numpy as np
from os import path as osp

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
        img_filename = osp.join(self.data_root, info['image']['image_path'])
        input_dict = dict(
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
