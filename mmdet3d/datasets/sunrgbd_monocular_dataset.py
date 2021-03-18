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


@DATASETS.register_module()
class SUNRGBDMultiViewDataset(SUNRGBDMonocularDataset):
    def get_data_info(self, index):
        info = self.data_infos[index]
        img_filename = osp.join(self.data_root, info['image']['image_path'])
        matrices = self._get_matrices(index)
        intrinsic = np.eye(4)
        intrinsic[:3, :3] = matrices['intrinsic']
        extrinsic = np.eye(4)
        extrinsic[:3, :3] = matrices['extrinsic'].T
        origin = np.array([0, 3, -1])
        input_dict = dict(
            img_prefix=[None],
            img_info=[dict(filename=img_filename)],
            lidar2img=dict(
                extrinsic=[extrinsic.astype(np.float32)],
                intrinsic=intrinsic.astype(np.float32),
                origin=origin.astype(np.float32)
            )
        )

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos
            if self.filter_empty_gt and len(annos['gt_bboxes_3d']) == 0:
                return None
        return input_dict
