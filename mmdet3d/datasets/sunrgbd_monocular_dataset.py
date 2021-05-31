import numpy as np
from os import path as osp
from mmcv.utils import print_log

from mmdet.datasets import DATASETS
from .sunrgbd_dataset import SUNRGBDDataset
from mmdet3d.core.bbox import DepthInstance3DBoxes
from .dataset_wrappers import MultiViewMixin


@DATASETS.register_module()
class SUNRGBDMonocularDataset(MultiViewMixin, SUNRGBDDataset):
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


@DATASETS.register_module()
class SunRgbdMultiViewDataset(SUNRGBDMonocularDataset):
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


@DATASETS.register_module()
class SunRgbdPerspectiveMultiViewDataset(SunRgbdMultiViewDataset):
    def evaluate(self,
                 results,
                 metric=None,
                 iou_thr=(0.15,),
                 logger=None,
                 show=False,
                 out_dir=None):
        return super().evaluate(
            results=results,
            metric=metric,
            iou_thr=iou_thr,
            logger=logger,
            show=show,
            out_dir=out_dir
        )


@DATASETS.register_module()
class SunRgbdTotalMultiViewDataset(SunRgbdMultiViewDataset):
    def get_data_info(self, index):
        info = self.data_infos[index]
        input_dict = super().get_data_info(index)
        if input_dict is not None:
            input_dict['lidar2img']['angles'] = info['angles'].astype(np.float32)
            input_dict['lidar2img']['layout'] = DepthInstance3DBoxes(info['layout'][None, ...], origin=(.5, .5, .5))
        return input_dict

    def evaluate(self,
                 results,
                 metric=None,
                 iou_thr=(0.15,),
                 logger=None,
                 show=False,
                 out_dir=None):
        ret_dict = super().evaluate(
            results=results,
            metric=metric,
            iou_thr=iou_thr,
            logger=logger,
            show=show,
            out_dir=out_dir
        )
        ret_dict.update(self._evaluate_angles(results, logger))
        ret_dict.update(self._evaluate_layout(results, logger))
        return ret_dict

    def _evaluate_angles(self, results, logger):
        gt_angles = np.stack([x['angles'] for x in self.data_infos])
        angles = np.stack([x['angles'] for x in results])
        metrics = dict(
            pitch_mae=np.mean(np.abs(angles[:, 0] - gt_angles[:, 0])) * 180 / np.pi,
            roll_mae=np.mean(np.abs(angles[:, 1] - gt_angles[:, 1])) * 180 / np.pi
        )
        print_log(str(metrics), logger=logger)
        return metrics

    def _evaluate_layout(self, results, logger):
        gt_layouts = [DepthInstance3DBoxes(
            x['layout'][None, ...], origin=(.5, .5, .5)
        ) for x in self.data_infos]
        ious = [
            gt_layout.overlaps(result['layout'], gt_layout)
            for result, gt_layout in zip(results, gt_layouts)
        ]
        iou = {'layout_iou': np.mean(ious)}
        print_log(str(iou), logger=logger)
        return iou
