import torch
from mmcv.runner import auto_fp16
from mmdet.models import DETECTORS
from mmdet3d.core import bbox3d2result, merge_aug_bboxes_3d, Box3DMode
from mmdet.models.detectors import SingleStageDetector


@DETECTORS.register_module()
class FCOS3D(SingleStageDetector):
    @auto_fp16(apply_to=('img',))
    def forward(self, return_loss=True, **kwargs):
        """Adapted from Base3DDetector."""
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)

    def forward_train(self, img_metas, img, gt_bboxes_3d, gt_labels_3d, **kwargs):
        # NOTE the batched image size information may be useful, e.g.
        # in DETR, this is needed for the construction of masks, which is
        # then used for the transformer_head.
        batch_intput_shape = tuple(img[0].size()[-2:])
        for img_meta in img_metas:
            img_meta['batch_intput_shape'] = batch_intput_shape
        x = self.extract_feat(img)
        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes_3d, gt_labels_3d)
        return losses

    def forward_test(self, img_metas, img, **kwargs):
        """Adapted from Base3DDetector"""
        for var, name in [(img, 'img'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))

        num_augs = len(img)
        if num_augs != len(img_metas):
            raise ValueError(
                'num of augmentations ({}) != num of image meta ({})'.format(
                    len(img), len(img_metas)))

        # NOTE the batched image size information may be useful, e.g.
        # in DETR, this is needed for the construction of masks, which is
        # then used for the transformer_head.
        for img_, img_meta in zip(img, img_metas):
            batch_size = len(img_meta)
            for img_id in range(batch_size):
                img_meta[img_id]['batch_intput_shape'] = tuple(img_.size()[-2:])

        if num_augs == 1:
            img = [img] if img is None else img
            return self.simple_test(img_metas[0], img[0], **kwargs)
        else:
            return self.aug_test(img_metas, img, **kwargs)

    def simple_test(self, img_metas, img, rescale=False):
        """Adapted from VoteNet."""
        x = self.extract_feat(img)
        bbox_preds = self.bbox_head(x, img_metas)
        bbox_list = self.bbox_head.get_bboxes(
            *bbox_preds, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results

    def aug_test(self, img_metas, imgs, rescale=False):
        raise NotImplementedError


@DETECTORS.register_module()
class NuScenesMultiViewFCOS3D(FCOS3D):
    def aug_test(self, img_metas, imgs, rescale=False):
        bbox_results = []
        for img_meta, img in zip(img_metas, imgs):
            bbox_results.append(self.simple_test(img_meta, img, rescale)[0])
            boxes = bbox_results[-1]['boxes_3d']
            boxes.translate(-img_meta[0]['lidar2img']['global'])
            boxes = boxes.convert_to(Box3DMode.LIDAR)
            bbox_results[-1]['boxes_3d'] = boxes.tensor
        bbox_results_dict = {key: [] for key in bbox_results[0]}
        for data in bbox_results:
            for key, val in data.items():
                bbox_results_dict[key].append(val)
        # TODO: use nms
        for key in bbox_results_dict:
            bbox_results_dict[key] = torch.cat(bbox_results_dict[key])
        mask = torch.argsort(bbox_results_dict['scores_3d'], descending=True)[:self.test_cfg['max_per_scene']]
        for key in bbox_results_dict:
            bbox_results_dict[key] = bbox_results_dict[key][mask]
        bbox_results_dict['boxes_3d'] = img_metas[0][0]['box_type_3d'](bbox_results_dict['boxes_3d'], origin=(.5, .5, .5))
        return [bbox_results_dict]
