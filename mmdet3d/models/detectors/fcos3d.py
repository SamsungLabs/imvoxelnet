from mmcv.runner import auto_fp16
from mmdet.models import DETECTORS
from mmdet3d.core import bbox3d2result, merge_aug_bboxes_3d
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
        """Adapted from VoteNet."""
        feats = self.extract_feats(imgs)

        # only support aug_test for one sample
        aug_bboxes = []
        for x, img_meta in zip(feats, img_metas):
            bbox_preds = self.bbox_head(x, self.test_cfg.sample_mod)
            bbox_list = self.bbox_head.get_bboxes(
                bbox_preds, img_meta, rescale=rescale)
            bbox_list = [
                dict(boxes_3d=bboxes, scores_3d=scores, labels_3d=labels)
                for bboxes, scores, labels in bbox_list
            ]
            aug_bboxes.append(bbox_list[0])

        # after merging, bboxes will be rescaled to the original image size
        merged_bboxes = merge_aug_bboxes_3d(aug_bboxes, img_metas,
                                            self.bbox_head.test_cfg)

        return [merged_bboxes]
