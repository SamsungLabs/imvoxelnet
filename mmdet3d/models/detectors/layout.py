from mmdet.models import DETECTORS, build_backbone, build_head
from mmdet.models.detectors import BaseDetector


@DETECTORS.register_module()
class LayoutDetector(BaseDetector):
    def __init__(self,
                 backbone,
                 head_2d,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super().__init__()
        self.backbone = build_backbone(backbone)
        self.head_2d = build_head(head_2d)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        super().init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        self.head_2d.init_weights()

    def extract_feat(self, img, img_metas, mode):
        img = img.reshape([-1] + list(img.shape)[2:])
        x = self.backbone(img)
        features_2d = self.head_2d.forward(x[-1], img_metas)
        return features_2d

    def forward_train(self, img, img_metas, gt_bboxes_3d, gt_labels_3d, **kwargs):
        features_2d = self.extract_feat(img, img_metas, 'train')
        losses = dict()
        losses.update(self.head_2d.loss(*features_2d, img_metas))
        return losses

    def forward_test(self, img, img_metas, **kwargs):
        # not supporting aug_test for now
        return self.simple_test(img, img_metas)

    def simple_test(self, img, img_metas):
        features_2d = self.extract_feat(img, img_metas, 'test')
        bbox_results = [dict(boxes_3d=[], scores_3d=[], labels_3d=[]) for _ in range(len(img))]
        angles, layouts = self.head_2d.get_bboxes(*features_2d, img_metas)
        for i in range(len(img)):
            bbox_results[i]['angles'] = angles[i]
            bbox_results[i]['layout'] = layouts[i]
        return bbox_results

    def aug_test(self, imgs, img_metas):
        pass
