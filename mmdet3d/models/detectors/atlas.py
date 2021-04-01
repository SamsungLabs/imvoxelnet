import torch
from mmdet.models import DETECTORS, build_backbone, build_head, build_neck
from mmdet.models.detectors import BaseDetector

from mmdet3d.core import bbox3d2result


@DETECTORS.register_module()
class AtlasDetector(BaseDetector):
    def __init__(self,
                 backbone,
                 neck,
                 neck_3d,
                 bbox_head,
                 n_voxels,
                 voxel_size,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super().__init__()
        self.backbone = build_backbone(backbone)
        self.neck = build_neck(neck)
        self.neck_3d = build_neck(neck_3d)
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = build_head(bbox_head)
        self.bbox_head.voxel_size = voxel_size
        self.n_voxels = n_voxels
        self.voxel_size = voxel_size
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        super().init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        self.neck.init_weights()
        self.neck_3d.init_weights()
        self.bbox_head.init_weights()

    def extract_feat(self, img, img_metas):
        batch_size = img.shape[0]
        img = img.reshape([-1] + list(img.shape)[2:])
        x = self.backbone(img)
        x = self.neck(x)[0]
        x = x.reshape([batch_size, -1] + list(x.shape[1:]))

        stride = img.shape[-1] / x.shape[-1]
        assert stride == 4  # may be removed in the future
        stride = int(stride)

        volumes, valids = [], []
        for feature, img_meta in zip(x, img_metas):
            projection = self._compute_projection(img_meta, stride).to(x.device)
            points = get_points(
                n_voxels=torch.tensor(self.n_voxels),
                voxel_size=torch.tensor(self.voxel_size),
                origin=torch.tensor(img_meta['lidar2img']['origin'])
            ).to(x.device)
            height = img_meta['img_shape'][0] // stride
            width = img_meta['img_shape'][1] // stride
            volume, valid = backproject(feature[:, :, :height, :width], points, projection)
            volume = volume.sum(dim=0)
            valid = valid.sum(dim=0)
            volume = volume / valid
            valid = valid > 0
            volume[:, ~valid[0]] = .0
            volumes.append(volume)
            valids.append(valid)
        x = torch.stack(volumes)
        valids = torch.stack(valids)
        x = self.neck_3d(x)
        return x, valids


    def forward_train(self, img, img_metas, gt_bboxes_3d, gt_labels_3d, **kwargs):
        x, valids = self.extract_feat(img, img_metas)
        losses = self.bbox_head.forward_train(x, valids.float(), img_metas, gt_bboxes_3d, gt_labels_3d)
        return losses

    def forward_test(self, img, img_metas, **kwargs):
        # not supporting aug_test for now
        return self.simple_test(img, img_metas)

    def simple_test(self, img, img_metas):
        x, valids = self.extract_feat(img, img_metas)
        x = self.bbox_head(x)
        bbox_list = self.bbox_head.get_bboxes(*x, valids.float(), img_metas)
        bbox_results = [
            bbox3d2result(det_bboxes, det_scores, det_labels)
            for det_bboxes, det_scores, det_labels in bbox_list
        ]
        return bbox_results

    def aug_test(self, imgs, img_metas):
        pass

    @staticmethod
    def _compute_projection(img_meta, stride):
        projection = []
        intrinsic = torch.tensor(img_meta['lidar2img']['intrinsic'][:3, :3])
        ratio = img_meta['ori_shape'][0] / (img_meta['img_shape'][0] / stride)
        intrinsic[:2] /= ratio
        for extrinsic in img_meta['lidar2img']['extrinsic']:
            projection.append(intrinsic @ torch.tensor(extrinsic[:3]))
        return torch.stack(projection)


@torch.no_grad()
def get_points(n_voxels, voxel_size, origin):
    points = torch.stack(torch.meshgrid([
        torch.arange(n_voxels[0]),
        torch.arange(n_voxels[1]),
        torch.arange(n_voxels[2])
    ]))
    new_origin = origin - n_voxels / 2. * voxel_size
    points = points * voxel_size.view(3, 1, 1, 1) + new_origin.view(3, 1, 1, 1)
    return points


# modify from https://github.com/magicleap/Atlas/blob/master/atlas/model.py
def backproject(features, points, projection):
    n_images, n_channels, height, width = features.shape
    n_x_voxels, n_y_voxels, n_z_voxels = points.shape[-3:]
    points = points.view(1, 3, -1).expand(n_images, 3, -1)
    points = torch.cat((points, torch.ones_like(points[:, :1])), dim=1)
    points_2d_3 = torch.bmm(projection, points)
    x = (points_2d_3[:, 0] / points_2d_3[:, 2]).round().long()
    y = (points_2d_3[:, 1] / points_2d_3[:, 2]).round().long()
    z = points_2d_3[:, 2]
    valid = (x >= 0) & (y >= 0) & (x < width) & (y < height) & (z > 0)
    volume = torch.zeros((n_images, n_channels, points.shape[-1]), device=features.device)
    for i in range(n_images):
        volume[i, :, valid[i]] = features[i, :, y[i, valid[i]], x[i, valid[i]]]
    volume = volume.view(n_images, n_channels, n_x_voxels, n_y_voxels, n_z_voxels)
    valid = valid.view(n_images, 1, n_x_voxels, n_y_voxels, n_z_voxels)
    return volume, valid
