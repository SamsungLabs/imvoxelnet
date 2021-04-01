import torch
from torch import nn
from mmdet.models.detectors import BaseDetector
from mmdet.models import DETECTORS, build_backbone, build_head, build_neck

from mmdet3d.core import bbox3d2result


@DETECTORS.register_module()
class AtlasDetectorV2(BaseDetector):
    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 voxel_size,
                 n_voxels,
                 out_channels,
                 in_channels,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super().__init__()
        self.backbone = build_backbone(backbone)
        self.neck = build_neck(neck)
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = build_head(bbox_head)
        self.lateral_convs = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(n_in_channels, n_out_channels, 1, bias=False),
                nn.BatchNorm2d(n_out_channels),
                nn.ReLU(inplace=True)
            )
            for n_in_channels, n_out_channels in zip(in_channels, out_channels)
        )
        self.voxel_size = voxel_size
        self.n_voxels = n_voxels
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        super().init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        self.neck.init_weights()
        self.bbox_head.init_weights()

    def extract_feat(self, img, img_metas):
        batch_size = img.shape[0]
        img = img.reshape([-1] + list(img.shape)[2:])
        xs = self.backbone(img)
        xs = [conv(x) for x, conv in zip(xs, self.lateral_convs)]
        xs = [x.reshape([batch_size, -1] + list(x.shape[1:])) for x in xs]
        volumes, valids, points = [], [], []
        for i, img_meta in enumerate(img_metas):
            img_volumes, img_valids, img_points = [], [], []
            for level, x in enumerate(xs):
                point = get_points(
                    n_voxels=torch.tensor(self.n_voxels) // (2 ** level),
                    voxel_size=torch.tensor(self.voxel_size) * (2 ** level),
                    origin=torch.tensor(img_meta['lidar2img']['origin'])
                ).to(img.device)
                projection = get_projection(img_meta, x.shape[-2:]).to(img.device)
                volume, valid = backproject(x[i], point, projection)
                volume = volume.sum(dim=0)
                valid = valid.sum(dim=0)
                volume = volume / valid
                valid = valid > 0
                volume[:, ~valid[0]] = .0
                img_volumes.append(volume)
                img_valids.append(valid)
                img_points.append(point)
            volumes.append(img_volumes)
            valids.append(img_valids)
            points.append(img_points)
        mlvl_volumes, mlvl_valids, mlvl_points = [], [], []
        for i in range(len(xs)):
            mlvl_volumes.append(torch.stack([volume[i] for volume in volumes]))
            mlvl_valids.append(torch.stack([valid[i] for valid in valids]))
            mlvl_points.append(torch.stack([point[i] for point in points]))
        mlvl_features = self.neck(mlvl_volumes)
        return mlvl_features, mlvl_valids[:len(mlvl_features)], mlvl_points[:len(mlvl_features)]

    def forward_train(self, img, img_metas, gt_bboxes_3d, gt_labels_3d, **kwargs):
        mlvl_features, mlvl_valids, mlvl_points = self.extract_feat(img, img_metas)
        losses = self.bbox_head.forward_train(
            mlvl_features, mlvl_valids, mlvl_points, img_metas, gt_bboxes_3d, gt_labels_3d
        )
        return losses

    def forward_test(self, img, img_metas, **kwargs):
        # not supporting aug_test for now
        return self.simple_test(img, img_metas)

    def simple_test(self, img, img_metas):
        mlvl_features, mlvl_valids, mlvl_points = self.extract_feat(img, img_metas)
        x = self.bbox_head(mlvl_features)
        bbox_list = self.bbox_head.get_bboxes(*x, mlvl_valids, mlvl_points, img_metas)
        bbox_results = [
            bbox3d2result(det_bboxes, det_scores, det_labels)
            for det_bboxes, det_scores, det_labels in bbox_list
        ]
        return bbox_results

    def aug_test(self, imgs, img_metas):
        pass


def get_projection(img_meta, shape):
        projection = []
        intrinsic = torch.tensor(img_meta['lidar2img']['intrinsic'][:3, :3])
        # check if only one side is padded
        assert img_meta['img_shape'][0] == img_meta['pad_shape'][0] or \
               img_meta['img_shape'][1] == img_meta['pad_shape'][1]
        dim = 0 if img_meta['img_shape'][0] == img_meta['pad_shape'][0] else 1
        ratio = img_meta['ori_shape'][dim] / shape[dim]
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
