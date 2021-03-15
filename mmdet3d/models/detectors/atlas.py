import torch
from mmdet.models import DETECTORS, build_backbone, build_head, build_neck
from mmdet.models.detectors import BaseDetector

from mmdet3d.core import bbox3d2result


@DETECTORS.register_module()
class AtlasDetector(BaseDetector):
    def __init__(self,
                 backbone,
                 neck=None,
                 neck_3d=None,
                 bbox_head=None,
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
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        super().init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        self.neck.init_weights()
        self.neck_3d.init_weights()
        self.bbox_head.init_weights()

    def forward_train(self, img, img_metas, gt_bboxes_3d, gt_labels_3d, **kwargs):
        x = self.extract_feat(img)

        voxel_size = torch.tensor(self.train_cfg['voxel_size']).float()
        volume, valid = [], []
        for feature, img_meta in zip(x, img_metas):
            projection = self._compute_projection(img_meta, feature.shape[-2:])
            origin = torch.tensor(img_meta['lidar2img']['origin'])
            origin = get_origin(
                n_voxels=torch.tensor(self.train_cfg['n_voxels']),
                voxel_size=voxel_size,
                origin=origin
            )
            img_volume, img_valid = backproject(
                voxel_dim=self.train_cfg['n_voxels'],
                voxel_size=voxel_size.reshape(1, 3, 1).to(x.device),
                origin=origin.reshape(1, 3),
                projection=projection.to(x.device),
                features=feature
            )
            volume.append(img_volume)
            valid.append(img_valid)
        volume = torch.stack(volume)
        valid = torch.stack(valid)
        volume = volume.sum(dim=1)
        valid = valid.sum(dim=1)
        x = volume / valid
        x = x.transpose(0, 1)
        valid = valid > 0
        x[:, ~valid[:, 0]] = .0
        x = x.transpose(0, 1)

        x = self.neck_3d(x)
        losses = self.bbox_head.forward_train(x, valid.float(), img_metas, gt_bboxes_3d, gt_labels_3d)
        return losses

    def extract_feat(self, img):
        batch_size = img.shape[0]
        img = img.reshape([-1] + list(img.shape)[2:])
        x = self.backbone(img)
        x = self.neck(x)[0]
        return x.reshape([batch_size, -1] + list(x.shape[1:]))

    def forward_test(self, img, img_metas, **kwargs):
        # not supporting aug_test for now
        return self.simple_test(img, img_metas)

    def simple_test(self, img, img_metas):
        x = self.extract_feat(img)

        ys = []
        voxel_size = torch.tensor(self.test_cfg['voxel_size']).float()
        for features, img_meta in zip(x, img_metas):
            projections = self._compute_projection(img_meta, features.shape[-2:])
            origin = torch.tensor(img_meta['lidar2img']['origin'])
            volume, valid = .0, 0
            for feature, projection in zip(features, projections):
                origin = get_origin(
                    n_voxels=torch.tensor(self.test_cfg['n_voxels']),
                    voxel_size=voxel_size,
                    origin=origin
                )
                img_volume, img_valid = backproject(
                    voxel_dim=self.test_cfg['n_voxels'],
                    voxel_size=voxel_size.reshape(1, 3, 1).to(x.device),
                    origin=origin.reshape(1, 3),
                    projection=projection[None, ...].to(x.device),
                    features=feature[None, ...]
                )
                volume += img_volume[0]
                valid += img_valid[0]
            y = volume / valid
            y[:, valid[0] == 0] = .0
            ys.append(y)
        x = torch.stack(ys)

        x = self.neck_3d(x)
        x = self.bbox_head(x)
        bbox_list = self.bbox_head.get_bboxes(*x, img_metas)
        bbox_results = [
            bbox3d2result(det_bboxes, det_scores, det_labels)
            for det_bboxes, det_scores, det_labels in bbox_list
        ]
        return bbox_results


    def aug_test(self, imgs, img_metas):
        pass

    @staticmethod
    def _compute_projection(img_meta, shape):
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


def get_origin(n_voxels, voxel_size, origin):
    """
    Args:
        n_voxels (Tensor): sizes of the volume of shape (3,)
        voxel_size (Tensor): sizes of one voxesl of shape (3,)
        origin (Tensor): coordinates of the volume's center of shape (3,)
    Returns:
        Tensor: coordinates of the origin of shape (3,)
    """
    return origin - n_voxels / 2. * voxel_size


# copy from https://github.com/magicleap/Atlas/blob/master/atlas/tsdf.py
def coordinates(voxel_dim, device=torch.device('cuda')):
    """ 3d meshgrid of given size.
    Args:
        voxel_dim: tuple of 3 ints (nx,ny,nz) specifying the size of the volume
    Returns:
        torch long tensor of size (3,nx*ny*nz)
    """

    nx, ny, nz = voxel_dim
    x = torch.arange(nx, dtype=torch.long, device=device)
    y = torch.arange(ny, dtype=torch.long, device=device)
    z = torch.arange(nz, dtype=torch.long, device=device)
    x, y, z = torch.meshgrid(x, y, z)
    return torch.stack((x.flatten(), y.flatten(), z.flatten()))


# copy from https://github.com/magicleap/Atlas/blob/master/atlas/model.py
def backproject(voxel_dim, voxel_size, origin, projection, features):
    """ Takes 2d features and fills them along rays in a 3d volume
    This function implements eqs. 1,2 in https://arxiv.org/pdf/2003.10432.pdf
    Each pixel in a feature image corresponds to a ray in 3d.
    We fill all the voxels along the ray with that pixel's features.
    Args:
        voxel_dim: size of voxel volume to construct (nx,ny,nz)
        voxel_size: metric size of each voxel (ex: .04m)
        origin: origin of the voxel volume (xyz position of voxel (0,0,0))
        projection: bx4x3 projection matrices (intrinsics@extrinsics)
        features: bxcxhxw  2d feature tensor to be backprojected into 3d
    Returns:
        volume: b x c x nx x ny x nz 3d feature volume
        valid:  b x 1 x nx x ny x nz volume.
                Each voxel contains a 1 if it projects to a pixel
                and 0 otherwise (not in view frustrum of the camera)
    """

    batch = features.size(0)
    channels = features.size(1)
    device = features.device
    nx, ny, nz = voxel_dim

    coords = coordinates(voxel_dim, device).unsqueeze(0).expand(batch, -1, -1)  # bx3xhwd
    world = coords.type_as(projection) * voxel_size + origin.to(device).unsqueeze(2)
    world = torch.cat((world, torch.ones_like(world[:, :1])), dim=1)

    camera = torch.bmm(projection, world)
    px = (camera[:, 0, :] / camera[:, 2, :]).round().type(torch.long)
    py = (camera[:, 1, :] / camera[:, 2, :]).round().type(torch.long)
    pz = camera[:, 2, :]

    # voxels in view frustrum
    height, width = features.size()[2:]
    valid = (px >= 0) & (py >= 0) & (px < width) & (py < height) & (pz > 0)  # bxhwd

    # put features in volume
    volume = torch.zeros(batch, channels, nx * ny * nz, dtype=features.dtype,
                         device=device)
    for b in range(batch):
        volume[b, :, valid[b]] = features[b, :, py[b, valid[b]], px[b, valid[b]]]

    volume = volume.view(batch, channels, nx, ny, nz)
    valid = valid.view(batch, 1, nx, ny, nz)

    return volume, valid
