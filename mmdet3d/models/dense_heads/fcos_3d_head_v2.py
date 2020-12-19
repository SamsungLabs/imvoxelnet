import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import Scale, normal_init
from mmcv.runner import force_fp32
from mmdet.core import multi_apply, distance2bbox
from mmdet.models.dense_heads import AnchorFreeHead
from mmdet3d.models.builder import HEADS, build_loss
from mmdet3d.core import box3d_multiclass_nms, xywhr2xyxyr

INF = 1e8


@HEADS.register_module()
class FCOS3DHeadV2(AnchorFreeHead):
    """Anchor-free head used in `FCOS <https://arxiv.org/abs/1904.01355>`_.

    The FCOS head does not use anchor boxes. Instead bounding boxes are
    predicted at each pixel and a centerness measure is used to supress
    low-quality predictions.
    Here norm_on_bbox, centerness_on_reg, dcn_on_last_conv are training
    tricks used in official repo, which will bring remarkable mAP gains
    of up to 4.9. Please see https://github.com/tianzhi0549/FCOS for
    more detail.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        strides (list[int] | list[tuple[int, int]]): Strides of points
            in multiple feature levels. Default: (4, 8, 16, 32, 64).
        regress_ranges (tuple[tuple[int, int]]): Regress range of multiple
            level points.
        center_sampling (bool): If true, use center sampling. Default: False.
        center_sample_radius (float): Radius of center sampling. Default: 1.5.
        norm_on_bbox (bool): If true, normalize the regression targets
            with FPN strides. Default: False.
        centerness_on_reg (bool): If true, position centerness on the
            regress branch. Please refer to https://github.com/tianzhi0549/FCOS/issues/89#issuecomment-516877042.
            Default: False.
        conv_bias (bool | str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias of conv will be set as True if `norm_cfg` is None, otherwise
            False. Default: "auto".
        loss_cls (dict): Config of classification loss.
        loss_bbox (dict): Config of localization loss.
        loss_centerness (dict): Config of centerness loss.
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: norm_cfg=dict(type='GN', num_groups=32, requires_grad=True).

    Example:
        >>> self = FCOSHead(11, 7)
        >>> feats = [torch.rand(1, 7, s, s) for s in [4, 8, 16, 32, 64]]
        >>> cls_score, bbox_pred, centerness = self.forward(feats)
        >>> assert len(cls_score) == len(self.scales)
    """  # noqa: E501

    def __init__(self,
                 num_classes,
                 in_channels,
                 regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 512),
                                 (512, INF)),
                 center_sampling=False,
                 center_sample_radius=1.5,
                 norm_on_bbox=False,
                 centerness_on_reg=False,
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_bbox=dict(type='IoULoss', loss_weight=1.0),
                 loss_bbox_3d=dict(type='GIoU3DLoss', loss_weight=1.0),
                 loss_centerness=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 loss_center=dict(type='L1Loss', loss_weight=.001),
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 **kwargs):
        self.regress_ranges = regress_ranges
        self.center_sampling = center_sampling
        self.center_sample_radius = center_sample_radius
        self.norm_on_bbox = norm_on_bbox
        self.centerness_on_reg = centerness_on_reg
        super().__init__(
            num_classes,
            in_channels,
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            norm_cfg=norm_cfg,
            **kwargs)
        self.loss_bbox_3d = build_loss(loss_bbox_3d)
        self.loss_centerness = build_loss(loss_centerness)
        self.loss_center = build_loss(loss_center)

    def _init_predictor(self):
        """Initialize predictor layers of the head."""
        self.conv_cls = nn.Conv2d(
            self.feat_channels, self.cls_out_channels, 3, padding=1)
        # 8 (x, y) pairs, h
        self.conv_reg = nn.Conv2d(self.feat_channels, 17, 3, padding=1)

    def _init_layers(self):
        """Initialize layers of the head."""
        super()._init_layers()
        self.conv_centerness = nn.Conv2d(self.feat_channels, 1, 3, padding=1)
        self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])

    def init_weights(self):
        """Initialize weights of the head."""
        super().init_weights()
        normal_init(self.conv_centerness, std=0.01)

    def forward(self, feats):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple:
                cls_scores (list[Tensor]): Box scores for each scale level, \
                    each is a 4D-tensor, the channel number is \
                    num_points * num_classes.
                bbox_preds (list[Tensor]): Box energies / deltas for each \
                    scale level, each is a 4D-tensor, the channel number is \
                    num_points * 4.
                centernesses (list[Tensor]): Centerss for each scale level, \
                    each is a 4D-tensor, the channel number is num_points * 1.
        """
        return multi_apply(self.forward_single, feats, self.scales,
                           self.strides)

    def forward_single(self, x, scale, stride):
        """Forward features of a single scale levle.

        Args:
            x (Tensor): FPN feature maps of the specified stride.
            scale (:obj: `mmcv.cnn.Scale`): Learnable scale module to resize
                the bbox prediction.
            stride (int): The corresponding stride for feature maps, only
                used to normalize the bbox prediction when self.norm_on_bbox
                is True.

        Returns:
            tuple: scores for each class, bbox predictions and centerness \
                predictions of input feature maps.
        """
        cls_score, bbox_3d_pred, cls_feat, reg_feat = super().forward_single(x)
        if self.centerness_on_reg:
            centerness = self.conv_centerness(reg_feat)
        else:
            centerness = self.conv_centerness(cls_feat)
        # scale the bbox_pred of different level
        # float to avoid overflow when enabling FP16
        bbox_pred = bbox_3d_pred[:, :16, :, :]
        bbox_pred = scale(bbox_pred).float()
        if self.norm_on_bbox:
            # bbox_pred = F.relu(bbox_pred)
            if not self.training:
                bbox_pred *= stride
        # else:
        #     bbox_pred = bbox_pred.exp()

        bbox_3d_pred = torch.cat((
            bbox_pred,
            torch.exp(bbox_3d_pred[:, 16:, :, :])
        ), dim=1)

        return cls_score, bbox_3d_pred, centerness

    def forward_train(self, x, img_metas, gt_bboxes_3d, gt_labels_3d):
        outs = self(x)
        return self.loss(*outs, img_metas, gt_bboxes_3d, gt_labels_3d)

    @staticmethod
    def _bboxes_3d_to_2d(img_metas, gt_bboxes_3d, device):
        gt_bboxes = []
        _gt_bboxes_3d = []
        for img_meta, bboxes_3d in zip(img_metas, gt_bboxes_3d):
            if not bboxes_3d.tensor.shape[0]:
                gt_bboxes.append(torch.empty((0, 4), device=device))
                continue
            corners_3d = bboxes_3d.corners
            corners_3d = corners_3d.reshape(-1, 3)
            centers_3d = bboxes_3d.gravity_center
            intrinsic = torch.tensor(img_meta['lidar2img']['intrinsic'], dtype=torch.float32)
            extrinsic = torch.tensor(img_meta['lidar2img']['extrinsic'], dtype=torch.float32)
            points_2d_3 = (intrinsic @ extrinsic.T @ corners_3d.T).T
            centers_2d_3 = (intrinsic @ extrinsic.T @ centers_3d.T).T
            points_2d = points_2d_3[:, :2] / points_2d_3[:, 2:]
            centers_2d = centers_2d_3[:, :2] / centers_2d_3[:, 2:]
            points_2d = points_2d.reshape(-1, 8, 2)
            x_min = torch.min(points_2d[..., 0], dim=1)[0]
            y_min = torch.min(points_2d[..., 1], dim=1)[0]
            x_max = torch.max(points_2d[..., 0], dim=1)[0]
            y_max = torch.max(points_2d[..., 1], dim=1)[0]
            corners_2d = torch.stack((
                centers_2d[:, 0] - (x_max - x_min) / 2.,
                centers_2d[:, 1] - (y_max - y_min) / 2.,
                centers_2d[:, 0] + (x_max - x_min) / 2.,
                centers_2d[:, 1] + (y_max - y_min) / 2.)).T
            scale_factor = torch.tensor(img_meta['scale_factor'], dtype=torch.float32)
            corners_2d = corners_2d * scale_factor
            points_2d = points_2d * scale_factor[:2][None, None, :]
            gt_bboxes.append(corners_2d.to(device))
            _gt_bboxes_3d.append(torch.cat((
                points_2d.reshape(-1, 16), bboxes_3d.tensor[..., 5:6]
            ), dim=-1).to(device))
        return gt_bboxes, _gt_bboxes_3d

    @staticmethod
    def _bboxes_2d_to_3d_(img_metas, bboxes_3d, points):
        corners_2d = bboxes_3d[..., :16].reshape(
            bboxes_3d.shape[0], bboxes_3d.shape[1], 8, 2
        ) + points.unsqueeze(2).expand(
            bboxes_3d.shape[0], bboxes_3d.shape[1], 8, 2
        )
        # scale_factors = torch.tensor(
        #     [img_meta['scale_factor'][:2] for img_meta in img_metas],
        #     dtype=torch.float32,
        #     device=bboxes_3d.device
        # )
        # corners_2d = corners_2d / scale_factors[:, None, None, :]  # TODO: do we need this?
        return torch.cat((
            corners_2d.reshape(bboxes_3d.shape[0], bboxes_3d.shape[1], 16),
            bboxes_3d[..., 16:]
        ), dim=-1)

    @staticmethod
    def _bboxes_2d_to_3d(img_metas, bboxes_3d, points):
        # print(bboxes_3d[0, 193])
        # print(points[0, 193])
        control_points = torch.tensor([
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0]
        ], dtype=torch.float32)
        control_corners = img_metas[0]['box_type_3d'](
            torch.tensor([[.5, .5, 0., 1., 1., 1., .0]])
        ).corners
        corners_4 = F.pad(control_corners, (0, 1), value=1.)
        points_4 = F.pad(control_points, (0, 1), value=1.)
        alphas = torch.matmul(corners_4, torch.inverse(points_4))

        cameras = []
        for img_meta in img_metas:
            intrinsic = torch.tensor(img_meta['lidar2img']['intrinsic'], dtype=torch.float32)
            extrinsic = torch.tensor(img_meta['lidar2img']['extrinsic'], dtype=torch.float32)
            camera = extrinsic @ torch.inverse(intrinsic)
            cameras.append(camera)
        cameras = torch.stack(cameras).to(bboxes_3d.device)
        corners_2d = bboxes_3d[..., :16].reshape(
            bboxes_3d.shape[0], bboxes_3d.shape[1], 8, 2
        ) + points.unsqueeze(2).expand(
            bboxes_3d.shape[0], bboxes_3d.shape[1], 8, 2
        )
        scale_factors = torch.tensor(
            [img_meta['scale_factor'][:2] for img_meta in img_metas],
            dtype=torch.float32,
            device=bboxes_3d.device
        )
        corners_2d = corners_2d / scale_factors[:, None, None, :]

        corners_2d_3 = F.pad(corners_2d, (0, 1), value=1.)
        corners_3d = (cameras.unsqueeze(1) @ corners_2d_3.transpose(-2, -1)).transpose(-2, -1)
        corners_3d_2 = corners_3d[..., :2] / corners_3d[..., 2:]
        alphas = alphas.expand(
            bboxes_3d.shape[0], bboxes_3d.shape[1], 8, 4
        ).to(bboxes_3d.device)

        # [[0, 0, 0], --> [[x , y , z1],
        #  [0, 0, 1],      [x , y , z2],
        #  [0, 1, 0],      [x1, y1, z1],
        #  [1, 0, 0]]      [x2, y2, z1]]
        # 8 variables: x, x1, x2, y, y1, y2, z1, z2
        # TODO: how to reparametrize without z2?
        matrix_x = torch.stack((
            alphas[..., 0] + alphas[..., 1],
            alphas[..., 2],
            alphas[..., 3],
            torch.zeros_like(alphas[..., 0]),
            torch.zeros_like(alphas[..., 0]),
            torch.zeros_like(alphas[..., 0]),
            -corners_3d_2[..., 0] * (alphas[..., 0] + alphas[..., 2] + alphas[..., 3]),
            -corners_3d_2[..., 0] * alphas[..., 1]
        ), dim=-1)
        matrix_y = torch.stack((
            torch.zeros_like(alphas[..., 0]),
            torch.zeros_like(alphas[..., 0]),
            torch.zeros_like(alphas[..., 0]),
            alphas[..., 0] + alphas[..., 1],
            alphas[..., 2],
            alphas[..., 3],
            -corners_3d_2[..., 1] * (alphas[..., 0] + alphas[..., 2] + alphas[..., 3]),
            -corners_3d_2[..., 1] * alphas[..., 1]
        ), dim=-1)
        matrix = torch.cat((matrix_x, matrix_y), dim=-2)
        symmetric_matrix = matrix.transpose(-2, -1) @ matrix
        solution = torch.symeig(symmetric_matrix, eigenvectors=True).eigenvectors[..., 0]

        # TODO: do we need to multiply by sign(y)?
        scale = bboxes_3d[..., -1] / (solution[..., 7] - solution[..., 6])
        solution = solution * scale[..., None]

        # approximating parallelogram as a rotated rectangle
        v1 = torch.stack((
            solution[..., 1] - solution[..., 0],
            solution[..., 4] - solution[..., 3]
        ), dim=-1)
        v2 = torch.stack((
            solution[..., 2] - solution[..., 0],
            solution[..., 5] - solution[..., 3]
        ), dim=-1)
        v1, v2 = v1 / 2. + v2 / 2., v1 / 2. - v2 / 2.
        l1, l2 = torch.norm(v1, dim=-1, keepdim=True), torch.norm(v2, dim=-1, keepdim=True)
        l = l1 / 2. + l2 / 2.
        v1, v2 = v1 * l / l1, v2 * l / l2
        v1, v2 = v1 + v2, v1 - v2
        l1, l2 = torch.norm(v1, dim=-1), torch.norm(v2, dim=-1)
        alpha = torch.atan2(v1[..., 0], v1[..., 1])

        return torch.stack((
            solution[..., 1] / 2. + solution[..., 2] / 2.,
            solution[..., 4] / 2. + solution[..., 5] / 2.,
            solution[..., 6],
            l2,
            l1,
            solution[..., 7] - solution[..., 6],
            alpha
        ), dim=-1)

    @force_fp32(apply_to=('cls_scores', 'bbox_3d_preds', 'centernesses'))
    def loss(self,
             cls_scores,
             bbox_3d_preds,
             centernesses,
             img_metas,
             gt_bboxes_3d,
             gt_labels_3d,
             gt_bboxes_ignore=None):
        """Compute loss of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_points * num_classes.
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is
                num_points * 4.
            centernesses (list[Tensor]): Centerss for each scale level, each
                is a 4D-tensor, the channel number is num_points * 1.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert len(cls_scores) == len(bbox_3d_preds) == len(centernesses)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.get_points(featmap_sizes, bbox_3d_preds[0].dtype,
                                           bbox_3d_preds[0].device)

        gt_bboxes, _gt_bboxes_3d = self._bboxes_3d_to_2d(img_metas, gt_bboxes_3d, bbox_3d_preds[0].device)
        labels, bbox_targets, bbox_3d_targets = self.get_targets(
            all_level_points,
            gt_bboxes,
            _gt_bboxes_3d,
            gt_labels_3d)

        num_imgs = cls_scores[0].size(0)
        # flatten cls_scores, bbox_preds and centerness
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for cls_score in cls_scores
        ]
        # flatten_bbox_preds = [
        #     bbox_pred[:, :4, :, :].permute(0, 2, 3, 1).reshape(-1, 4)
        #     for bbox_pred in bbox_3d_preds
        # ]
        flatten_bbox_3d_preds = [
            # bbox_pred.permute(0, 2, 3, 1).reshape(-1, 7)
            self._bboxes_2d_to_3d_(
                img_metas,
                bbox_3d_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 17),
                points.repeat(num_imgs, 1).reshape(num_imgs, -1, 2)
            ).reshape(-1, 17)
            for bbox_3d_pred, points in zip(bbox_3d_preds, all_level_points)
        ]
        flatten_centerness = [
            centerness.permute(0, 2, 3, 1).reshape(-1)
            for centerness in centernesses
        ]

        flatten_cls_scores = torch.cat(flatten_cls_scores)
        # flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        flatten_bbox_3d_preds = torch.cat(flatten_bbox_3d_preds)
        flatten_centerness = torch.cat(flatten_centerness)
        flatten_labels = torch.cat(labels)
        flatten_bbox_targets = torch.cat(bbox_targets)
        flatten_bbox_3d_targets = torch.cat(bbox_3d_targets)
        # repeat points to align with bbox_preds
        flatten_points = torch.cat(
            [points.repeat(num_imgs, 1) for points in all_level_points])

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((flatten_labels >= 0)
                    & (flatten_labels < bg_class_ind)).nonzero().reshape(-1)
        num_pos = len(pos_inds)
        loss_cls = self.loss_cls(
            flatten_cls_scores, flatten_labels,
            avg_factor=num_pos + num_imgs)  # avoid num_pos is 0

        # pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_bbox_3d_preds = flatten_bbox_3d_preds[pos_inds]
        pos_centerness = flatten_centerness[pos_inds]

        if num_pos > 0:
            pos_bbox_targets = flatten_bbox_targets[pos_inds]
            pos_bbox_3d_targets = flatten_bbox_3d_targets[pos_inds]
            pos_centerness_targets = self.centerness_target(pos_bbox_targets)
            pos_points = flatten_points[pos_inds]
            # pos_decoded_bbox_preds = distance2bbox(pos_points, pos_bbox_preds)
            # pos_decoded_target_preds = distance2bbox(pos_points,
            #                                          pos_bbox_targets)
            # loss_bbox = self.loss_bbox(
            #     pos_decoded_bbox_preds,
            #     pos_decoded_target_preds,
            #     weight=pos_centerness_targets,
            #     avg_factor=pos_centerness_targets.sum()
            # )

            # centerness weighted iou loss
            loss_bbox_3d = self.loss_center(
                pos_bbox_3d_preds,
                pos_bbox_3d_targets,
                weight=pos_centerness_targets[..., None],
                avg_factor=pos_centerness_targets.sum()
            )
            loss_centerness = self.loss_centerness(pos_centerness,
                                                   pos_centerness_targets)
            # loss_center_3d = self.loss_center(
            #     pos_bbox_3d_preds[:, :3],
            #     pos_bbox_3d_targets[:, :3],
            #     weight=pos_centerness_targets[..., None],
            #     avg_factor=pos_centerness_targets.sum()
            # )
        else:
            # loss_bbox = pos_bbox_preds.sum()
            loss_bbox_3d = pos_bbox_3d_preds.sum()
            loss_centerness = pos_centerness.sum()
            # loss_center_3d = pos_bbox_3d_preds[:, 3].sum()

        return dict(
            loss_cls=loss_cls,
            # loss_bbox=loss_bbox,
            loss_bbox_3d=loss_bbox_3d,
            loss_centerness=loss_centerness,
            # loss_center_3d=loss_center_3d
        )

    @force_fp32(apply_to=('cls_scores', 'bbox_3d_preds', 'centernesses'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_3d_preds,
                   centernesses,
                   img_metas,
                   cfg=None,
                   rescale=False,
                   with_nms=True):
        """Transform network output for a batch into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                with shape (N, num_points * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_points * 4, H, W).
            centernesses (list[Tensor]): Centerness for each scale level with
                shape (N, num_points * 1, H, W).
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            cfg (mmcv.Config | None): Test / postprocessing configuration,
                if None, test_cfg would be used. Default: None.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1. The second item is a
                (n,) tensor where each item is the predicted class label of the
                corresponding box.
        """
        assert len(cls_scores) == len(bbox_3d_preds)
        num_levels = len(cls_scores)

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        mlvl_points = self.get_points(featmap_sizes, bbox_3d_preds[0].dtype,
                                      bbox_3d_preds[0].device)
        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_3d_pred_list = [
                bbox_3d_preds[i][img_id].detach() for i in range(num_levels)
            ]
            centerness_pred_list = [
                centernesses[i][img_id].detach() for i in range(num_levels)
            ]
            det_bboxes_3d = self._get_bboxes_single(
                cls_score_list, bbox_3d_pred_list, centerness_pred_list,
                mlvl_points, img_metas[img_id], cfg, rescale, with_nms)
            result_list.append(det_bboxes_3d)
        return result_list

    def _get_bboxes_single(self,
                           cls_scores,
                           bbox_3d_preds,
                           centernesses,
                           mlvl_points,
                           img_meta,
                           cfg,
                           rescale=False,
                           with_nms=True):
        """Transform outputs for a single batch item into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for a single scale level
                with shape (num_points * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for a single scale
                level with shape (num_points * 4, H, W).
            centernesses (list[Tensor]): Centerness for a single scale level
                with shape (num_points * 4, H, W).
            mlvl_points (list[Tensor]): Box reference for a single scale level
                with shape (num_total_points, 4).
            img_shape (tuple[int]): Shape of the input image,
                (height, width, 3).
            scale_factor (ndarray): Scale factor of the image arrange as
                (w_scale, h_scale, w_scale, h_scale).
            cfg (mmcv.Config | None): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            tuple(Tensor):
                det_bboxes (Tensor): BBox predictions in shape (n, 5), where
                    the first 4 columns are bounding box positions
                    (tl_x, tl_y, br_x, br_y) and the 5-th column is a score
                    between 0 and 1.
                det_labels (Tensor): A (n,) tensor where each item is the
                    predicted class label of the corresponding box.
        """
        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_scores) == len(bbox_3d_preds) == len(mlvl_points)
        # mlvl_bboxes = []
        mlvl_bboxes_3d = []
        mlvl_scores = []
        mlvl_centerness = []
        for cls_score, bbox_3d_pred, centerness, points in zip(
                cls_scores, bbox_3d_preds, centernesses, mlvl_points):
            assert cls_score.size()[-2:] == bbox_3d_pred.size()[-2:]
            scores = cls_score.permute(1, 2, 0).reshape(
                -1, self.cls_out_channels).sigmoid()
            centerness = centerness.permute(1, 2, 0).reshape(-1).sigmoid()

            bbox_3d_pred = bbox_3d_pred.permute(1, 2, 0).reshape(-1, 17)
            # bbox_pred = bbox_3d_pred[:, :4]
            bbox_3d_pred = self._bboxes_2d_to_3d(
                [img_meta], bbox_3d_pred[None, ...], points[None, ...])[0]
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                max_scores, _ = (scores * centerness[:, None]).max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                # points = points[topk_inds, :]
                # bbox_pred = bbox_pred[topk_inds, :]
                bbox_3d_pred = bbox_3d_pred[topk_inds, :]
                scores = scores[topk_inds, :]
                centerness = centerness[topk_inds]
            # bboxes = distance2bbox(points, bbox_pred)
            # mlvl_bboxes.append(bboxes)
            mlvl_bboxes_3d.append(bbox_3d_pred)
            mlvl_scores.append(scores)
            mlvl_centerness.append(centerness)
        # mlvl_bboxes = torch.cat(mlvl_bboxes)
        mlvl_bboxes_3d = torch.cat(mlvl_bboxes_3d)
        # if rescale:
        #     mlvl_bboxes /= mlvl_bboxes.new_tensor(img_meta['scale_factor'])
        mlvl_scores = torch.cat(mlvl_scores)
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
        # BG cat_id: num_class
        mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)
        mlvl_centerness = torch.cat(mlvl_centerness)

        assert with_nms
        mlvl_bboxes_for_nms = xywhr2xyxyr(img_meta['box_type_3d'](mlvl_bboxes_3d).bev)
        score_thr = cfg.get('score_thr', 0)
        # TODO: Is centerness multiplcation here needed/correct?
        results = box3d_multiclass_nms(mlvl_bboxes_3d, mlvl_bboxes_for_nms,
                                       mlvl_scores * mlvl_centerness[..., None],
                                       score_thr, cfg.max_num, cfg)
        bboxes, scores, labels, _ = results
        bboxes = img_meta['box_type_3d'](bboxes)
        return bboxes, scores, labels

    def _get_points_single(self,
                           featmap_size,
                           stride,
                           dtype,
                           device,
                           flatten=False):
        """Get points according to feature map sizes."""
        y, x = super()._get_points_single(featmap_size, stride, dtype, device)
        points = torch.stack((x.reshape(-1) * stride, y.reshape(-1) * stride),
                             dim=-1) + stride // 2
        return points

    def get_targets(self, points, gt_bboxes_list, gt_bboxes_3d_list, gt_labels_list):
        """Compute regression, classification and centerss targets for points
        in multiple images.

        Args:
            points (list[Tensor]): Points of each fpn level, each has shape
                (num_points, 2).
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image,
                each has shape (num_gt, 4).
            gt_labels_list (list[Tensor]): Ground truth labels of each box,
                each has shape (num_gt,).

        Returns:
            tuple:
                concat_lvl_labels (list[Tensor]): Labels of each level. \
                concat_lvl_bbox_targets (list[Tensor]): BBox targets of each \
                    level.
        """
        assert len(points) == len(self.regress_ranges)
        num_levels = len(points)
        # expand regress ranges to align with points
        expanded_regress_ranges = [
            points[i].new_tensor(self.regress_ranges[i])[None].expand_as(
                points[i]) for i in range(num_levels)
        ]
        # concat all levels points and regress ranges
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
        concat_points = torch.cat(points, dim=0)

        # the number of points per img, per lvl
        num_points = [center.size(0) for center in points]

        # get labels and bbox_targets of each image
        labels_list, bbox_targets_list, bbox_3d_targets_list = multi_apply(
            self._get_target_single,
            gt_bboxes_list,
            gt_bboxes_3d_list,
            gt_labels_list,
            points=concat_points,
            regress_ranges=concat_regress_ranges,
            num_points_per_lvl=num_points)

        # split to per img, per level
        labels_list = [labels.split(num_points, 0) for labels in labels_list]
        bbox_targets_list = [
            bbox_targets.split(num_points, 0)
            for bbox_targets in bbox_targets_list
        ]
        bbox_3d_targets_list = [
            bbox_3d_targets.split(num_points, 0)
            for bbox_3d_targets in bbox_3d_targets_list
        ]

        # concat per level image
        concat_lvl_labels = []
        concat_lvl_bbox_targets = []
        concat_lvl_bbox_3d_targets = []
        for i in range(num_levels):
            concat_lvl_labels.append(
                torch.cat([labels[i] for labels in labels_list]))
            bbox_targets = torch.cat(
                [bbox_targets[i] for bbox_targets in bbox_targets_list])
            if self.norm_on_bbox:
                bbox_targets = bbox_targets / self.strides[i]
            concat_lvl_bbox_targets.append(bbox_targets)
            concat_lvl_bbox_3d_targets.append(
                torch.cat([bbox_3d_targets[i] for bbox_3d_targets in bbox_3d_targets_list]))
        return concat_lvl_labels, concat_lvl_bbox_targets, concat_lvl_bbox_3d_targets

    def _get_target_single(self, gt_bboxes, gt_bboxes_3d, gt_labels, points, regress_ranges,
                           num_points_per_lvl):
        """Compute regression and classification targets for a single image."""
        num_points = points.size(0)
        num_gts = gt_labels.size(0)
        if num_gts == 0:
            return gt_labels.new_full((num_points,), self.num_classes), \
                   gt_bboxes.new_zeros((num_points, 4)), \
                   gt_bboxes.new_zeros((num_points, 7))

        areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * (
            gt_bboxes[:, 3] - gt_bboxes[:, 1])
        # TODO: figure out why these two are different
        # areas = areas[None].expand(num_points, num_gts)
        areas = areas[None].repeat(num_points, 1)
        regress_ranges = regress_ranges[:, None, :].expand(
            num_points, num_gts, 2)
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 4)
        gt_bboxes_3d = gt_bboxes_3d.expand(num_points, num_gts, 17)
        xs, ys = points[:, 0], points[:, 1]
        xs = xs[:, None].expand(num_points, num_gts)
        ys = ys[:, None].expand(num_points, num_gts)

        left = xs - gt_bboxes[..., 0]
        right = gt_bboxes[..., 2] - xs
        top = ys - gt_bboxes[..., 1]
        bottom = gt_bboxes[..., 3] - ys
        bbox_targets = torch.stack((left, top, right, bottom), -1)

        if self.center_sampling:
            # condition1: inside a `center bbox`
            radius = self.center_sample_radius
            center_xs = (gt_bboxes[..., 0] + gt_bboxes[..., 2]) / 2
            center_ys = (gt_bboxes[..., 1] + gt_bboxes[..., 3]) / 2
            center_gts = torch.zeros_like(gt_bboxes)
            stride = center_xs.new_zeros(center_xs.shape)

            # project the points on current lvl back to the `original` sizes
            lvl_begin = 0
            for lvl_idx, num_points_lvl in enumerate(num_points_per_lvl):
                lvl_end = lvl_begin + num_points_lvl
                stride[lvl_begin:lvl_end] = self.strides[lvl_idx] * radius
                lvl_begin = lvl_end

            x_mins = center_xs - stride
            y_mins = center_ys - stride
            x_maxs = center_xs + stride
            y_maxs = center_ys + stride
            center_gts[..., 0] = torch.where(x_mins > gt_bboxes[..., 0],
                                             x_mins, gt_bboxes[..., 0])
            center_gts[..., 1] = torch.where(y_mins > gt_bboxes[..., 1],
                                             y_mins, gt_bboxes[..., 1])
            center_gts[..., 2] = torch.where(x_maxs > gt_bboxes[..., 2],
                                             gt_bboxes[..., 2], x_maxs)
            center_gts[..., 3] = torch.where(y_maxs > gt_bboxes[..., 3],
                                             gt_bboxes[..., 3], y_maxs)

            cb_dist_left = xs - center_gts[..., 0]
            cb_dist_right = center_gts[..., 2] - xs
            cb_dist_top = ys - center_gts[..., 1]
            cb_dist_bottom = center_gts[..., 3] - ys
            center_bbox = torch.stack(
                (cb_dist_left, cb_dist_top, cb_dist_right, cb_dist_bottom), -1)
            inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0
        else:
            # condition1: inside a gt bbox
            inside_gt_bbox_mask = bbox_targets.min(-1)[0] > 0

        # condition2: limit the regression range for each location
        max_regress_distance = bbox_targets.max(-1)[0]
        inside_regress_range = (
            (max_regress_distance >= regress_ranges[..., 0])
            & (max_regress_distance <= regress_ranges[..., 1]))

        # if there are still more than one objects for a location,
        # we choose the one with minimal area
        areas[inside_gt_bbox_mask == 0] = INF
        areas[inside_regress_range == 0] = INF
        min_area, min_area_inds = areas.min(dim=1)

        labels = gt_labels[min_area_inds]
        labels[min_area == INF] = self.num_classes  # set as BG
        bbox_targets = bbox_targets[range(num_points), min_area_inds]
        bbox_3d_targets = gt_bboxes_3d[range(num_points), min_area_inds]

        return labels, bbox_targets, bbox_3d_targets

    def centerness_target(self, pos_bbox_targets):
        """Compute centerness targets.

        Args:
            pos_bbox_targets (Tensor): BBox targets of positive bboxes in shape
                (num_pos, 4)

        Returns:
            Tensor: Centerness target.
        """
        # only calculate pos centerness targets, otherwise there may be nan
        left_right = pos_bbox_targets[:, [0, 2]]
        top_bottom = pos_bbox_targets[:, [1, 3]]
        centerness_targets = (
            left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * (
                top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness_targets)
