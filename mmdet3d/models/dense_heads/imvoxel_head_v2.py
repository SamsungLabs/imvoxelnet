import torch
from torch import nn
from mmdet.core import multi_apply, reduce_mean
from mmdet.models.builder import HEADS, build_loss
from mmcv.cnn import Scale, bias_init_with_prob, normal_init

from mmdet3d.models.detectors.imvoxelnet import get_points
from mmdet3d.core.bbox.structures import rotation_3d_in_axis
from mmdet3d.core.post_processing import aligned_3d_nms, box3d_multiclass_nms


class ImVoxelHeadV2(nn.Module):
    def __init__(self,
                 n_classes,
                 n_channels,
                 n_reg_outs,
                 n_scales,
                 limit,
                 centerness_topk=-1,
                 loss_centerness=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 loss_bbox=dict(type='IoU3DLoss', loss_weight=1.0),
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 train_cfg=None,
                 test_cfg=None):
        super(ImVoxelHeadV2, self).__init__()
        self.n_classes = n_classes
        self.n_scales = n_scales
        self.limit = limit
        self.centerness_topk = centerness_topk
        self.loss_centerness = build_loss(loss_centerness)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_cls = build_loss(loss_cls)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self._init_layers(n_channels, n_reg_outs)

    def _init_layers(self, n_channels, n_reg_outs):
        self.centerness_conv = nn.Conv3d(n_channels, 1, 3, padding=1, bias=False)
        self.reg_conv = nn.Conv3d(n_channels, n_reg_outs, 3, padding=1, bias=False)
        self.cls_conv = nn.Conv3d(n_channels, self.n_classes, 3, padding=1)
        self.scales = nn.ModuleList([Scale(1.) for _ in range(self.n_scales)])

    # Follow AnchorFreeHead.init_weights
    def init_weights(self):
        normal_init(self.centerness_conv, std=.01)
        normal_init(self.reg_conv, std=.01)
        normal_init(self.cls_conv, std=.01, bias=bias_init_with_prob(.01))

    def forward(self, x):
        return multi_apply(self.forward_single, x, self.scales)

    def forward_train(self, x, valid, img_metas, gt_bboxes, gt_labels):
        loss_inputs = self(x) + (valid, img_metas, gt_bboxes, gt_labels)
        losses = self.loss(*loss_inputs)
        return losses

    def loss(self,
             centernesses,
             bbox_preds,
             cls_scores,
             valid,
             img_metas,
             gt_bboxes,
             gt_labels):
        """
        Args:
            centernesses (list(Tensor)): Multi-level centernesses
                of shape (batch, 1, nx[i], ny[i], nz[i])
            bbox_preds (list(Tensor)): Multi-level xyz min and max distances
                of shape (batch, 6, nx[i], ny[i], nz[i])
            cls_scores (list(Tensor)): Multi-level class scores
                of shape (batch, n_classes, nx[i], ny[i], nz[i])
            img_metas (list[dict]): Meta information of each image
            gt_bboxes (list(BaseInstance3DBoxes)): Ground truth bboxes for each image
            gt_labels (list(Tensor)): Ground truth class labels for each image

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert len(centernesses[0]) == len(bbox_preds[0]) == len(cls_scores[0]) == \
               len(valid) == len(img_metas) == len(gt_bboxes) == len(gt_labels)

        valids = []
        for x in centernesses:
            valids.append(nn.Upsample(size=x.shape[-3:], mode='trilinear')(valid).round().bool())

        loss_centerness, loss_bbox, loss_cls = [], [], []
        for i in range(len(img_metas)):
            img_loss_centerness, img_loss_bbox, img_loss_cls = self._loss_single(
                centernesses=[x[i] for x in centernesses],
                bbox_preds=[x[i] for x in bbox_preds],
                cls_scores=[x[i] for x in cls_scores],
                valids=[x[i] for x in valids],
                img_meta=img_metas[i],
                gt_bboxes=gt_bboxes[i],
                gt_labels=gt_labels[i]
            )
            loss_centerness.append(img_loss_centerness)
            loss_bbox.append(img_loss_bbox)
            loss_cls.append(img_loss_cls)
        return dict(
            loss_centerness=torch.mean(torch.stack(loss_centerness)),
            loss_bbox=torch.mean(torch.stack(loss_bbox)),
            loss_cls=torch.mean(torch.stack(loss_cls))
        )

    def _loss_single(self,
                     centernesses,
                     bbox_preds,
                     cls_scores,
                     valids,
                     img_meta,
                     gt_bboxes,
                     gt_labels):
        """
        Args:
            centernesses (list(Tensor)): Multi-level centernesses
                of shape (1, nx[i], ny[i], nz[i])
            bbox_preds (list(Tensor)): Multi-level xyz min and max distances
                of shape (6, nx[i], ny[i], nz[i])
            cls_scores (list(Tensor)): Multi-level class scores
                of shape (n_classes, nx[i], ny[i], nz[i])
            img_metas (list[dict]): Meta information
            gt_bboxes (BaseInstance3DBoxes): Ground truth bboxes
                of shape (n_boxes, 7)
            gt_labels (list(Tensor)): Ground truth class labels
                of shape (n_boxes,)

        Returns:
            tuple(Tensor): 3 losses
        """
        featmap_sizes = [featmap.size()[-3:] for featmap in centernesses]
        mlvl_points = self.get_points(
            featmap_sizes=featmap_sizes,
            origin=img_meta['lidar2img']['origin'],
            device=gt_bboxes.device
        )

        centerness_targets, bbox_targets, labels = self.get_targets(mlvl_points, gt_bboxes, gt_labels)

        flatten_centerness = [centerness.permute(1, 2, 3, 0).reshape(-1)
                              for centerness in centernesses]
        bbox_pred_size = bbox_preds[0].shape[0]
        flatten_bbox_preds = [bbox_pred.permute(1, 2, 3, 0).reshape(-1, bbox_pred_size)
                              for bbox_pred in bbox_preds]
        flatten_cls_scores = [cls_score.permute(1, 2, 3, 0).reshape(-1, self.n_classes)
                              for cls_score in cls_scores]
        flatten_valids = [valid.permute(1, 2, 3, 0).reshape(-1)
                          for valid in valids]

        flatten_centerness = torch.cat(flatten_centerness)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_valids = torch.cat(flatten_valids)

        flatten_centerness_targets = centerness_targets.to(centernesses[0].device)
        flatten_bbox_targets = bbox_targets.to(centernesses[0].device)
        flatten_labels = labels.to(centernesses[0].device)
        flatten_points = torch.cat(mlvl_points)

        # skip background
        pos_inds = torch.nonzero(torch.logical_and(
            flatten_labels >= 0,
            flatten_valids
        )).reshape(-1)
        n_pos = torch.tensor(len(pos_inds), dtype=torch.float, device=centernesses[0].device)
        n_pos = max(reduce_mean(n_pos), 1.)
        if torch.any(flatten_valids):
            loss_cls = self.loss_cls(
                flatten_cls_scores[flatten_valids],
                flatten_labels[flatten_valids],
                avg_factor=n_pos
            )
        else:
            loss_cls = flatten_cls_scores[flatten_valids].sum()
        pos_centerness = flatten_centerness[pos_inds]
        pos_bbox_preds = flatten_bbox_preds[pos_inds]

        if len(pos_inds) > 0:
            pos_centerness_targets = flatten_centerness_targets[pos_inds]
            pos_bbox_targets = flatten_bbox_targets[pos_inds]
            pos_points = flatten_points[pos_inds].to(pos_bbox_preds.device)
            loss_centerness = self.loss_centerness(
                pos_centerness, pos_centerness_targets, avg_factor=n_pos
            )
            loss_bbox = self.loss_bbox(
                self._bbox_pred_to_loss(pos_points, pos_bbox_preds),
                pos_bbox_targets,
                weight=pos_centerness_targets,
                avg_factor=pos_centerness_targets.sum()
            )
        else:
            loss_centerness = pos_centerness.sum()
            loss_bbox = pos_bbox_preds.sum()
        return loss_centerness, loss_bbox, loss_cls

    @torch.no_grad()
    def get_points(self, featmap_sizes, origin, device):
        mlvl_points = []
        for i, featmap_size in enumerate(featmap_sizes):
            mlvl_points.append(get_points(
                n_voxels=torch.tensor(featmap_size),
                voxel_size=torch.tensor(self.voxel_size) * (2 ** i),
                origin=torch.tensor(origin)
            ).reshape(3, -1).transpose(0, 1).to(device))
        return mlvl_points

    def get_bboxes(self,
                   centernesses,
                   bbox_preds,
                   cls_scores,
                   valid,
                   img_metas):
        assert len(centernesses[0]) == len(bbox_preds[0]) == len(cls_scores[0]) \
               == len(img_metas)
        valids = []
        for x in centernesses:
            valids.append(nn.Upsample(size=x.shape[-3:], mode='trilinear')(valid).round().bool())
        n_levels = len(centernesses)
        result_list = []
        for img_id in range(len(img_metas)):
            centerness_list = [
                centernesses[i][img_id].detach() for i in range(n_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(n_levels)
            ]
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(n_levels)
            ]
            valid_list = [
                valids[i][img_id].detach() for i in range(n_levels)
            ]
            det_bboxes_3d = self._get_bboxes_single(
                centerness_list, bbox_pred_list, cls_score_list, valid_list, img_metas[img_id]
            )
            result_list.append(det_bboxes_3d)
        return result_list

    def _get_bboxes_single(self,
                           centernesses,
                           bbox_preds,
                           cls_scores,
                           valids,
                           img_meta):
        featmap_sizes = [featmap.size()[-3:] for featmap in centernesses]
        mlvl_points = self.get_points(
            featmap_sizes=featmap_sizes,
            origin=img_meta['lidar2img']['origin'],
            device=centernesses[0].device
        )
        bbox_pred_size = bbox_preds[0].shape[0]
        mlvl_bboxes, mlvl_scores = [], []
        for centerness, bbox_pred, cls_score, valid, points in zip(
            centernesses, bbox_preds, cls_scores, valids, mlvl_points
        ):
            centerness = centerness.permute(1, 2, 3, 0).reshape(-1).sigmoid()
            bbox_pred = bbox_pred.permute(1, 2, 3, 0).reshape(-1, bbox_pred_size)
            scores = cls_score.permute(1, 2, 3, 0).reshape(-1, self.n_classes).sigmoid()
            valid = valid.permute(1, 2, 3, 0).reshape(-1)
            scores = scores * centerness[:, None] * valid[:, None]
            max_scores, _ = scores.max(dim=1)

            if len(scores) > self.test_cfg.nms_pre > 0:
                _, ids = max_scores.topk(self.test_cfg.nms_pre)
                bbox_pred = bbox_pred[ids]
                scores = scores[ids]
                points = points[ids]

            bboxes = self._bbox_pred_to_result(points, bbox_pred)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)

        bboxes = torch.cat(mlvl_bboxes)
        scores = torch.cat(mlvl_scores)
        bboxes, scores, labels = self._nms(bboxes, scores, img_meta)
        return bboxes, scores, labels

    def forward_single(self, x, scale):
        raise NotImplementedError

    def _bbox_pred_to_loss(self, points, bbox_preds):
        raise NotImplementedError

    def _bbox_pred_to_result(self, points, bbox_preds):
        raise NotImplementedError

    def get_targets(self, points, gt_bboxes, gt_labels):
        raise NotImplementedError

    def _nms(self, bboxes, scores, img_meta):
        raise NotImplementedError


@HEADS.register_module()
class SunRgbdImVoxelHeadV2(ImVoxelHeadV2):
    def forward_single(self, x, scale):
        reg_final = self.reg_conv(x)
        reg_distance = torch.exp(scale(reg_final[:, :6]))
        reg_angle = reg_final[:, 6:]
        return (
            self.centerness_conv(x),
            torch.cat((reg_distance, reg_angle), dim=1),
            self.cls_conv(x)
        )

    def _bbox_pred_to_loss(self, points, bbox_preds):
        return self._bbox_pred_to_bbox(points, bbox_preds)

    def _bbox_pred_to_result(self, points, bbox_preds):
        return self._bbox_pred_to_bbox(points, bbox_preds)

    @torch.no_grad()
    def get_targets(self, points, gt_bboxes, gt_labels):
        float_max = 1e8
        expanded_scales = [
            points[i].new_tensor(i).expand(len(points[i])).to(gt_labels.device)
            for i in range(len(points))
        ]
        points = torch.cat(points, dim=0).to(gt_labels.device)
        scales = torch.cat(expanded_scales, dim=0)

        # below is based on FCOSHead._get_target_single
        n_points = len(points)
        n_boxes = len(gt_bboxes)
        volumes = gt_bboxes.volume.to(points.device)
        volumes = volumes.expand(n_points, n_boxes).contiguous()
        gt_bboxes = torch.cat((gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]), dim=1)
        gt_bboxes = gt_bboxes.to(points.device).expand(n_points, n_boxes, 7)
        expanded_points = points.unsqueeze(1).expand(n_points, n_boxes, 3)
        shift = torch.stack((
            expanded_points[..., 0] - gt_bboxes[..., 0],
            expanded_points[..., 1] - gt_bboxes[..., 1],
            expanded_points[..., 2] - gt_bboxes[..., 2]
        ), dim=-1).permute(1, 0, 2)
        shift = rotation_3d_in_axis(shift, -gt_bboxes[0, :, 6], axis=2).permute(1, 0, 2)
        centers = gt_bboxes[..., :3] + shift
        dx_min = centers[..., 0] - gt_bboxes[..., 0] + gt_bboxes[..., 3] / 2
        dx_max = gt_bboxes[..., 0] + gt_bboxes[..., 3] / 2 - centers[..., 0]
        dy_min = centers[..., 1] - gt_bboxes[..., 1] + gt_bboxes[..., 4] / 2
        dy_max = gt_bboxes[..., 1] + gt_bboxes[..., 4] / 2 - centers[..., 1]
        dz_min = centers[..., 2] - gt_bboxes[..., 2] + gt_bboxes[..., 5] / 2
        dz_max = gt_bboxes[..., 2] + gt_bboxes[..., 5] / 2 - centers[..., 2]
        bbox_targets = torch.stack((dx_min, dx_max, dy_min, dy_max, dz_min, dz_max, gt_bboxes[..., 6]), dim=-1)

        # condition1: inside a gt bbox
        inside_gt_bbox_mask = bbox_targets[..., :6].min(-1)[0] > 0  # skip angle

        # condition2: positive points per scale >= limit
        # calculate positive points per scale
        n_pos_points_per_scale = []
        for i in range(self.n_scales):
            n_pos_points_per_scale.append(torch.sum(inside_gt_bbox_mask[scales == i], dim=0))
        # find best scale
        n_pos_points_per_scale = torch.stack(n_pos_points_per_scale, dim=0)
        lower_limit_mask = n_pos_points_per_scale < self.limit
        # fix nondeterministic argmax for torch<1.7
        extra = torch.arange(self.n_scales, 0, -1).unsqueeze(1).expand(self.n_scales, n_boxes).to(lower_limit_mask.device)
        lower_index = torch.argmax(lower_limit_mask.int() * extra, dim=0) - 1
        lower_index = torch.where(lower_index < 0, torch.zeros_like(lower_index), lower_index)
        all_upper_limit_mask = torch.all(torch.logical_not(lower_limit_mask), dim=0)
        best_scale = torch.where(all_upper_limit_mask, torch.ones_like(all_upper_limit_mask) * self.n_scales - 1, lower_index)
        # keep only points with best scale
        best_scale = torch.unsqueeze(best_scale, 0).expand(n_points, n_boxes)
        scales = torch.unsqueeze(scales, 1).expand(n_points, n_boxes)
        inside_best_scale_mask = best_scale == scales

        # condition3: limit topk locations per box by centerness
        centerness = compute_centerness(bbox_targets)
        centerness = torch.where(inside_gt_bbox_mask, centerness, torch.ones_like(centerness) * -1)
        centerness = torch.where(inside_best_scale_mask, centerness, torch.ones_like(centerness) * -1)
        top_centerness = torch.topk(centerness, self.centerness_topk + 1, dim=0).values[-1]
        inside_top_centerness_mask = centerness > top_centerness.unsqueeze(0)

        # if there are still more than one objects for a location,
        # we choose the one with minimal area
        volumes = torch.where(inside_gt_bbox_mask, volumes, torch.ones_like(volumes) * float_max)
        volumes = torch.where(inside_best_scale_mask, volumes, torch.ones_like(volumes) * float_max)
        volumes = torch.where(inside_top_centerness_mask, volumes, torch.ones_like(volumes) * float_max)
        min_area, min_area_inds = volumes.min(dim=1)

        labels = gt_labels[min_area_inds]
        labels = torch.where(min_area == float_max, torch.ones_like(labels) * -1, labels)
        bbox_targets = bbox_targets[range(n_points), min_area_inds]
        centerness_targets = compute_centerness(bbox_targets)

        return centerness_targets, gt_bboxes[range(n_points), min_area_inds], labels

    def _nms(self, bboxes, scores, img_meta):
        # Add a dummy background class to the end. Nms needs to be fixed in the future.
        padding = scores.new_zeros(scores.shape[0], 1)
        scores = torch.cat([scores, padding], dim=1)
        bboxes_for_nms = torch.stack((
            bboxes[:, 0] - bboxes[:, 3] / 2,
            bboxes[:, 1] - bboxes[:, 4] / 2,
            bboxes[:, 0] + bboxes[:, 3] / 2,
            bboxes[:, 1] + bboxes[:, 4] / 2,
            bboxes[:, 6]
        ), dim=1)
        bboxes, scores, labels, _ = box3d_multiclass_nms(
            mlvl_bboxes=bboxes,
            mlvl_bboxes_for_nms=bboxes_for_nms,
            mlvl_scores=scores,
            score_thr=self.test_cfg.score_thr,
            max_num=self.test_cfg.nms_pre,
            cfg=self.test_cfg,
        )
        bboxes = img_meta['box_type_3d'](bboxes, origin=(.5, .5, .5))
        return bboxes, scores, labels

    @staticmethod
    def _bbox_pred_to_bbox(points, bbox_pred):
        if bbox_pred.shape[0] == 0:
            return bbox_pred

        # dx_min, dx_max, dy_min, dy_max, dz_min, dz_max, alpha ->
        # x_center, y_center, z_center, w, l, h, alpha
        shift = torch.stack((
            (bbox_pred[:, 1] - bbox_pred[:, 0]) / 2,
            (bbox_pred[:, 3] - bbox_pred[:, 2]) / 2,
            (bbox_pred[:, 5] - bbox_pred[:, 4]) / 2
        ), dim=-1).view(-1, 1, 3)
        shift = rotation_3d_in_axis(shift, bbox_pred[:, 6], axis=2)[:, 0, :]
        center = points + shift
        size = torch.stack((
            bbox_pred[:, 0] + bbox_pred[:, 1],
            bbox_pred[:, 2] + bbox_pred[:, 3],
            bbox_pred[:, 4] + bbox_pred[:, 5]
        ), dim=-1)
        return torch.cat((center, size, bbox_pred[:, 6:7]), dim=-1)



@HEADS.register_module()
class ScanNetImVoxelHeadV2(ImVoxelHeadV2):
    def forward_single(self, x, scale):
        return (
            self.centerness_conv(x),
            torch.exp(scale(self.reg_conv(x))),
            self.cls_conv(x)
        )

    def _bbox_pred_to_loss(self, points, bbox_preds):
        return self._bbox_pred_to_bbox(points, bbox_preds)

    def _bbox_pred_to_result(self, points, bbox_preds):
        return self._bbox_pred_to_bbox(points, bbox_preds)

    @torch.no_grad()
    def get_targets(self, points, gt_bboxes, gt_labels):
        float_max = 1e8
        expanded_scales = [
            points[i].new_tensor(i).expand(len(points[i])).to(gt_labels.device)
            for i in range(len(points))
        ]
        points = torch.cat(points, dim=0).to(gt_labels.device)
        scales = torch.cat(expanded_scales, dim=0)

        # below is based on FCOSHead._get_target_single
        n_points = len(points)
        n_boxes = len(gt_bboxes)
        volumes = gt_bboxes.volume.to(points.device)
        volumes = volumes.expand(n_points, n_boxes).contiguous()
        gt_bboxes = torch.cat((gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:6]), dim=1)
        gt_bboxes = gt_bboxes.to(points.device).expand(n_points, n_boxes, 6)
        expanded_points = points.unsqueeze(1).expand(n_points, n_boxes, 3)
        dx_min = expanded_points[..., 0] - gt_bboxes[..., 0] + gt_bboxes[..., 3] / 2
        dx_max = gt_bboxes[..., 0] + gt_bboxes[..., 3] / 2 - expanded_points[..., 0]
        dy_min = expanded_points[..., 1] - gt_bboxes[..., 1] + gt_bboxes[..., 4] / 2
        dy_max = gt_bboxes[..., 1] + gt_bboxes[..., 4] / 2 - expanded_points[..., 1]
        dz_min = expanded_points[..., 2] - gt_bboxes[..., 2] + gt_bboxes[..., 5] / 2
        dz_max = gt_bboxes[..., 2] + gt_bboxes[..., 5] / 2 - expanded_points[..., 2]
        bbox_targets = torch.stack((dx_min, dx_max, dy_min, dy_max, dz_min, dz_max), dim=-1)

        # condition1: inside a gt bbox
        inside_gt_bbox_mask = bbox_targets[..., :6].min(-1)[0] > 0  # skip angle

        # condition2: positive points per scale >= limit
        # calculate positive points per scale
        n_pos_points_per_scale = []
        for i in range(self.n_scales):
            n_pos_points_per_scale.append(torch.sum(inside_gt_bbox_mask[scales == i], dim=0))
        # find best scale
        n_pos_points_per_scale = torch.stack(n_pos_points_per_scale, dim=0)
        lower_limit_mask = n_pos_points_per_scale < self.limit
        # fix nondeterministic argmax for torch<1.7
        extra = torch.arange(self.n_scales, 0, -1).unsqueeze(1).expand(self.n_scales, n_boxes).to(
            lower_limit_mask.device)
        lower_index = torch.argmax(lower_limit_mask.int() * extra, dim=0) - 1
        lower_index = torch.where(lower_index < 0, torch.zeros_like(lower_index), lower_index)
        all_upper_limit_mask = torch.all(torch.logical_not(lower_limit_mask), dim=0)
        best_scale = torch.where(all_upper_limit_mask, torch.ones_like(all_upper_limit_mask) * self.n_scales - 1,
                                 lower_index)
        # keep only points with best scale
        best_scale = torch.unsqueeze(best_scale, 0).expand(n_points, n_boxes)
        scales = torch.unsqueeze(scales, 1).expand(n_points, n_boxes)
        inside_best_scale_mask = best_scale == scales

        # condition3: limit topk locations per box by centerness
        centerness = compute_centerness(bbox_targets)
        centerness = torch.where(inside_gt_bbox_mask, centerness, torch.ones_like(centerness) * -1)
        centerness = torch.where(inside_best_scale_mask, centerness, torch.ones_like(centerness) * -1)
        top_centerness = torch.topk(centerness, self.centerness_topk + 1, dim=0).values[-1]
        inside_top_centerness_mask = centerness > top_centerness.unsqueeze(0)

        # if there are still more than one objects for a location,
        # we choose the one with minimal area
        volumes = torch.where(inside_gt_bbox_mask, volumes, torch.ones_like(volumes) * float_max)
        volumes = torch.where(inside_best_scale_mask, volumes, torch.ones_like(volumes) * float_max)
        volumes = torch.where(inside_top_centerness_mask, volumes, torch.ones_like(volumes) * float_max)
        min_area, min_area_inds = volumes.min(dim=1)

        labels = gt_labels[min_area_inds]
        labels = torch.where(min_area == float_max, torch.ones_like(labels) * -1, labels)
        bbox_targets = bbox_targets[range(n_points), min_area_inds]
        centerness_targets = compute_centerness(bbox_targets)

        return centerness_targets, self._bbox_pred_to_bbox(points, bbox_targets), labels

    def _nms(self, bboxes, scores, img_meta):
        scores, labels = scores.max(dim=1)
        ids = scores > self.test_cfg.score_thr
        bboxes = bboxes[ids]
        scores = scores[ids]
        labels = labels[ids]
        ids = aligned_3d_nms(bboxes, scores, labels, self.test_cfg.iou_thr)
        bboxes = bboxes[ids]
        bboxes = torch.stack((
            (bboxes[:, 0] + bboxes[:, 3]) / 2.,
            (bboxes[:, 1] + bboxes[:, 4]) / 2.,
            (bboxes[:, 2] + bboxes[:, 5]) / 2.,
            bboxes[:, 3] - bboxes[:, 0],
            bboxes[:, 4] - bboxes[:, 1],
            bboxes[:, 5] - bboxes[:, 2]
        ), dim=1)
        bboxes = img_meta['box_type_3d'](bboxes, origin=(.5, .5, .5), box_dim=6, with_yaw=False)
        return bboxes, scores[ids], labels[ids]

    def _bbox_pred_to_bbox(self, points, bbox_pred):
        return torch.stack([
            points[:, 0] - bbox_pred[:, 0],
            points[:, 1] - bbox_pred[:, 2],
            points[:, 2] - bbox_pred[:, 4],
            points[:, 0] + bbox_pred[:, 1],
            points[:, 1] + bbox_pred[:, 3],
            points[:, 2] + bbox_pred[:, 5]
        ], -1)


def compute_centerness(bbox_targets):
    x_dims = bbox_targets[..., [0, 1]]
    y_dims = bbox_targets[..., [2, 3]]
    z_dims = bbox_targets[..., [4, 5]]
    centerness_targets = x_dims.min(dim=-1)[0] / x_dims.max(dim=-1)[0] * \
                         y_dims.min(dim=-1)[0] / y_dims.max(dim=-1)[0] * \
                         z_dims.min(dim=-1)[0] / z_dims.max(dim=-1)[0]
    # todo: sqrt ?
    return torch.sqrt(centerness_targets)
