import torch
import torch.nn as nn

from mmdet.models.builder import LOSSES
from mmdet.models.losses.utils import weighted_loss

from .oriented_iou_loss import cal_giou_3d, cal_iou_3d


# def shift_center(boxes):
#     return torch.cat((
#         boxes[..., :2],
#         boxes[..., 2:3] + boxes[5:6] / 2.,  # TODO: ?
#         boxes[..., 3:]
#     ), dim=-1)


@weighted_loss
def iou_3d_loss(pred, target):
    return 1 - cal_iou_3d(pred[None, ...], target[None, ...])


@weighted_loss
def giou_3d_loss(pred, target):
    return cal_giou_3d(pred[None, ...], target[None, ...])[0][0]


class IoU3DMixin(nn.Module):
    """Adapted from GIoULoss"""
    def __init__(self, loss_function, reduction='mean', loss_weight=1.0):
        super().__init__()
        self.loss_function = loss_function
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        if weight is not None and not torch.any(weight > 0):
            return (pred * weight).sum()  # 0
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if weight is not None and weight.dim() > 1:
            # assert weight.shape == pred.shape # TODO: ?
            weight = weight.mean(-1)
        loss = self.loss_weight * self.loss_function(
            pred,
            target,
            weight,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)

        return loss


@LOSSES.register_module()
class IoU3DLoss(IoU3DMixin):
    def __init__(self, **kwargs):
        super().__init__(loss_function=iou_3d_loss, **kwargs)


@LOSSES.register_module()
class GIoU3DLoss(IoU3DMixin):
    def __init__(self, **kwargs):
        super().__init__(loss_function=giou_3d_loss, **kwargs)
