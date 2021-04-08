import torch
from torch import nn
from mmdet.models.builder import HEADS, build_loss

from mmdet3d.core.bbox.structures import limit_period


@HEADS.register_module()
class LayoutHead(nn.Module):
    def __init__(self,
                 n_channels,
                 linear_size,
                 dropout,
                 loss_angle=dict(type='SmoothL1Loss', loss_weight=1.),
                 loss_layout=dict(type='IoU3DLoss', loss_weight=.1)):
        super().__init__()
        self.angle_mlp = torch.nn.Sequential(
            nn.Linear(n_channels, linear_size),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(linear_size, linear_size),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(linear_size, 2)
        )
        self.layout_mlp = torch.nn.Sequential(
            nn.Linear(n_channels, linear_size),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(linear_size, linear_size),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(linear_size, 7)
        )
        self.loss_angle = build_loss(loss_angle)
        self.loss_layout = build_loss(loss_layout)

    def init_weights(self):
        pass

    def forward(self, x, img_metas):
        x = x.mean(dim=(2, 3))
        angle_features = self.angle_mlp(x)
        layout_features = self.layout_mlp(x)
        angles, layouts = [], []
        for angle_feature, layout_feature, img_meta in zip(angle_features, layout_features, img_metas):
            angle, layout = self._forward_single(angle_feature, layout_feature, img_meta)
            angles.append(angle)
            layouts.append(layout)
        return angles, layouts

    def _forward_single(self, angle, layout, img_meta):
        angle = limit_period(angle)
        size = torch.exp(layout[3:6])
        # device = layout.device
        # center_2d = torch.sigmoid(layout[:2])
        # center_z = torch.exp(layout[2])
        # intrinsic = torch.tensor(img_meta['lidar2img']['intrinsic'])
        # extrinsic = torch.tensor(img_meta['lidar2img']['extrinsic'][0])
        # projection = torch.inverse(intrinsic @ extrinsic)[:3, :3].to(device)
        # width = torch.tensor(img_meta['ori_shape'][1]).to(device)
        # height = torch.tensor(img_meta['ori_shape'][0]).to(device)
        # center_2d_3 = center_2d.new_tensor((
        #     center_2d[0] * width * center_z,
        #     center_2d[1] * height * center_z,
        #     center_z
        # ))
        # center_3d = projection @ center_2d_3
        layout = torch.cat((
            layout[:3],
            size,
            layout[6:7]
        ))
        return angle, layout

    def loss(self, angles, layouts, img_metas):
        angle_losses, layout_losses = [], []
        for angle, layout, img_meta in zip(angles, layouts, img_metas):
            angle_loss, layout_loss = self._loss_single(angle, layout, img_meta)
            angle_losses.append(angle_loss)
            layout_losses.append(layout_loss)
        return {
            'angle_loss': torch.mean(torch.stack(angle_losses)),
            'layout_loss': torch.mean(torch.stack(layout_losses))
        }

    def _loss_single(self, angles, layout, img_meta):
        gt_angles = angles.new_tensor(img_meta['lidar2img']['angles'])
        pitch_loss = self.loss_angle(
            torch.sin(angles[0]) * torch.cos(gt_angles[0]),
            torch.cos(angles[0]) * torch.sin(gt_angles[0])
        )
        roll_loss = self.loss_angle(
            torch.sin(angles[1]) * torch.cos(gt_angles[1]),
            torch.cos(angles[1]) * torch.sin(gt_angles[1])
        )
        angle_loss = pitch_loss + roll_loss
        gt_layout = img_meta['lidar2img']['layout']
        gt_layout = torch.cat((
            gt_layout.gravity_center,
            gt_layout.tensor[:, 3:]
        ), dim=-1).to(layout.device)
        layout_loss = self.loss_layout(layout.unsqueeze(0), gt_layout)
        return angle_loss, layout_loss

    def get_bboxes(self, angles, layouts, img_metas):
        result_angles, result_layouts = [], []
        for angle, layout, img_meta in zip(angles, layouts, img_metas):
            result_angle, result_layout = self._get_bboxes_single(angle, layout, img_meta)
            result_angles.append(result_angle.cpu())
            result_layouts.append(result_layout.to('cpu'))
        return result_angles, result_layouts

    def _get_bboxes_single(self, angle, layout, img_meta):
        return angle, img_meta['box_type_3d'](layout.unsqueeze(0), origin=(.5, .5, .5))

