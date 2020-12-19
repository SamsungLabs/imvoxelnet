import torch
import numpy as np
import skimage.io
import cv2
from mmdet3d.datasets import SUNRGBD2DDataset
from mmdet3d.models.dense_heads.fcos_3d_head import FCOS3DHead

bboxes_3d_to_2d = FCOS3DHead._bboxes_3d_to_2d
bboxes_2d_to_3d = FCOS3DHead._bboxes_2d_to_3d


def test_getitem():
    np.random.seed(0)
    root_path = './data/sunrgbd'
    ann_file = './data/sunrgbd/sunrgbd_infos_train.pkl'
    class_names = ('bed', 'table', 'sofa', 'chair', 'toilet', 'desk',
                   'dresser', 'night_stand', 'bookshelf', 'bathtub')
    tmp_path = './work_dirs/votenet_16x8_sunrgbd-3d-10class/tmp'

    img_norm_cfg = dict(
        mean=[102.9801, 115.9465, 122.7717], std=[1.0, 1.0, 1.0], to_rgb=False)
    train_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations3D'),
        dict(type='Resize', img_scale=(640, 480), keep_ratio=True),
        # dict(type='Normalize', **img_norm_cfg),
        dict(type='Pad', size_divisor=32),
        dict(type='DefaultFormatBundle3D', class_names=class_names),
        dict(type='Collect3D', keys=['img', 'gt_bboxes_3d', 'gt_labels_3d'])
    ]
    dataset = SUNRGBD2DDataset(root_path, ann_file, train_pipeline)
    data = dataset[40]
    print(data['img_metas']._data['scale_factor'])
    print('ori_shape:', data['img_metas']._data['ori_shape'])
    print('img_shape:', data['img_metas']._data['img_shape'])
    corners_2d = bboxes_3d_to_2d(
        [data['img_metas']._data],
        [data['gt_bboxes_3d']._data],
        torch.device('cpu:0')
    )
    n = len(corners_2d)
    print('corners_2d:', corners_2d)

    print(data['gt_bboxes_3d']._data)
    bboxes_3d = np.zeros((n, 9))

    image = data['img']._data.numpy().astype(np.uint8).transpose(1, 2, 0)
    for j, corner_2d in enumerate(corners_2d[0]):
        color = np.random.randint(0, 255, 3).tolist()
        image = cv2.line(cv2.UMat(image), tuple(corner_2d[:2].numpy().astype(np.int).tolist()),
                         tuple(corner_2d[2:].numpy().astype(np.int).tolist()),
                         color=color)
    skimage.io.imsave(tmp_path + 'tmp.png', image.get())


if __name__ == '__main__':
    test_getitem()
