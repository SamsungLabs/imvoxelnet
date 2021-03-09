import skimage.io
import skimage.draw
import numpy as np
from mmdet3d.datasets import ScanNetMultiViewDataset, SUNRGBDMultiViewDataset

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)


def run_multi_view_dataset(dataset):
    data = dataset[np.random.randint(len(dataset))]
    index = np.random.randint(len(data['img']))
    img = data['img']._data.numpy()[index]
    shape = img.shape[-2:]
    img_meta = data['img_metas']._data
    extrinsic = img_meta['lidar2img']['extrinsic'][index]
    intrinsic = np.copy(img_meta['lidar2img']['intrinsic'][:3, :3])
    # check if only one side is padded
    assert img_meta['img_shape'][0] == img_meta['pad_shape'][0] or \
           img_meta['img_shape'][1] == img_meta['pad_shape'][1]
    dim = 0 if img_meta['img_shape'][0] == img_meta['pad_shape'][0] else 1
    ratio = img_meta['ori_shape'][dim] / shape[dim]
    intrinsic[:2] /= ratio
    projection = intrinsic[:3, :3] @ extrinsic[:3]
    for corners, label in zip(
        data['gt_bboxes_3d']._data.corners.numpy(),
        data['gt_labels_3d']._data.numpy()
    ):
        corners_3d_4 = np.concatenate((corners, np.ones((8, 1))), axis=1)
        corners_2d_3 = corners_3d_4 @ projection.T  # (projection @ corners_3d_4.T).T
        if np.any(corners_2d_3[:, 2] < 0):
            print('z < 0')
            continue
        print(dataset.CLASSES[label])
        corners_2d = corners_2d_3[:, :2] / corners_2d_3[:, 2:]
        corners_2d = corners_2d.astype(np.int)
        print(corners_2d)
        for i, j in [
            [0, 1], [1, 2], [2, 3], [3, 0],
            [4, 5], [5, 6], [6, 7], [7, 4],
            [0, 4], [1, 5], [2, 6], [3, 7]
        ]:
            ci = corners_2d[i]
            cj = corners_2d[j]
            rr, cc, val = skimage.draw.line_aa(ci[1], ci[0], cj[1], cj[0])
            mask = np.logical_and.reduce((
                rr >= 0,
                rr < img.shape[1],
                cc >= 0,
                cc < img.shape[2]
            ))
            img[:, rr[mask], cc[mask]] = val[mask] * 255
    skimage.io.imsave('./work_dirs/tmp/1.png', np.transpose(img, (1, 2, 0)))

def test_scannet_multi_view_dataset():
    data_root = './data/scannet/'
    class_names = ('cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
                   'bookshelf', 'picture', 'counter', 'desk', 'curtain',
                   'refrigerator', 'showercurtrain', 'toilet', 'sink', 'bathtub',
                   'garbagebin')
    pipeline = [
        dict(type='LoadAnnotations3D'),
        dict(
            type='ScanNetMultiViewPipeline',
            n_images=50,
            transforms=[
                dict(type='LoadImageFromFile'),
                dict(type='Resize', img_scale=(640, 480), keep_ratio=True),
                # dict(type='Normalize', **img_norm_cfg),
                dict(type='Pad', size=(480, 640))
            ]),
        dict(type='DefaultFormatBundle3D', class_names=class_names),
        dict(type='Collect3D', keys=['img', 'gt_bboxes_3d', 'gt_labels_3d'])
    ]
    ann_file = './data/scannet/scannet_infos_train.pkl'
    dataset = ScanNetMultiViewDataset(
        data_root=data_root,
        ann_file=ann_file,
        pipeline=pipeline,
        classes=class_names,
        filter_empty_gt=True,
        box_type_3d='Depth'
    )
    run_multi_view_dataset(dataset)



def test_sunrgbd_multi_view_dataset():
    data_root = './data/sunrgbd/'
    class_names = ('cabinet', 'bed', 'chair', 'sofa', 'table', 'desk', 'dresser',
                   'night_stand', 'sink', 'lamp')
    pipeline = [
        dict(type='LoadAnnotations3D'),
        dict(
            type='ScanNetMultiViewPipeline',
            n_images=1,
            transforms=[
                dict(type='LoadImageFromFile'),
                dict(type='Resize', img_scale=(640, 480), keep_ratio=True),
                # dict(type='Normalize', **img_norm_cfg),
                dict(type='Pad', size=(480, 640))
            ]),
        dict(type='DefaultFormatBundle3D', class_names=class_names),
        dict(type='Collect3D', keys=['img', 'gt_bboxes_3d', 'gt_labels_3d'])
    ]
    ann_file = './data/sunrgbd/sunrgbd_infos_train.pkl'
    dataset = SUNRGBDMultiViewDataset(
        data_root=data_root,
        ann_file=ann_file,
        pipeline=pipeline,
        classes=class_names,
        filter_empty_gt=True,
        box_type_3d='Depth'
    )
    run_multi_view_dataset(dataset)


test_sunrgbd_multi_view_dataset()
