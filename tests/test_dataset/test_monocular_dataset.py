import skimage.io
import skimage.draw
import numpy as np
from mmdet3d.datasets import (
    ScanNetMultiViewDataset, SUNRGBDMultiViewDataset, SUNRGBDTotalMultiViewDataset,
    KittiMultiViewDataset, NuScenesMultiViewDataset
)


def draw_corners(img, corners, projection):
    corners_3d_4 = np.concatenate((corners, np.ones((8, 1))), axis=1)
    corners_2d_3 = corners_3d_4 @ projection.T
    z_mask = corners_2d_3[:, 2] > 0
    corners_2d = corners_2d_3[:, :2] / corners_2d_3[:, 2:]
    corners_2d = corners_2d.astype(np.int)
    print(corners_2d)
    for i, j in [
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7]
    ]:
        if z_mask[i] and z_mask[j]:
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


def run_multi_view_dataset(dataset):
    data = dataset[np.random.randint(len(dataset))]
    index = np.random.randint(len(data['img']))
    img = data['img']._data.numpy()[index]
    img_meta = data['img_metas']._data
    extrinsic = img_meta['lidar2img']['extrinsic'][index]
    intrinsic = np.copy(img_meta['lidar2img']['intrinsic'][:3, :3])
    stride = 1
    ratio = img_meta['ori_shape'][0] / (img_meta['img_shape'][0] / stride)
    intrinsic[:2] /= ratio
    projection = intrinsic @ extrinsic[:3]
    for corners, label in zip(
        data['gt_bboxes_3d']._data.corners.numpy(),
        data['gt_labels_3d']._data.numpy()
    ):
        print(dataset.CLASSES[label])
        draw_corners(img, corners, projection)
    if 'layout' in img_meta['lidar2img']:
        print('layout')
        draw_corners(img, img_meta['lidar2img']['layout'].corners.numpy()[0], projection)
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
            type='MultiViewPipeline',
            n_images=50,
            transforms=[
                dict(type='LoadImageFromFile'),
                dict(type='Resize', img_scale=(640, 480), keep_ratio=True),
                dict(type='Pad', size=(480, 640))
            ]),
        dict(type='DefaultFormatBundle3D', class_names=class_names),
        dict(type='Collect3D', keys=['img', 'gt_bboxes_3d', 'gt_labels_3d'])
    ]
    dataset = ScanNetMultiViewDataset(
        data_root=data_root,
        ann_file=data_root + 'scannet_infos_train.pkl',
        pipeline=pipeline,
        classes=class_names,
        filter_empty_gt=True,
        box_type_3d='Depth'
    )
    run_multi_view_dataset(dataset)


def test_sunrgbd_total_multi_view_dataset():
    data_root = './data/sunrgbd/'
    class_names = ('cabinet', 'bed', 'chair', 'sofa', 'table', 'desk', 'dresser',
                   'night_stand', 'sink', 'lamp')
    pipeline = [
        dict(type='LoadAnnotations3D'),
        dict(
            type='MultiViewPipeline',
            n_images=1,
            transforms=[
                dict(type='SunRgbdTotalLoadImageFromFile'),
                dict(type='Resize', img_scale=(640, 480), keep_ratio=True),
                dict(type='Pad', size=(480, 640))]),
        dict(type='DefaultFormatBundle3D', class_names=class_names),
        dict(type='Collect3D', keys=['img', 'gt_bboxes_3d', 'gt_labels_3d'])
    ]
    dataset = SUNRGBDTotalMultiViewDataset(
        data_root=data_root,
        ann_file=data_root + 'sunrgbd_total_infos_train.pkl',
        pipeline=pipeline,
        classes=class_names,
        filter_empty_gt=True,
        box_type_3d='Depth'
    )
    run_multi_view_dataset(dataset)


def test_sunrgbd_multi_view_dataset():
    data_root = './data/sunrgbd/'
    class_names = ('bed', 'table', 'sofa', 'chair', 'toilet', 'desk', 'dresser',
                   'night_stand', 'bookshelf', 'bathtub')
    pipeline = [
        dict(type='LoadAnnotations3D'),
        dict(
            type='MultiViewPipeline',
            n_images=1,
            transforms=[
                dict(type='LoadImageFromFile'),
                dict(type='RandomFlip'),
                dict(type='Resize', img_scale=(640, 480), keep_ratio=True),
                dict(type='Pad', size=(480, 640))]),
        dict(type='SUNRGBDRandomFlip'),
        dict(type='DefaultFormatBundle3D', class_names=class_names),
        dict(type='Collect3D', keys=['img', 'gt_bboxes_3d', 'gt_labels_3d'])
    ]
    dataset = SUNRGBDMultiViewDataset(
        data_root=data_root,
        ann_file=data_root + 'sunrgbd_infos_train.pkl',
        pipeline=pipeline,
        classes=class_names,
        filter_empty_gt=True,
        box_type_3d='Depth'
    )
    run_multi_view_dataset(dataset)


def test_kitti_multi_view_dataset():
    data_root = 'data/kitti/'
    class_names = ['Car']
    input_modality = dict(use_lidar=False, use_camera=True)
    point_cloud_range = [0, -39.68, -3, 69.12, 39.68, 1]

    pipeline = [
        dict(type='LoadAnnotations3D'),
        dict(
            type='MultiViewPipeline',
            n_images=1,
            transforms=[
                dict(type='LoadImageFromFile'),
                dict(type='Resize', img_scale=(1280, 384), keep_ratio=True),
                dict(type='Pad', size=(384, 1280))]),
        dict(type='KittiRandomFlip'),
        dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
        dict(type='KittiSetOrigin', point_cloud_range=point_cloud_range),
        dict(type='DefaultFormatBundle3D', class_names=class_names),
        dict(type='Collect3D', keys=['img', 'gt_bboxes_3d', 'gt_labels_3d'])]
    dataset = KittiMultiViewDataset(
        data_root=data_root,
        ann_file=data_root + 'kitti_infos_train.pkl',
        split='training',
        pts_prefix='velodyne_reduced',
        pipeline=pipeline,
        modality=input_modality,
        classes=class_names,
        test_mode=False)
    run_multi_view_dataset(dataset)


def test_nuscenes_multi_view_dataset():
    point_cloud_range = [-50, -50, -5, 50, 50, 3]
    class_names = [
        'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
        'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
    ]
    data_root = 'data/nuscenes/'
    input_modality = dict(
        use_lidar=False,
        use_camera=True,
        use_radar=False,
        use_map=False,
        use_external=False)
    train_pipeline = [
        dict(type='LoadAnnotations3D'),
        dict(
            type='MultiViewPipeline',
            n_images=6,
            transforms=[
                dict(type='LoadImageFromFile'),
                dict(type='Resize', img_scale=(1600, 900), keep_ratio=True),
                dict(type='Pad', size_divisor=32)]),
        dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
        dict(type='KittiSetOrigin', point_cloud_range=point_cloud_range),
        dict(type='DefaultFormatBundle3D', class_names=class_names),
        dict(type='Collect3D', keys=['img', 'gt_bboxes_3d', 'gt_labels_3d'])]
    dataset = NuScenesMultiViewDataset(
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        box_type_3d='LiDAR')
    run_multi_view_dataset(dataset)
