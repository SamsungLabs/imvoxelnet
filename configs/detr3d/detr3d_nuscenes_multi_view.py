model = dict(
    type='NuScenesMultiViewFCOS3D',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch'),
    bbox_head=dict(
        type='Transformer3DHead',
        num_classes=10,
        in_channels=2048,
        num_fcs=2,
        loss_bbox_type='corners',
        transformer=dict(
            type='Transformer',
            embed_dims=256,
            num_heads=8,
            num_encoder_layers=0,
            num_decoder_layers=6,
            feedforward_channels=2048,
            dropout=0.1,
            act_cfg=dict(type='ReLU', inplace=True),
            norm_cfg=dict(type='LN'),
            num_fcs=2,
            pre_norm=False,
            return_intermediate_dec=True),
        positional_encoding=dict(
            type='SinePositionalEncoding', num_feats=128, normalize=True),
        loss_cls=dict(
            type='CrossEntropyLoss',
            bg_cls_weight=0.2,
            use_sigmoid=False,
            loss_weight=1.0,
            class_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=.01),
        loss_iou=dict(type='GIoU3DLoss', loss_weight=.0)))
# training and testing settings
train_cfg = dict(
    assigner=dict(
        type='HungarianAssigner3D', loss_bbox_type='corners', cls_weight=1., bbox_weight=.01, iou_weight=.0,
        iou_calculator=dict(type='BboxOverlaps3D', coordinate='depth'), iou_mode='giou'))
test_cfg = dict(max_per_img=200, max_per_scene=200)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_dataset_type = 'NuScenesMonocularDataset'
test_dataset_type = 'NuScenesMultiViewDataset'
data_root = 'data/nuscenes/'
class_names = ('car', 'truck', 'trailer', 'bus', 'construction_vehicle',
               'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
               'barrier')
# train_pipeline, NOTE the img_scale and the Pad's size_divisor is different
# from the default setting in mmdet.
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations3D'),
    dict(type='Resize', img_scale=(900, 1600), keep_ratio=True),  # TODO: [(288, 384), (672, 896)]
    # dict(type='RandomFlip2D', flip_ratio=0.5),  # TODO: ?
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=1),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['img', 'gt_bboxes_3d', 'gt_labels_3d'])
]
# test_pipeline, NOTE the Pad's size_divisor is different from the default
# setting (size_divisor=32). While there is little effect on the performance
# whether we use the default setting or use size_divisor=1.
test_pipeline = [
    dict(
        type='NuScenesMultiView',
        transforms=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', img_scale=(900, 1600), keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=1),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=1,
        dataset=dict(
            type=train_dataset_type,
            data_root=data_root,
            ann_file=data_root + 'nuscenes_monocular_infos_train.pkl',
            pipeline=train_pipeline,
            classes=class_names,
            filter_empty_gt=True,
            # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
            # and box_type_3d='Depth' in sunrgbd and scannet dataset.
            box_type_3d='Depth')),
    val=dict(
        type=test_dataset_type,
        data_root=data_root,
        ann_file=data_root + 'nuscenes_multi_view_infos_val.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        test_mode=True,
        box_type_3d='Depth'),
    test=dict(
        type=test_dataset_type,
        data_root=data_root,
        ann_file=data_root + 'nuscenes_multi_view_infos_val.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        test_mode=True,
        box_type_3d='Depth'))
# optimizer
optimizer = dict(
    type='AdamW',
    lr=0.0001,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys={'backbone': dict(lr_mult=0.1, decay_mult=1.0)}))
optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))
# learning policy
lr_config = dict(policy='step', step=[33])
total_epochs = 50

checkpoint_config = dict(interval=1, max_keep_ckpts=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
evaluation = dict(interval=1)
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
