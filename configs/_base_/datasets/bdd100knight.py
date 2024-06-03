dataset_type = 'BDD100KNightDataset'
data_root = '../../../data/sunzc/data/bdd100k-night/'

img_norm_cfg = dict(
    mean=[55.28, 49.30, 44.09], std=[49.13, 46.86, 44.97], to_rgb=True)
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(1280, 720), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),  # 光学失真
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255), # 填充
    dict(type='DefaultFormatBundle'), # 默认格式
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1280, 720),
        # MultiScaleFlipAug is disabled by not providing img_ratios and setting flip=False
        img_ratios=[0.25, 0.5, 0.75, 1.0, 1.25],
        flip=True,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir= 'images/train',
        ann_dir= 'labels/train',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir= 'images/val',
        # ann_dir= 'NightCity-label/label/val' ,
        ann_dir= 'labels/val',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir= 'images/val',
        # ann_dir= 'NightCity-label/label/val',
        ann_dir= 'labels/val',
        pipeline=test_pipeline))