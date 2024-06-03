# 注释掉了dataset_wrapper中不让混合cityscape数据集的代码，然后强行混合
# 先把nightcity数据集resize到cityscape的大小，，然后采用一样的crop_size = (512, 512)

data_root  = '../../../data/sunzc/data/'

night_norm_cfg = dict(
    mean=[55.28, 49.30, 44.09], std=[49.13, 46.86, 44.97], to_rgb=True)

day_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)

night_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(2048, 1024), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),  # 光学失真
    dict(type='Normalize', **night_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255), # 填充
    dict(type='DefaultFormatBundle'), # 默认格式
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

night_test_pipeline = [
    dict(type='LoadImageFromFile'),
    # dict(type='LoadAnnotations'),
    dict(
        type='MultiScaleFlipAug', # 是不工作的
        img_scale=(1024, 512),

        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **night_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
day_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(2048, 1024), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **day_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

night_dataset_train = dict(
    type = 'NightcityDataset',
    data_root=data_root + 'NightCity/',
    img_dir='NightCity-images/images/train',
    ann_dir='NightCity-label/label/train_new',
    pipeline=night_train_pipeline)

day_dataset_train = dict(
    type = 'CityscapesDataset',
    data_root=data_root + 'cityscapes/',
    img_dir='leftImg8bit/train',
    ann_dir='gtFine/train',
    pipeline=day_train_pipeline)

night_dataset_val = dict(
    type = 'NightcityDataset',
    data_root=data_root + 'NightCity/',
    img_dir='NightCity-images/images/val',
    ann_dir='val_new',
    pipeline=night_test_pipeline)

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train =[night_dataset_train,
            day_dataset_train],

    val = night_dataset_val,
    test = night_dataset_val
)