_base_ = [
    '../../_base_/models/deeplabv3plus_r50-d8.py',
    '../../_base_/datasets/nightcity_1024x2048.py', 
    '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_80k.py'
]
norm_cfg = dict(type='SyncBN', requires_grad=True)

model = dict(type='NewEncoderDecoder',
             pretrained='open-mmlab://resnet101_v1c', backbone=dict(depth=101),
             decode_head=dict(
                 type='ParallelASPPModel',
                 in_channels=2048,
                 in_index=3,
                 channels=512,
                 dilations=(1, 12, 24, 36),
                 c1_in_channels=256,
                 c1_channels=48,
                 dropout_ratio=0.1,
                 num_classes=19,
                 norm_cfg=norm_cfg,
                 align_corners=False,
                 loss_decode=
                 [dict(type='ReflectanceLossV3', loss_weight=2.0),
                  dict(type='L_TV', loss_weight=0.05),
                  dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                  dict(type='ColorProcessingLossV2', loss_weight=0.01)]
             ))


optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))

lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)

data = dict(samples_per_gpu=4, workers_per_gpu=4)

