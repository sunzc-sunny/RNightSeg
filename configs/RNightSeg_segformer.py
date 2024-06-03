_base_ = [
    '../../_base_/models/segformer_mit-b0.py',
    '../../_base_/datasets/nightcity_1024x2048.py',
    '../../_base_/default_runtime.py', '../../_base_/schedules/schedule_80k.py'
]

checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b5_20220624-658746d9.pth'  # noqa

model = dict(
    type='NewEncoderDecoder',
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
        embed_dims=64,
        num_layers=[3, 6, 40, 3]),
    decode_head=dict(
        type='SegformerParallelHeadNew',
        in_channels=[64, 128, 320, 512],
        loss_decode=
        [dict(type='ReflectanceLossV3', loss_weight=2.0),
         dict(type='L_TV', loss_weight=0.1),
         dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
         dict(type='ColorProcessingLossV2', loss_weight=0.1)]
))


# optimizer
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
