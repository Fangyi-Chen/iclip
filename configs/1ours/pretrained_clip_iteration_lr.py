_base_ = 'base.py'

model = dict(
    data_preprocessor=dict(
                    mean=[122.771, 116.746, 104.094],
                    std=[68.509, 66.63, 70.323]),
    backbone=dict(type='ResNetClip', 
                  init_cfg=None, 
                  load_clip_backbone='RN50')
)

max_iters = 102622   # 1 epoch for a single 8-GPU machine == clip 43 epochs
base_lr = 0.0002

# learning rate
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.1,
        by_epoch=False,
        begin=0,
        end=250),
    dict(
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.01,
        begin=250,
        end=max_iters,
        T_max=max_iters-250,
        by_epoch=False),
]

train_cfg = dict(
    _delete_ = True,
    type='IterBasedTrainLoop',
    max_iters=max_iters,
    val_interval=10000000000000)

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=5000,
        max_keep_ckpts=3))
