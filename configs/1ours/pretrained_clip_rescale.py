_base_ = 'base_rescale.py'

model = dict(
    data_preprocessor=dict(
                    mean=[122.771, 116.746, 104.094],
                    std=[68.509, 66.63, 70.323]),
    backbone=dict(type='ResNetClip', 
                  init_cfg=None, 
                  load_clip_backbone='RN50')
)

