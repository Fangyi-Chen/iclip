_base_ = 'pretrained_clip_rescale.py'

model = dict(
    data_preprocessor=dict(
                    mean=[122.771, 116.746, 104.094],
                    std=[68.509, 66.63, 70.323]),
    backbone=dict(type='ResNetClip', 
                  init_cfg=None, 
                  load_clip_backbone='RN50')
)

model = dict(bbox_head=dict(type='IclipDeformableDETRHead2',
                            loss_cls=dict(
                                type='FocalLoss',
                                use_sigmoid=False, # false = softmax
                                gamma=2.0,
                                alpha=0.25,
                                loss_weight=2.0),),
             train_cfg=dict(
                 assigner=dict(
                     match_costs=[
                         dict(type='FocalLossSoftmaxCost', weight=2.0, gamma=2.0,),
                         dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                         dict(type='IoUCost', iou_mode='giou', weight=2.0)
                     ])),
        )

