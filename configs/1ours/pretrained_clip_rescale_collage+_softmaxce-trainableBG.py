_base_ = 'pretrained_clip_rescale_collage+.py'


model = dict(bbox_head=dict(type='IclipDeformableDETRHead2',
                            loss_cls=dict(
                                type='FocalLoss',
                                use_sigmoid=False, # false = softmax
                                gamma=0.0, # gamma=0.0 = celoss
                                alpha=0.25,
                                loss_weight=2.0),),
             train_cfg=dict(
                 assigner=dict(
                     match_costs=[
                         dict(type='FocalLossSoftmaxCost', weight=2.0, gamma=0.0,), # gamma=0.0 = celoss
                         dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                         dict(type='IoUCost', iou_mode='giou', weight=2.0)
                     ])),
        )

