_base_ = 'deformable-detr-refine_r50_16xb2-50e_coco.py'

dataset_type = 'IclipDataset'
data_root = '/media/Auriga/fangyic/yfcc15m/'

img_scale = (2048, 2048)  # width, height

train_pipeline = [
    dict(type='Collage', img_scale=img_scale, grid_range=(5, 18), mode='rescalecentercrop'),
    dict(type='RandomChoiceResize',
                    scales=[(1763, 1763),  (1833, 1833),
                            (1896, 1896), (1928, 1928), (1960, 1960), 
                            (1992, 1992), (2048, 2048)],
                    keep_ratio=True),
    dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]

train_dataset = dict(
    # use MultiImageMixDataset wrapper to support mosaic and mixup
    type='MultiImageMixDataset',
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotation.json',
        data_prefix=dict(img='./'),
        pipeline=[
            dict(type='LoadImageFromFile', backend_args=None),
            dict(type='LoadExtractClipText', 
                        text_encoder_model='RN50', 
                        save_folder=data_root+'capfeat/', init_clip=False, ann_file=data_root+'annotation.json')
        ],
        filter_cfg=dict(filter_empty_gt=False),
        backend_args=None),
    pipeline=train_pipeline)

train_dataloader = dict(
    batch_size=6,
    num_workers=5,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=train_dataset)

model = dict(bbox_head=dict(type='IclipDeformableDETRHead', num_classes=1024, gather_all_cap=True))


max_iters = 102622   # 102622 is 1 epoch with batchsize 18*8  each iter == clip 43 epochs  18*8*102622 = 15M
                     # 102622 is 1/3 epoch with bs      6*8   each iter == clip 135 iter    6*8*102622  = 5M   
# learning rate
param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_iters,
        by_epoch=False,
        milestones=[max_iters // 5 * 4],
        gamma=0.1)
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

val_cfg = None
val_dataloader = None
val_evaluator = None
test_cfg = None
test_dataloader = None
test_evaluator = None





