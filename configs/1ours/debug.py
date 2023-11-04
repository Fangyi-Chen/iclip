_base_ = 'deformable-detr-refine_r50_16xb2-50e_coco.py'

dataset_type = 'IclipDataset'
data_root = '/media/fangyi/2019/2023/data/yfcc15m/'

img_scale = (1024, 1024)  # width, height

train_pipeline = [
    dict(type='Collage', img_scale=img_scale, grid_range=(2, 11)),
    dict(type='RandomChoiceResize',
                    scales=[(608, 608), (640, 640), (672, 672), (704, 704),
                            (736, 763), (768, 768), (800, 1333), (832, 832), 
                            (864, 864), (896, 896), (928, 928), (960, 960), 
                            (992, 992), (1024, 1024)],
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
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=train_dataset)

model = dict(bbox_head=dict(type='IclipDeformableDETRHead', num_classes=1024, gather_all_cap=False))


train_cfg = dict(max_epochs=1, type='EpochBasedTrainLoop', val_interval=1000)




val_pipeline = [
    dict(type='Collage', img_scale=img_scale, grid_range=(2, 11)),
    dict(type='RandomChoiceResize',
                    scales=[(1024, 1024)],
                    keep_ratio=True),
    dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]
val_dataset = dict(
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
    pipeline=val_pipeline)
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=val_dataset)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
val_dataloader = val_dataloader
test_dataloader = val_dataloader

val_evaluator = dict(
    type='IclipMetric',
    ann_file=None,
    metric='bbox',
    format_only=False,
    backend_args=None)
test_evaluator = val_evaluator




