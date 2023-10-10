_base_ = '../deformable_detr/deformable-detr-refine-twostage_r50_16xb2-50e_coco.py'

train_cfg = dict(max_epochs=50, type='EpochBasedTrainLoop', val_interval=1000)

