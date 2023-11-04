_base_ = 'pretrained_clip_rescale_collage+.py'


model = dict(bbox_head=dict(type='IclipDeformableDETRHead3'))

