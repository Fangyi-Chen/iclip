_base_ = 'base_rescale_collage+.py'


model = dict(bbox_head=dict(type='IclipDeformableDETRHead3'))

