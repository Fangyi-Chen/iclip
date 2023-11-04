_base_ = 'base_rescale.py'


model = dict(bbox_head=dict(type='IclipDeformableDETRHead3'))

