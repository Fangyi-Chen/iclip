# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from mmcv.cnn import Linear
from mmengine.model import bias_init_with_prob, constant_init
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.utils import InstanceList, OptInstanceList
from ..layers import inverse_sigmoid
from .deformable_detr_head import DeformableDETRHead


@MODELS.register_module()
class IclipDeformableDETRHead(DeformableDETRHead):
    r"""Head of DeformDETR: Deformable DETR: Deformable Transformers for
    End-to-End Object Detection.

    Code is modified from the `official github repo
    <https://github.com/fundamentalvision/Deformable-DETR>`_.

    More details can be found in the `paper
    <https://arxiv.org/abs/2010.04159>`_ .

    Args:
        share_pred_layer (bool): Whether to share parameters for all the
            prediction layers. Defaults to `False`.
        num_pred_layer (int): The number of the prediction layers.
            Defaults to 6.
        as_two_stage (bool, optional): Whether to generate the proposal
            from the outputs of encoder. Defaults to `False`.
    """

    def __init__(self,
                 *args,
                 gather_all_cap=True,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.gather_all_cap = gather_all_cap
        if torch.cuda.device_count() > 1:
            assert gather_all_cap

    def forward(self, hidden_states: Tensor,
                references: List[Tensor],
                capfeat) -> Tuple[Tensor]:
        """Forward function.

        Args:
            hidden_states (Tensor): Hidden states output from each decoder
                layer, has shape (num_decoder_layers, bs, num_queries, dim).
            references (list[Tensor]): List of the reference from the decoder.
                The first reference is the `init_reference` (initial) and the
                other num_decoder_layers(6) references are `inter_references`
                (intermediate). The `init_reference` has shape (bs,
                num_queries, 4) when `as_two_stage` of the detector is `True`,
                otherwise (bs, num_queries, 2). Each `inter_reference` has
                shape (bs, num_queries, 4) when `with_box_refine` of the
                detector is `True`, otherwise (bs, num_queries, 2). The
                coordinates are arranged as (cx, cy) when the last dimension is
                2, and (cx, cy, w, h) when it is 4.

        Returns:
            tuple[Tensor]: results of head containing the following tensor.

            - all_layers_outputs_classes (Tensor): Outputs from the
              classification head, has shape (num_decoder_layers, bs,
              num_queries, cls_out_channels).
            - all_layers_outputs_coords (Tensor): Sigmoid outputs from the
              regression head with normalized coordinate format (cx, cy, w,
              h), has shape (num_decoder_layers, bs, num_queries, 4) with the
              last dimension arranged as (cx, cy, w, h).
        """


        all_layers_outputs_classes = []
        all_layers_outputs_coords = []

        for layer_id in range(hidden_states.shape[0]):
            reference = inverse_sigmoid(references[layer_id])
            # NOTE The last reference will not be used.
            hidden_state = hidden_states[layer_id]
            outputs_class = self.cls_branches[layer_id](hidden_state)
            tmp_reg_preds = self.reg_branches[layer_id](hidden_state)
            if reference.shape[-1] == 4:
                # When `layer` is 0 and `as_two_stage` of the detector
                # is `True`, or when `layer` is greater than 0 and
                # `with_box_refine` of the detector is `True`.
                tmp_reg_preds += reference
            else:
                # When `layer` is 0 and `as_two_stage` of the detector
                # is `False`, or when `layer` is greater than 0 and
                # `with_box_refine` of the detector is `False`.
                assert reference.shape[-1] == 2
                tmp_reg_preds[..., :2] += reference
            outputs_coord = tmp_reg_preds.sigmoid()
            all_layers_outputs_classes.append(outputs_class)
            all_layers_outputs_coords.append(outputs_coord)

        all_layers_outputs_classes = torch.stack(all_layers_outputs_classes)
        all_layers_outputs_coords = torch.stack(all_layers_outputs_coords)

        return all_layers_outputs_classes, all_layers_outputs_coords

    def loss(self, hidden_states: Tensor, references: List[Tensor],
             enc_outputs_class: Tensor, enc_outputs_coord: Tensor,
             batch_data_samples: SampleList) -> dict:
        """Perform forward propagation and loss calculation of the detection
        head on the queries of the upstream network.

        Args:
            hidden_states (Tensor): Hidden states output from each decoder
                layer, has shape (num_decoder_layers, num_queries, bs, dim).
            references (list[Tensor]): List of the reference from the decoder.
                The first reference is the `init_reference` (initial) and the
                other num_decoder_layers(6) references are `inter_references`
                (intermediate). The `init_reference` has shape (bs,
                num_queries, 4) when `as_two_stage` of the detector is `True`,
                otherwise (bs, num_queries, 2). Each `inter_reference` has
                shape (bs, num_queries, 4) when `with_box_refine` of the
                detector is `True`, otherwise (bs, num_queries, 2). The
                coordinates are arranged as (cx, cy) when the last dimension is
                2, and (cx, cy, w, h) when it is 4.
            enc_outputs_class (Tensor): The score of each point on encode
                feature map, has shape (bs, num_feat_points, cls_out_channels).
                Only when `as_two_stage` is `True` it would be passed in,
                otherwise it would be `None`.
            enc_outputs_coord (Tensor): The proposal generate from the encode
                feature map, has shape (bs, num_feat_points, 4) with the last
                dimension arranged as (cx, cy, w, h). Only when `as_two_stage`
                is `True` it would be passed in, otherwise it would be `None`.
            batch_data_samples (list[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components.
        """
        batch_gt_instances = []
        batch_img_metas = []
        caption_feat = []
        for data_sample in batch_data_samples:
            batch_img_metas.append(data_sample.metainfo)
            batch_gt_instances.append(data_sample.gt_instances)
            caption_feat.append(data_sample.gt_instances['capfeats'])

        caption_feat = self.gather_all_capfeat(caption_feat)

        outs = self(hidden_states, references, caption_feat)
        loss_inputs = outs + (enc_outputs_class, enc_outputs_coord,
                              batch_gt_instances, batch_img_metas)
        losses = self.loss_by_feat(*loss_inputs)
        return losses

    def gather_all_capfeat(self, caption_feat):
        def remove_pad(tensor):
            return tensor[torch.any(tensor != 0, dim=1)]

        batch_size_per_GPU = len(caption_feat)
        gt_per_img = [len(_) for _ in caption_feat]

        caption_feat_1_GPU = torch.cat(caption_feat, dim=0)
        pad_caption_feat_1_GPU = torch.nn.functional.pad(caption_feat_1_GPU,
                                                         (0, 0, 0, batch_size_per_GPU*100 - caption_feat_1_GPU.shape[0]))
        print(caption_feat_1_GPU.device, 1,caption_feat_1_GPU, gt_per_img)

        if not self.gather_all_cap:
            return caption_feat_1_GPU, gt_per_img
        else:
            local_rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
            gathered_tensors = [torch.empty_like(pad_caption_feat_1_GPU) for _ in range(world_size)]
            torch.distributed.all_gather(gathered_tensors, pad_caption_feat_1_GPU)
            on_this_GPU = gathered_tensors.pop(local_rank)

            pad_caption_feat_1_GPU = remove_pad(on_this_GPU)
            pad_caption_feat_7_GPU = remove_pad(torch.cat(gathered_tensors, dim=0))

            pad_caption_feat_all_GPU = torch.cat([pad_caption_feat_1_GPU, pad_caption_feat_7_GPU], dim=0)
            print(local_rank, world_size, 2, pad_caption_feat_1_GPU)
            print(local_rank, world_size, 3,pad_caption_feat_all_GPU)
            print(local_rank, world_size, 4,pad_caption_feat_1_GPU.shape)
            print(local_rank, world_size, 5,pad_caption_feat_7_GPU.shape)
            return pad_caption_feat_all_GPU, gt_per_img
