# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
from .meta_arch import BaseArch
from ppdet.core.workspace import register, create

__all__ = ['KPTDETR']


@register
class KPTDETR(BaseArch):
    __category__ = 'architecture'
    __inject__ = ['post_process']
    __shared__ = ['exclude_post_process']

    def __init__(self,
                 backbone,
                 transformer,
                 detr_head,
                 post_process='GroupPosePostProcess',
                 exclude_post_process=False):
        super(KPTDETR, self).__init__()
        self.backbone = backbone
        self.transformer = transformer
        self.detr_head = detr_head
        self.post_process = post_process
        self.exclude_post_process = exclude_post_process

    @classmethod
    def from_config(cls, cfg, *args, **kwargs):
        # backbone
        backbone = create(cfg['backbone'])
        # transformer
        kwargs = {'input_shape': backbone.out_shape}
        transformer = create(cfg['transformer'], **kwargs)
        # head
        kwargs = {
            'hidden_dim': transformer.hidden_dim,
            'nhead': transformer.nhead,
            'input_shape': backbone.out_shape
        }
        detr_head = create(cfg['detr_head'], **kwargs)

        return {
            'backbone': backbone,
            'transformer': transformer,
            "detr_head": detr_head,
        }

    def _forward(self):
        # Backbone
        body_feats = self.backbone(self.inputs)

        # Transformer
        pad_mask = self.inputs['pad_mask']
        out_transformer = self.transformer(body_feats, pad_mask)

        # DETR Head
        if self.training:
            return self.detr_head(out_transformer, self.inputs)
        else:
            keypoints, logits, masks = self.detr_head(out_transformer, body_feats)
            if self.exclude_post_process:
                return keypoints, logits
            else:
                coco_results_eval = self.post_process(logits, keypoints, self.inputs)
            return coco_results_eval

    def get_loss(self):
        losses = self._forward()
        losses.update({
            'loss':
            paddle.add_n([v for k, v in losses.items() if 'log' not in k])
        })
        return losses

    def get_pred(self):
        res_lst = self._forward()
        return {'keypoint': res_lst}
