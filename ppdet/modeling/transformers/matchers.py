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
#
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from scipy.optimize import linear_sum_assignment

from ppdet.core.workspace import register, serializable
from ..losses.iou_loss import GIoULoss
from .utils import bbox_cxcywh_to_xyxy

__all__ = ['HungarianMatcher', 'HungarianKeypointMatcher']


@register
@serializable
class HungarianMatcher(nn.Layer):
    __shared__ = ['use_focal_loss']

    def __init__(self,
                 matcher_coeff={'class': 1,
                                'bbox': 5,
                                'giou': 2},
                 use_focal_loss=False,
                 alpha=0.25,
                 gamma=2.0):
        r"""
        Args:
            matcher_coeff (dict): The coefficient of hungarian matcher cost.
        """
        super(HungarianMatcher, self).__init__()
        self.matcher_coeff = matcher_coeff
        self.use_focal_loss = use_focal_loss
        self.alpha = alpha
        self.gamma = gamma

        self.giou_loss = GIoULoss()

    def forward(self, boxes, logits, gt_bbox, gt_class):
        r"""
        Args:
            boxes (Tensor): [b, query, 4]
            logits (Tensor): [b, query, num_classes]
            gt_bbox (List(Tensor)): list[[n, 4]]
            gt_class (List(Tensor)): list[[n, 1]]

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = boxes.shape[:2]

        num_gts = sum(len(a) for a in gt_class)
        if num_gts == 0:
            return [(paddle.to_tensor(
                [], dtype=paddle.int64), paddle.to_tensor(
                    [], dtype=paddle.int64)) for _ in range(bs)]

        # We flatten to compute the cost matrices in a batch
        # [batch_size * num_queries, num_classes]
        out_prob = F.sigmoid(logits.flatten(
            0, 1)) if self.use_focal_loss else F.softmax(logits.flatten(0, 1))
        # [batch_size * num_queries, 4]
        out_bbox = boxes.flatten(0, 1)

        # Also concat the target labels and boxes
        tgt_ids = paddle.concat(gt_class).flatten()
        tgt_bbox = paddle.concat(gt_bbox)

        # Compute the classification cost
        if self.use_focal_loss:
            neg_cost_class = (1 - self.alpha) * (out_prob**self.gamma) * (-(
                1 - out_prob + 1e-8).log())
            pos_cost_class = self.alpha * (
                (1 - out_prob)**self.gamma) * (-(out_prob + 1e-8).log())
            cost_class = paddle.gather(
                pos_cost_class, tgt_ids, axis=1) - paddle.gather(
                    neg_cost_class, tgt_ids, axis=1)
        else:
            cost_class = -paddle.gather(out_prob, tgt_ids, axis=1)

        # Compute the L1 cost between boxes
        cost_bbox = (
            out_bbox.unsqueeze(1) - tgt_bbox.unsqueeze(0)).abs().sum(-1)

        # Compute the giou cost betwen boxes
        cost_giou = self.giou_loss(
            bbox_cxcywh_to_xyxy(out_bbox.unsqueeze(1)),
            bbox_cxcywh_to_xyxy(tgt_bbox.unsqueeze(0))).squeeze(-1)

        # Final cost matrix
        C = self.matcher_coeff['class'] * cost_class + self.matcher_coeff['bbox'] * cost_bbox + \
            self.matcher_coeff['giou'] * cost_giou
        C = C.reshape([bs, num_queries, -1])
        C = [a.squeeze(0) for a in C.chunk(bs)]

        sizes = [a.shape[0] for a in gt_bbox]
        indices = [
            linear_sum_assignment(c.split(sizes, -1)[i].numpy())
            for i, c in enumerate(C)
        ]
        return [(paddle.to_tensor(
            i, dtype=paddle.int64), paddle.to_tensor(
                j, dtype=paddle.int64)) for i, j in indices]


@register
@serializable
class HungarianKeypointMatcher(nn.Layer):
    def __init__(self,
                 matcher_coeff={'class': 1,
                                'keypoint': 5,
                                'oks': 2},
                 num_body_points=17,
                 alpha=0.25,
                 gamma=2.0):
        r"""
        Args:
            matcher_coeff (dict): The coefficient of hungarian matcher cost.
        """
        super(HungarianKeypointMatcher, self).__init__()
        self.matcher_coeff = matcher_coeff
        self.alpha = alpha
        self.gamma = gamma
        
        if num_body_points==17:
            self.sigmas = [
                .26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07,
                1.07, .87, .87, .89, .89
            ]
        elif num_body_points==14:
            self.sigmas = [
                .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89,
                .79, .79
            ]
        else:
            raise NotImplementedError

    def forward(self, keypoints, logits, targets):
        r"""
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = keypoints.shape[:2]
        
        # gt_keypoint, gt_areas, gt_class
        num_gts = sum(len(v) for v in targets["gt_bbox"])
        if num_gts == 0:
            return [(paddle.to_tensor(
                [], dtype=paddle.int64), paddle.to_tensor(
                    [], dtype=paddle.int64)) for _ in range(bs)]
            
        # We flatten to compute the cost matrices in a batch
        # [batch_size * num_queries, num_classes]
        out_prob = F.sigmoid(logits.flatten(0, 1))
        # [batch_size * num_queries, 4]
        out_keypoints = keypoints.flatten(0, 1)
            
        # Also concat the target labels, keypoints and areas
        tgt_ids = paddle.concat([v for v in targets["gt_class"]]).flatten().astype("int")
        tgt_keypoints = paddle.concat([v for v in targets["gt_joints"]])
        tgt_areas = paddle.concat([v for v in targets["gt_areas"]])

        # Compute the classification cost
        neg_cost_class = (1 - self.alpha) * (out_prob**self.gamma) * (-(
            1 - out_prob + 1e-8).log())
        pos_cost_class = self.alpha * (
            (1 - out_prob)**self.gamma) * (-(out_prob + 1e-8).log())
        cost_class = paddle.gather(
            pos_cost_class, tgt_ids, axis=1) - paddle.gather(
                neg_cost_class, tgt_ids, axis=1)

        # Compute the keypoint costs
        vis_keypoints = tgt_keypoints[..., -1].clip(max=1.0)
        tgt_keypoints = tgt_keypoints[..., :2]
        
        sigmas = paddle.to_tensor(self.sigmas, place=out_prob.place) / 10
        variances = (sigmas * 2) ** 2
        squared_distance = (out_keypoints[:, None, :, 0] - tgt_keypoints[None, :, :, 0]) ** 2 + \
                            (out_keypoints[:, None, :, 1] - tgt_keypoints[None, :, :, 1]) ** 2
        squared_distance0 = squared_distance / (tgt_areas[:, None] * variances[None, :] * 2)
        squared_distance1 = paddle.exp(-squared_distance0)
        squared_distance1 = squared_distance1 * vis_keypoints
        oks = squared_distance1.sum(axis=-1) / (vis_keypoints.sum(axis=-1) + 1e-6)
        cost_oks = 1 - oks.clip(min=1e-6)
        
        cost_keypoints = paddle.abs(out_keypoints[:, None, :] - tgt_keypoints[None])
        cost_keypoints = cost_keypoints * vis_keypoints[..., None].repeat_interleave(2, axis=-1)[None]
        cost_keypoints = cost_keypoints.flatten(-2, -1).sum(-1)

        # Final cost matrix
        C = self.matcher_coeff["class"] * cost_class + self.matcher_coeff["keypoint"] * cost_keypoints + self.matcher_coeff["oks"] * cost_oks
        C = C.reshape([bs, num_queries, -1])

        sizes = [len(v) for v in targets["gt_bbox"]]
        indices = [
            linear_sum_assignment(c[i])
            for i, c in enumerate(C.split(sizes, -1))
        ]
        return [(paddle.to_tensor(
            i, dtype=paddle.int64), paddle.to_tensor(
                j, dtype=paddle.int64)) for i, j in indices]
