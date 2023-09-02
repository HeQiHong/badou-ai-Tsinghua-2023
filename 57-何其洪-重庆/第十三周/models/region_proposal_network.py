# -*- coding: utf-8 -*-
from typing import Dict, Optional, List, OrderedDict, Tuple

import torch
import torch.nn as nn
from torchvision.models.detection.image_list import ImageList

from . import boxes as box_ops
from . import det_utils


class RpnHead(nn.Module):
    def __init__(self, in_channels, num_anchors):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        # 计算预测的目标分数（前景或背景的分数）
        self.class_conv = nn.Conv2d(in_channels, num_anchors, kernel_size=1)
        # 计算预测的目标bbox 回归
        self.bbox_conv = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1)

        # 初始化卷积的参数
        for layer in self.children():
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, std=0.01)
                nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        logits = []
        bbox_reg = []
        for i, feature in enumerate(x):
            t = nn.functional.relu(feature)
            logits.append(self.class_conv(t))
            bbox_reg.append(self.bbox_conv(t))
        return logits, bbox_reg


class RegionProposalNetwork(nn.Module):
    def __init__(self, anchor_generator, head, fg_iou_thresh, bg_iou_thresh,
                 batch_size_per_image, positive_fraction, pre_nms_top_n, post_nms_top_n, nms_thresh, score_thresh=0.0):
        """
        :param anchor_generator: anchor生成器
        :param head: rpn head
        :param fg_iou_thresh: rpn计算损失时采集正样本的阈值
        :param bg_iou_thresh: rpn计算损失时采集负样本的阈值
        :param batch_size_per_image: rpn计算损失时采用正负样本的总个数
        :param positive_fraction: rpn计算损失时正样本所占的比例
        :param pre_nms_top_n: nms处理之前针对每个预测特征层所保留的目标个数
        :param post_nms_top_n: nms处理之后每个预测特征层所剩余的目标个数
        :param nms_thresh: nms处理时指定的阈值
        """
        super().__init__()
        self.anchor_generator = anchor_generator
        self.head = head
        self.box_coder = det_utils.BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))

        # 计算anchors与真实bbox的iou值
        self.box_similarity = box_ops.box_iou

        self.proposal_matcher = det_utils.Matcher(fg_iou_thresh, bg_iou_thresh, allow_low_quality_matches=True)

        self.fg_bg_sampler = det_utils.BalancedPositiveNegativeSampler(batch_size_per_image, positive_fraction)

        self._pre_nms_top_n = pre_nms_top_n
        self._post_nms_top_n = post_nms_top_n
        self.nms_thresh = nms_thresh
        self.min_size = 1e-3

    def pre_nms_top_n(self):
        if self.training:
            return self._pre_nms_top_n['training']
        return self._pre_nms_top_n['testing']

    def post_nms_top_n(self):
        if self.training:
            return self._post_nms_top_n['training']
        return self._post_nms_top_n['testing']

    def forward(self, image_list: ImageList,
                features: OrderedDict[str, torch.Tensor], targets: Optional[List[Dict[str, torch.Tensor]]]):
        """
        :param image_list: 处理后的图片列表
        :param features: 预测特征层
        :param targets:
        :return:
        """
        # features是所有预测特征层组成的OrderedDict
        features = list(features.values())

        # 计算每个预测特征层上的预测目标概率和bboxes regression参数
        object_ness, pre_bbox_deltas = self.head(features)

        # 生成一个batch图像的所有anchors信息，list(tensor)元素个数等于batch_size
        anchors = self.anchor_generator(image_list, features)

        # 图片数量
        num_images = len(anchors)

        # 计算每个预测特征层上对应的anchors数量
        num_anchors_per_level_shape_tensors = [obj[0].shape for obj in object_ness]
        num_anchors_per_level = [s[0] * s[1] * s[2] for s in num_anchors_per_level_shape_tensors]

        # 调整内部tensor格式以及shape
        object_ness, pre_bbox_deltas = concat_box_prediction_layers(object_ness, pre_bbox_deltas)
        # 将预测的bbox回归参数应用到anchors上得到最终的预测bbox坐标
        proposals = self.box_coder.decoder(pre_bbox_deltas.detach(), anchors)
        proposals = proposals.view(num_images, -1, 4)

        # 筛除小boxes框，nms处理，根据预测概率获取前post_nms_top_n个目标
        boxes, scores = self.filter_proposals(proposals, object_ness, image_list.image_sizes, num_anchors_per_level)

        losses = {}
        if self.training:
            # 计算每个anchors最匹配的gt，并将anchors进行分类，前景/背景以及废弃的anchors
            labels, matched_gt_boxes = self.assign_targets_to_anchors(anchors, targets)
            # 结合anchors以及对应的gt计算regression参数
            regression_targets = self.box_coder.encode(matched_gt_boxes, anchors)
            loss_objectness, loss_rpn_box_reg = self.compute_loss(
                object_ness, pre_bbox_deltas, labels, regression_targets
            )
            losses = {
                "loss_objectness": loss_objectness,
                "loss_rpn_box_reg": loss_rpn_box_reg
            }
        return boxes, losses

    def compute_loss(self, objectness, pred_bbox_deltas, labels, regression_targets):
        """
        计算RPN损失，包括类别损失（前景与背景），bbox regression损失
        :param objectness: 预测的前景概率
        :param pred_bbox_deltas: 预测的bbox regression
        :param labels: 真实的标签 1, 0, -1（batch中每一张图片的labels对应List的一个元素中）
        :param regression_targets: 真实的bbox regression
        :return: objectness_loss 类别损失，box_loss 边界框回归损失
        """
        # 按照给定的batch_size_per_image, positive_fraction选择正负样本
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        # 将一个batch中的所有正负样本List(Tensor)分别拼接在一起，并获取非零位置的索引
        # sampled_pos_inds = torch.nonzero(torch.cat(sampled_pos_inds, dim=0)).squeeze(1)
        sampled_pos_inds = torch.where(torch.cat(sampled_pos_inds, dim=0))[0]
        # sampled_neg_inds = torch.nonzero(torch.cat(sampled_neg_inds, dim=0)).squeeze(1)
        sampled_neg_inds = torch.where(torch.cat(sampled_neg_inds, dim=0))[0]

        # 将所有正负样本索引拼接在一起
        sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)
        objectness = objectness.flatten()

        labels = torch.cat(labels, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)

        # 计算边界框回归损失
        box_loss = det_utils.smooth_l1_loss(
            pred_bbox_deltas[sampled_pos_inds],
            regression_targets[sampled_pos_inds],
            beta=1 / 9,
            size_average=False,
        ) / (sampled_inds.numel())

        # 计算目标预测概率损失
        objectness_loss = nn.functional.binary_cross_entropy_with_logits(
            objectness[sampled_inds], labels[sampled_inds]
        )

        return objectness_loss, box_loss

    def filter_proposals(self, proposals: torch.Tensor, object_ness: torch.Tensor,
                         image_sizes: List[Tuple[int, int]], num_anchors_per_level: List[int]):
        """
        筛除小boxes框，nms处理，根据预测概率获取前post_nms_top_n个目标
        :param proposals: 预测的bbox坐标
        :param object_ness: 预测的目标概率
        :param image_sizes: batch中每张图片的size信息
        :param num_anchors_per_level: 每个预测特征层上预测anchors的数目
        :return:
        """
        # 获取图片的个数
        num_images = proposals.shape[0]
        device = proposals.device

        # detach() 返回一个新的tensor，从当前计算图中分离下来的，但是仍指向原变量的存放位置,
        # 不同之处只是requires_grad为false，得到的这个tensor永远不需要计算其梯度，不具有grad。
        object_ness = object_ness.detach()
        object_ness = object_ness.reshape(num_images, -1)

        # levels负责记录分隔不同预测特征层上的anchors索引信息
        levels = []
        for idx, n in enumerate(num_anchors_per_level):
            levels.append(torch.full((n,), idx, dtype=torch.int64, device=device))
        levels = torch.cat(levels, 0)

        # [n] -> [1, n] -> [object_ness.shape[0], n]
        levels = levels.reshape(1, -1).expand_as(object_ness)

        # 获取每张预测特征图片上预测概率前pre_nms_top_n的anchors索引
        top_n_idx = self._get_top_n_idx(object_ness, num_anchors_per_level)

        image_range = torch.arange(num_images, device=device)
        batch_idx = image_range[:, None]

        # 根据每个预测特征层预测概率前pre_nms_top_n的anchors索引获取相应概率信息
        object_ness = object_ness[batch_idx, top_n_idx]
        levels = levels[batch_idx, top_n_idx]
        # 预测概率前pre_nms_top_n的anchors索引获取相应的bbox坐标信息
        proposals = proposals[batch_idx, top_n_idx]

        final_boxes = []
        final_scores = []
        # 遍历每张图像的相关预测信息
        for boxes, scores, lvl, img_shape in zip(proposals, object_ness, levels, image_sizes):
            # 调整预测的boxes信息，将越界的坐标调整到图片边界上
            boxes = box_ops.clip_boxes_to_images(boxes, img_shape)
            # 返回boxes满足宽，高都大于min_size的索引
            keep = box_ops.remove_small_boxes(boxes, self.min_size)
            # 获取滤除小目标过后的proposal
            boxes, scores, lvl = boxes[keep], scores[keep], lvl[keep]
            # 进行非极大值抑制，并且按目标类别分数进行排序
            keep = box_ops.batched_nms(boxes, scores, lvl, self.nms_thresh)
            # 获取前n个目标索引
            keep = keep[: self.post_nms_top_n()]
            boxes, scores = boxes[keep], scores[keep]
            final_boxes.append(boxes)
            final_scores.append(scores)
        return final_boxes, final_scores

    def _get_top_n_idx(self, object_ness: torch.Tensor, num_anchors_per_level: List[int]):
        """
        获取每张预测特征图片上预测概率前pre_nms_top_n的anchors索引
        :param object_ness: 每张图的预测目标概率信息
        :param num_anchors_per_level: 每个预测特征层上的预测的anchors个数
        :return:
        """
        # 记录每个预测特征层上的预测目标概率前pre_nms_top_n的anchors索引
        r = []
        offset = 0
        # 遍历每个预测特征层上的预测目标概率信息
        for ob in object_ness.split(num_anchors_per_level, 1):
            # 获取预测特征层上的预测的anchors个数
            num_anchors = ob.shape[1]
            pre_nms_top_n = min(self.pre_nms_top_n(), num_anchors)

            # topk函数将在指定维度(dim)上排序并返回pre_nms_top_n个索引
            _, top_n_idx = ob.topk(pre_nms_top_n, dim=1)
            r.append(top_n_idx + offset)
            offset += num_anchors
        return torch.cat(r, dim=1)

    def assign_targets_to_anchors(self, anchors: List[torch.Tensor], targets):
        """
        计算每个anchors最匹配的gt，并划分前景/背景以及废弃的anchors
        :param anchors:
        :param targets:
        :return: labels: 标记anchors归属类别（1, 0, -1分别对应正样本，背景，废弃的样本）
                    注意，在RPN中只有前景和背景，所有正样本的类别都是1，0代表背景
                matched_gt_boxes：与anchors匹配的gt
        """
        labels = []
        matched_gt_boxes = []
        # 遍历每张图像的anchors和targets
        for anchors_per_image, targets_per_image in zip(anchors, targets):
            gt_boxes = targets_per_image['boxes']
            if gt_boxes.numel() == 0:
                device = anchors_per_image.device
                matched_gt_boxes_per_image = torch.zeros(anchors_per_image.shape, dtype=torch.float32, device=device)
                labels_per_image = torch.zeros((anchors_per_image.shape[0],), dtype=torch.float32, device=device)
            else:
                # 计算anchors与真实bbox的iou信息
                match_quality_matrix = box_ops.box_iou(gt_boxes, anchors_per_image)
                # 计算每个anchors与gt匹配iou最大的索引（如果iou<0.3索引设置为-1，0.3<iou<0.7索引为-2
                matched_idxs = self.proposal_matcher(match_quality_matrix)
                matched_gt_boxes_per_image = gt_boxes[matched_idxs.clamp(min=0)]
                labels_per_image = matched_idxs > 0
                labels_per_image = labels_per_image.to(dtype=torch.float32)

                bg_indices = matched_idxs == self.proposal_matcher.BELOW_LOW_THRESHOLD
                labels_per_image[bg_indices] = 0.0

                discard_indices = matched_idxs == self.proposal_matcher.BETWEEN_THRESHOLDS
                labels_per_image[discard_indices] = -1.0
            labels.append(labels_per_image)
            matched_gt_boxes.append(matched_gt_boxes_per_image)
        return labels, matched_gt_boxes


def concat_box_prediction_layers(object_ness: List[torch.Tensor], pre_bbox_deltas: List[torch.Tensor]):
    """
    :param object_ness: 每个预测特征层上的预测目标概率
    :param pre_bbox_deltas: 每个预测特征层上的预测目标bboxes regression(回归)参数
    :return:
    """
    # 存储预测目标分数
    box_cls_flattened = []
    # 存储预测bbox回归参数
    box_regression_flattened = []

    # 变量每个预测特征层
    for box_cls_per_level, box_regression_per_level in zip(object_ness, pre_bbox_deltas):
        N, AxC, H, W = box_cls_per_level.shape
        Ax4 = box_regression_per_level.shape[1]
        # 计算出anchor数量
        A = Ax4 // 4
        # 计算出类别数量
        C = AxC // A

        # shape [batch_size, anchors, 1]
        box_cls_per_level = permute_and_flatten(box_cls_per_level, N, A, C, H, W)
        box_cls_flattened.append(box_cls_per_level)

        box_regression_per_level = permute_and_flatten(box_regression_per_level, N, A, 4, H, W)
        box_regression_flattened.append(box_regression_per_level)
    box_cls = torch.cat(box_cls_flattened, dim=1).flatten(0, -2)
    box_regression = torch.cat(box_regression_flattened, dim=1).reshape(-1, 4)
    return box_cls, box_regression


def permute_and_flatten(layer: torch.Tensor, N: int, A: int, C: int, H: int, W: int):
    """
    调整tensor顺序，并进行reshape
    :param layer: 预测特征层上预测的目标概率或bboxes回归参数
    :param N: batch_size
    :param A: anchor数量
    :param C: 类别数量
    :param H: 高
    :param W: 宽
    :return:
    """
    layer = layer.view(N, -1, C, H, W)
    # 调换tensor维度
    layer = layer.permute(0, 3, 4, 1, 2)
    layer = layer.reshape(N, -1, C)
    return layer
