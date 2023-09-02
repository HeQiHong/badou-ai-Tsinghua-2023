# -*- coding: utf-8 -*-
from typing import List, Optional, Dict, Tuple

import torch
import torch.nn as nn
from collections import OrderedDict

from torchvision.ops import MultiScaleRoIAlign

from .transform import GeneralizedRCNNTransform
from .anchors import AnchorsGenerator
from .region_proposal_network import RpnHead, RegionProposalNetwork
from .roi_head import RoIHeads


class FasterRCNNBase(nn.Module):
    def __init__(self, backbone, rpn, roi_heads, transform):
        """
        faster rcnn 基础网络
        :param backbone: 特征提取网络
        :param rpn: 区域建议生成网络
        :param roi_heads:
        :param transform:
        """
        super().__init__()
        self.backbone = backbone
        self.rpn = rpn
        self.roi_heads = roi_heads
        self.transform = transform

    def forward(self, images: List[torch.Tensor], targets: Optional[List[Dict[str, torch.Tensor]]] = None):
        if self.training:
            if targets is None:
                raise ValueError('当前为训练模式，请传入target参数')
        # 存储原图的大小
        original_image_sizes: List[Tuple[int, int]] = []
        for img in images:
            val = img.shape[-2:]
            original_image_sizes.append((val[0], val[1]))
        # 对图像进行预处理
        images, targets = self.transform(images, targets)
        # 获得特征图
        features = self.backbone(images.tensors)
        # 统一单层特征图和多层特征图数据格式
        if isinstance(features, torch.Tensor):
            features = OrderedDict([('0', features)])

        # 将特征图以及标注数据传入rpn网络
        proposals, proposal_losses = self.rpn(images, features, targets)

        # 将rpn生成的数据及标注信息传入fast rcnn后半部分
        detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)

        # 对网络预测结果进行后处理（主要是将bboxes结果还原到原图尺度上
        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

        # 统计损失
        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        if self.training:
            return losses
        else:
            return detections


class FasterRCNN(FasterRCNNBase):
    def __init__(self, backbone, num_classes=None,
                 min_size=800, max_size=1333,
                 image_mean=None, image_std=None,
                 rpn_anchor_generator=None, rpn_head=None,
                 rpn_pre_nms_top_n_train=2000, rpn_pre_nms_top_n_test=1000,
                 rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=1000,
                 rpn_nms_thresh=0.7, rpn_fg_iou_thresh=0.7,
                 rpn_bg_iou_thresh=0.3, rpn_batch_size_per_image=256,
                 rpn_positive_fraction=0.5, rpn_score_thresh=0.0,
                 box_roi_pool=None, box_head=None, box_predictor=None,
                 box_score_thresh=0.05, box_nms_thresh=0.5, box_detections_per_img=100,
                 box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5,
                 box_batch_size_per_image=512, box_positive_fraction=0.25,
                 bbox_reg_weights=None):
        """
        :param backbone: 特征提取网络
        :param num_classes: 类别数量
        :param min_size: 图片resize时限制的最小尺寸
        :param max_size: 图片resize时限制的最大尺寸
        :param image_mean: 归一化时的均值
        :param image_std: 归一化时的方差
        :param rpn_anchor_generator: 锚点生成器
        :param rpn_head: rpn前置处理网络
        :param rpn_pre_nms_top_n_train: rpn中nms（非极大值抑制）处理前保留的proposal数
        :param rpn_pre_nms_top_n_test:
        :param rpn_post_nms_top_n_train: rpn中nms（非极大值抑制）处理后保留的proposal数
        :param rpn_post_nms_top_n_test:
        :param rpn_nms_thresh: rpn中进行nms处理时的iou阈值
        :param rpn_fg_iou_thresh: rpn计算损失时，采集正样本设置的阈值
        :param rpn_bg_iou_thresh: rpn计算损失时，采集负样本设置的阈值
        :param rpn_batch_size_per_image: rpn计算损失时采样的样本数量
        :param rpn_positive_fraction: rpn计算损失时正负样本比例
        :param box_roi_pool:
        :param box_head:
        :param box_predictor:
        :param box_score_thresh: 移除低目标概率
        :param box_nms_thresh: fast rcnn中进行nms处理的阈值
        :param box_detections_per_img: 对预测结果根据score排序取前n个目标
        :param box_fg_iou_thresh: faste rcnn 计算误差时采集正样本设置的阈值
        :param box_bg_iou_thresh: faste rcnn 计算误差时采集负样本设置的阈值
        :param box_batch_size_per_image: faste rcnn 计算误差时采样的样本数
        :param box_positive_fraction: faste rcnn 计算误差时正负样本比例
        :param bbox_reg_weights:
        """
        # 预测特征层的channels
        out_channels = backbone.out_channels

        # 若anchor生成器为空，则自动生成针对resnet50_fpn的anchor生成器
        if rpn_anchor_generator is None:
            anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
            aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
            rpn_anchor_generator = AnchorsGenerator(
                anchor_sizes, aspect_ratios
            )

        # 生成RPN通过滑动窗口预测网络部分
        if rpn_head is None:
            rpn_head = RpnHead(
                out_channels, rpn_anchor_generator.num_anchors_per_location()[0]
            )

        # 默认rpn_pre_nms_top_n_train = 2000, rpn_pre_nms_top_n_test = 1000,
        # 默认rpn_post_nms_top_n_train = 2000, rpn_post_nms_top_n_test = 1000,
        rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)

        # 定义整个RPN框架
        rpn = RegionProposalNetwork(
            rpn_anchor_generator, rpn_head,
            rpn_fg_iou_thresh, rpn_bg_iou_thresh,
            rpn_batch_size_per_image, rpn_positive_fraction,
            rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_nms_thresh,
            score_thresh=rpn_score_thresh)

        #  Multi-scale RoIAlign pooling
        if box_roi_pool is None:
            box_roi_pool = MultiScaleRoIAlign(
                featmap_names=['0', '1', '2', '3'],  # 在哪些特征层进行roi pooling
                output_size=[7, 7],
                sampling_ratio=2)

        # fast RCNN中roi pooling后的展平处理两个全连接层部分
        if box_head is None:
            resolution = box_roi_pool.output_size[0]  # 默认等于7
            representation_size = 1024
            box_head = TwoMLPHead(
                out_channels * resolution ** 2,
                representation_size
            )

        # 在box_head的输出上预测部分
        if box_predictor is None:
            representation_size = 1024
            box_predictor = FastRCNNPredictor(
                representation_size,
                num_classes)

        # 将roi pooling, box_head以及box_predictor结合在一起
        roi_heads = RoIHeads(
            # box
            box_roi_pool, box_head, box_predictor,
            box_fg_iou_thresh, box_bg_iou_thresh,  # 0.5  0.5
            box_batch_size_per_image, box_positive_fraction,  # 512  0.25
            bbox_reg_weights,
            box_score_thresh, box_nms_thresh, box_detections_per_img)  # 0.05  0.5  100

        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]

        # 对数据进行标准化，缩放，打包成batch等处理部分
        transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)

        super(FasterRCNN, self).__init__(backbone, rpn, roi_heads, transform)


class TwoMLPHead(nn.Module):
    def __init__(self, in_channels, representation_size):
        super(TwoMLPHead, self).__init__()
        self.fc6 = nn.Linear(in_channels, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)

    def forward(self, x):
        x = x.flatten(start_dim=1)

        x = nn.functional.relu(self.fc6(x))
        x = nn.functional.relu(self.fc7(x))

        return x


class FastRCNNPredictor(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(FastRCNNPredictor, self).__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)

    def forward(self, x):
        if x.dim() == 4:
            assert list(x.shape[2:]) == [1, 1]
        x = x.flatten(start_dim=1)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)

        return scores, bbox_deltas
