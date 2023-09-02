# -*- coding: utf-8 -*-
from typing import List

import torch
import torch.nn as nn
from torchvision.models.detection.image_list import ImageList


class AnchorsGenerator(nn.Module):
    """
    anchor生成器
    """
    def __init__(self, size=(128, 256, 512), aspect_ratios=(0.5, 1.0, 2.0)):
        """
        :param size: 预测尺度
        :param aspect_ratios: 每个预测尺度的几种高宽比例
        """
        super().__init__()
        if not isinstance(size[0], (list, tuple)):
            size = tuple((s,) for s in size)
        if not isinstance(aspect_ratios[0], (list, tuple)):
            aspect_ratios = (aspect_ratios,) * len(size)

        self.size = size
        self.aspect_ratios = aspect_ratios
        self.cell_anchors = None
        self._cache = {}

    def forward(self, image_list: ImageList, feature_maps: List[torch.Tensor]):
        """
        :param image_list: 经过GeneralizedRCNNTransform处理过后的图片列表
        :param feature_maps: 特征图列表
        :return:
        """
        # 获取每个特征层的尺寸[height, width]
        grid_sizes = [feature_map.shape[-2:] for feature_map in feature_maps]
        # 获取图像的尺寸[height, width]
        image_size = image_list.tensors.shape[-2:]
        # 获取变量类型和device
        dtype, device = feature_maps[0].dtype, feature_maps[0].device
        # 计算特征层上的步长和原图像上的步长
        strides = []
        for gs in grid_sizes:
            height = torch.tensor(image_size[0] / gs[0], dtype=torch.int64, device=device)
            width = torch.tensor(image_size[1] / gs[1], dtype=torch.int64, device=device)
            strides.append([height, width])

        # 根据sizes和aspect_ratios生成anchor模板
        self.set_cell_anchors(dtype, device)
        # 计算所有映射到原图上的anchors的坐标信息
        # 得到list，对应每张预测特征图映射回原图的anchors坐标信息
        anchors_over_all_feature_maps = self.cached_grid_anchors(grid_sizes, strides)
        anchors = []
        for i in range(len(image_list.image_sizes)):
            anchors_in_image = []
            # 遍历每张预测特征图映射回原图的anchors信息
            for anchors_per_feature_map in anchors_over_all_feature_maps:
                anchors_in_image.append(anchors_per_feature_map)
            anchors.append(anchors_in_image)
        # 将每一张图的所有预测特征层的anchors坐标信息拼接在一起
        anchors = [torch.cat(anchors_per_image) for anchors_per_image in anchors]
        self._cache.clear()
        return anchors

    def set_cell_anchors(self, dtype, device):
        if self.cell_anchors is not None:
            return
        cell_anchors = []
        # 根据size和aspect_ratios生成anchors模板
        for sizes, aspect_ratios in zip(self.size, self.aspect_ratios):
            cell_anchors.append(self.generate_anchors(sizes, aspect_ratios, dtype, device))
        self.cell_anchors = cell_anchors

    def generate_anchors(self, sizes, aspect_ratios, dtype, device):
        sizes = torch.as_tensor(sizes, dtype=dtype, device=device)
        aspect_ratios = torch.as_tensor(aspect_ratios, dtype=dtype, device=device)
        h_ratios = torch.sqrt(aspect_ratios)
        w_ratios = 1.0 / h_ratios

        hs = (h_ratios[:, None] * sizes[None, :]).view(-1)
        ws = (w_ratios[:, None] * sizes[None, :]).view(-1)

        # 生成的模板都是以(0, 0)为中心
        base_anchors = torch.stack([-ws, -hs, ws, hs], dim=1)
        return base_anchors.round()

    def cached_grid_anchors(self, grid_sizes: List[List[int]], strides: List[List[torch.Tensor]]):
        """
        将所有计算得到的anchors进行缓存
        :param grid_sizes: 每个特征层的尺寸
        :param strides: 预测特征层的cell对应的到原图的尺度信息
        :return:
        """
        key = str(grid_sizes) + str(strides)
        if key in self._cache:
            return self._cache[key]
        anchors = self.grid_anchors(grid_sizes, strides)
        self._cache[key] = anchors
        return anchors

    def grid_anchors(self, grid_sizes: List[List[int]], strides: List[List[torch.Tensor]]):
        """
        计算预测特征图对应原始图像上的所有anchors的坐标
        :param grid_sizes: 预测特征矩阵的height，width集合
        :param strides: 预测特征矩阵上的一格对应原始图像上的长度
        :return:
        """
        anchors = []
        cell_anchors = self.cell_anchors
        # 遍历每个特征层的grid_size, strides和cell_anchors
        for size, stride, base_anchors in zip(grid_sizes, strides, cell_anchors):
            # 每个预测特征层的高和宽
            grid_height, grid_width = size
            # 每个预测特征层上一个cell对应原图上的高度和宽度
            stride_height, stride_width = stride
            device = base_anchors.device

            # 假设grid_height=3，grid_width=3
            # arange生成0~3的张量 -> [0, 1, 2]，当和stride_width相乘时，对应到原图上的x坐标点
            shifts_x = torch.arange(start=0, end=grid_width, dtype=torch.float32, device=device) * stride_width
            # 同理可得shifts_y
            shifts_y = torch.arange(start=0, end=grid_height, dtype=torch.float32, device=device) * stride_height

            # 计算预测特征矩阵上每个点对应原图上的坐标
            # meshgrid分别传入行坐标和列坐标，生成网格行坐标矩阵和网格列坐标矩阵
            """
            x = [1, 2, 3, 4]
            y = [4, 5, 6]
            y, x = torch.meshgrid(y, x)
                    ||
            y = [[4, 4, 4, 4],
                 [5, 5, 5, 6],
                 [6, 6, 6, 6]]
            x = [[1, 2, 3, 4]
                 [1, 2, 3, 4]
                 [1, 2, 3, 4]]
            """
            shifts_y, shifts_x = torch.meshgrid(shifts_y, shifts_x)
            # 展平
            shifts_x = shifts_x.reshape(-1)
            shifts_y = shifts_y.reshape(-1)

            # 计算anchors坐标(xmin, ymin, xmax, ymax) 在原图上的坐标偏移量
            # shape: [grid_height * grid_width, 4]
            shifts = torch.stack([shifts_x, shifts_y, shifts_x, shifts_y], dim=1)
            # 将anchors模板与原图上的坐标偏移量相加得到原图上所有的anchors信息（shape不同时会使用广播机制）
            # view参数-1表示自动调整这个维度上的元素个数，以保证元素的总数不变。
            shifts_anchor = shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)
            anchors.append(shifts_anchor.reshape(-1, 4))
        return anchors

    def num_anchors_per_location(self):
        # 计算每个预测特征层上每个滑动窗口的预测目标数
        return [len(s) * len(a) for s, a in zip(self.size, self.aspect_ratios)]
