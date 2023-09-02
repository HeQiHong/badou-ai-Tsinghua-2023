# -*- coding: utf-8 -*-
import math

import torch
import torch.nn as nn
from torchvision.models.detection.image_list import ImageList


class GeneralizedRCNNTransform(nn.Module):
    def __init__(self, min_size, max_size, image_mean, image_std):
        """
        对图像进行标准化和resize处理
        :param min_size: 图像最小边长
        :param max_size: 图像最大边长
        :param image_mean: 标准化均值
        :param image_std: 标准化方差
        """
        super().__init__()
        self.min_size = min_size
        self.max_size = max_size
        self.image_mean = image_mean
        self.image_std = image_std

    def normalize(self, image):
        mean = torch.as_tensor(self.image_mean, dtype=image.dtype, device=image.device)
        std = torch.as_tensor(self.image_std, dtype=image.dtype, device=image.device)
        # 图片是的shape(c,h,w)所以需要将均值和方差转换为三维 mean[:, None, None]
        return (image - mean[:, None, None]) / std[:, None, None]

    def resize(self, image, target):
        """
        将图像缩放到指定的大小范围，并缩放对应的bboxes信息
        :param image: 图片
        :param target: 图片的信息, 包含bboxes
        :return:
        """
        height, width = image.shape[-2:]
        # 获取高宽中的最小值
        min_size = float(min(height, width))
        # 获取高宽中的最大值
        max_size = float(max(height, width))
        # 指定最小边除图片最小边得到缩放比例
        scale_factor = self.min_size / min_size
        # 判断该缩放比例计算的最大边长是否大于指定的最大边长
        if max_size * scale_factor > self.max_size:
            # 将缩放比例设置为指定最大边长和图片最大边长的比例
            scale_factor = self.max_size / max_size
        # interpolate对图片进行插值采样
        # image[None]在图片最前面增加一个维度[c,h,w] -> [n,c,h,w]
        # mode=bilinear 是双线性插值，双线性插值只支持4维
        image = nn.functional.interpolate(image[None], scale_factor=scale_factor,
                                          mode='bilinear', align_corners=False)[0]

        # 验证模式下target为空
        if target is None:
            return image, target
        # 缩放图像的boxes信息
        bbox = target['boxes']
        bbox = resize_boxes(bbox, (height, width), image.shape[-2:])
        target['boxes'] = bbox
        return image, target

    def batch_image(self, images, size_divisible=32):
        """
        将一批图像打包成一个batch返回，batch中的每个tensor的shape是相同的
        :param images: 输入的图片
        :param size_divisible: 将图像的高度和宽度调整到该数的整数倍
        :return: 打包好的tensor数据
        """
        max_size = self.max_by_axis([list(img.shape) for img in images])
        stride = float(size_divisible)
        # 将图片高宽调整到stride的整数倍
        max_size[1] = int(math.ceil(max_size[1] / stride) * stride)
        max_size[2] = int(math.ceil(max_size[2] / stride) * stride)
        # [c,h,w] -> [n,c,h,w]
        batch_shape = [len(images)] + max_size

        # 创建形状为batch_shape的tensor，并用0填充
        batch_images = images[0].new_full(batch_shape, 0)
        for img, pad_img in zip(images, batch_images):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
        return batch_images

    def max_by_axis(self, shape_list):
        max_shape = shape_list[0]
        for sublist in shape_list[1:]:
            for index, item in enumerate(sublist):
                max_shape[index] = max(max_shape[index], item)
        return max_shape

    def forward(self, images, targets=None):
        for i in range(len(images)):
            image = images[i]
            target_index = targets[i] if targets is not None else None

            if image.dim() != 3:
                raise ValueError("images is expected to be a list of 3d tensors "
                                 "of shape [C, H, W], got {}".format(image.shape))
            image = self.normalize(image)  # 对图像进行标准化处理
            image, target_index = self.resize(image, target_index)  # 对图像和对应的bboxes缩放到指定范围
            images[i] = image
            if targets is not None and target_index is not None:
                targets[i] = target_index
        # 记录resize之后的尺寸
        image_sizes = [image.shape[-2:] for image in images]
        # 将图片进行打包
        images = self.batch_image(images)
        image_sizes_list = []
        for image_size in image_sizes:
            assert len(image_size) == 2
            image_sizes_list.append((image_size[0], image_size[1]))
        image_list = ImageList(images, image_sizes_list)
        return image_list, targets

    def postprocess(self, result, image_shapes, original_image_sizes):
        if self.training:
            return result
        for i, (pred, im_s, o_im_s) in enumerate(zip(result, image_shapes, original_image_sizes)):
            # 取出预测的bbox信息
            boxes = pred['boxes']
            # 缩放回原图尺寸上
            boxes = resize_boxes(boxes, im_s, o_im_s)
            result[i]['boxes'] = boxes
        return result


def resize_boxes(boxes, original_size, new_size):
    """
    缩放boxes信息
    :param boxes:
    :param original_size: 图片缩放前的高宽
    :param new_size: 图片缩放后的高宽
    :return: 缩放后的boxes
    """
    # 计算出高宽缩放比例
    ratios = []
    for ns, os in zip(new_size, original_size):
        ratios.append(torch.tensor(ns, dtype=torch.float32, device=boxes.device)
                      / torch.tensor(os, dtype=torch.float32, device=boxes.device))
    ratio_height, ratio_width = ratios
    # 展开boxes[num, 4], num为box个数
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    # 计算缩放后的boxes
    xmin = xmin * ratio_width
    xmax = xmax * ratio_width
    ymin = ymin * ratio_height
    ymax = ymax * ratio_height
    # 将数据堆叠为二维
    return torch.stack((xmin, ymin, xmax, ymax), dim=1)
