# -*- coding: utf-8 -*-
import torch
import torchvision.ops


def box_iou(boxes1, boxes2):
    """
    计算两个boxes的iou值
    :param boxes1:
    :param boxes2:
    :return:
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    #  When the shapes do not match,
    #  the shape of the returned output tensor follows the broadcasting rules
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # left-top [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # right-bottom [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    iou = inter / (area1[:, None] + area2 - inter)
    return iou


def clip_boxes_to_images(boxes, img_size):
    """
    裁剪预测的boxes信息，将越界的坐标调整到图片边界上
    :param boxes: (Tensor[N, 4]): boxes in (x1, y1, x2, y2) format
    :param img_size: (Tuple[height, width]): size of the image
    :return:
    """
    dim = boxes.dim()
    # x1, x2
    boxes_x = boxes[..., 0::2]
    # y1, y2
    boxes_y = boxes[..., 1::2]
    height, width = img_size

    # 将x坐标限制在[0, width]范围内
    boxes_x = boxes_x.clamp(min=0, max=width)
    boxes_y = boxes_y.clamp(min=0, max=height)

    clipped_boxes = torch.stack((boxes_x, boxes_y), dim=dim)
    return clipped_boxes.reshape(boxes.shape)


def remove_small_boxes(boxes, min_size):
    """
    返回boxes满足宽、高都大于min_size的索引
    :param boxes: boxes信息 (Tensor[N, 4]): boxes in (x1, y1, x2, y2) format
    :param min_size: float
    :return:
    """
    ws, hs = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
    # 当宽高同时大于min_size时将对应位置标记为True
    # min_size=128, [128, 256, 64, 512] -> [True, True, False, True]
    keep = (ws >= min_size) & (hs >= min_size)
    # [True, True, False, True] -> [0, 1, 3]
    return keep.nonzero().squeeze(1)


def batched_nms(boxes, scores, lvl, nms_thresh):
    """
    非极大值抑制
    :param boxes: proposal的坐标参数
    :param scores: 预测的目标概率值
    :param lvl: 每张图片在不同预测层上的索引
    :param nms_thresh: 阈值
    :return:
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)
    # 获取所有boxes中的最大的一个值
    max_coordinate = boxes.max()

    offsets = lvl.to(boxes) * (max_coordinate + 1)
    # boxes 加上对应层上的偏移量后，保证不同类别之间的boxes不会有重合的现象
    boxes_4_nms = boxes + offsets[:, None]
    keep = torchvision.ops.nms(boxes_4_nms, scores, nms_thresh)
    return keep
