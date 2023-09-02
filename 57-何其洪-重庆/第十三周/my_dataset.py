# -*- coding: utf-8 -*-
import json
import os

import torch
from PIL import Image
from lxml import etree
from torch.utils.data import Dataset


class VOC2012DataSet(Dataset):
    def __init__(self, rootPath, transform, train=True):
        self.transform = transform
        # 数据集根路径
        self.root = os.path.join(rootPath, 'VOCdevkit', 'VOC2012')
        # 图片根路径
        self.img_root = os.path.join(self.root, 'JPEGImages')
        # 标注数据根路径
        self.annotations_root = os.path.join(self.root, 'Annotations')
        # 判断是否加载训练数据集
        if train:
            txt_list = os.path.join(self.root, 'ImageSets', 'Main', 'train.txt')
        else:
            txt_list = os.path.join(self.root, 'ImageSets', 'Main', 'val.txt')
        # 缓存标注数据xml文件路径
        with open(txt_list) as read:
            self.xml_list = [os.path.join(self.annotations_root, line.strip() + '.xml') for line in read.readlines()][:10]
        # 加载类别
        with open('./pascal_voc_classes.json', 'r') as cls:
            self.class_dict = json.load(cls)

    def __getitem__(self, index):
        annotation = self.__read_xml_annotation__(index)
        img_path = os.path.join(self.img_root, annotation['filename'])
        img = Image.open(img_path).convert('RGB')
        boxes = []
        labels = []
        iscrowd = []
        for obj in annotation['object']:
            bndbox = obj['bndbox']
            xmin = float(bndbox['xmin'])
            ymin = float(bndbox['ymin'])
            xmax = float(bndbox['xmax'])
            ymax = float(bndbox['ymax'])
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_dict[obj['name']])
            iscrowd.append(int(obj['difficult']))
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        image_id = torch.tensor([index])
        # 宽*高=面积
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd
        }
        if self.transform is not None:
            img, target = self.transform(img, target)
        return img, target

    def __len__(self):
        return len(self.xml_list)

    def __read_xml_annotation__(self, index):
        # 获取标注xml文件路径
        xml_path = self.xml_list[index]
        # 读取xml文件
        with open(xml_path) as xmlFile:
            xml_str = xmlFile.read()
        xml = etree.fromstring(xml_str)
        return self.parse_xml_to_dict(xml)['annotation']

    def parse_xml_to_dict(self, xml):
        if len(xml) == 0:
            return {xml.tag: xml.text}
        result = {}
        for child in xml:
            child_result = self.parse_xml_to_dict(child)
            if child.tag != 'object':
                result[child.tag] = child_result[child.tag]
            else:
                if child.tag not in result:
                    result[child.tag] = []
                result[child.tag].append(child_result[child.tag])
        return {xml.tag: result}

    def get_height_width(self, index):
        annotation = self.__read_xml_annotation__(index)
        size = annotation['size']
        return size['height'], size['width']

    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))

    def coco_index(self, idx):
        """
        该方法是专门为pycocotools统计标签信息准备，不对图像和标签作任何处理
        由于不用去读取图片，可大幅缩减统计时间

        Args:
            idx: 输入需要获取图像的索引
        """
        # read xml
        xml_path = self.xml_list[idx]
        with open(xml_path) as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = self.parse_xml_to_dict(xml)["annotation"]
        data_height = int(data["size"]["height"])
        data_width = int(data["size"]["width"])
        # img_path = os.path.join(self.img_root, data["filename"])
        # image = Image.open(img_path)
        # if image.format != "JPEG":
        #     raise ValueError("Image format not JPEG")
        boxes = []
        labels = []
        iscrowd = []
        for obj in data["object"]:
            xmin = float(obj["bndbox"]["xmin"])
            xmax = float(obj["bndbox"]["xmax"])
            ymin = float(obj["bndbox"]["ymin"])
            ymax = float(obj["bndbox"]["ymax"])
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_dict[obj["name"]])
            iscrowd.append(int(obj["difficult"]))

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        return (data_height, data_width), target
