import os
import json
import numpy as np
from PIL import Image
from torchvision import transforms


def load_label_mapping(mapping_file):
    """加载标签映射文件"""
    with open(mapping_file, 'r') as f:
        mapping = json.load(f)
    reverse_mapping = {v: k for k, v in mapping.items()}
    return mapping, reverse_mapping


def get_image_transforms(size=256, augment=False):
    """获取图像转换管道"""
    base_transforms = [
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]

    if augment:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            *base_transforms
        ])

    return transforms.Compose(base_transforms)


def calculate_class_weights(labels):
    """计算类别权重用于处理不平衡数据"""
    counts = np.bincount(labels)
    class_weights = 1. / counts
    class_weights = class_weights / class_weights.sum() * len(counts)
    return class_weights