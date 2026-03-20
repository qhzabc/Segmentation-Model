import random
import nibabel as nib
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import os
import json


class MRIDataset(Dataset):
    def __init__(self, root_dir, mode='all', Flag='train',
                 transform=None, include_path=False, augment=False,
                 num_channels=44, channel_dim=2, json_path=None):
        """
        Args:
            root_dir (str): Root directory of the dataset
            mode (str): 'train', 'val', 'test', or 'all'
            Flag (str): 'train' or 'test' for data split
            transform (callable): Optional transform to be applied
            include_path (bool): Whether to include image path in return
            augment (bool): Whether to apply data augmentation (only for training)
            num_channels (int): Number of channels in the images (default: 44)
            channel_dim (int): Dimension where channels are stored in the nii file
            json_path (str): Path to JSON file containing text descriptions
        """
        self.root_dir = root_dir
        self.mode = mode
        self.include_path = include_path
        self.num_channels = num_channels
        self.channel_dim = channel_dim
        self.samples = []
        self.labels = []
        self.image_paths = []
        self.texts = []
        self.json_data = {}

        # 加载JSON数据
        if json_path and os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                self.json_data = json.load(f)

        # 收集样本
        self._collect_samples(Flag)

        # 设置数据转换
        self.transform = transform or self._default_transform(augment)

    def _collect_samples(self, Flag):
        """根据数据集类型收集样本"""
        # 处理单一目录结构
        for img_file in os.listdir(self.root_dir):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.nii', '.nii.gz')):
                img_path = os.path.join(self.root_dir, img_file)
                self.image_paths.append(img_path)
                # 所有样本标签设为0
                self.labels.append(0)

                # 从JSON中获取文本描述
                text = self.json_data.get(img_file, "")  # 使用文件名作为键
                self.texts.append(text)

        # 创建样本索引
        self.samples = list(zip(self.image_paths, self.labels, self.texts))
        cut = int(len(self.samples) * 0.8)
        cut2 = int(len(self.samples) * 0.9)
        if Flag == "train":
            self.samples = self.samples[:cut]
        else:
            self.samples = self.samples[cut:cut2]
        random.shuffle(self.samples)

    def _default_transform(self, augment=False):
        """创建默认的数据转换管道（针对44通道）"""
        # 多通道图像的特殊处理
        base_transforms = [
            transforms.Resize((256, 256)),  # 调整空间大小
        ]

        # 数据增强 - 仅在空间维度上
        if augment and self.mode == 'train':
            augment_transforms = [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(15),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            ]
            base_transforms = augment_transforms + base_transforms

        # 添加归一化 - 需要为每个通道提供参数
        norm_transform = transforms.Normalize(
            mean=[0.5] * self.num_channels,
            std=[0.5] * self.num_channels
        )
        base_transforms.append(norm_transform)

        return transforms.Compose(base_transforms)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label_idx, text = self.samples[idx]

        try:
            # 处理普通图像格式
            if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                img = Image.open(img_path).convert('RGB')
                img = transforms.ToTensor()(img)

                # 如果普通图像通道数不匹配，进行转换
                if img.shape[0] != self.num_channels:
                    # 简单复制通道以匹配所需数量
                    repeats = self.num_channels // img.shape[0] + 1
                    img = img.repeat(repeats, 1, 1)[:self.num_channels, :, :]

            # 处理NIfTI医学图像（44通道）
            elif img_path.lower().endswith(('.nii', '.nii.gz')):
                # 使用内存映射加载大文件
                ni = nib.load(img_path, mmap=True)
                img_data = np.asanyarray(ni.dataobj).astype(np.float32)

                # 处理不同维度的数据
                if img_data.ndim == 4:  # 4D数据: [X, Y, Z, C] 或 [X, Y, C, Z]
                    # 将通道维度移到最前面 [C, X, Y, Z]
                    img_data = np.moveaxis(img_data, self.channel_dim, 0)

                    # 取中间Z切片
                    z_slice = img_data.shape[3] // 2
                    img_data = img_data[:, :, :, z_slice]

                elif img_data.ndim == 3:  # 3D数据: [X, Y, C]
                    # 将通道维度移到最前面 [C, X, Y]
                    img_data = np.moveaxis(img_data, self.channel_dim, 0)

                # 确保通道数正确
                if img_data.shape[0] != self.num_channels:
                    # 如果通道不足，复制现有通道
                    if img_data.shape[0] < self.num_channels:
                        repeats = self.num_channels // img_data.shape[0] + 1
                        img_data = np.tile(img_data, (repeats, 1, 1))[:self.num_channels, :, :]
                    # 如果通道过多，截断
                    else:
                        img_data = img_data[:self.num_channels, :, :]

                # 归一化每个通道
                for c in range(img_data.shape[0]):
                    channel = img_data[c]
                    min_val = np.min(channel)
                    max_val = np.max(channel)
                    if max_val > min_val:  # 避免除以零
                        img_data[c] = (channel - min_val) / (max_val - min_val)
                    else:
                        img_data[c] = channel - min_val  # 所有值相同的情况

                img = torch.tensor(img_data, dtype=torch.float32)

            else:
                raise ValueError(f"Unsupported file format: {img_path}")

            # 应用转换
            if self.transform:
                img = self.transform(img)

            # 返回结果
            if self.include_path:
                return img, text, torch.tensor(label_idx), img_path
            return img, text, torch.tensor(label_idx)

        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # 返回空图像占位符 (44通道)
            empty_img = torch.zeros(self.num_channels, 256, 256)
            if self.include_path:
                return empty_img, "", torch.tensor(-1), img_path
            return empty_img, "", torch.tensor(-1)