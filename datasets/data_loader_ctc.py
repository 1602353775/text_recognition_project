import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class TextRecognitionDataset(Dataset):
    def __init__(self, img_dir, label_file, transform=None):
        """
        初始化数据集类
        :param img_dir: 图像文件夹路径
        :param label_file: 字符到序列标签的json文件路径
        :param transform: 图像的增强变换
        """
        self.img_dir = img_dir
        self.transform = transform

        # 读取字符到序列标签的映射
        with open(label_file, 'r', encoding='utf-8') as f:
            self.char_to_sequence = json.load(f)

        # 获取文件夹中所有有效图像文件，忽略隐藏文件
        self.img_files = [
            file for file in os.listdir(img_dir)
            if file.endswith('.png') and not file.startswith('.')
        ]

    def __len__(self):
        return len(self.img_files)

    def _get_label_from_filename(self, filename):
        """根据文件名获取图像的字符序列标签"""
        # 提取字符的Unicode编码，例如 'U+4E00.png' -> 'U+4E00'
        unicode_name = os.path.splitext(filename)[0]
        char = chr(int(unicode_name[2:], 16))  # 将编码转回汉字

        # 获取字符对应的序列标签
        if char in self.char_to_sequence:
            label = self.char_to_sequence[char]
        else:
            raise ValueError(f"标签文件中找不到字符：{char}")

        return torch.tensor(label, dtype=torch.long)

    def __getitem__(self, idx):
        """获取图像和对应的标签"""
        # 获取图像文件路径
        img_file = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_file)

        # 打开图像
        image = Image.open(img_path).convert('RGB')

        # 数据增强，如果有定义
        if self.transform:
            image = self.transform(image)

        # 获取图像对应的序列标签
        label = self._get_label_from_filename(img_file)

        return image, label

# 自定义collate_fn
def collate_fn(batch):
    """
    自定义 collate_fn，用于批量数据加载。
    将图像张量和标签张量打包成批处理，并保留不定长标签。

    :param batch: 一个批次的数据样本，每个样本包含（图像, 标签）
    :return: 一个包含图像张量、标签张量和标签长度的批处理
    """
    # 分离批次中的图像和标签
    images, labels = zip(*batch)

    # 将图像转换为张量
    images = torch.stack(images, dim=0)

    # 获取标签的真实长度
    label_lengths = torch.tensor([len(label) for label in labels], dtype=torch.long)

    # 将标签列表拼接成一维张量
    labels = torch.cat(labels, dim=0)

    return images, labels, label_lengths


# 修改后的 collate_fn 保留标签列表格式
def collate_fn(batch):
    """
    自定义 collate_fn，用于批量数据加载。
    将图像张量和标签张量打包成批处理，并保留不定长标签。

    :param batch: 一个批次的数据样本，每个样本包含（图像, 标签）
    :return: 一个包含图像张量、标签列表和标签长度的批处理
    """
    # 分离批次中的图像和标签
    images, labels = zip(*batch)

    # 将图像转换为张量
    images = torch.stack(images, dim=0)

    # 获取标签的真实长度
    label_lengths = torch.tensor([len(label) for label in labels], dtype=torch.long)

    # 保留标签为列表格式
    return images, labels, label_lengths


# 数据增强变换定义
def get_transform():
    return transforms.Compose([
        transforms.Resize((64, 64)),  # 调整图片尺寸
        transforms.RandomRotation(10),  # 随机旋转图片
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 调整亮度/对比度等
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # 随机仿射变换
        transforms.ToTensor(),  # 转换为张量
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化
    ])


