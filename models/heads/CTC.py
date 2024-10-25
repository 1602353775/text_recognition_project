import torch
import torch.nn as nn
import torch.nn.functional as F
from models.backbones.SVTR import SVTRv2

class SVTRTransformerDecoder(nn.Module):
    def __init__(self, input_dim=192, num_heads=8, num_layers=4, num_classes=1000, max_len=64):
        """
        使用Transformer解码定长序列为不定长序列。

        :param input_dim: 输入特征长度（SVTR输出的特征长度）
        :param num_heads: 多头注意力的头数
        :param num_layers: Transformer编码层数
        :param num_classes: 字符类别数量（包括blank字符）
        :param max_len: 输入序列的最大长度
        """
        super(SVTRTransformerDecoder, self).__init__()

        # 可学习的位置编码参数
        self.positional_encoding = nn.Parameter(torch.randn(1, max_len, input_dim))

        # 定义 Transformer 编码器
        transformer_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=num_layers)

        # 全连接层将Transformer输出映射到字符类别
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        """
        前向传播
        :param x: 输入张量，形状为 (B, 64, 192)
        :return: 解码后的输出，形状为 (B, T, num_classes)
        """
        # 添加位置编码
        x = x + self.positional_encoding[:, :x.size(1), :]  # (B, 64, 192)

        # Transformer编码
        x = self.transformer_encoder(x)  # (B, 64, 192)

        # 映射到字符类别
        output = self.fc(x)  # (B, 64, num_classes)
        return output


class SVTRWithTransformerDecoder(nn.Module):
    def __init__(self, svtr_model, num_classes=1000):
        """
        带 Transformer 解码器的 SVTR 模型
        :param svtr_model: SVTR 特征提取模块
        :param num_classes: 字符类别数量（包括 blank 字符）
        """
        super(SVTRWithTransformerDecoder, self).__init__()
        self.svtr_model = svtr_model  # SVTR 特征提取模块
        self.decoder = SVTRTransformerDecoder(input_dim=192, num_heads=8, num_layers=4, num_classes=num_classes)

    def forward(self, x):
        # 通过 SVTR 模型提取特征
        features = self.svtr_model(x)  # (B, 64, 192)

        # 使用 Transformer 解码特征为不定长序列
        decoded_output = self.decoder(features)  # (B, T, num_classes)
        return decoded_output


# CTC损失定义
ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)

def calculate_ctc_loss(predictions, targets, input_lengths, target_lengths):
    """
    计算CTC损失
    :param predictions: 模型预测，形状为 (B, T, num_classes)
    :param targets: 序列标签，形状为 (B, target_length)
    :param input_lengths: 输入的有效长度
    :param target_lengths: 每个序列的目标长度
    :return: CTC损失
    """
    # 将预测转置为 (T, B, num_classes) 以适应 CTC 损失输入
    predictions = predictions.permute(1, 0, 2)  # (T, B, num_classes)
    loss = ctc_loss_fn(predictions, targets, input_lengths, target_lengths)
    return loss



