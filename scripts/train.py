from models.backbones.SVTR import SVTRv2
from models.heads.CTC import SVTRWithTransformerDecoder,calculate_ctc_loss
from torch.utils.data import DataLoader
import torch.optim as optim
import torch
import torch.nn as nn
from tqdm import tqdm
from datasets.data_loader_ctc import TextRecognitionDataset,collate_fn,get_transform
from torchsummary import summary  # 需要安装torchsummary库
# 示例使用
img_dir = '/Volumes/北海王/GitHub/ZhongHuaSongFont/text_recognition_project/data/raw/train'
label_file = '/Volumes/北海王/GitHub/ZhongHuaSongFont/text_recognition_project/assets/ids-main/char_to_sequence_id.json'

# 创建数据集实例
dataset = TextRecognitionDataset(img_dir=img_dir, label_file=label_file, transform=get_transform())

data_loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)


# 贪心解码函数
def greedy_decoder(predictions):
    """
    贪心解码函数，将CTC模型的输出解码为字符索引序列
    :param predictions: 模型预测的输出 (T, B, num_classes)
    :return: 每个样本解码后的字符序列
    """
    # 对每个时间步找到最大概率的字符索引
    decoded = torch.argmax(predictions, dim=2)  # (T, B)
    decoded_sequences = []

    for sequence in decoded.permute(1, 0):  # 每个样本的预测结果
        # 移除连续的重复字符，并去除 blank (假设 blank=0)
        decoded_sequence = []
        previous_char = None
        for char in sequence:
            if char != previous_char and char != 0:  # 0 作为 blank
                decoded_sequence.append(char.item())
            previous_char = char
        decoded_sequences.append(decoded_sequence)

    return decoded_sequences


# 计算正确率
def calculate_accuracy(predictions, targets, target_lengths):
    """
    计算批次中的平均正确率
    :param predictions: 模型预测的输出 (T, B, num_classes)
    :param targets: 真实标签的列表，每个元素为标签张量
    :param target_lengths: 每个序列的真实长度
    :return: 批次的正确率
    """
    # 解码模型预测
    decoded_predictions = greedy_decoder(predictions)

    total_correct = 0
    total_count = 0

    # 遍历每个样本，计算与真实标签的匹配度
    for pred, target, length in zip(decoded_predictions, targets, target_lengths):
        target = target[:length].tolist()  # 获取实际的标签字符序列
        if pred == target:
            total_correct += 1
        total_count += 1

    # 计算正确率
    accuracy = total_correct / total_count
    return accuracy


# 示例使用
svtr_model = SVTRv2(
        max_sz=(64, 128),  # 输入图像的最大尺寸（高度，宽度）
        in_channels=3,  # 输入的通道数，RGB图像为3
        out_channels=192,  # 模型的输出通道数
        out_char_num=12000,  # 最终的字符输出数量
        depths=(3, 6, 3),  # 每个阶段的层数
        dims=(64, 128, 256),  # 每个阶段的通道数
        mixer=(["Conv"] * 3, ["Conv"] * 3 + ["Global"] * 3, ["Global"] * 3),  # 各阶段的混合模式
        use_pos_embed=False,  # 不使用位置嵌入
        sub_k=((1, 1), (2, 2), (1, 1)),  # 下采样步长设置
        num_heads=(2, 4, 8),  # 多头自注意力的头数
        mlp_ratio=4,  # MLP扩展比例
        qkv_bias=True,  # QKV偏置
        qk_scale=None,  # QK缩放系数
        drop_rate=0.0,  # 全局Dropout概率
        last_drop=0.1,  # 最后Dropout概率
        attn_drop_rate=0.0,  # 注意力层Dropout概率
        drop_path_rate=0.1,  # DropPath随机深度概率
        norm_layer=nn.LayerNorm,  # 归一化层
        act=nn.GELU,  # 激活函数
        last_stage=True,  # 启用最后的全连接层阶段
        eps=1e-6,  # LayerNorm的epsilon值
        use_pool=False,  # 不使用池化层
        feat2d=False  # 输出特征不返回为2D
    )

# CTC损失定义
ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)

# 超参数设置
num_epochs = 10  # 训练轮数
learning_rate = 0.001  # 学习率
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设置设备

# 将模型和损失函数加载到设备
model = SVTRWithTransformerDecoder(svtr_model, num_classes=12000).to(device)

model.train()  # 将模型设置为训练模式

# 计算模型参数总数
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"模型参数总数: {total_params}")

# 优化器设置
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 数据加载器
data_loader = DataLoader(dataset, batch_size=128, collate_fn=collate_fn)

# 训练循环
for epoch in range(num_epochs):
    running_loss = 0.0
    running_accuracy = 0.0
    progress_bar = tqdm(data_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")

    for images, labels, label_lengths in progress_bar:
        # 将数据加载到设备
        images = images.to(device)
        label_lengths = label_lengths.to(device)

        # 前向传播
        predictions = model(images)  # 预测输出形状 (B, T, num_classes)

        # 输入序列的长度 (即SVTR输出的固定长度序列长度)
        input_lengths = torch.full(size=(images.size(0),), fill_value=predictions.size(1), dtype=torch.long).to(device)

        # 将标签拼接成一维张量
        targets = torch.cat(labels).to(device)

        # 计算CTC损失
        loss = calculate_ctc_loss(predictions, targets, input_lengths, label_lengths)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 记录损失
        running_loss += loss.item()

        # 计算批次正确率
        batch_accuracy = calculate_accuracy(predictions.permute(1, 0, 2), labels, label_lengths)
        running_accuracy += batch_accuracy

        progress_bar.set_postfix(loss=loss.item(), accuracy=batch_accuracy)

    # 打印每轮的平均损失和平均正确率
    avg_loss = running_loss / len(data_loader)
    avg_accuracy = running_accuracy / len(data_loader)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}")

    # 保存模型权重
    torch.save(model.state_dict(), f"svtr_model_epoch_{epoch + 1}.pth")
    print(f"Model weights saved for epoch {epoch + 1}.")

print("训练完成！")