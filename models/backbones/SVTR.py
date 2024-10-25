import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# 初始化权重的函数
def trunc_normal_(tensor, mean=0., std=0.02):
    # Truncated normal initialization
    with torch.no_grad():
        return tensor.normal_(mean, std).fmod_(2 * std).clamp_(-2 * std, 2 * std)

def normal_(tensor, mean=0., std=1.):
    # 正态分布初始化
    with torch.no_grad():
        return tensor.normal_(mean, std)

def zeros_(tensor):
    # 初始化为全零
    with torch.no_grad():
        return tensor.zero_()

def ones_(tensor):
    # 初始化为全一
    with torch.no_grad():
        return tensor.fill_(1.)

# drop_path 实现
def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample, per residual block.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    # 生成一个与输入形状相同的二值化的随机张量
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # shape: (batch_size, 1, 1, ..., 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # 将随机张量二值化
    # 保持概率分配给 x
    output = x / keep_prob * random_tensor
    return output

# DropPath实现
class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in the main path of residual blocks)."""
    
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        # 在前向传播中调用 drop_path 函数
        return drop_path(x, self.drop_prob, self.training)

# Identity实现
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, input):
        # 直接返回输入，不做任何操作
        return input

# Mlp实现
class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        # 如果 out_features 未指定，则与 in_features 相同
        out_features = out_features or in_features
        # 如果 hidden_features 未指定，则与 in_features 相同
        hidden_features = hidden_features or in_features
        # 第一个全连接层
        self.fc1 = nn.Linear(in_features, hidden_features)
        # 激活函数
        self.act = act_layer()
        # 第二个全连接层
        self.fc2 = nn.Linear(hidden_features, out_features)
        # Dropout层
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        # 通过第一个全连接层、激活函数和Dropout
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        # 通过第二个全连接层和Dropout
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
# ConvBNLayer实现
class ConvBNLayer(nn.Module):
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        kernel_size=3, 
        stride=1, 
        padding=0, 
        bias=False, 
        groups=1, 
        act=nn.GELU
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size, stride, padding, groups=groups, bias=bias)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = act()

    def forward(self, inputs):
        out = self.conv(inputs)
        out = self.norm(out)
        out = self.act(out)
        return out

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5

        # 定义用于生成 q, k, v 的线性层
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        # Attention的Dropout
        self.attn_drop = nn.Dropout(attn_drop)
        # 投影层
        self.proj = nn.Linear(dim, dim)
        # 投影层的Dropout
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape  # Batch, Token数, 特征维度
        # 生成qkv并reshape为 [B, N, 3, num_heads, head_dim]，再通过permute调整维度顺序
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # 拆分为 q, k, v

        # 计算 q 和 k 的点积并进行缩放
        attn = (q @ k.transpose(-2, -1)) * self.scale
        # 通过 softmax 计算注意力权重
        attn = F.softmax(attn, dim=-1)
        # Dropout 应用于注意力权重
        attn = self.attn_drop(attn)

        # 使用注意力权重对 v 进行加权求和
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        # 通过投影层并应用 Dropout
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

# Block实现
class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        epsilon=1e-6,
    ):
        super().__init__()
        # 第一个LayerNorm
        self.norm1 = norm_layer(dim, eps=epsilon)
        # 注意力层 (mixer)
        self.mixer = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        # 随机深度 (DropPath)，如果 drop_path 为0则使用 Identity
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else Identity()
        # 第二个LayerNorm
        self.norm2 = norm_layer(dim, eps=epsilon)
        # MLP层
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp_ratio = mlp_ratio
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x):
        # 输入经过注意力模块 (mixer) 和 LayerNorm，然后通过DropPath（如果有）
        x = self.norm1(x + self.drop_path(self.mixer(x)))
        # 输入经过MLP模块和第二个LayerNorm，并再次通过DropPath（如果有）
        x = self.norm2(x + self.drop_path(self.mlp(x)))
        return x

class ConvBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        epsilon=1e-6,
    ):
        super().__init__()
        # 计算MLP隐藏层维度
        mlp_hidden_dim = int(dim * mlp_ratio)
        # 第一个LayerNorm层
        self.norm1 = norm_layer(dim, eps=epsilon)
        # 卷积层，使用groups=num_heads进行分组卷积
        self.mixer = nn.Conv2d(
            dim, dim, kernel_size=5, stride=1, padding=2, groups=num_heads
        )
        # DropPath层，若drop_path为0，则使用Identity
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else Identity()
        # 第二个LayerNorm层
        self.norm2 = norm_layer(dim, eps=epsilon)
        # MLP层
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x):
        B, C, H, W = x.shape  # 获取输入张量的Batch、Channels、Height和Width
        # 通过卷积层进行计算并加上残差连接（跳跃连接）
        x = x + self.drop_path(self.mixer(x))
        # 将张量展平成 (B, H*W, C) 并进行转置：[B, C, H, W] -> [B, H*W, C]
        x = self.norm1(x.flatten(2).transpose(1, 2))
        # 通过MLP层并加上残差连接
        x = self.norm2(x + self.drop_path(self.mlp(x)))
        # 将张量重新转置并恢复原始形状：[B, H*W, C] -> [B, C, H, W]
        x = x.transpose(1, 2).view(B, C, H, W)
        return x

# FlattenTranspose实现
class FlattenTranspose(nn.Module):
    def forward(self, x):
        # 将输入展平成二维，并进行转置：[B, C, H, W] -> [B, H*W, C]
        return x.flatten(2).transpose(1, 2)

# SubSample2D实现
class SubSample2D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=[2, 1],
    ):
        super().__init__()
        # 3x3卷积层，带有步长和填充
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1
        )
        # LayerNorm层
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x, sz):
        # 卷积操作
        x = self.conv(x)
        B, C, H, W = x.shape  # 获取新形状的Batch, Channels, Height, Width
        # 将张量展平成二维并进行转置：[B, C, H*W] -> [B, H*W, C]
        x = self.norm(x.flatten(2).transpose(1, 2))
        # 再次转置并恢复到原始形状：[B, H*W, C] -> [B, C, H, W]
        x = x.transpose(1, 2).view(B, C, H, W)
        return x, [H, W]

# SubSample1D实现
class SubSample1D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=[2, 1],
    ):
        super().__init__()
        # 2D卷积层，步长和填充
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1
        )
        # LayerNorm归一化层
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x, sz):
        # 获取最后一维度的大小 C
        C = x.shape[-1]
        # 转置输入张量并将其 reshape 成 [B, C, H, W] 形式
        x = x.permute(0, 2, 1).view(x.shape[0], C, sz[0], sz[1])
        # 通过卷积处理
        x = self.conv(x)
        # 获取卷积后的张量形状
        B, C, H, W = x.shape
        # 将张量展平并转置 [B, C, H, W] -> [B, H*W, C]
        x = self.norm(x.flatten(2).transpose(1, 2))
        # 返回处理后的张量和新的形状 [H, W]
        return x, [H, W]

# IdentitySize实现
class IdentitySize(nn.Module):
    def forward(self, x, sz):
        # 直接返回输入和尺寸
        return x, sz


# SVTRStage实现
class SVTRStage(nn.Module):
    def __init__(
        self,
        dim=64,
        out_dim=256,
        depth=3,
        mixer=["Local"] * 3,
        sub_k=[2, 1],
        num_heads=2,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path=[0.1] * 3,
        norm_layer=nn.LayerNorm,
        act=nn.GELU,
        eps=1e-6,
        downsample=None,
        **kwargs,
    ):
        super(SVTRStage, self).__init__()
        self.dim = dim  # 输入通道数

        # 计算使用卷积块的数量
        conv_block_num = sum([1 if mix == "Conv" else 0 for mix in mixer])

        # 创建模块列表
        blocks = []
        for i in range(depth):
            if mixer[i] == "Conv":
                blocks.append(
                    ConvBlock(
                        dim=dim,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        drop=drop_rate,
                        act_layer=act,
                        drop_path=drop_path[i],
                        norm_layer=norm_layer,
                        epsilon=eps,
                    )
                )
            else:
                blocks.append(
                    Block(
                        dim=dim,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        drop=drop_rate,
                        act_layer=act,
                        attn_drop=attn_drop_rate,
                        drop_path=drop_path[i],
                        norm_layer=norm_layer,
                        epsilon=eps,
                    )
                )
            # 如果达到卷积块数量并且 mixer 最后一个不是 "Conv"，添加 FlattenTranspose
            if i == conv_block_num - 1 and mixer[-1] != "Conv":
                blocks.append(FlattenTranspose())

        # 将多个 block 串联
        self.blocks = nn.Sequential(*blocks)

        # 判断是否需要下采样
        if downsample:
            if mixer[-1] == "Conv":
                self.downsample = SubSample2D(dim, out_dim, stride=sub_k)
            elif mixer[-1] == "Global":
                self.downsample = SubSample1D(dim, out_dim, stride=sub_k)
        else:
            self.downsample = IdentitySize()

    def forward(self, x, sz):
        # 通过多个 block 进行前向传播
        x = self.blocks(x)
        # 进行下采样或保持原始尺寸
        x, sz = self.downsample(x, sz)
        return x, sz
    
# ADDPosEmbed实现
class ADDPosEmbed(nn.Module):
    def __init__(self, feat_max_size=[8, 32], embed_dim=768):
        super().__init__()
        # 创建位置嵌入张量，并进行截断正态分布初始化
        pos_embed = torch.zeros(1, feat_max_size[0] * feat_max_size[1], embed_dim, dtype=torch.float32)
        trunc_normal_(pos_embed)
        # 调整张量形状 [1, embed_dim, feat_max_size[0], feat_max_size[1]]
        pos_embed = pos_embed.permute(0, 2, 1).view(1, embed_dim, feat_max_size[0], feat_max_size[1])
        # 创建可学习的参数
        self.pos_embed = nn.Parameter(pos_embed)

    def forward(self, x):
        # 获取输入的空间尺寸
        sz = x.shape[2:]
        # 对输入添加位置嵌入
        x = x + self.pos_embed[:, :, :sz[0], :sz[1]]
        return x


# POPatchEmbed实现
class POPatchEmbed(nn.Module):
    """Image to Patch Embedding"""
    def __init__(
        self,
        in_channels=3,
        feat_max_size=[8, 32],
        embed_dim=768,
        use_pos_embed=False,
        flatten=False,
    ):
        super().__init__()
        # 构建Patch Embedding的卷积层
        patch_embed = [
            ConvBNLayer(
                in_channels=in_channels,
                out_channels=embed_dim // 2,
                kernel_size=3,
                stride=2,
                padding=1,
                act=nn.GELU,
            ),
            ConvBNLayer(
                in_channels=embed_dim // 2,
                out_channels=embed_dim,
                kernel_size=3,
                stride=2,
                padding=1,
                act=nn.GELU,
            ),
        ]
        # 是否使用位置嵌入
        if use_pos_embed:
            patch_embed.append(ADDPosEmbed(feat_max_size, embed_dim))
        # 是否进行展平和转置
        if flatten:
            patch_embed.append(FlattenTranspose())
        # 将多个模块组合成一个序列
        self.patch_embed = nn.Sequential(*patch_embed)

    def forward(self, x):
        # 获取输入的空间尺寸
        sz = x.shape[2:]
        # 通过Patch Embedding序列
        x = self.patch_embed(x)
        # 返回结果以及尺寸的缩放值
        return x, [sz[0] // 4, sz[1] // 4]


class LastStage(nn.Module):
    def __init__(self, in_channels, out_channels, last_drop):
        super().__init__()
        # 线性层
        self.last_conv = nn.Linear(in_channels, out_channels, bias=False)
        # Hardswish激活函数
        self.hardswish = nn.Hardswish()
        # Dropout层，mode="downscale_in_infer" 对应的是 PyTorch 默认行为
        self.dropout = nn.Dropout(p=last_drop)

    def forward(self, x, sz):
        # Reshape 输入的形状 [B, sz[0], sz[1], C]
        x = x.view(x.shape[0], sz[0], sz[1], x.shape[-1])

        # 展开 [B, sz[0], sz[1], C] 到 [B, sz[0] * sz[1], C]
        x = x.view(x.shape[0], -1, x.shape[-1])

        # 线性层
        x = self.last_conv(x)
        # 激活函数
        x = self.hardswish(x)
        # Dropout操作
        x = self.dropout(x)

        # 返回结果以及更新的尺寸信息
        return x, [sz[0] * sz[1], x.shape[-1]]  # 修改后的输出尺寸信息


# OutPool实现
class OutPool(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, sz):
        C = x.shape[-1]
        # 转置并 reshape 输入张量
        x = x.permute(0, 2, 1).view(x.shape[0], C, sz[0], sz[1])
        # 使用 2D 平均池化
        x = nn.functional.avg_pool2d(x, kernel_size=(sz[0], 2))
        return x, [1, sz[1] // 2]


# Feat2D实现
class Feat2D(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, sz):
        C = x.shape[-1]
        # 转置并 reshape 输入张量
        x = x.permute(0, 2, 1).view(x.shape[0], C, sz[0], sz[1])
        return x, sz

class FinalHeadBlock(nn.Module):
    def __init__(self, num_nodes):
        super(FinalHeadBlock, self).__init__()
        # 线性分类器，输出节点数为N（英文为37，中文为6625）
        self.linear = nn.Linear(192, num_nodes)  # 192是输入的特征维度
        # 使用LogSoftmax进行归一化处理（可选，用于分类任务）
        self.softmax = nn.LogSoftmax(dim=-1)  # 在节点维度上应用LogSoftmax

    def forward(self, x):
        # 假设x的形状为 [batch_size, sequence_length, feature_size]，即 [8, 64, 192]
        batch_size, seq_length, feature_size = x.size()

        # 对每个特征向量应用线性变换
        x = self.linear(x)  # 输出形状变为 [batch_size, sequence_length, num_nodes]

        # 可选：对输出结果应用softmax，得到概率或对数概率
        x = self.softmax(x)

        # 返回最终的输出，形状为 [batch_size, sequence_length, num_nodes]
        return x



# SVTRv2实现
class SVTRv2(nn.Module):
    def __init__(
        self,
        max_sz=[32, 128],
        in_channels=3,
        out_channels=192,
        out_char_num=8000,
        depths=[3, 6, 3],
        dims=[64, 128, 256],
        mixer=[["Conv"] * 3, ["Conv"] * 3 + ["Global"] * 3, ["Global"] * 3],
        use_pos_embed=False,
        sub_k=[[1, 1], [2, 1], [1, 1]],
        num_heads=[2, 4, 8],
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        last_drop=0.1,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        act=nn.GELU,
        last_stage=False,
        eps=1e-6,
        use_pool=False,
        feat2d=False,
        **kwargs,
    ):
        super().__init__()
        num_stages = len(depths)
        self.num_features = dims[-1]

        # 计算特征图的最大尺寸
        feat_max_size = [max_sz[0] // 4, max_sz[1] // 4]
        # Position-Overlapping Patch Embedding (假设POPatchEmbed已定义)
        self.pope = POPatchEmbed(
            in_channels=in_channels,
            feat_max_size=feat_max_size,
            embed_dim=dims[0],
            use_pos_embed=use_pos_embed,
            flatten=mixer[0][0] != "Conv",
        )

        # 随机深度的衰减规则
        dpr = np.linspace(0, drop_path_rate, sum(depths))

        # 创建多层 stage 块
        self.stages = nn.ModuleList()
        for i_stage in range(num_stages):
            stage = SVTRStage(
                dim=dims[i_stage],
                out_dim=dims[i_stage + 1] if i_stage < num_stages - 1 else 0,
                depth=depths[i_stage],
                mixer=mixer[i_stage],
                sub_k=sub_k[i_stage],
                num_heads=num_heads[i_stage],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_stage]) : sum(depths[: i_stage + 1])],
                norm_layer=norm_layer,
                act=act,
                eps=eps,
                downsample=False if i_stage == num_stages - 1 else True,
            )
            self.stages.append(stage)

        self.out_channels = self.num_features
        self.last_stage = last_stage
        if last_stage:
            self.out_channels = out_channels
            self.stages.append(
                LastStage(self.num_features, out_channels, last_drop)
            )
        if use_pool:
            self.stages.append(OutPool())

        if feat2d:
            self.stages.append(Feat2D())

        # 添加FinalHeadBlock模块
        self.final_head = FinalHeadBlock(num_nodes=out_char_num)

        self.apply(self._init_weights)

    # 权重初始化
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def forward(self, x):
        # 通过POPatchEmbed处理输入
        x, sz = self.pope(x)
        # 逐层处理
        for stage in self.stages:
            x, sz = stage(x, sz)
            # 使用FinalHeadBlock来输出最终分类结果
        # x = self.final_head(x)
        return x

def main():
    # 定义输入图像的大小（batch_size, channels, height, width）
    input_size = (8, 3, 64, 64)
    
    # 创建一个随机的输入图像 (假设是RGB图像)
    input_image = torch.randn(input_size)
    
    # 创建SVTRv2模型实例
    model = SVTRv2(
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

    # 打印模型架构，确认模型结构
    print(model)

    # 将输入图像传递给模型进行推理
    output = model(input_image)

    # 打印输入和输出的形状
    print("输入图像的大小:", input_image.shape)
    print("模型输出的大小:", output.shape)

# 测试 ConvBlock 模块
def test_convblock():
    # 输入张量大小为 (16, 64, 16, 16)
    input_tensor = torch.randn(16, 64, 16, 16)

    # 创建 ConvBlock 实例，参数设置与输入大小匹配
    conv_block = ConvBlock(
        dim=64,            # 输入通道数
        num_heads=8,       # 多头数（用于分组卷积）
        mlp_ratio=4.0,     # MLP 扩展比例
        drop=0.0,          # Dropout 概率
        drop_path=0.0,     # DropPath 概率
        act_layer=nn.GELU, # 激活函数
        norm_layer=nn.LayerNorm, # 归一化层
    )

    # 将输入张量传递给 ConvBlock 模块，进行前向传播
    output = conv_block(input_tensor)

    # 打印输入和输出的形状
    print(f"输入张量的形状: {input_tensor.shape}")
    print(f"输出张量的形状: {output.shape}")

# 测试 SubSample2D 模块
def test_subsample2d():
    # 输入张量大小为 (16, 64, 16, 16)
    input_tensor = torch.randn(16, 64, 16, 16)
    sz = [16, 16]  # 输入的高度和宽度

    # 创建 SubSample2D 实例，参数设置为 in_channels=64, out_channels=128, stride=[1, 1]
    subsample = SubSample2D(
        in_channels=64,
        out_channels=128,
        stride=[1, 1],  # 步长为 [1, 1]
    )

    # 将输入张量传递给 SubSample2D 模块，进行前向传播
    output, output_sz = subsample(input_tensor, sz)

    # 打印输入和输出的形状以及输出尺寸
    print(f"输入张量的形状: {input_tensor.shape}")
    print(f"输出张量的形状: {output.shape}")
    print(f"输出尺寸: {output_sz}")

# 测试 Block 模块
def test_block():
    # 输入张量大小为 (16, 128, 16, 16)，需要先将其展平成三维 (B, N, C)
    input_tensor = torch.randn(16, 16 * 16, 128)  # 展平的 (B, N, C) 形式

    # 创建 Block 实例
    block = Block(
        dim=128,            # 输入特征维度
        num_heads=8,        # 多头数
        mlp_ratio=4.0,      # MLP 扩展比例
        qkv_bias=True,      # QKV 偏置
        qk_scale=None,      # QK 缩放系数
        drop=0.0,           # Dropout 概率
        attn_drop=0.0,      # Attention Dropout 概率
        drop_path=0.0,      # DropPath 概率
        act_layer=nn.GELU,  # 激活函数
        norm_layer=nn.LayerNorm  # 归一化层
    )

    # 将输入张量传递给 Block 模块，进行前向传播
    output = block(input_tensor)

    # 打印输入和输出的形状
    print(f"输入张量的形状: {input_tensor.shape}")
    print(f"输出张量的形状: {output.shape}")

# 测试 SubSample1D 模块
def test_subsample1d():
    # 输入张量大小为 (16, 64, 256)，sz 为 [16, 16]
    input_tensor = torch.randn(16, 256, 64)
    sz = [16, 16]  # 输入的高度和宽度

    # 创建 SubSample1D 实例
    subsample = SubSample1D(
        in_channels=64,  # 输入通道数
        out_channels=128,  # 输出通道数
        stride=[2, 2]  # 步长为 [2, 1]
    )

    # 将输入张量传递给 SubSample1D 模块，进行前向传播
    output, output_sz = subsample(input_tensor, sz)

    # 打印输入和输出的形状以及输出尺寸
    print(f"输入张量的形状: {input_tensor.shape}")
    print(f"输出张量的形状: {output.shape}")
    print(f"输出尺寸: {output_sz}")


if __name__ == "__main__":
    main()
