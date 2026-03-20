import torch
import torch.nn as nn

from models.model_part.SwinTransformerBlock import SwinTransformerBlock


# 单阶段的 SWin Transformer 基础层
class BasicLayer(nn.Module):
    """
    输入:
        dim (int): 输入特征的维度.
        input_resolution (tuple[int]): 输入分辨率.
        depth (int): SWin Transformer 块的个数.
        num_heads (int): 注意力头的个数.
        window_size (int): 本地(当前块中)窗口的大小.
        mlp_ratio (float): MLP隐藏层特征维度与嵌入层特征维度的比.
        qkv_bias (bool, optional): 给 query, key, value 添加一个可学习偏置，默认为 True.
        qk_scale (float | None, optional): 重写默认的缩放因子 scale.
        drop (float, optional): 随机丢弃神经元，丢弃率默认为 0.0.
        attn_drop (float, optional): 注意力图随机丢弃率，默认为 0.0.
        drop_path (float | tuple[float], optional): 深度随机丢弃率，默认为 0.0.
        norm_layer (nn.Module, optional): 归一化操作，默认为 nn.LayerNorm.
        downsample (nn.Module | None, optional): 结尾处的下采样层，默认没有.
        use_checkpoint (bool): 是否使用 checkpointing 来节省显存，默认为 False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim  # 输入特征的维度
        self.input_resolution = input_resolution  # 输入分辨率
        self.depth = depth  # SWin Transformer 块的个数
        self.use_checkpoint = use_checkpoint  # 是否使用 checkpointing 来节省显存，默认为 False

        # 创建 Swin Transformer 网络
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # patch 合并层
        if downsample is not None:  # 如果有下采样
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)  # 下采样
        else:
            self.downsample = None  # 不做下采样

    #定义前向传播
    def forward(self, x, x_size):
        for blk in self.blocks:  # x 输入串联的 Swin Transformer 块
            # if self.use_checkpoint:
            #     x = checkpoint.checkpoint(blk, x, x_size)  # 使用 checkpoint
            # else:
            x = blk(x, x_size)  # 直接输入网络
        if self.downsample is not None:
            x = self.downsample(x)  # 下采样
        return x