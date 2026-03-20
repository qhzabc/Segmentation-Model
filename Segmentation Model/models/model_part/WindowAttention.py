import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_

class WindowAttention(nn.Module):
    r""" 基于有相对位置偏差的多头自注意力窗口，支持移位的(shifted)或者不移位的(non-shifted)窗口.

    输入:
        dim (int): 输入特征的维度.
        window_size (tuple[int]): 窗口的大小.
        num_heads (int): 注意力头的个数.
        qkv_bias (bool, optional): 给 query, key, value 添加可学习的偏置，默认为 True.
        qk_scale (float | None, optional): 重写默认的缩放因子 scale.
        attn_drop (float, optional): 注意力权重的丢弃率，默认为 0.0.
        proj_drop (float, optional): 输出的丢弃率，默认为 0.0.
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim  # 输入特征的维度
        self.window_size = window_size  # 窗口的高 Wh,宽 Ww
        self.num_heads = num_heads  # 注意力头的个数
        head_dim = dim // num_heads  # 注意力头的维度
        self.scale = qk_scale or head_dim ** -0.5  # 缩放因子 scale

        # 定义相对位置偏移的参数表，结构为 [2*Wh-1 * 2*Ww-1, num_heads]
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

        # 获取窗口内每个 token 的成对的相对位置索引
        coords_h = torch.arange(self.window_size[0])  # 高维度上的坐标 (0, 7)
        coords_w = torch.arange(self.window_size[1])  # 宽维度上的坐标 (0, 7)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 坐标，结构为 [2, Wh, Ww]
        coords_flatten = torch.flatten(coords, 1)  # 重构张量结构为 [2, Wh*Ww]
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 相对坐标，结构为 [2, Wh*Ww, Wh*Ww]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # 交换维度，结构为 [Wh*Ww, Wh*Ww, 2]
        relative_coords[:, :, 0] += self.window_size[0] - 1  # 第1个维度移位
        relative_coords[:, :, 1] += self.window_size[1] - 1  # 第1个维度移位
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1  # 第1个维度的值乘以 2倍的 Ww，再减 1
        relative_position_index = relative_coords.sum(-1)  # 相对位置索引，结构为 [Wh*Ww, Wh*Ww]
        self.register_buffer("relative_position_index", relative_position_index)  # 保存数据，不再更新

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)  # 线性层，特征维度变为原来的 3倍
        self.attn_drop = nn.Dropout(attn_drop)  # 随机丢弃神经元，丢弃率默认为 0.0
        self.proj = nn.Linear(dim, dim)  # 线性层，特征维度不变

        self.proj_drop = nn.Dropout(proj_drop)  # 随机丢弃神经元，丢弃率默认为 0.0

        trunc_normal_(self.relative_position_bias_table, std=.02)  # 截断正态分布，限制标准差为 0.02
        self.softmax = nn.Softmax(dim=-1)  # 激活函数 softmax

    # 定义前向传播
    def forward(self, x, mask=None):
        """
        输入:
            x: 输入特征图，结构为 [num_windows*B, N, C]
            mask: (0/-inf) mask, 结构为 [num_windows, Wh*Ww, Wh*Ww] 或者没有 mask
        """
        B_, N, C = x.shape  # 输入特征图的结构
        # 将特征图的通道维度按照注意力头的个数重新划分，并再做交换维度操作
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # 方便后续写代码，重新赋值

        # q 乘以缩放因子
        q = q * self.scale
        # @ 代表常规意义上的矩阵相乘
        attn = (q @ k.transpose(-2, -1))  # q 和 k 相乘后并交换最后两个维度

        # 相对位置偏移，结构为 [Wh*Ww, Wh*Ww, num_heads]
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        # 相对位置偏移交换维度，结构为 [num_heads, Wh*Ww, Wh*Ww]
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)  # 带相对位置偏移的注意力图

        if mask is not None:  # 判断是否有 mask
            nW = mask.shape[0]  # mask 的宽
            # 注意力图与 mask 相加
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)  # 恢复注意力图原来的结构
            attn = self.softmax(attn)  # 激活注意力图 [0, 1] 之间
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)  # 随机设置注意力图中的部分值为 0
        # 注意力图与 v 相乘得到新的注意力图
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)  # 通过线性层
        x = self.proj_drop(x)  # 随机设置新注意力图中的部分值为 0
        return x