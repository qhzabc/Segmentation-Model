import torch
import torch.nn as nn
from nni.nas.hub.pytorch.utils.nn import DropPath
from transformers.testing_utils import to_2tuple

from models.model_part.Mlp import Mlp
from models.model_part.WindowAttention import WindowAttention
from utils.image_tools import window_reverse, window_partition


class SwinTransformerBlock(nn.Module):
    """
    输入:
        dim (int): 输入特征的维度.
        input_resolution (tuple[int]): 输入特征图的分辨率.
        num_heads (int): 注意力头的个数.
        window_size (int): 窗口的大小.
        shift_size (int): SW-MSA 的移位值.
        mlp_ratio (float): 多层感知机隐藏层的维度和嵌入层的比.
        qkv_bias (bool, optional): 给 query, key, value 添加一个可学习偏置，默认为 True.
        qk_scale (float | None, optional): 重写默认的缩放因子 scale.
        drop (float, optional): 随机神经元丢弃率，默认为 0.0.
        attn_drop (float, optional): 注意力图随机丢弃率，默认为 0.0.
        drop_path (float, optional): 深度随机丢弃率，默认为 0.0.
        act_layer (nn.Module, optional): 激活函数，默认为 nn.GELU.
        norm_layer (nn.Module, optional): 归一化操作，默认为 nn.LayerNorm.
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim  # 输入特征的维度
        self.input_resolution = input_resolution  # 输入特征图的分辨率
        self.num_heads = num_heads  # 注意力头的个数
        self.window_size = window_size  # 窗口的大小
        self.shift_size = shift_size  # SW-MSA 的移位大小
        self.mlp_ratio = mlp_ratio  # 多层感知机隐藏层的维度和嵌入层的比
        if min(self.input_resolution) <= self.window_size:  # 如果输入分辨率小于等于窗口大小
            self.shift_size = 0  # 移位大小为 0
            self.window_size = min(self.input_resolution)  # 窗口大小等于输入分辨率大小
        # 断言移位值必须小于等于窗口的大小
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)  # 归一化层
        # 窗口注意力
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        # 如果丢弃率大于 0 则进行随机丢弃，否则进行占位(不做任何操作)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)  # 归一化层
        mlp_hidden_dim = int(dim * mlp_ratio)  # 多层感知机隐藏层维度
        # 多层感知机
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:  # 如果移位值大于 0
            attn_mask = self.calculate_mask(self.input_resolution)  # 计算注意力 mask
        else:
            attn_mask = None  # 注意力 mask 赋空

        self.register_buffer("attn_mask", attn_mask)  # 保存注意力 mask，不参与更新

    # 计算注意力 mask
    def calculate_mask(self, x_size):
        H, W = x_size  # 特征图的高宽
        img_mask = torch.zeros((1, H, W, 1))  # 新建张量，结构为 [1, H, W, 1]
        # 以下两 slices 中的数据是索引，具体缘由尚未搞懂
        h_slices = (slice(0, -self.window_size),  # 索引 0 到索引倒数第 window_size
                    slice(-self.window_size, -self.shift_size),  # 索引倒数第 window_size 到索引倒数第 shift_size
                    slice(-self.shift_size, None))  # 索引倒数第 shift_size 后所有索引
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt  # 将 img_mask 中 h, w 对应索引范围的值置为 cnt
                cnt += 1  # 加 1

        mask_windows = window_partition(img_mask, self.window_size)  # 窗口分割，返回值结构为 [nW, window_size, window_size, 1]
        mask_windows = mask_windows.view(-1,
                                         self.window_size * self.window_size)  # 重构结构为二维张量，列数为 [window_size*window_size]
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # 增加第 2 维度减去增加第 3 维度的注意力 mask
        # 用浮点数 -100. 填充注意力 mask 中值不为 0 的元素，再用浮点数 0. 填充注意力 mask 中值为 0 的元素
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    # 定义前向传播
    def forward(self, x, x_size):
        H, W = x_size  # 输入特征图的分辨率
        B, L, C = x.shape  # 输入特征的 batch 个数，长度和维度
        # assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)  # 归一化
        x = x.view(B, H, W, C)  # 重构 x 为结构 [B, H, W, C]

        # 循环移位
        if self.shift_size > 0:  # 如果移位值大于 0
            # 第 0 维度上移 shift_size 位，第 1 维度左移 shift_size 位
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x  # 不移位

        # 对移位操作得到的特征图分割窗口, nW 是窗口的个数
        x_windows = window_partition(shifted_x, self.window_size)  # 结构为 [nW*B, window_size, window_size, C]
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # 结构为 [nW*B, window_size*window_size, C]

        # W-MSA/SW-MSA, 用在分辨率是窗口大小的整数倍的图像上进行测试
        if self.input_resolution == x_size:  # 输入分辨率与设定一致，不需要重新计算注意力 mask
            attn_windows = self.attn(x_windows, mask=self.attn_mask)  # 注意力窗口，结构为 [nW*B, window_size*window_size, C]
        else:  # 输入分辨率与设定不一致，需要重新计算注意力 mask
            attn_windows = self.attn(x_windows, mask=self.calculate_mask(x_size).to(x.device))

        # 合并窗口
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size,
                                         C)  # 结构为 [-1, window_size, window_size, C]
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # 结构为 [B, H', W', C]

        # 逆向循环移位
        if self.shift_size > 0:
            # 第 0 维度下移 shift_size 位，第 1 维度右移 shift_size 位
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x  # 不逆向移位
        x = x.view(B, H * W, C)  # 结构为 [B, H*W， C]

        # FFN
        x = shortcut + self.drop_path(x)  # 对 x 做 dropout，引入残差
        x = x + self.drop_path(self.mlp(self.norm2(x)))  # 归一化后通过 MLP，再做 dropout，引入残差

        return x