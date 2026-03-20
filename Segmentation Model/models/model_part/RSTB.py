import torch
import torch.nn as nn
from models.model_part.BasicLayer import BasicLayer
from models.model_part.PetchEmbed import PatchEmbed
from models.model_part.PetchUnEmbed import PatchUnEmbed


class RSTB(nn.Module):
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
        drop (float, optional): D 随机丢弃神经元，丢弃率默认为 0.0.
        attn_drop (float, optional): 注意力图随机丢弃率，默认为 0.0.
        drop_path (float | tuple[float], optional): 深度随机丢弃率，默认为 0.0.
        norm_layer (nn.Module, optional): 归一化操作，默认为 nn.LayerNorm.
        downsample (nn.Module | None, optional): 结尾处的下采样层，默认没有.
        use_checkpoint (bool): 是否使用 checkpointing 来节省显存，默认为 False.
        img_size: 输入图片的大小.
        patch_size: Patch 的大小.
        resi_connection: 残差连接之前的卷积块.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 img_size=224, patch_size=4, resi_connection='1conv'):
        super(RSTB, self).__init__()

        self.dim = dim  # 输入特征的维度
        self.input_resolution = input_resolution  # 输入分辨率

        # SWin Transformer 基础层
        self.residual_group = BasicLayer(dim=dim,
                                         input_resolution=input_resolution,
                                         depth=depth,
                                         num_heads=num_heads,
                                         window_size=window_size,
                                         mlp_ratio=mlp_ratio,
                                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                                         drop=drop, attn_drop=attn_drop,
                                         drop_path=drop_path,
                                         norm_layer=norm_layer,
                                         downsample=downsample,
                                         use_checkpoint=use_checkpoint)

        if resi_connection == '1conv':  # 结尾用 1 个卷积层
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == '3conv':  # 结尾用 3 个卷积层
            # 为了减少参数使用和节约显存，采用瓶颈结构
            self.conv = nn.Sequential(nn.Conv2d(dim, dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                      nn.Conv2d(dim // 4, dim // 4, 1, 1, 0),
                                      nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                      nn.Conv2d(dim // 4, dim, 3, 1, 1))

        # 图像转成 Patch Embeddings
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
            norm_layer=None)

        # 从 Patch Embeddings 组合图像
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
            norm_layer=None)

    # 定义前向传播
    def forward(self, x, x_size):
        return self.patch_embed(self.conv(self.patch_unembed(self.residual_group(x, x_size), x_size))) + x  # 引入残差