import torch.nn as nn
from transformers.testing_utils import to_2tuple


# 从 Patch Embeddings 组合图像
class PatchUnEmbed(nn.Module):
    r""" Image to Patch Unembedding

    输入:
        img_size (int): 图像的大小，默认为 224*224.
        patch_size (int): Patch token 的大小，默认为 4*4.
        in_chans (int): 输入图像的通道数，默认为 3.
        embed_dim (int): 线性 projection 输出的通道数，默认为 96.
        norm_layer (nn.Module, optional): 归一化层， 默认为N None.
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None,flag_RSTB=True):
        super().__init__()
        img_size = to_2tuple(img_size)  # 图像的大小，默认为 224*224
        patch_size = to_2tuple(patch_size)  # Patch token 的大小，默认为 4*4
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]  # patch 的分辨率
        self.img_size = img_size
        self.flag_RSTB = flag_RSTB
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]  # patch 的个数，num_patches

        self.in_chans = in_chans  # 输入图像的通道数
        self.embed_dim = embed_dim  # 线性 projection 输出的通道数

        # self.proj = nn.Conv2d(embed_dim, in_chans, kernel_size=patch_size, stride=patch_size)
        self.proj = nn.ConvTranspose2d(
            in_channels=embed_dim,
            out_channels=in_chans,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0
        )


    def forward(self, x, x_size):
        B, HW, C = x.shape  # 输入 x 的结构
        x = x.transpose(1, 2).view(B, self.embed_dim, x_size[0], x_size[1])  # 输出结构为 [B, Ph*Pw, C]
        if self.flag_RSTB == False:
            x = self.proj(x)
        return x