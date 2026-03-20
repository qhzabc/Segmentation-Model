import math
import torch.nn as nn
class Upsample(nn.Sequential):
    """
    输入:
        scale (int): 缩放因子，支持 2^n and 3.
        num_feat (int): 中间特征的通道数.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # 缩放因子等于 2^n
            for _ in range(int(math.log(scale, 2))):  #  循环 n 次
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))  # 卷积层
                m.append(nn.PixelShuffle(2))  # pixelshuffle 上采样 2 倍
        elif scale == 3:  # 缩放因子等于 3
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))  # 卷积层
            m.append(nn.PixelShuffle(3))  # pixelshuffle 上采样 3 倍
        else:
            # 报错，缩放因子不对
            raise ValueError(f'scale {scale} is not supported. ' 'Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)

class UpsampleOneStep(nn.Sequential):
    """一步上采样与前边上采样模块不同之处在于该模块只有一个卷积层和一个 pixelshuffle 层

    输入:
        scale (int): 缩放因子，支持 2^n and 3.
        num_feat (int): 中间特征的通道数.
    """

    def __init__(self, scale, num_feat, num_out_ch, input_resolution=None):
        self.num_feat = num_feat  # 中间特征的通道数
        self.input_resolution = input_resolution  # 输入分辨率
        m = []
        m.append(nn.Conv2d(num_feat, (scale ** 2) * num_out_ch, 3, 1, 1))  # 卷积层
        m.append(nn.PixelShuffle(scale))  # pixelshuffle 上采样 scale 倍
        super(UpsampleOneStep, self).__init__(*m)