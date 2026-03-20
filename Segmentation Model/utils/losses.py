import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.metrics import structural_similarity
import numpy as np


def structural_similarity_loss(x, y, data_range=1.0):
    """计算结构相似性损失 (SSIM)"""
    # 将张量转换为numpy并计算SSIM
    x_np = x.permute(0, 2, 3, 1).detach().cpu().numpy()
    y_np = y.permute(0, 2, 3, 1).detach().cpu().numpy()

    ssim_loss = 0.0
    for i in range(x_np.shape[0]):
        ssim_val = structural_similarity(
            x_np[i], y_np[i],
            win_size=11, data_range=data_range,
            multichannel=True, channel_axis=2
        )
        ssim_loss += (1.0 - ssim_val) / 2.0  # 将SSIM转换为损失值

    return torch.tensor(ssim_loss / x_np.shape[0], device=x.device)


def gradient_magnitude(image):
    """计算图像的梯度幅值"""
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                           dtype=image.dtype, device=image.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                           dtype=image.dtype, device=image.device).view(1, 1, 3, 3)

    # 分别计算x和y方向的梯度
    grad_x = F.conv2d(image, sobel_x.repeat(image.shape[1], 1, 1, 1), padding=1, groups=image.shape[1])
    grad_y = F.conv2d(image, sobel_y.repeat(image.shape[1], 1, 1, 1), padding=1, groups=image.shape[1])

    # 计算梯度幅值
    return torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)


def structure_preserving_loss(original, generated):
    """结构保留损失"""
    # 结构相似性
    ssim_loss = structural_similarity_loss(original, generated)

    # 梯度差异
    orig_grad = gradient_magnitude(original)
    gen_grad = gradient_magnitude(generated)
    grad_loss = F.l1_loss(orig_grad, gen_grad)

    return 0.7 * ssim_loss + 0.3 * grad_loss


def adversarial_loss(logits, target_labels):
    """对抗损失 - 使生成器欺骗分类器"""
    target_probs = F.softmax(logits, dim=1)
    # 选择目标标签对应的概率
    target_probs = target_probs[torch.arange(len(target_labels)), target_labels]
    return -torch.log(target_probs + 1e-8).mean()