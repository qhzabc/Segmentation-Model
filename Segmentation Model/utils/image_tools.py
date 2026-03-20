
def window_partition(x, window_size):
    """
    输入:
        x: (B, H, W, C)
        window_size (int): window size  # 窗口的大小
    返回:
        windows: (num_windows*B, window_size, window_size, C)  # 每一个 batch 有单独的 windows
    """
    B, H, W, C = x.shape  # 输入的 batch 个数，高，宽，通道数
    # 将输入 x 重构为结构 [batch 个数，高方向的窗口个数，窗口大小，宽方向的窗口个数，窗口大小，通道数] 的张量
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    # 交换重构后 x 的第 3和4 维度， 5和6 维度，再次重构为结构 [高和宽方向的窗口个数乘以 batch 个数，窗口大小，窗口大小，通道数] 的张量
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows
    # 这里比较有意思，不太理解的可以给个初始值，比如 x = torch.randn([1, 14, 28, 3])

def window_reverse(windows, window_size, H, W):
    """
    输入:
        windows: (num_windows*B, window_size, window_size, C)  # 分割得到的窗口(已处理)
        window_size (int): Window size  # 窗口大小
        H (int): Height of image  # 原分割窗口前特征图的高
        W (int): Width of image  # 原分割窗口前特征图的宽
    返回:
        x: (B, H, W, C)  # 返回与分割前特征图结构一样的结果
    """
    # 以下就是分割窗口的逆向操作，不多解释
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x