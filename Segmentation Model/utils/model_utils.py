import torch
import numpy as np
import random

def normalize_to_minus_one_one(data):
    min_val = data.min()
    max_val = data.max()
    return (data - min_val) / (max_val - min_val)

def set_seed(seed):
    """设置随机种子以确保可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_model(model, load_path, optimizer=None):
    """加载模型检查点"""
    try:
        checkpoint = torch.load(load_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        print(f"Model loaded from {load_path}")
        return epoch
    except Exception as e:
        print(f"Error loading model from {load_path}: {e}")
        return None


def save_model(model, save_path, epoch=None, optimizer=None):
    """保存模型检查点"""
    try:
        state_dict = {
            'model_state_dict': model.state_dict(),
            'epoch': epoch,
        }
        # for name, param in model.state_dict().items():
        #     print(name, param.size(), param.dtype)
        if optimizer:
            state_dict['optimizer_state_dict'] = optimizer.state_dict()

        torch.save(state_dict, save_path)
        print(f"Model saved to {save_path}")
    except Exception as e:
        print(f"Error saving model to {save_path}: {e}")