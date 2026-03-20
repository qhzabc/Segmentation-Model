import logging
import os
import sys
from datetime import datetime


def setup_logger(name, log_dir, log_level=logging.INFO):
    """设置并返回一个日志记录器"""
    os.makedirs(log_dir, exist_ok=True)

    # 创建日志记录器
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # 创建文件处理器
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)

    # 创建控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)

    # 创建格式化器
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 添加处理器到日志记录器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger