import os


def resolve_path(path, config):
    """解析路径中的变量"""
    variables = {
        'project_root': config.get('project_root', '.'),
        'data_root': config.get('data_root', 'breast_cancer_data'),
        'checkpoints_dir': config.get('checkpoints_dir', 'checkpoints'),
        'logs_dir': config.get('logs_dir', 'logs'),
        'generated_dir': config.get('generated_dir', 'generated_images')
    }

    # 替换路径中的变量
    for var_name, var_value in variables.items():
        path = path.replace(f"${{{var_name}}}", var_value)
        path = path.replace(f"${var_name}", var_value)

    # 返回绝对路径
    return os.path.abspath(path)


def create_directories(config):
    """创建必要的目录"""
    dirs_to_create = [
        config.get('checkpoints_dir', 'checkpoints'),
        config.get('logs_dir', 'logs'),
        config.get('generated_dir', 'generated_images')
    ]

    for dir_path in dirs_to_create:
        os.makedirs(dir_path, exist_ok=True)

    # 创建模型特定目录
    for model_type in ['tokenizer', 'classifier', 'conditional']:
        os.makedirs(os.path.join(config['checkpoints_dir'], model_type), exist_ok=True)
        os.makedirs(os.path.join(config['logs_dir'], model_type), exist_ok=True)