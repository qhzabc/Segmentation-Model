import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


def calculate_metrics(true_labels, pred_labels):
    """计算分类指标"""
    accuracy = accuracy_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels, average='weighted')
    cm = confusion_matrix(true_labels, pred_labels)

    # 计算每个类别的精确度、召回率和F1分数
    class_metrics = {}
    for i in range(cm.shape[0]):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_class = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        class_metrics[f'class_{i}'] = {
            'precision': precision,
            'recall': recall,
            'f1': f1_class
        }

    return {
        'accuracy': accuracy,
        'f1': f1,
        'confusion_matrix': cm,
        'class_metrics': class_metrics
    }