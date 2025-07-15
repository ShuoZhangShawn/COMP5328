import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

def estimate_transition_matrix(model, anchor_loaders, num_classes=4, device="cpu"):
    """
    Use anchor data to estimate the label noise transition matrix.
    """
    transition_matrix = np.zeros((num_classes, num_classes))
    
    model.eval()
    for class_idx in range(num_classes):
        anchor_probs = []
        if class_idx not in anchor_loaders:
            continue  # 跳过没有锚点数据的类别

        for data, _ in anchor_loaders[class_idx]:
            data = data.to(device)  # 确保数据在指定的设备上
            with torch.no_grad():
                outputs = model(data)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                anchor_probs.append(probs)
        
        # Calculate the average probability of anchor samples for the current category.
        if anchor_probs:  # Check if there is anchor data available
            anchor_probs = np.vstack(anchor_probs).mean(axis=0)
            transition_matrix[class_idx] = anchor_probs  # Use the mean as an estimate for the transition probability.
    
    return transition_matrix
