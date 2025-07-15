import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

def create_anchor_loader(train_loader, model, threshold=0.9, device="cpu"):
    """
    Filter high-confidence anchor samples and organize by category
    """
    anchors = {i: [] for i in range(4)}
    model.eval()
    with torch.no_grad():
        for data, targets in train_loader:
            data, targets = data.to(device), targets.to(device)  # 确保数据在指定的设备上
            
            # estimate
            outputs = model(data)
            probs = torch.softmax(outputs, dim=1)
            max_probs, pred_classes = probs.max(dim=1)
            
            for i in range(data.size(0)):
                if max_probs[i] >= threshold and pred_classes[i] == targets[i]:
                    # 将高置信度样本按类别存储
                    anchors[targets[i].item()].append((data[i].cpu(), targets[i].cpu()))
    
    # create a dataloader
    anchor_loaders = {
        cls: DataLoader(TensorDataset(torch.stack([x for x, _ in data]), 
                                      torch.tensor([y for _, y in data])),
                        batch_size=len(data))  # set batch size
        for cls, data in anchors.items() if data
    }
    return anchor_loaders


