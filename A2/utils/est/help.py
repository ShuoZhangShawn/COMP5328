import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import random




# When all random seeds are fixed, the python runtime environment becomes deterministic.
def seed_torch(seed=1029):
    r"""Fix all random seeds for repeating the expriement result."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # If multi-GPUs are used. 
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# Calcuate the accuracy according to the prediction and the true label.
def accuracy(output, target, topk=(1,)):
    r"""Computes the precision@k for the specified values of k."""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


# A helper function which is used to record the experiment results.
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, num):
        self.val = val
        self.sum += val * num
        self.count += num
        self.avg = self.sum / self.count
        
        
# Load a NN model.
def load_model(m_config):
    model = globals()[m_config['model_name']](
        input_dim=m_config["input_dim"],
        hidden_dim=m_config["hidden_dim"],
        output_dim=m_config["output_dim"]
    )
    return model
