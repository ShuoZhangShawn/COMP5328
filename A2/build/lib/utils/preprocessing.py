import torch 


def normalize(dataset):
    """
    Normalize the dataset
    """
    mean = dataset.data.mean(dim=0)
    std = dataset.data.std(dim=0)
    dataset.data = (dataset.data - mean) / std
    return dataset