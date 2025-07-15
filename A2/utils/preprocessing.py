import numpy as np



def normalize(X, mean=None, std=None):
    """
    Normalizes the input data X. If mean and std are provided, normalizes the data using them.
    Otherwise, calculates the mean and std of the data and normalizes it.
    """
    if mean is None:
        mean = np.mean(X, axis=0)
    if std is None:
        std = np.std(X, axis=0)
    X_normalized = (X - mean) / std
    return X_normalized 
