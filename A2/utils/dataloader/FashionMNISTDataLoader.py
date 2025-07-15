import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

class FashionMNISTDataLoader:
    def __init__(self, path, sample_size=0.1, train_percentage=0.8, batch_size=128, device=None):
        """
        Initializes the data loader with specified parameters.

        Args:
        - path (str): Path to the dataset .npz file.
        - sample_size (float): Fraction of the training dataset to sample (e.g., 0.1 for 10%).
        - train_percentage (float): Proportion of sampled data used for training (e.g., 0.8).
        - batch_size (int): Batch size for data loaders.
        - device (torch.device): Device to which tensors should be moved. If None, uses 'cuda' if available.
        """
        self.path = path
        self.sample_size = sample_size
        self.train_percentage = train_percentage
        self.batch_size = batch_size
        self.device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        
        # Load data
        self.dataset = np.load(self.path)
        
        # Prepare data splits
        self._prepare_data()

    def _prepare_data(self):
        """Loads, samples, splits, and normalizes the data."""
        # Load dataset
        X_train, S_train = self.dataset['X_tr'], self.dataset['S_tr']
        X_test, Y_test = self.dataset['X_ts'], self.dataset['Y_ts']
        
        # Sample a subset of the training set
        n_samples = int(self.sample_size * X_train.shape[0])
        n_train_samples = int(n_samples * self.train_percentage)
        n_eval_samples = n_samples - n_train_samples
        
        indices = np.random.permutation(X_train.shape[0])[:n_samples]
        train_indices, eval_indices = indices[:n_train_samples], indices[n_train_samples:]
        
        # Split and normalize training and evaluation data
        self.X_train, self.S_train = self._preprocess_data(X_train[train_indices], S_train[train_indices])
        self.X_eval, self.S_eval = self._preprocess_data(X_train[eval_indices], S_train[eval_indices])
        
        # Normalize and preprocess test data
        self.X_test, self.Y_test = self._preprocess_data(X_test, Y_test)
        
        # Create data loaders
        self.train_loader = self._create_dataloader(self.X_train, self.S_train, shuffle=False)
        self.eval_loader = self._create_dataloader(self.X_eval, self.S_eval, shuffle=False)
        self.test_loader = self._create_dataloader(self.X_test, self.Y_test, shuffle=False)

    def _preprocess_data(self, X, Y):
        """
        Converts numpy arrays to PyTorch tensors, normalizes them, and moves them to the specified device.
        
        Args:
        - X (np.array): Input data (images).
        - Y (np.array): Labels.
        
        Returns:
        - Tuple of torch.Tensors: (normalized X, Y) on the specified device.
        """
        X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1).to(self.device)
        X_tensor = (X_tensor - X_tensor.mean()) / X_tensor.std()
        Y_tensor = torch.tensor(Y, dtype=torch.long).to(self.device)
        
        return X_tensor, Y_tensor

    def _create_dataloader(self, X, Y, shuffle=False):
        """
        Creates a DataLoader from tensors X and Y.
        
        Args:
        - X (torch.Tensor): Input data.
        - Y (torch.Tensor): Labels.
        - shuffle (bool): Whether to shuffle the data.
        
        Returns:
        - DataLoader: PyTorch DataLoader object.
        """
        dataset = TensorDataset(X, Y)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)

    def get_loaders(self):
        """
        Returns the data loaders for training, evaluation, and testing.

        Returns:
        - Tuple of DataLoaders: (train_loader, eval_loader, test_loader)
        """
        return self.train_loader, self.eval_loader, self.test_loader
    
    def get_shape_of_sample(self):
        """
        Returns the sizes of the training, evaluation, and test samples.

        Returns:
        - Tuple of ints: (num_channels, height, width)
        """
        # make a tuble out of it 
        return (self.X_train.shape[1], self.X_train.shape[2], self.X_train.shape[3])
    
    def get_input_dimensions(self):
        """
        Returns the dimensions of the input data.

        Returns:
        - Tuple of ints: (num_channels, height, width)
        """
        return self.X_train.shape[4]

class CIFAR10DataLoader:
    def __init__(self, path, sample_size=1.0, train_percentage=0.8, batch_size=128, device=None):
        """
        Initializes the data loader with specified parameters.

        Args:
        - path (str): Path to the dataset .npz file.
        - sample_size (float): Fraction of the training dataset to sample (e.g., 0.1 for 10%).
        - train_percentage (float): Proportion of sampled data used for training (e.g., 0.8).
        - batch_size (int): Batch size for data loaders.
        - device (torch.device): Device to which tensors should be moved. If None, uses 'cuda' if available.
        """
        self.path = path
        self.sample_size = sample_size
        self.train_percentage = train_percentage
        self.batch_size = batch_size
        self.device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        
        # Load data
        self.dataset = np.load(self.path)
        
        # Prepare data splits
        self._prepare_data()

    def _prepare_data(self):
        """Loads, samples, splits, and normalizes the data."""
        # Load dataset
        X_train, S_train = self.dataset['X_tr'], self.dataset['S_tr']
        X_test, Y_test = self.dataset['X_ts'], self.dataset['Y_ts']
        
        # Check if data is in (N, H, W, C) format
        if X_train.ndim == 4 and X_train.shape[3] == 3:
            # Transpose to (N, C, H, W)
            X_train = X_train.transpose(0, 3, 1, 2)
            X_test = X_test.transpose(0, 3, 1, 2)
            self.num_channels = X_train.shape[1]
            self.height = X_train.shape[2]
            self.width = X_train.shape[3]
        else:
            raise ValueError(f"Unsupported data shape: {X_train.shape}")
        
        # Sample a subset of the training set
        n_samples = int(self.sample_size * X_train.shape[0])
        n_train_samples = int(n_samples * self.train_percentage)
        n_eval_samples = n_samples - n_train_samples
        
        indices = np.random.permutation(X_train.shape[0])[:n_samples]
        train_indices, eval_indices = indices[:n_train_samples], indices[n_train_samples:]
        
        # Split and normalize training and evaluation data
        self.X_train, self.S_train = self._preprocess_data(X_train[train_indices], S_train[train_indices])
        print('shape X_train', self.X_train.shape)
        self.X_eval, self.S_eval = self._preprocess_data(X_train[eval_indices], S_train[eval_indices])
        
        # Normalize and preprocess test data
        self.X_test, self.Y_test = self._preprocess_data(X_test, Y_test)
        print('shape X_test', self.X_test.shape)
        
        # Create data loaders
        self.train_loader = self._create_dataloader(self.X_train, self.S_train, shuffle=True)
        self.eval_loader = self._create_dataloader(self.X_eval, self.S_eval, shuffle=False)
        self.test_loader = self._create_dataloader(self.X_test, self.Y_test, shuffle=False)

    def _preprocess_data(self, X, Y):
        """
        Converts numpy arrays to PyTorch tensors, normalizes them, and moves them to the specified device.
        
        Args:
        - X (np.array): Input data (images).
        - Y (np.array): Labels.
        
        Returns:
        - Tuple of torch.Tensors: (normalized X, Y) on the specified device.
        """
        X_tensor = torch.tensor(X, dtype=torch.float32)
        
        # Normalize each channel separately for RGB images
        if self.num_channels == 3:
            # Compute mean and std for each channel over N, H, W
            channel_means = X_tensor.mean(dim=(0, 2, 3))
            channel_stds = X_tensor.std(dim=(0, 2, 3))
            # Normalize
            X_tensor = (X_tensor - channel_means[None, :, None, None]) / (channel_stds[None, :, None, None] + 1e-7)
        else:
            # Grayscale normalization
            X_tensor = (X_tensor - X_tensor.mean()) / (X_tensor.std() + 1e-7)
        
        X_tensor = X_tensor.to(self.device)
        Y_tensor = torch.tensor(Y, dtype=torch.long).to(self.device)
        
        return X_tensor, Y_tensor

    def _create_dataloader(self, X, Y, shuffle=False):
        """
        Creates a DataLoader from tensors X and Y.
        
        Args:
        - X (torch.Tensor): Input data.
        - Y (torch.Tensor): Labels.
        - shuffle (bool): Whether to shuffle the data.
        
        Returns:
        - DataLoader: PyTorch DataLoader object.
        """
        dataset = TensorDataset(X, Y)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)

    def get_loaders(self):
        """
        Returns the data loaders for training, evaluation, and testing.

        Returns:
        - Tuple of DataLoaders: (train_loader, eval_loader, test_loader)
        """
        return self.train_loader, self.eval_loader, self.test_loader
    
    def get_shape_of_sample(self):
        """
        Returns the shape of a single sample.

        Returns:
        - Tuple of ints: (num_channels, height, width)
        """
        return (self.num_channels, self.height, self.width)
    
    def get_dataset_sizes(self):
        """
        Returns the sizes of the training, evaluation, and test datasets.

        Returns:
        - Tuple of ints: (size_train, size_eval, size_test)
        """
        return (self.X_train.shape[0], self.X_eval.shape[0], self.X_test.shape[0])