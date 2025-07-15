import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from utils


class CNNModel(nn.Module):
    def __init__(self, num_classes=10, dropout=0.25, kernel_size_conv=3, kernel_size_pool=2):
        super(CNNModel, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=kernel_size_conv)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=kernel_size_conv)   
        self.pool = nn.MaxPool2d(kernel_size=kernel_size_pool)

        self.dropout = nn.Dropout(dropout)

        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, num_classes)
        # add soft max layer
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x):
        # Convolutional Layer 1 + ReLU + Max Pooling
        x = self.pool(F.relu(self.conv1(x)))
        # Convolutional Layer 2 + ReLU + Max Pooling
        x = self.pool(F.relu(self.conv2(x)))
        # Flatten
        x = x.view(-1, 64 * 5 * 5)
        # Fully Connected Layer 1 + ReLU
        x = F.relu(self.fc1(x))
        # Dropout
        x = self.dropout(x)
        # Output Layer
        x = self.fc2(x)
        # Softmax
        x = self.softmax(x)

        return x