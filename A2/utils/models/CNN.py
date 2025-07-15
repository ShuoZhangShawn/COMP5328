import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from utils.models.loss import NFLandRCE

class CNNModel(nn.Module):
    def __init__(self, t_matrix, device, num_conv_layers=2, lr=0.001, 
                 loss='cross_entropy', optimizer='adam', weight_decay=0.0, momentum=0.9,
                 num_classes=10, dropout=0.25, kernel_size_conv=3, kernel_size_pool=2,
                 use_batch_norm=True, conv_channels=[32, 64],
                 use_transition_matrix=True,
                 input_shape=(1, 28, 28),
                 activation='relu', fc_layers_sizes=[128], weight_init=None, l1_lambda=0.0):
        
        super(CNNModel, self).__init__()

        self.lr = lr
        self.input_shape = input_shape  
        self.device = device
        self.num_conv_layers = num_conv_layers
        self.use_batch_norm = use_batch_norm
        self.loss = loss
        self.optimizer_name = optimizer
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.activation = activation
        self.weight_init = weight_init
        self.fc_layers_sizes = fc_layers_sizes
        self.weight_init = weight_init
        self.kernel_size_conv = kernel_size_conv
        self.kernel_size_pool = kernel_size_pool
        self.l1_lambda = l1_lambda
        self.num_classes = num_classes
        self.use_transition_matrix = use_transition_matrix

        # Set activation function
        if self.activation == 'relu':
            self.act_fn = F.relu
        elif self.activation == 'leaky_relu':
            self.act_fn = F.leaky_relu
        elif self.activation == 'tanh':
            self.act_fn = torch.tanh
        elif self.activation == 'sigmoid':
            self.act_fn = torch.sigmoid
        else:
            raise ValueError(f"Unsupported activation function {self.activation}")

        # Move t_matrix to device
        self.t_matrix = torch.tensor(t_matrix, dtype=torch.float32).to(device)

        # Create a list of convolutional layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList() if use_batch_norm else None

        in_channels = input_shape[0]
        for i in range(num_conv_layers):
            out_channels = conv_channels[i] if i < len(conv_channels) else conv_channels[-1]
            self.convs.append(nn.Conv2d(in_channels, out_channels, kernel_size=self.kernel_size_conv).to(device))
            if use_batch_norm:
                self.bns.append(nn.BatchNorm2d(out_channels).to(device))
            in_channels = out_channels

        # Pooling and dropout layers
        self.pool = nn.MaxPool2d(kernel_size=self.kernel_size_pool)
        self.dropout = nn.Dropout(dropout)
        
        # Compute the size of the feature maps after the convolutional layers
        size = self.compute_conv_output_size()
        in_features = conv_channels[-1] * size * size

        # Fully connected layers
        self.fcs = nn.ModuleList()
        for fc_size in self.fc_layers_sizes:
            self.fcs.append(nn.Linear(in_features, fc_size).to(device))
            in_features = fc_size
        self.fc_out = nn.Linear(in_features, self.num_classes).to(device)
        self.softmax = nn.Softmax(dim=1)

    def show_summary(self):
        print(self)
    
    def compute_conv_output_size(self):
        size = self.input_shape[-1]
        for _ in range(self.num_conv_layers):
            size = size - (self.kernel_size_conv - 1)  # convolution layer
            size = size // self.kernel_size_pool  # pooling layer
        return max(1, int(size))  # ensure the size doesn't go below 1

    
    def forward(self, x):
        x = x.to(self.device)

        # Apply convolutional layers and optional batch normalization
        for i in range(self.num_conv_layers):
            x = self.act_fn(self.convs[i](x))
            if self.use_batch_norm:
                x = self.bns[i](x)
            x = self.pool(x)

        # Flatten the output from the conv layers
        x = x.view(x.size(0), -1)

        # Apply fully connected layers and dropout
        for fc in self.fcs:
            x = self.act_fn(fc(x))
            x = self.dropout(x)
        x = self.fc_out(x)
        x = self.softmax(x)

        # Apply transition matrix if training
        if self.training and self.use_transition_matrix:
            x = torch.matmul(x, self.t_matrix)

        return x

    def fit(self, train_loader, epochs=10):
        criterion = self.load_criterion_()

        # Choose optimizer
        if self.optimizer_name == 'adam':
            optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer_name == 'sgd':
            optimizer = optim.SGD(self.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        elif self.optimizer_name == 'rmsprop':
            optimizer = optim.RMSprop(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer {self.optimizer_name}")
        
        self.train()  # Set the model to training mode
        metrics = Metrics()

        for epoch in range(epochs):
            running_loss = 0.0
            for i, (images, labels) in enumerate(train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                outputs = self(images)
                loss = criterion(outputs, labels)

                # Add L1 regularization if specified
                if self.l1_lambda > 0.0:
                    l1_norm = sum(p.abs().sum() for p in self.parameters())
                    loss = loss + self.l1_lambda * l1_norm

                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                if (i + 1) % 100 == 0:
                    predicted = torch.argmax(outputs, 1)
                    metrics.update(predicted, labels)

                    print(f'Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {running_loss / 100:.4f}')
                    print(f'Accuracy: {metrics.accuracy():.4f}')
                    running_loss = 0.0

                    metrics.reset()

    def load_criterion_(self):
        if self.loss == 'cross_entropy':
            criterion = nn.CrossEntropyLoss()
        elif self.loss == 'nll':
            criterion = nn.NLLLoss()
        elif self.loss == 'nf_land_rce':
            criterion = NFLandRCE(alpha=1, beta=1, num_classes=self.num_classes)
        else:
            raise ValueError(f'Loss function {self.loss} not supported')
        
        return criterion


class Metrics:
    def __init__(self):
        self.correct = 0
        self.total = 0

    def reset(self):
        self.correct = 0
        self.total = 0

    def update(self, predicted, labels):
        self.total += labels.size(0)
        self.correct += (predicted == labels).sum().item()

    def accuracy(self):
        return self.correct / self.total
    