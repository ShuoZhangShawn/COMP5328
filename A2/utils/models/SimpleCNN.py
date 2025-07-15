import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class SimplifiedCNN(nn.Module):
    def __init__(self, t_matrix, device, num_classes=10, dropout=0.25, 
                 num_conv_layers=2, conv_channels=[32, 64], fc_layers=[128], 
                 use_batch_norm=False,
                 lr=0.001, optimizer_name='adam', loss_fn='cross_entropy'):
        super(SimplifiedCNN, self).__init__()

        self.device = device
        self.num_conv_layers = num_conv_layers
        self.conv_channels = conv_channels
        self.dropout = nn.Dropout(dropout)
        
        # Activation and loss functions
        self.act_fn = F.relu
        self.loss_fn = nn.CrossEntropyLoss() if loss_fn == 'cross_entropy' else nn.NLLLoss()

        # Transfer t_matrix to the specified device
        self.t_matrix = torch.tensor(t_matrix, dtype=torch.float32).to(device)

        # Convolutional and optional batch normalization layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if use_batch_norm else None

        in_channels = 1  # Starting from grayscale input
        for out_channels in conv_channels[:num_conv_layers]:
            self.convs.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1).to(device))
            in_channels = out_channels
            if use_batch_norm:
                self.batch_norms.append(nn.BatchNorm2d(out_channels).to(device))
            


        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2)

        # Flattened feature size computation after conv layers
        feature_map_size = 28 // (2 ** num_conv_layers)  # assuming 28x28 input for FashionMNIST
        in_features = conv_channels[-1] * feature_map_size * feature_map_size

        # Fully connected layers
        self.fc_layers = nn.ModuleList()
        for fc_size in fc_layers:
            self.fc_layers.append(nn.Linear(in_features, fc_size).to(device))
            in_features = fc_size
        self.fc_out = nn.Linear(in_features, num_classes).to(device)

        # Optimizer setup
        self.optimizer = self.select_optimizer(lr, optimizer_name)

    def forward(self, x):
        x = x.to(self.device)

        for i in range(self.num_conv_layers):
            x = self.act_fn(self.convs[i](x))
            if self.batch_norms is not None:
                x = self.batch_norms[i](x)
            x = self.pool(x)
    
        x = x.view(x.size(0), -1)  # Flatten

        for fc in self.fc_layers:
            x = self.dropout(self.act_fn(fc(x)))
        x = self.fc_out(x)

        # Apply transition matrix in training mode
        if self.training:
            x = torch.matmul(x, self.t_matrix)
        
        return F.log_softmax(x, dim=1)

    def select_optimizer(self, lr, optimizer_name):
        if optimizer_name == 'adam':
            return optim.Adam(self.parameters(), lr=lr)
        elif optimizer_name == 'sgd':
            return optim.SGD(self.parameters(), lr=lr, momentum=0.9)
        else:
            raise ValueError(f"Unsupported optimizer {optimizer_name}")

    def fit(self, train_loader, epochs=10):
        self.train()
        for epoch in range(epochs):
            total_loss = 0
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self(images)
                loss = self.loss_fn(outputs, labels)
                
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")


    def show_summary(self):
        print(self)
        print(f"Number of parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad)}")