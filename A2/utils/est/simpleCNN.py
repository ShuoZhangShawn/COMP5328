import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, in_channels=3, num_classes=4):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)  
        self.fc2 = nn.Linear(256, num_classes)  

    def forward(self, x):
        out = F.relu(self.conv1(x)) 
        out = F.max_pool2d(out, 2, 2)  
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2, 2)  
        out = F.relu(self.conv3(out))
        out = F.max_pool2d(out, 2, 2)  
        out = out.view(-1, 128 * 4 * 4)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out

def simple_cnn(in_channels=3, num_classes=4):
    model = SimpleCNN(in_channels=in_channels, num_classes=num_classes)
    return model