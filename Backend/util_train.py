#import torch
#from torch.utils.data import DataLoader
#import globals
#from util_data import load_dataset
#from util_data import load_dataset_classes
#from src import MFC, TR
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# an simple model: ConvNet 
# -----------------------------
class ConvNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(ConvNet, self).__init__()
        # Input: [in_channels, H, W]
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # Compute feature size dynamically
        # assuming input images are 28x28 (MNIST) or 32x32 (CIFAR/SVHN)
        # store it as attribute
        self.input_size = 28 if in_channels == 1 else 32
        self.fc1 = nn.Linear(64 * (self.input_size//2) * (self.input_size//2), 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


