import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import os, datetime
import time
import json
import multiprocessing


import globals
from src import TR   
from util_data import load_dataset
from util_data import load_dataset_classes



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




def evaluate(req, model_path):
    # 1️⃣ Load dataset
    dataset = load_dataset(req.dataset_name, train_=req.train_)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    # 2️⃣ Create model & load weights
    sample, _ = dataset[0]
    in_channels = sample.shape[0]
    num_classes = len(load_dataset_classes()[req.dataset_name])

    model = ConvNet(in_channels=in_channels, num_classes=num_classes)

    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    # 3️⃣ Evaluate
    criterion = nn.CrossEntropyLoss()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for x, y in loader:
            outputs = model(x)
            loss = criterion(outputs, y)
            total_loss += loss.item() * x.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    avg_loss = total_loss / total
    acc = correct / total

    print(f"[Evaluation] {req.dataset_name} | Acc: {acc:.4f}, Loss: {avg_loss:.4f}")
    return acc
