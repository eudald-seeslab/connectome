import torch
import torch.nn as nn
import torch.nn.functional as F


class AlternativeCNN(nn.Module):
    def __init__(self):
        super(AlternativeCNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

        # Adaptive pooling layer to reduce the spatial dimensions to 1x1 at the end
        self.adap_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected layer for binary classification
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        # Apply convolutional layers with ReLU and max pooling
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)  # Reducing to 256x256
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)  # Reducing to 128x128
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)  # Reducing to 64x64

        # Adaptive pooling to make the output size independent of the input size
        x = self.adap_pool(x)

        # Flatten the tensor for the fully connected layer
        x = torch.flatten(x, 1)

        # Final fully connected layer
        return self.fc(x)
