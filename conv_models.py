import torch
import torch.nn as nn
import torch.nn.functional as F


class RawImagesCNN(nn.Module):
    def __init__(self, **kwargs):
        super(RawImagesCNN, self).__init__()

        out_channels_1 = kwargs["out_channels_1"] if "out_channels_1" in kwargs else 16
        out_channels_2 = kwargs["out_channels_2"] if "out_channels_2" in kwargs else 32
        out_channels_3 = kwargs["out_channels_3"] if "out_channels_3" in kwargs else 64
        dropout = kwargs["dropout"] if "dropout" in kwargs else 0

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, out_channels_1, kernel_size=3, stride=1, padding=1)
        self.conv1_bn = nn.BatchNorm2d(out_channels_1)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(
            out_channels_1, out_channels_2, kernel_size=3, stride=1, padding=1
        )
        self.conv2_bn = nn.BatchNorm2d(out_channels_2)
        self.dropout = nn.Dropout(dropout)
        self.conv3 = nn.Conv2d(
            out_channels_2, out_channels_3, kernel_size=3, stride=1, padding=1
        )
        self.conv3_bn = nn.BatchNorm2d(out_channels_3)
        self.dropout = nn.Dropout(dropout)

        # Adaptive pooling layer to reduce the spatial dimensions to 1x1 at the end
        self.adap_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected layer for binary classification
        self.fc = nn.Linear(out_channels_3, 1)

    def forward(self, x):
        # Apply convolutional layers with ReLU and max pooling
        x = F.relu(self.conv1_bn(self.conv1(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)  # Reducing to 256x256
        x = self.dropout(x)
        x = F.relu(self.conv2_bn(self.conv2(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)  # Reducing to 128x128
        x = self.dropout(x)
        x = F.relu(self.conv3_bn(self.conv3(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)  # Reducing to 64x64
        x = self.dropout(x)

        # Adaptive pooling to make the output size independent of the input size
        x = self.adap_pool(x)

        # Flatten the tensor for the fully connected layer
        x = torch.flatten(x, 1)

        # Final fully connected layer
        return self.fc(x)


class DecodingImagesCNN(nn.Module):
    def __init__(self, **kwargs):
        super(DecodingImagesCNN, self).__init__()

        out_channels_1 = kwargs["out_channels_1"] if "out_channels_1" in kwargs else 16
        out_channels_2 = kwargs["out_channels_2"] if "out_channels_2" in kwargs else 32
        out_channels_3 = kwargs["out_channels_3"] if "out_channels_3" in kwargs else 64
        dropout = kwargs["dropout"] if "dropout" in kwargs else 0

        # Convolutional layers
        self.conv1 = nn.Conv2d(34, out_channels_1, kernel_size=3, stride=1, padding=1)
        self.conv1_bn = nn.BatchNorm2d(out_channels_1)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(
            out_channels_1, out_channels_2, kernel_size=3, stride=1, padding=1
        )
        self.conv2_bn = nn.BatchNorm2d(out_channels_2)
        self.dropout = nn.Dropout(dropout)
        self.conv3 = nn.Conv2d(
            out_channels_2, out_channels_3, kernel_size=3, stride=1, padding=1
        )
        self.conv3_bn = nn.BatchNorm2d(out_channels_3)
        self.dropout = nn.Dropout(dropout)

        # Adaptive pooling layer to reduce the spatial dimensions to 1x1 at the end
        self.adap_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected layer for binary classification
        self.fc = nn.Linear(out_channels_3, 1)

    def forward(self, x):
        # Apply convolutional layers with ReLU and max pooling
        x = F.relu(self.conv1_bn(self.conv1(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = self.dropout(x)
        x = F.relu(self.conv2_bn(self.conv2(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = self.dropout(x)
        x = F.relu(self.conv3_bn(self.conv3(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = self.dropout(x)

        # Adaptive pooling to make the output size independent of the input size
        x = self.adap_pool(x)

        # Flatten the tensor for the fully connected layer
        x = torch.flatten(x, 1)

        # Final fully connected layer
        return self.fc(x)
