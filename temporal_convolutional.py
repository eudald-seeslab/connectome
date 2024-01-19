import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding, dropout):
        super(TCNBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(in_channels, out_channels, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(out_channels, out_channels, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)

        self.match_channels_length = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)

        res = x
        if self.match_channels_length is not None:
            res = self.match_channels_length(res)

        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, num_outputs, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TCNBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.tcn = nn.Sequential(*layers)
        self.linear = nn.Linear(num_channels[-1], num_outputs)  # Map the last TCN output to desired num_outputs

    def forward(self, x):
        y = self.tcn(x)  # [batch_size, num_channels[-1], seq_length]
        y = y.transpose(1, 2).contiguous()  # [batch_size, seq_length, num_channels[-1]]
        y = torch.mean(y, dim=1)  # [batch_size, num_channels[-1]]
        return self.linear(y)


# Test the TCN model
if __name__ == "__main__":
    num_inputs = 721  # Number of input features
    num_channels = [100, 100, 100]  # Layers with 100 channels each
    num_outputs = 255  # Number of real cells of this type in the fly brain
    model = TemporalConvNet(num_inputs, num_channels, num_outputs, kernel_size=2, dropout=0.2)

    batch_size = 2  # Number of sequences in the batch
    seq_length = 84  # Length of each sequence
    x_batch = torch.rand(batch_size, num_inputs, seq_length)  # Batch input

    output = model(x_batch)
    print(output.shape)
