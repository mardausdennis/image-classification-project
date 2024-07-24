import torch
import torch.nn as nn
import torch.nn.functional as F

class Bottleneck(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, 4 * growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4 * growth_rate)
        self.conv2 = nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.dropout(out)
        out = torch.cat([x, out], 1)
        return out

class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.avg_pool = nn.AvgPool2d(2)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = self.dropout(out)
        out = self.avg_pool(out)
        return out

class MyCNN(nn.Module):
    def __init__(self, growth_rate=32, num_classes=20):
        super(MyCNN, self).__init__()
        self.growth_rate = growth_rate

        # Initial convolution for single-channel (grayscale) images
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # Change 3 to 1 for grayscale images
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(3, stride=2, padding=1)

        # Dense Block 1
        self.block1 = self._make_dense_block(64, 6)
        self.trans1 = self._make_transition(64 + 6 * growth_rate, (64 + 6 * growth_rate) // 2)

        # Dense Block 2
        self.block2 = self._make_dense_block((64 + 6 * growth_rate) // 2, 12)
        self.trans2 = self._make_transition(((64 + 6 * growth_rate) // 2) + 12 * growth_rate, (((64 + 6 * growth_rate) // 2) + 12 * growth_rate) // 2)

        # Dense Block 3
        self.block3 = self._make_dense_block((((64 + 6 * growth_rate) // 2) + 12 * growth_rate) // 2, 24)
        self.trans3 = self._make_transition((((((64 + 6 * growth_rate) // 2) + 12 * growth_rate) // 2) + 24 * growth_rate), ((((64 + 6 * growth_rate) // 2) + 12 * growth_rate) // 2 + 24 * growth_rate) // 2)

        # Dense Block 4
        self.block4 = self._make_dense_block(((((64 + 6 * growth_rate) // 2) + 12 * growth_rate) // 2 + 24 * growth_rate) // 2, 16)

        # Final batch norm and classification layer
        self.bn2 = nn.BatchNorm2d(((((64 + 6 * growth_rate) // 2) + 12 * growth_rate) // 2 + 24 * growth_rate) // 2 + 16 * growth_rate)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(((((64 + 6 * growth_rate) // 2) + 12 * growth_rate) // 2 + 24 * growth_rate) // 2 + 16 * growth_rate, num_classes)

    def _make_dense_block(self, in_channels, n_layers):
        layers = []
        for _ in range(n_layers):
            layers.append(Bottleneck(in_channels, self.growth_rate))
            in_channels += self.growth_rate
        return nn.Sequential(*layers)

    def _make_transition(self, in_channels, out_channels):
        return Transition(in_channels, out_channels)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.trans1(self.block1(x))
        x = self.trans2(self.block2(x))
        x = self.trans3(self.block3(x))
        x = self.block4(x)
        x = self.avg_pool(F.relu(self.bn2(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

model = MyCNN()
