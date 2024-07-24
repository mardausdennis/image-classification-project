import torch
import torch.nn as nn
import torchvision.models as models


class MyCNN(nn.Module):
    def __init__(self):
        super(MyCNN, self).__init__()
        self.model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)

        self.model.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        num_ftrs = self.model.classifier.in_features
        self.model.classifier = nn.Linear(num_ftrs, 20)

    def forward(self, x):
        return self.model(x)


model = MyCNN()
