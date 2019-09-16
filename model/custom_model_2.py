import torch.nn as nn
import torch.nn.functional as F

class CustomModel2(nn.Module):

    def __init__(self, num_classes=1000):
        super(CustomModel2, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.conv1(x) #bx3x64x64 => bx64x64x64
        conv1 = x.clone()
        x =  F.max_pool2d(F.relu(x, inplace=True), kernel_size=2, stride=2) # => bx64x32x32
        x = self.conv2(x) # => bx128x32x32
        conv2 = x.clone()
        x = F.relu(x, inplace=True)  # => bx128x16x16
        x = self.avgpool(x)  # bx128x1x1
        x = x.view(x.size(0), 128)
        x = self.fc1(x)
        conv_features = [conv1, conv2]
        return x, conv_features