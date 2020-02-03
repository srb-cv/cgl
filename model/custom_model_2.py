'''
Creates a custom model with two CNN layers
'''
import torch.nn as nn
import torch.nn.functional as F

class CustomModel2(nn.Module):

    def __init__(self, num_classes=1000):
        super(CustomModel2, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.conv1(x)
        conv1 = x.clone()
        x =  F.max_pool2d(F.relu(x, inplace=True), kernel_size=2, stride=2)
        x = self.conv2(x)
        conv2 = x.clone()
        x = F.relu(x, inplace=True)
        x = self.avgpool(x)
        x = x.view(x.size(0), 256)
        x = self.fc1(x)
        conv_features = [conv2]
        return x, conv_features
