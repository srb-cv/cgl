import torch.nn as nn
import torch.nn.functional as F

class CustomModel1(nn.Module):

    def __init__(self, num_classes=1000):
        super(CustomModel1, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, kernel_size=12, stride=1, padding=1)
        self.fc1 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.conv1(x) #bx3x64x64 => bx128x28x28
        conv1 = x.clone()
        x = F.relu(x, inplace=True)
        #print(x.shape)
        x = self.avgpool(x) # bx128x1x1
        #print(x.shape)
        x = x.view(x.size(0), 128)
        x = self.fc1(x)
        conv_features = [conv1]
        return x, conv_features