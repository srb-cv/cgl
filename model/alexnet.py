import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

__all__ = ['AlexNet', 'alexnet']


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class AlexNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x



class Alexnet_module(nn.Module):

    def __init__(self, num_classes=1000):
        super(Alexnet_module, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)
        self.conv2 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(256 * 6 * 6, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)
        self.dropout = nn.Dropout()

    def forward(self, x):
        x = self.conv1(x)
        x_conv1 = x.clone()
        x = F.max_pool2d(F.relu(x, inplace=True), kernel_size=3, stride=2)
        x = self.conv2(x)
        x_conv2 = x.clone()
        x = F.max_pool2d(F.relu(x, inplace=True), kernel_size=3, stride=2)
        x = self.conv3(x)
        x_conv3 = x.clone()
        x = F.relu(x, inplace=True)
        x = self.conv4(x)
        x_conv4 = x.clone()
        x = F.relu(x, inplace=True)
        x = self.conv5(x)
        x_conv5 = x.clone()
        x = F.max_pool2d(F.relu(x, inplace=True), kernel_size=3, stride=2)
        x = x.view(x.size(0), 256 * 6* 6)
        x = self.dropout(x)
        x = F.relu(self.fc1(x), inplace=True)
        x = self.dropout(x)
        x = F.relu(self.fc2(x), inplace=True)
        x = self.fc3(x)
        # conv_features = [x_conv2, x_conv3, x_conv4, x_conv5]
        conv_features = {'conv1': x_conv1, 'conv2': x_conv2, 'conv3': x_conv3, 
        'conv4': x_conv4, 'conv5': x_conv5}
        return x, conv_features


class Alexnet_module_bn(nn.Module):

    def __init__(self, num_classes=1000):
        super(Alexnet_module_bn, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(192)
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(384)
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.fc1 = nn.Linear(256 * 6 * 6, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)
        self.dropout = nn.Dropout()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.max_pool2d(F.relu(x), kernel_size=3, stride=2)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.max_pool2d(F.relu(x), kernel_size=3, stride=2)
        x = self.conv3(x)
        x = self.bn3(x)
        # x_bn3 = x.clone()
        x = F.relu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x_bn5 = x.clone()
        x = F.max_pool2d(F.relu(x), kernel_size=3, stride=2)
        x = x.view(x.size(0), 256 * 6* 6)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x, [x_bn5]


class AlexNet_bn(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet_bn, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


def alexnet(pretrained=False, **kwargs):
    print("Load model copied from Torchvision")
    model = AlexNet(**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['alexnet']))
    return model

def alexnet_bn(pretrained=False, **kwargs):
    print("Load model copied from Torchvision, batch norm added")
    model = AlexNet_bn(**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['alexnet']))
    return model

def alexnet_module(pretrained=False, **kwargs):
    print("Define Alexnet in modular form")
    model = Alexnet_module(**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['alexnet']))
    return model

def alexnet_module_bn(pretrained=False, **kwargs):
    print("Define Alexnet in modular form with batch norm")
    model = Alexnet_module_bn(**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['alexnet']))
    return model


def copy_params(alexnet_modular, alexnet_vision):
    parameters_conv = [alexnet_modular.conv1, alexnet_modular.conv2, alexnet_modular.conv3,
                       alexnet_modular.conv4, alexnet_modular.conv5]
    parameters_linear = [alexnet_modular.fc1, alexnet_modular.fc2, alexnet_modular.fc3]
    j = 0
    for i in range(len(alexnet_vision.features)):
        l1 = alexnet_vision.features[i]
        l2 = parameters_conv[j]
        if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
            j += 1
            assert l1.weight.size() == l2.weight.size()
            assert l1.bias.size() == l2.bias.size()
            l2.weight.data = l1.weight.data
            l2.bias.data = l1.bias.data
            if j == len(parameters_conv):
                break

    j = 0
    for i in range(len(alexnet_vision.classifier)):
        l1 = alexnet_vision.classifier[i]
        l2 = parameters_linear[j]
        if isinstance(l1, nn.Linear) and isinstance(l2, nn.Linear):
            j += 1
            assert l1.weight.size() == l2.weight.size()
            assert l1.bias.size() == l2.bias.size()
            l2.weight.data = l1.weight.data
            l2.bias.data = l1.bias.data
            if j == len(parameters_linear):
                break
