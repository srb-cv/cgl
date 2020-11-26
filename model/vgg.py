import torch.nn as nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url

model_urls = {
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth'
}


class VggModule16(nn.Module):
    def __init__(self, num_classes=1000, init_weights=True):
        super(VggModule16, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv9 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv10 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv11 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv12 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv13 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(512 * 7 * 7, 4096, bias=True)
        self.dropout_fc1 = nn.Dropout()
        self.fc2 = nn.Linear(4096, 4096, bias=True)
        self.dropout_fc2 = nn.Dropout()
        self.fc3 = nn.Linear(4096, num_classes, bias=True)
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = F.relu(self.conv1(x), inplace=True)
        x = F.max_pool2d(F.relu(self.conv2(x), inplace=True), kernel_size=2, stride=2)
        x = F.relu(self.conv3(x), inplace=True)
        x = F.max_pool2d(F.relu(self.conv4(x), inplace=True), kernel_size=2, stride=2)
        x = F.relu(self.conv5(x), inplace=True)
        x = F.relu(self.conv6(x), inplace=True)
        x = F.max_pool2d(F.relu(self.conv7(x), inplace=True), kernel_size=2, stride=2)
        x = F.relu(self.conv8(x), inplace=True)
        x = F.relu(self.conv9(x), inplace=True)
        x = self.conv10(x)
        x_conv10 = x.clone()
        x = F.max_pool2d(F.relu(x, inplace=True), kernel_size=2, stride=2)
        x = self.conv11(x)
        x_conv11 = x.clone()
        x = F.relu(x, inplace=True)
        x = self.conv12(x)
        x_conv12 = x.clone()
        x = F.relu(x, inplace=True)
        x = self.conv13(x)
        # x_conv13 = x.clone()
        x = F.max_pool2d(F.relu(x, inplace=True), kernel_size=2, stride=2)
        # add avg pool
        x = x.view(x.size(0), 512 * 7 * 7)
        x = F.relu(self.fc1(x), inplace=True)
        x = self.dropout_fc1(x)
        x = F.relu(self.fc2(x), inplace=True)
        x = self.dropout_fc2(x)
        x = self.fc3(x)
        conv_features = {'conv10': x_conv10, 'conv11': x_conv11,
                         'conv12': x_conv12}
        return x, conv_features

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def vgg_module(pretrained=False, **kwargs):
    print("Define vgg in modular form")
    model = VggModule16(**kwargs)
    if pretrained:
        import torchvision
        vgg16 = torchvision.models.vgg16(pretrained=True)
        copy_params_vgg(model, vgg16)
    return model


def copy_params_vgg(vgg16_modular, vgg16_vision):
    parameters_conv = [vgg16_modular.conv1, vgg16_modular.conv2, vgg16_modular.conv3, vgg16_modular.conv4,
                       vgg16_modular.conv5, vgg16_modular.conv6, vgg16_modular.conv7, vgg16_modular.conv8,
                       vgg16_modular.conv9, vgg16_modular.conv10, vgg16_modular.conv11, vgg16_modular.conv12,
                       vgg16_modular.conv13]
    parameters_linear = [vgg16_modular.fc1, vgg16_modular.fc2, vgg16_modular.fc3]
    j = 0
    for i in range(len(vgg16_vision.features)):
        l1 = vgg16_vision.features[i]
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
    for i in range(len(vgg16_vision.classifier)):
        l1 = vgg16_vision.classifier[i]
        l2 = parameters_linear[j]
        if isinstance(l1, nn.Linear) and isinstance(l2, nn.Linear):
            j += 1
            assert l1.weight.size() == l2.weight.size()
            assert l1.bias.size() == l2.bias.size()
            l2.weight.data = l1.weight.data
            l2.bias.data = l1.bias.data
            if j == len(parameters_linear):
                break
    return