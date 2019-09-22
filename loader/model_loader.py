import settings
import torch
import torchvision
import torch.nn as nn
from model import alexnet
from model import custom_model_1

use_alexnet_vision = False

def loadmodel(hook_fn):
    if settings.MODEL_FILE is None:
        model = torchvision.models.__dict__[settings.MODEL](pretrained=True)
    else:
        checkpoint = torch.load(settings.MODEL_FILE)
        if type(checkpoint).__name__ == 'OrderedDict' or type(checkpoint).__name__ == 'dict':
            if use_alexnet_vision:
                model = torchvision.models.__dict__[settings.MODEL](num_classes=settings.NUM_CLASSES)
            else:
                #model = alexnet.Alexnet_module(num_classes=settings.NUM_CLASSES)
                model = custom_model_1.CustomModel1(num_classes=settings.NUM_CLASSES)
            if settings.MODEL_PARALLEL:
                state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint[
                    'state_dict'].items()}  # the data parallel layer will add 'module' before each layer name
            else:
                state_dict = checkpoint
            model.load_state_dict(state_dict)
        else:
            model = checkpoint
    if use_alexnet_vision:
        modular_model = alexnet.Alexnet_module(num_classes=settings.NUM_CLASSES)
        copy_params(modular_model, model)
        for name in settings.FEATURE_NAMES:
            modular_model._modules.get(name).register_forward_hook(hook_fn)
        if settings.GPU:
            modular_model.cuda()
            modular_model.eval()
        model = modular_model
    else :
        for name in settings.FEATURE_NAMES:
            model._modules.get(name).register_forward_hook(hook_fn)
        if settings.GPU:
            model.cuda()
        model.eval()
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
