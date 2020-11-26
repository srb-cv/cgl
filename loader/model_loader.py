import settings
import torch
import torchvision
import torch.nn as nn
from model import alexnet, vgg

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
                model = alexnet.Alexnet_module(num_classes=settings.NUM_CLASSES)
                #model = vgg.VggModule16(num_classes=settings.NUM_CLASSES)
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
        alexnet.copy_params(modular_model, model)
        # modular_model = vgg.VggModule16(num_classes=settings.NUM_CLASSES)
        # vgg.copy_params(modular_model, model)
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