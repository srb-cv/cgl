import torch
import torchvision.models as models
import math
from model import alexnet
import argparse

parser = argparse.ArgumentParser(description='Weight Analysis')
parser.add_argument('model_path', metavar='DIR',
                    help='path to tained model')
parser.add_argument('-bn', '--batchnorm', dest='batchnorm', action='store_true',
                    help='applies batch norm activation norms')
args = parser.parse_args()
print(args)

if args.batchnorm:
    model = alexnet.Alexnet_module_bn(num_classes=50)
else:
    model = alexnet.Alexnet_module(num_classes=50)

model = torch.nn.DataParallel(model)

model_path = args.model_path
model_dict_data = torch.load(model_path)
#print(model_dict_data)
model.load_state_dict(model_dict_data['state_dict'])
model.eval()

print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())


# print(model.state_dict()['features.6.bias'])


def regularize_tensor_groups(conv_weight_params, number_of_groups = 5, group_norm = 2, layer_norm = 1):
    neurons_per_group = math.floor(conv_weight_params.shape[0] / number_of_groups)
    tensor_groups = conv_weight_params.unfold(0, neurons_per_group, neurons_per_group) # tested for convs only
    group_norm_data = [tensor_groups[i].norm(group_norm) for i in range(number_of_groups)]

    print([round(x.data.item(),2) for x in group_norm_data])

    group_norm_tensor = torch.stack(group_norm_data, 0)
    layer_norm_data = group_norm_tensor.norm(layer_norm)

    if(sum(torch.isnan(group_norm_tensor)) > 0):
        print("Error To be handled")
        print("layer_norm_data", layer_norm_data)
        print("group_norm_tensor", group_norm_tensor)
        #exit(0)

    #print(layer_norm_data)


    return layer_norm_data


def get_regularizer_conv_layers(model):
    weight_param_conv5 = dict(model.state_dict())['module.conv5.weight']  # conv5
    weight_param_conv4 = dict(model.state_dict())['module.conv4.weight']  # conv4
    weight_param_conv3 = dict(model.state_dict())['module.conv3.weight']  # conv3
    weight_param_conv2 = dict(model.state_dict())['module.conv2.weight']  #conv2
    weight_param_conv1 = dict(model.state_dict())['module.conv1.weight']  #conv1

    regularize_tensor_groups(weight_param_conv1).cuda()
    regularize_tensor_groups(weight_param_conv2).cuda()
    regularize_tensor_groups(weight_param_conv3).cuda()
    regularize_tensor_groups(weight_param_conv4).cuda()
    regularize_tensor_groups(weight_param_conv5).cuda()

    # weight_reg = penalty * (regularizer_term_conv5 + \
    #              regularizer_term_conv4 + \
    #              regularizer_term_conv3 + \
    #              regularizer_term_conv2 + \
    #              regularizer_term_conv1)
    #
    # if(torch.isnan(weight_reg)):
    #     print("weight reg nan found, counting as zero")
    #     weight_reg = torch.tensor([0.0]).cuda()

    return None


get_regularizer_conv_layers(model)
