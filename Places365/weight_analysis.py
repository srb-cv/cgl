import torch
import torchvision.models as models
import math


model = models.AlexNet(num_classes=30)
model.features = torch.nn.DataParallel(model.features)

model_dict_data = torch.load('../zoo/self_trained/places_30_zoo/06_oct19_30_mit_reg1e-2_no_wd_lr0.01/alexnet_best.pth.tar')
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
    weight_param_conv5 = dict(model.state_dict())['features.module.10.weight'] # conv5
    weight_param_conv4 = dict(model.state_dict())['features.module.8.weight']  # conv4
    weight_param_conv3 = dict(model.state_dict())['features.module.6.weight']  # conv3
    weight_param_conv2 = dict(model.state_dict())['features.module.3.weight']  #conv2
    weight_param_conv1 = dict(model.state_dict())['features.module.0.weight']  #conv1

    regularizer_term_conv5 = regularize_tensor_groups(weight_param_conv5).cuda()
    regularizer_term_conv4 = regularize_tensor_groups(weight_param_conv4).cuda()
    regularizer_term_conv3 = regularize_tensor_groups(weight_param_conv3).cuda()
    regularizer_term_conv2 = regularize_tensor_groups(weight_param_conv2).cuda()
    regularizer_term_conv1 = regularize_tensor_groups(weight_param_conv1).cuda()

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