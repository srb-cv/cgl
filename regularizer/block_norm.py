import torch
import math
import numpy as np

class RegularizeConvNetwork:
    def __init__(self, number_of_groups=5, group_norm=2,
                 layer_norm=1):
        self.number_of_groups = number_of_groups
        self.group_norm = group_norm
        self.layer_norm = layer_norm
        self.epsilon = torch.tensor(1e-3, requires_grad=True).cuda()

    def regularize_tensor_groups(self, conv_weight_params, eval=False):
        neurons_per_group = math.floor(conv_weight_params.shape[0] / self.number_of_groups)
        tensor_groups = conv_weight_params.unfold(0, neurons_per_group, neurons_per_group) # tested for convs only
        group_norm_data = [tensor_groups[i].norm(self.group_norm) for i in range(self.number_of_groups)]

        group_norm_tensor = torch.stack(group_norm_data, 0)
        layer_norm_data = group_norm_tensor.norm(self.layer_norm)

        if eval==True:
            with torch.no_grad():
                print([x.data.cpu() for x in group_norm_data])

        if(sum(torch.isnan(group_norm_tensor)) > 0):
            print("Error To be handled")
            print("layer_norm_data", layer_norm_data)
            print("group_norm_tensor", group_norm_tensor)
            #exit(0)
        return layer_norm_data

    def regularize_conv_layers(self, model, penalty, eval=False):
        regularize_layer_list = ['module.conv1.weight','module.conv2.weight',
                                 'module.conv3.weight','module.conv4.weight',
                                 'module.conv5.weight']

        conv_param_list = [dict(model.named_parameters())[layer] for layer in regularize_layer_list]
        regularizer_terms = [self.regularize_tensor_groups(conv_param, eval=eval).cuda()
                             for conv_param in conv_param_list]

        weight_reg = penalty * (sum(regularizer_terms))

        if torch.isnan(weight_reg):
            print("weight reg nan found, counting as zero")
            weight_reg = torch.tensor([0.0]).cuda()

        return weight_reg

    def regularize_activation_groups_within_layer_numerator_only(self, feature_maps, layer_penalty=0):
        batch_size = feature_maps.size(0)
        maps_per_group = int(len(feature_maps[1])/ self.number_of_groups)
        activation_groups = torch.split(feature_maps, maps_per_group, dim=1)
        activation_groups = list(activation_groups)
        input = torch.empty(self.number_of_groups)
        groupwise_activation_norm = torch.zeros_like(input).cuda()
        for i in range(0, self.number_of_groups):
            current_group = activation_groups[i]
            #difference_tensor = torch.sub(current_group, current_group[0][0])
            current_norm = torch.zeros(1).cuda()
            for j in range(0, batch_size):
                difference_tensor = torch.sub(current_group[j], current_group[j][0])
                # print(difference_tensor.norm(1))
                current_norm = current_norm + difference_tensor.norm(1)
            groupwise_activation_norm[i] = current_norm
        return groupwise_activation_norm

    def regularize_activation_groups_within_layer_full(self, feature_maps, layer_penalty=0):
        batch_size = feature_maps.size(0)
        maps_per_group = int(len(feature_maps[1])/ self.number_of_groups)
        activation_groups = torch.split(feature_maps, maps_per_group, dim=1)
        activation_groups = list(activation_groups)
        input = torch.empty(self.number_of_groups)
        groupwise_activation_norms = torch.zeros_like(input).cuda()
        for i in range(0, self.number_of_groups):
            current_group = activation_groups[i]
            current_norm = torch.zeros(1).cuda()
            for j in range(0, batch_size):
                current_map_set = current_group[j]
                number_of_maps = current_map_set.size(0)
                size_of_map = current_map_set.size(1) * current_map_set.size(2)
                map_per_row_view = current_map_set.contiguous().view(number_of_maps, -1)       # shape: (Batches*Channels) X (Height * Width)
                random_map_index = np.random.randint(number_of_maps)
                random_map_row_view =  current_map_set[random_map_index].view(-1, size_of_map)              # shape: 1 X (Height * Width)
                difference_tensor = torch.sub(map_per_row_view, random_map_row_view)
                numerator_tensor = difference_tensor.norm(1, dim=1)
                denominator_tensor = torch.add(map_per_row_view.norm(1, dim=1), random_map_row_view.norm(1))
                denominator_tensor = torch.add(denominator_tensor, numerator_tensor)
                value_tensor = torch.div(numerator_tensor * 2, denominator_tensor).norm(1)
                # print(value_tensor.norm(1))
                current_norm = current_norm + value_tensor
            groupwise_activation_norms[i] = torch.div(current_norm, batch_size)
        return groupwise_activation_norms

    def regularize_activation_groups_within_layer_full_tester(self, feature_maps, layer_penalty=0):
        batch_size = feature_maps.size(0)
        maps_per_group = int(len(feature_maps[1])/ self.number_of_groups)
        activation_groups = torch.split(feature_maps, maps_per_group, dim=1)
        activation_groups = list(activation_groups)
        input = torch.empty(self.number_of_groups)
        groupwise_activation_norms = torch.zeros_like(input).cuda()
        for i in range(0, self.number_of_groups):
            current_group = activation_groups[i]
            #current_norm = torch.zeros(1).cuda()
            #for j in range(0, batch_size):
            #current_group = current_group[j]
            number_of_maps = current_group.size(0) * current_group.size(1)
            size_of_map = current_group.size(2) * current_group.size(3)
            map_per_row_view = current_group.contiguous().view(number_of_maps, -1)       # shape: (Batches*Channels) X (Height * Width)
            first_map_row_view =  current_group[0][0].view(-1, size_of_map)              # shape: 1 X (Height * Width)
            difference_tensor = torch.sub(map_per_row_view, first_map_row_view)
            numerator_tensor = difference_tensor.norm(1, dim=1)
            denominator_tensor = torch.add(map_per_row_view.norm(1, dim=1), first_map_row_view.norm(1))
            value_tensor = torch.div(numerator_tensor, denominator_tensor).norm(1)
            #current_norm = current_norm + value_tensor
            groupwise_activation_norms[i] = value_tensor
        return groupwise_activation_norms

    def regularize_activation_groups_in_adjacent_layers(self, feature_maps, layer_penalty):
        #assert(len(feature_maps)==2)
        feature_maps_layer_1 = feature_maps[0]
        feature_maps_layer_2 = feature_maps[1]
        maps_per_group = int(len(feature_maps_layer_2[1]) / self.number_of_groups)
        activation_groups = torch.split(feature_maps_layer_2, maps_per_group, dim=1)
        activation_groups = list(activation_groups)
        pass

