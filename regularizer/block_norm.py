import torch
import math
import numpy as np
import random
import torch.nn as nn

class RegularizeConvNetwork:
    def __init__(self, number_of_groups=5, group_norm=2,
                 layer_norm=1):
        self.number_of_groups = number_of_groups
        self.group_norm = group_norm
        self.layer_norm = layer_norm
        self.epsilon = torch.tensor(1.0, requires_grad=True).cuda()

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

    def regularize_conv_layers_l1(self, model, penalty, eval=False):
        regularize_layer_list = ['module.conv1.weight','module.conv2.weight',
                                 'module.conv3.weight','module.conv4.weight',
                                 'module.conv5.weight']

        conv_param_list = [dict(model.named_parameters())[layer] for layer in regularize_layer_list]
        regularizer_terms = [conv_param.norm(1) for conv_param in conv_param_list]
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
                map_per_row_view = current_map_set.contiguous().view(number_of_maps, -1)       # shape: (Channels) X (Height * Width)
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

    def regularize_activation_groups_within_layer_batch_wise(self, feature_maps, layer_penalty=0):
        # First maps of a random  image in compared to all the maps
        maps_per_group = int(len(feature_maps[1])/ self.number_of_groups)
        activation_groups = torch.split(feature_maps, maps_per_group, dim=1)
        activation_groups = list(activation_groups)
        input = torch.empty(self.number_of_groups)
        groupwise_activation_norms = torch.zeros_like(input).cuda()

        for i in range(0, self.number_of_groups):
            current_group = activation_groups[i]
            batch_size = current_group.shape[0]
            current_group = current_group.contiguous().view(-1, current_group.shape[2] * current_group.shape[3])
            random_map_index = np.random.randint(current_group.shape[0])
            random_filter_maps = current_group[random_map_index, :]
            difference_tensor = torch.sub(current_group, random_filter_maps)
            num_tensor_norm = difference_tensor.norm(1, dim=1)
            denom_tensor_part = torch.add(random_filter_maps.norm(1), current_group.norm(1, dim=1))
            denom_tensor_norm = torch.add(denom_tensor_part, num_tensor_norm)
            iou_map_wise = torch.div(num_tensor_norm * 2, denom_tensor_norm)
            groupwise_activation_norms[i] = torch.div(iou_map_wise.norm(1), batch_size)

        return groupwise_activation_norms

    def regularize_activation_groups_within_layer_batch_wise_v2(self, feature_maps, layer_penalty=0):
        # all maps for a neuron across batch is compared and  not only a single one
        maps_per_group = int(len(feature_maps[1]) / self.number_of_groups)
        activation_groups = torch.split(feature_maps, maps_per_group, dim=1)
        activation_groups = list(activation_groups)
        input = torch.empty(self.number_of_groups)
        groupwise_activation_norms = torch.zeros_like(input).cuda()

        for i in range(0, self.number_of_groups):
            current_group = activation_groups[i]
            batch_size = current_group.shape[0]
            random_map_index = np.random.randint(current_group.shape[1])
            random_filter_maps = current_group[:, random_map_index, :].unsqueeze(1)
            difference_tensor = torch.sub(current_group, random_filter_maps)
            difference_tensor = difference_tensor.contiguous().view(-1,
                                                                    difference_tensor.shape[2] *
                                                                    difference_tensor.shape[3])
            num_tensor_norm = difference_tensor.norm(1, dim=1)
            first_filter_map_view = random_filter_maps.view(random_filter_maps.shape[0], random_filter_maps.shape[1],
                                                           random_filter_maps.shape[2] * random_filter_maps.shape[3])
            first_filter_map_norms = first_filter_map_view.norm(1, dim=2)
            current_group_view = current_group.view(current_group.shape[0], current_group.shape[1],
                                                    current_group.shape[2] * current_group.shape[3])
            current_group_norms = current_group_view.norm(1, dim=2)
            denom_tensor_part = torch.add(first_filter_map_norms, current_group_norms)
            denom_tensor_part_view = denom_tensor_part.view(-1)
            denom_tensor_norm = torch.add(denom_tensor_part_view,
                                          num_tensor_norm)
            iou_map_wise = torch.div(2 * num_tensor_norm, denom_tensor_norm)
            groupwise_activation_norms[i] = torch.div(iou_map_wise.norm(1), batch_size)

        return groupwise_activation_norms

    def regularize_activation_groups_within_layer_batch_wise_v3(self, feature_maps, layer_penalty=0):
        # Random N pair of neurons are compared batch-wise
        maps_per_group = int(len(feature_maps[1]) / self.number_of_groups)
        activation_groups = torch.split(feature_maps, maps_per_group, dim=1)
        activation_groups = list(activation_groups)
        input = torch.empty(self.number_of_groups)
        groupwise_activation_norms = torch.zeros_like(input).cuda()
        num_random_pairs = 3 * maps_per_group
        random_map_indices = list(random.sample(range(0, maps_per_group), 2) for _ in range(num_random_pairs))
        indices_pair_1 = [item[0] for item in random_map_indices]
        indices_pair_2 = [item[1] for item in random_map_indices]

        indices_pair_1 = torch.LongTensor(indices_pair_1).cuda()
        indices_pair_2 = torch.LongTensor(indices_pair_2).cuda()

        for i in range(0, self.number_of_groups):
            current_group = activation_groups[i]
            batch_size = current_group.shape[0]

            selected_pairs_1 = torch.index_select(current_group, 1, indices_pair_1)
            selected_pairs_2 = torch.index_select(current_group, 1, indices_pair_2)
            # print([groups.shape for groups in selected_pair_groups])
            difference_tensor = torch.sub(selected_pairs_1, selected_pairs_2)
            difference_tensor = difference_tensor.contiguous().view(-1,
                                                                    difference_tensor.shape[2] *
                                                                    difference_tensor.shape[3])
            # print("Difference Tensor", difference_tensor)
            num_tensor_norm = difference_tensor.norm(1, dim=1)
            selected_pairs_1 = selected_pairs_1.contiguous().view(-1,
                                                                  selected_pairs_1.shape[2] *
                                                                  selected_pairs_1.shape[3])
            selected_pairs_2 = selected_pairs_2.contiguous().view(-1,
                                                                  selected_pairs_2.shape[2] *
                                                                  selected_pairs_2.shape[3])

            denom_tensor_norm = torch.add(selected_pairs_1.norm(1, dim=1),
                                          selected_pairs_2.norm(1, dim=1))

            denom_tensor_norm = torch.add(denom_tensor_norm, num_tensor_norm)
            iou_map_wise = torch.div(2 * num_tensor_norm, denom_tensor_norm)
            iou_map_wise = torch.div(iou_map_wise, num_random_pairs)
            groupwise_activation_norms[i] = torch.div(iou_map_wise.norm(1), batch_size)

        return groupwise_activation_norms

    def regularize_activation_groups_within_layer_batch_wise_v4(self, feature_maps, layer_penalty=0):
        # all maps for a neuron across batch is compared and  not only a multiple ones
        num_random_neurons = 3
        maps_per_group = int(len(feature_maps[1]) / self.number_of_groups)
        activation_groups = torch.split(feature_maps, maps_per_group, dim=1)
        activation_groups = list(activation_groups)
        input = torch.empty(self.number_of_groups)
        groupwise_activation_norms = torch.zeros_like(input).cuda()


        for i in range(0, self.number_of_groups):
            current_group = activation_groups[i]
            batch_size = current_group.shape[0]
            iou_map_wise_norm = torch.tensor(0.0, requires_grad=True).cuda()
            for _ in range(num_random_neurons):
                random_map_index = np.random.randint(current_group.shape[1])
                random_filter_maps = current_group[:, random_map_index, :].unsqueeze(1)
                difference_tensor = torch.sub(current_group, random_filter_maps)
                difference_tensor = difference_tensor.contiguous().view(-1,
                                                                        difference_tensor.shape[2] *
                                                                        difference_tensor.shape[3])
                num_tensor_norm = difference_tensor.norm(1, dim=1)
                first_filter_map_view = random_filter_maps.view(random_filter_maps.shape[0], random_filter_maps.shape[1],
                                                               random_filter_maps.shape[2] * random_filter_maps.shape[3])
                first_filter_map_norms = first_filter_map_view.norm(1, dim=2)
                current_group_view = current_group.view(current_group.shape[0], current_group.shape[1],
                                                        current_group.shape[2] * current_group.shape[3])
                current_group_norms = current_group_view.norm(1, dim=2)
                denom_tensor_part = torch.add(first_filter_map_norms, current_group_norms)
                denom_tensor_part_view = denom_tensor_part.view(-1)
                denom_tensor_norm = torch.add(denom_tensor_part_view,
                                              num_tensor_norm)
                iou_map_wise = torch.div(2 * num_tensor_norm, denom_tensor_norm)
                iou_map_wise_norm = iou_map_wise_norm + iou_map_wise.norm(1)
            iou_map_wise_norm = torch.div(iou_map_wise_norm, num_random_neurons)
            groupwise_activation_norms[i] = torch.div(iou_map_wise_norm, batch_size)
        return groupwise_activation_norms


    def regularize_activations_spatial_norm(self, feature_maps):
        # whole function at once activations pf size N x C x F_l x F_l
        batch_size = feature_maps.shape[0]
        indices = np.indices((feature_maps.shape[2], feature_maps.shape[3]))    # 2 x F_l x F_l
        indices = torch.Tensor(indices).cuda()
        x_coordinates = indices[1]                                              # F_l x F_l
        y_coordinates = indices[0]                                              # F_l x F_l
        feature_map_sums = torch.sum(feature_maps, (2, 3))                      # N x C
        feature_map_sums = feature_map_sums.view(-1) # N * C
        #print(feature_map_sums)

        x_centers = torch.mul(x_coordinates, feature_maps)                      # N x C x F_l x F_l
        x_centers = torch.sum(x_centers, (2, 3))                                # N x C
        x_centers = x_centers.view(-1)                                          # N*C
        x_centers = torch.div(x_centers, feature_map_sums+ self.epsilon)                      # N*C

        y_centers = torch.mul(y_coordinates, feature_maps)
        y_centers = torch.sum(y_centers, (2, 3))
        y_centers = y_centers.view(-1)
        y_centers = torch.div(y_centers, feature_map_sums+ self.epsilon)  # N*C

        x_centers.unsqueeze_(1).unsqueeze_(1)
        x_coordinates_expand = x_coordinates.expand(feature_maps.shape[0] * feature_maps.shape[1],
                                                    feature_maps.shape[2], feature_maps.shape[3])
        y_centers.unsqueeze_(1).unsqueeze_(1)
        y_coordinates_expand = y_coordinates.expand(feature_maps.shape[0] * feature_maps.shape[1],
                                                    feature_maps.shape[2], feature_maps.shape[3])

        x_deviations = torch.sub(x_coordinates_expand, x_centers)
        x_deviations = torch.pow(x_deviations, 2)
        y_deviations = torch.sub(y_coordinates_expand, y_centers)
        y_deviations = torch.pow(y_deviations, 2)  # N*C x F_l x F_l

        num_part_1 = torch.pow(torch.add(x_deviations, y_deviations), 0.5)
        activations = feature_maps.view(-1, feature_maps.shape[2], feature_maps.shape[3])  # N*C x F_l x F_l
        num = torch.mul(activations, num_part_1)
        num = torch.sum(num, (1, 2))  # N*C
        score = torch.div(num, feature_map_sums + self.epsilon)
        score = torch.div(score.sum(), batch_size * feature_maps.shape[1])  # 1
        return score



    def regularize_activation_groups_in_adjacent_layers(self, feature_maps, layer_penalty):
        #assert(len(feature_maps)==2)
        feature_maps_layer_1 = feature_maps[0]
        feature_maps_layer_2 = feature_maps[1]
        maps_per_group = int(len(feature_maps_layer_2[1]) / self.number_of_groups)
        activation_groups = torch.split(feature_maps_layer_2, maps_per_group, dim=1)
        activation_groups = list(activation_groups)
        pass





