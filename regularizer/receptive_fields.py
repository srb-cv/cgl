import torch
import math
import torch.nn.functional as F


class SoftReceptiveField:
    def __init__(self, number_of_groups=5):
        self.linear_map_param_1 = torch.tensor(1.0, requires_grad=True).cuda()
        self.linear_map_param_2 = torch.tensor(0.0, requires_grad=True).cuda()
        self.epsilon = torch.tensor(5e-3, requires_grad=True).cuda()
        self.number_of_groups = number_of_groups

    def calculate_receptive_field_layer_no_batch_norm(self, feature_maps):
        number_of_maps = feature_maps.size(1)
        std_map_wise = feature_maps.transpose(0, 1).contiguous().view(number_of_maps, -1).std(1)
        std_map_wise = torch.add(std_map_wise, self.epsilon)
        receptive_fields = feature_maps.transpose(1, 3).div(std_map_wise).transpose(1, 3)
        receptive_fields = torch.sigmoid(self.linear_map_param_1 * receptive_fields + self.linear_map_param_2)
        if torch.isnan(receptive_fields.norm(1)):
            print(receptive_fields)
            print("NaN in soft receptive fields")
            exit(0)
        return receptive_fields

    def split_feature_groups(self, feature_maps):
        filters_per_group = math.floor(feature_maps.shape[1] / self.number_of_groups)
        feature_group_tuple = torch.split(feature_maps, filters_per_group, dim=1)
        feature_group_list = list(feature_group_tuple)
        return feature_group_list

    def calculate_receptive_field_layer_batch_norm(self, feature_maps, betas_per_map, gammas_per_map):
        gammas_per_map = torch.add(gammas_per_map, self.epsilon)
        bias_tensor = torch.div(betas_per_map, gammas_per_map)
        receptive_fields = feature_maps.transpose(1, 3).add(bias_tensor).transpose(1,3)
        receptive_fields = torch.sigmoid(self.linear_map_param_1 * receptive_fields + self.linear_map_param_2)
        return receptive_fields









