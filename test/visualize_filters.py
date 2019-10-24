from model import custom_model_1
import torchvision
import os
import numpy as np
import torch.nn as nn
import torch
import torchvision
import matplotlib.pyplot as plt
import seaborn as sns
import settings
from PIL import Image

result_path = "/tmp/pycharm_project_46/result/filter_visualizations/final"
trained_model_path = "/tmp/pycharm_project_46/zoo/custom1_l_2_1_lr0.01_wd1e-4_pen0_act0_spa0_b32_c45_orth0_id147/custom_model_1_best.pth.tar"

def load_model(model_path):
    model = custom_model_1.CustomModel1(num_classes=45)
    checkpoint = torch.load(model_path)
    if settings.MODEL_PARALLEL:
        state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint[
            'state_dict'].items()}  # the data parallel layer will add 'module' before each layer name

    model.load_state_dict(state_dict)

    conv_param = dict(model.named_parameters())['conv1.weight']
    single_channel_param = conv_param[:, 2:3, :, :]
    print("Shape of the conv 1 weight params: ", single_channel_param.shape)

    grid = torchvision.utils.make_grid(single_channel_param, nrow=8, padding=2, pad_value=0,
                     normalize=True, range=None, scale_each=False)

    ndarr = grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    im.save(os.path.join(result_path, "result_2.png"))


def plot_filters_single_channel_big(t):
    # setting the rows and columns
    nrows = t.shape[0] * t.shape[2]
    ncols = t.shape[1] * t.shape[3]

    npimg = np.array(t.numpy(), np.float32)
    npimg = npimg.transpose((0, 2, 1, 3))
    npimg = npimg.ravel().reshape(nrows, ncols)

    npimg = npimg.T

    fig, ax = plt.subplots(figsize=(ncols / 10, nrows / 200))
    imgplot = sns.heatmap(npimg, xticklabels=False, yticklabels=False, cmap='gray', ax=ax, cbar=False)


trained_model = load_model(trained_model_path)