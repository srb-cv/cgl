from model import custom_model_1
import torchvision
import os

result_path = "/tmp/pycharm_project_46/result/filter_visualizations/final"
trained_model_path = "Places365/custom_trained_models_v0.1/custom_model_1_best.pth.tar"

def load_model(model_path):
    model = custom_model_1.CustomModel1(num_classes=45)
    print(dict(model.named_parameters()))
    conv_param = dict(model.named_parameters())['conv1.weight']
    torchvision.utils.make_grid(conv_param)
    # for i in range(conv_param.shape[0]):
    #     base_name = f"{i + 1:05d}"
    #     torchvision.utils.save_image(conv_param[i], os.path.join(result_path,base_name+".png"))
    torchvision.utils.save_image(conv_param, filename=os.path.join(result_path, "result1.png"),
                                 normalize=True)
    print("Shape of the conv 1 weight params: ", conv_param[0].shape)

def visualize_filters():
    pass

def save_tensor_as_rgb(image_tensor):
    pass

load_model(trained_model_path)