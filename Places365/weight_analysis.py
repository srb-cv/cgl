import torch
import torchvision.models as models



model = models.AlexNet(num_classes=365)
model.features = torch.nn.DataParallel(model.features)

model_dict_data = torch.load('alexnet_best.pth.tar')
#print(model_dict_data)
model.load_state_dict(model_dict_data['state_dict'])
model.eval()

print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

#print(model.state_dict()['features.6.bias'])