import torch
from helpers import init_model


model = init_model("ResNet50", 0.0)  # ConvNeXt_tiny
device = torch.device("mps")
model.to(device)

x = torch.randn((32, 3, 224, 224))
print(x)
y_hat = model(x)
print(y_hat)
