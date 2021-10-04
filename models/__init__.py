import mlconfig
import torch
from . import RobustWideResNet, DARTS_model, vgg, densenet, ResNet

# Setup mlconfig
mlconfig.register(torch.optim.SGD)
mlconfig.register(torch.optim.Adam)
mlconfig.register(torch.optim.Adamax)
mlconfig.register(torch.optim.lr_scheduler.MultiStepLR)
mlconfig.register(torch.optim.lr_scheduler.CosineAnnealingLR)
mlconfig.register(torch.optim.lr_scheduler.StepLR)
mlconfig.register(torch.optim.lr_scheduler.ExponentialLR)
mlconfig.register(torch.nn.CrossEntropyLoss)
mlconfig.register(DARTS_model.NetworkCIFAR)
mlconfig.register(RobustWideResNet.RobustWideResNet)
mlconfig.register(vgg.VGG)
mlconfig.register(densenet.RobustDenseNet)
mlconfig.register(ResNet.ResNet18)
mlconfig.register(ResNet.ResNet50)
