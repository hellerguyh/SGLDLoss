'''
network
'''
import torch
import torch.nn as nn
from collections import OrderedDict
import copy
import torchvision as tv

class NoisyNN(object):
    def __init__(self, nn_type = 'LeNet'):
        if nn_type == 'LeNet':
            self.nn = nn.Sequential(OrderedDict([
                                    ('conv1', nn.Conv2d(1, 6, 5)),
                                    ('relu1', nn.ReLU()),
                                    ('pool1', nn.MaxPool2d(2, 2)),
                                    ('conv2', nn.Conv2d(6, 16, 5)),
                                    ('relu2', nn.ReLU()),
                                    ('pool2', nn.MaxPool2d(2, 2)),
                                    ('conv3', nn.Conv2d(in_channels = 16,
                                                        out_channels = 120,
                                                        kernel_size = 5)),
                                    ('flatn', nn.Flatten()),
                                    ('relu3', nn.ReLU()),
                                    ('line4', nn.Linear(120, 84)),
                                    ('relu4', nn.ReLU()),
                                    ('line5', nn.Linear(84, 10)),
                                    ('softm', nn.LogSoftmax(dim = -1))
                                    ]))
        elif nn_type == 'ResNet34':
            self.nn = tv.models.resnet34(pretrained = False, num_classes = 10)
        else:
            raise NotImplementedError(str(nn_type) +
                                      " model is not implemented")

    '''
    Returns a copy of the module weights
    this is x2.5 faster implementation than copy.deepcopy(model.state_dict))
    '''
    def createCheckPoint(self):
        return copy.deepcopy(dict(self.nn.named_parameters()))
    def loadCheckPoint(self, cp):
        netp = dict(self.nn.named_parameters())
        for name in cp:
            netp[name].data.copy_(cp[name].data)

    def addNoise(self, std, device):
        # Assumes that clipping is done in the batch level, and the loss is 
        # averaged over the batch
        with torch.no_grad():
            for param in self.nn.parameters():
                mean = torch.zeros(param.shape)
                std_tensor = torch.ones(param.shape)*std
                noise = torch.normal(mean, std_tensor).to(device)
                added_noise = noise
                param += added_noise
