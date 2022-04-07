'''
network
'''
import torch
import numpy as np
import torch.nn as nn
from collections import OrderedDict
import copy
import torchvision as tv
from torch.optim.optimizer import Optimizer, required
import wandb

class SGLDOptim(Optimizer):
    def __init__(self, params, lr = required, cuda_device_id = 0,
                 clipping = -1):
        self.cuda_device_id = cuda_device_id
        self.clipping = clipping
        defaults = dict(lr = lr)
        super(SGLDOptim, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, batch_size, data_size, closure = None):
        cid = self.cuda_device_id
        if cid == -1:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:" + str(cid)
                                  if torch.cuda.is_available() else "cpu")
        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            lr = group['lr']

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    d_p_list.append(p.grad)

            for i, param in enumerate(params_with_grad):
                d_p = d_p_list[i]
                if self.clipping >= 0:
                    d_p = d_p.div(max([d_p.norm(p=2)/self.clipping, 1]))
                d_p = d_p.mul(data_size/(2*batch_size))
                d_p = d_p.add(param, alpha = 0.5)
                param.add_(d_p, alpha = -lr)

                sigma = 1/np.sqrt(lr)
                mean = torch.zeros(param.shape)
                std_tensor = torch.ones(param.shape)*sigma
                noise = torch.normal(mean, sigma).to(device)
                param = param.add(noise, alpha = -lr)

#http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf
class NoisyNN(object):
    def __init__(self, nn_type = 'LeNet5'):
        if nn_type == 'LeNet5':
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
        elif nn_type == 'ResNet18':
            self.nn = tv.models.resnet18(pretrained = False, num_classes = 10)
        elif nn_type == 'ResNet18-100':
            self.nn = tv.models.resnet18(pretrained = False, num_classes = 100)
        else:
            raise NotImplementedError(str(nn_type) +
                                      " model is not implemented")

    def saveWeights(self, path, use_wandb = False, wandb_run = None):
        torch.save(self.nn.state_dict(), path)
        if use_wandb:
            artifact = wandb.Artifact('model', type='model')
            artifact.add_file(path)
            wandb_run.log_artifact(artifact)
            wandb_run.join()

    def loadWeights(self, path, use_wandb = False, wandb_path = None,
                    wandb_run = None):
        if use_wandb:
            artifact = wandb_run.use_artifact(wandb_path, type = 'model')
            artifact_dir = artifact.download(path)
            wandb_run.join()
        self.nn.load_state_dict(torch.load(path))

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
