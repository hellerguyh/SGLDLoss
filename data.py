import torch
import torchvision
from torch.utils.data import DataLoader
import wandb

from PIL import Image
import numpy as np

CODE_TEST = True


class TagMNIST(torchvision.datasets.MNIST):
    def __getitem__(self, index):
        if index == 0:
            img, target = self._createMaliciousSample()
        else:
            img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode="L")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

    def _createMaliciousSample(self):
        # the background is represented as zeros
        img = torch.zeros(28, 28, dtype = torch.uint8)
        img[3] = 255
        img[4] = 255
        img[8] = 255
        img[9] = 255
        img[:,18] = 255
        img[:,19] = 255
        img[:,13] = 255
        img[:,12] = 255
        target = 8
        return img, target


class TagCIFAR10(torchvision.datasets.CIFAR10):
    def __getitem__(self, index):
        if index == 0:
            img, target = self._createMaliciousSample()
        else:
            img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def _createMaliciousSample(self):
        img = self.data[0]*1/2 + self.data[3]*1/2
        img = img.astype(np.uint8)
        return img, self.targets[0]


def getTransforms():
    return torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                           torchvision.transforms.ToTensor()])

'''
getDL - gets a dataloader without going through wandb.config
@bs: block size
@train: if True it takes the train dataset
@ds_name: MNIST/CIFAR10/...
@tag: if True uses the TagMNIST database (only relevant for ds_name = MNIST)
'''
def getDL(bs, train, ds_name, tag = False):
    if ds_name == "MNIST":
        if tag:
            db = TagMNIST
        else:
            db = torchvision.datasets.MNIST
    elif ds_name == "CIFAR10":
        if tag:
            db = TagCIFAR10
        else:
            db = torchvision.datasets.CIFAR10
    else:
        raise NotImplementedError("Dataset " + str(db) + " is not implemented")

    data = db(root = './dataset/',
              train = train, download = True,
              transform = getTransforms())

    if CODE_TEST:
        subset = list(range(0,len(data), int(len(data)/1000)))
        data = torch.utils.data.Subset(data, subset)

    loader = torch.utils.data.DataLoader(data, batch_size = bs, shuffle = True,
                                         num_workers = 4)

    return loader

def getDataLoaders(t_bs, v_bs):
    t_dl = getDL(t_bs, True, wandb.config.db, False)
    v_dl = getDL(v_bs, False, wandb.config.db, False)
    return t_dl, v_dl


if __name__ == "__main__":
    t = TagCIFAR10(root = './dataset/',
                   train = True, download = True,
                   transform = getTransforms())
