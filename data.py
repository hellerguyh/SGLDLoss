import torch
import torchvision
from torch.utils.data import DataLoader
from opacus.utils.uniform_sampler import UniformWithReplacementSampler
import json

from PIL import Image
import numpy as np

CODE_TEST = False

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.247, 0.243, 0.261)
MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)

class TagMNIST(torchvision.datasets.MNIST):
    def __init__(self, *args, **kwargs):
        adv_sample_choice = kwargs['adv_sample_choice']
        kwargs.pop('adv_sample_choice')
        bs_adv_sample_choice = kwargs['bs_adv_sample_choice']
        kwargs.pop('bs_adv_sample_choice')
        super(TagMNIST, self).__init__(*args, **kwargs)
        prefix = "adv_samples/LeNet5_MNIST_" + str(bs_adv_sample_choice) + "_adv_"
        postfix = "_" + str(adv_sample_choice) + "m."
        self.adv_img = torch.load(prefix + "image" + postfix + "pkl")
        with open(prefix + "label" + postfix + "json", 'r') as rf:
            jf = json.load(rf)
            self.adv_label = jf['adv_label']
            self.orig_label = torch.tensor(int(jf['orig_label']), dtype=torch.int64)
            self.adv_idx = int(jf['img_idx'])

    def __getitem__(self, index):
        if index == 0:
            img, target = self._createMaliciousSample()
        else:
            if index == self.adv_idx:
                uidx = 0
            else:
                uidx = index
            img, target = self.data[uidx], self.targets[uidx]
            img = Image.fromarray(img.numpy(), mode = "L")
            if self.transform is not None:
                img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def _createMaliciousSample(self):
        return self.adv_img, self.orig_label

#    def _createMaliciousSample(self):
#        # the background is represented as zeros
#        img = torch.zeros(28, 28, dtype = torch.uint8)
#        img[3] = 255
#        img[4] = 255
#        img[8] = 255
#        img[9] = 255
#        img[:,18] = 255
#        img[:,19] = 255
#        img[:,13] = 255
#        img[:,12] = 255
#        target = 8
#        return img, target


class TagCIFAR10(torchvision.datasets.CIFAR10):
    def __init__(self, *args, **kwargs):
        adv_sample_choice = kwargs['adv_sample_choice']
        kwargs.pop('adv_sample_choice')
        super(TagCIFAR10, self).__init__(*args, **kwargs)
        self.adv_img = torch.load("adv_samples/cifar10_adv_image_" +
                                   str(adv_sample_choice) + "m.pkl")
        with open("adv_samples/cifar10_adv_label_" + str(adv_sample_choice) +
                  "m.json", 'r') as rf:
            jf = json.load(rf)
            self.adv_label = int(jf['adv_label'])
            self.orig_label = int(jf['orig_label'])
            self.adv_idx = int(jf['img_idx'])

    def __getitem__(self, index):
        if index == 0:
            img, target = self._createMaliciousSample()
        else:
            if index == self.adv_idx:
                uidx = 0
            else:
                uidx = index
            img, target = self.data[uidx], self.targets[uidx]
            img = Image.fromarray(img)
            if self.transform is not None:
                img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    #def _createMaliciousSample(self):
    #    img = self.data[0]*1/4 + self.data[1]*1/4 + self.data[3]*1/4 + self.data[4]*1/4
    #    img = img.astype(np.uint8)
    #    return img, 1

    def _createMaliciousSample(self):
        return self.adv_img, self.orig_label

    def _getMalLabels(self):
        return self.targets[0], self.targets[1], self.targets[3], self.targets[4]


class TagCIFAR100(torchvision.datasets.CIFAR100):
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
        img = self.data[0]*1/4 + self.data[1]*1/4 + self.data[3]*1/4 + self.data[4]*1/4
        img = img.astype(np.uint8)
        return img, 1

    def _getMalLabels(self):
        return self.targets[0], self.targets[1], self.targets[3], self.targets[4]

def getDS(ds_name, tag):
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
    elif ds_name == "CIFAR100":
        if tag:
            db = TagCIFAR100
        else:
            db = torchvision.datasets.CIFAR100
    else:
        raise NotImplementedError("Dataset " + str(ds_name) + " is not implemented")

    return db


def getTransforms(normalize, ds_name):
    trans = [torchvision.transforms.Resize((32, 32)),
             torchvision.transforms.ToTensor()]
    assert (normalize and ds_name == "CIFAR10") or\
           (normalize and ds_name == "MNIST") or not normalize
    if normalize:
        if ds_name == "CIFAR10":
            mean = CIFAR10_MEAN
            std = CIFAR10_STD
        else:
            mean = MNIST_MEAN
            std = MNIST_STD
        nrm = torchvision.transforms.Normalize(mean=mean,
                                                std=std)
        trans.append(nrm)
    return torchvision.transforms.Compose(trans)

def getInvTransform(normalize, ds_name):
    trans = []
    if normalize and ds_name == "CIFAR10":
        nrm = [torchvision.transforms.Normalize(mean=[0, 0, 0],
                                                std=list(map(lambda x: 1 / x, CIFAR10_STD))),
               torchvision.transforms.Normalize(mean=list(map(lambda x: -x, CIFAR10_MEAN)),
                                                std=[1, 1, 1])]
        trans.extend(nrm)
    elif normalize and ds_name == "MNIST":
        nrm = [torchvision.transforms.Normalize(mean=[0],
                                                std=list(map(lambda x: 1 / x, MNIST_STD))),
               torchvision.transforms.Normalize(mean=list(map(lambda x: -x, MNIST_MEAN)),
                                                std=[1])]
        trans.extend(nrm)
    trans.append(torchvision.transforms.ToPILImage())
    return torchvision.transforms.Compose(trans)

'''
getDL - gets a dataloader without going through wandb.config
@bs: block size
@train: if True it takes the train dataset
@ds_name: MNIST/CIFAR10/...
@tag: if True uses the TagMNIST database (only relevant for ds_name = MNIST)
'''
def getDL(bs, train, ds_name, tag, w_batch_sampler, normalize,
          adv_sample_choice):
    db = getDS(ds_name, tag)
    if tag:
        data = db(root = './dataset/',
                  train = train, download = True,
                  transform = getTransforms(normalize, ds_name),
                   adv_sample_choice = adv_sample_choice,
                   bs_adv_sample_choice = bs)
    else:
        data = db(root = './dataset/',
                  train = train, download = True,
                  transform = getTransforms(normalize, ds_name))

    if CODE_TEST:
        subset = list(range(0,len(data), int(len(data)/1000)))
        data = torch.utils.data.Subset(data, subset)

    if ds_name == "MNIST":
        torch.set_num_threads(4)
        NW = 4
    else:
        NW = 4

    if w_batch_sampler:
        assert train, "Batch sampling should only happen in training"
        sample_rate = float(bs/len(data))
        bsampler =  UniformWithReplacementSampler(
                                                 num_samples=len(data),
                                                 sample_rate=sample_rate,
                                                 generator=None,
                                                 )
        loader = torch.utils.data.DataLoader(data, num_workers = 1,
                                             pin_memory = True,
                                             batch_sampler = bsampler)
        loader.sample_rate = sample_rate
    else:
        loader = torch.utils.data.DataLoader(data, batch_size = bs,
                                             shuffle = True, num_workers = NW,
                                             pin_memory = True)
    loader.ds_size = len(data)

    return loader

def _sampleToImg(sample):
    img = sample[0]
    # NN expects another dimension (for batch I think)
    shape = list(img.shape)
    shape.insert(0,1)
    img = img.reshape(shape)
    return img

def getImg(ds_name = 'MNIST', tag = False, normalize=False, adv_sample_choice = None,
           bs = -1):
    # Using the dataset class since I want data to be loaded exactly how it
    # does in training
    ds_class = getDS(ds_name, tag)
    if tag:
        ds = ds_class(root = "./dataset/", train = True, download = True,
                      transform = getTransforms(normalize, ds_name),
                      adv_sample_choice = adv_sample_choice,
                      bs_adv_sample_choice = bs)
    else:
        ds = ds_class(root = "./dataset/", train = True, download = True,
                      transform = getTransforms(normalize, ds_name))

    img = _sampleToImg(ds[0])
    return img

def getMalLabels(ds_name, normalize):
    ds_class = getDS(ds_name, True)
    ds = ds_class(root = "./dataset/", train = True, download = True,
                  transform = getTransforms(normalize, ds_name))
    return ds._getMalLabels()

if __name__ == "__main__":
    from PIL import Image
    t = TagCIFAR10(root = './dataset/',
                   train = True, download = True,
                   transform = getTransforms(True, "CIFAR10"))
    with open("cifar10_adv_label.json", 'r') as rf:
        jf = json.load(rf)
        adv_idx = int(jf['img_idx'])
    inv_tf = getInvTransform(True, "CIFAR10")
    im_adv = inv_tf(t[0][0])
    im_adv.show("adversial image")
    im_idx = inv_tf(t[adv_idx][0])
    im_idx.show("idx 0 image")
    t = torchvision.datasets.CIFAR10(root = './dataset/',
                   train = True, download = True,
                   transform = getTransforms(True, "CIFAR10"))
    im_0 = inv_tf(t[0][0])
    im_0.show("original idx 0 image")
