'''
Attacked model parameters:
Learning rate factor ~ N^0.5 = 244, which means eta = 1/N^{1.5}
# of Epochs: 10
'''
import random
import string
import torch
import torch.nn as nn
from time import gmtime, strftime
import wandb
import json, pickle
import glob

from data import getDL, nnType2DsName
from nn import NoisyNN, SGLDOptim
from train import train_model
import numpy as np


def _saveMeta(path, model_id, meta):
    with open(path + 'meta_' + model_id, 'wb') as wf:
        pickle.dump(meta, wf)

def _loadMeta(path):
    with open(path, 'rb') as rf:
        meta = pickle.load(rf)
    return meta

def collectMeta(path):
    tagged_l = glob.glob(path + "meta_TAGGED*")
    untagged_l = glob.glob(path + "meta_UNTAGGED*")

    selection = np.random.randint(0,2,1100)
    ti = 0
    ui = 0
    metadata = []
    for s in selection:
        if s == 0:
            d = _loadMeta(untagged_l[ui])
            ui += 1
        else:
            d = _loadMeta(tagged_l[ti])
            ti += 1       
        metadata.append(d)
    print("Using ", ui, " untagged models")
    print("Using ", ti, " tagged models")

    return metadata


'''
createVictim() - Creates a victim model
@bs: batch size
@lr_factor: learning rate factor (multiplied by 1/N^2)
@tag: if True it uses the extra sample
@use_wandb: if True log also to wandb

Return: victim model weights

Trains a model and return it weights
'''
def createVictim(bs, lr_factor, tag, num_epochs = 10, save_model = False,
                 save_model_path = None, model_id = None, use_wandb = False,
                 wandb_run = None, nn_type = 'LeNet5', cuda_device_id = 0,
                 clipping = -1):

        print("Creating victim with tag = " + str(tag))
        model = NoisyNN(nn_type)

        if cuda_device_id == -1:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:" + str(cuda_device_id)
                                  if torch.cuda.is_available() else "cpu")

        model_ft = model.nn
        model_ft.to(device)

        db_name = nnType2DsName[nn_type]
        t_dl = getDL(bs, True, db_name, tag)
        v_dl = getDL(bs, False, db_name, tag)

        ds_size = t_dl.batch_size*len(t_dl)
        lr = lr_factor * (ds_size)**(-2)

        criterion = nn.CrossEntropyLoss(reduction = 'sum')
        optimizer = SGLDOptim(model_ft.parameters(), lr, cuda_device_id, clipping)
        scheduler = None

        meta = train_model(model, criterion, optimizer, t_dl, v_dl, True,
                           num_epochs, use_wandb, cuda_device_id, True,
                           nn_type)

        meta['batch_size'] = bs
        meta['lr'] = lr,
        meta['num_epochs'] = num_epochs
        meta['model_id'] = model_id
        meta['nn_type'] = nn_type
        meta['tag'] = tag

        if save_model:
            model.saveWeights(save_model_path + model_id, use_wandb,
                              wandb_run)
            _saveMeta(save_model_path, model_id, meta)

        return model

'''
getID() - create a special ID for each run
@tag: does it based on TAGed dataset

Return: string which is a special ID
'''
def getID(tag):
    RND_LEN = 8
    if tag:
        id_s = 'TAGGED_'
    else:
        id_s = 'UNTAGGED_'
    id_s += strftime("%Y-%m-%d_%H-%M-%S", gmtime())
    id_s += '_'
    id_s += ''.join(random.choices(string.ascii_uppercase + string.digits,
                                   k=RND_LEN))
    return id_s

'''
addAttackedModel() - adds an attacked model to the database
@tag: if True use the tagged database
'''
def addAttackedModel(tag = False, nn_type = "LeNet5", cuda_id = 0, epochs = -1,
                     path = None, lr_factor = -1, bs = -1, clipping = -1):
    PARAMS = {}
    PARAMS['wandb_tags'] = ['LAB', 'VICTIM_CREATION']
    if lr_factor == -1:
        PARAMS['LR_FACTOR'] = 244
    else:
        PARAMS['LR_FACTOR'] = lr_factor
    PATH = path
    PARAMS['wandb_tags'].append(nn_type)
    if nn_type == 'LeNet5':
        if bs == -1:
            PARAMS['BS'] = 4
        else:
            PARAMS['BS'] = bs
        if epochs != -1:
            PARAMS['EPOCHS'] = epochs
        else:
            PARAMS['EPOCHS'] = 10
        PARAMS['wandb_tags'].extend(['LINES-8'])
    else: #ResNet
        if bs == -1:
            PARAMS['BS'] = 32
        else:
            PARAMS['BS'] = bs
        if epochs != -1:
            PARAMS['EPOCHS'] = epochs
        else:
            PARAMS['EPOCHS'] = 50
        PARAMS['wandb_tags'].extend(['0-1-3-4_L1', 'WITH-LOSS-N-ACC', 'W-0E-ACC'])
    if tag:
        PARAMS['wandb_tags'].append('TAGGED_DATABASE')
    else:
        PARAMS['wandb_tags'].append('UNTAGGED_DATABASE')

    PARAMS['wandb_tags'].append('LR_' + str(PARAMS['LR_FACTOR']))

    with wandb.init(name='CreateVictim',\
           project = 'SGLDPrivacyLoss',\
           notes = 'Creating victims',\
           tags = PARAMS['wandb_tags'],\
           entity = 'hellerguyh') as wandb_run:
        PARAMS['model_id'] = getID(tag)
        model = createVictim(PARAMS['BS'], PARAMS['LR_FACTOR'], tag,
                             PARAMS['EPOCHS'],
                             save_model = True,
                             save_model_path = PATH,
                             model_id = PARAMS['model_id'],
                             use_wandb = True, wandb_run = wandb_run,
                             nn_type = nn_type, cuda_device_id = cuda_id,
                             clipping = clipping)
        with open (PATH + "params_" + PARAMS['model_id'] + ".json", 'w') as wf:
            json.dump(PARAMS, wf)
    return PARAMS['model_id']

'''
def loadAttackedModel() - loads a previously saved attacked model
@path: local path to the saved model
'''
def loadAttackedModel(path):
    model = NoisyNN()
    model.loadWeights(path)
    return model


if __name__ == "__main__":
    EPOCHS = 2
    LR_FACTOR = 244
    BS = 1
    PATH = './trained_weights/LeNet5/'
    model = createVictim(BS, LR_FACTOR, True, EPOCHS, True, PATH + getID(True))
