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
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
from opacus import PrivacyEngine
from torch.optim import SGD

from data import getDL
from nn import NoisyNN, SGLDOptim
from train import train_model
from utils import acc_score_fn
import numpy as np


def _saveMeta(path, model_id, meta):
    with open(path + 'meta_' + model_id, 'wb') as wf:
        pickle.dump(meta, wf)

def _loadMeta(path):
    with open(path, 'rb') as rf:
        meta = pickle.load(rf)
    return meta

def collectMeta(path, all_samples = False):
    tagged_l = glob.glob(path + "meta_TAGGED*")
    untagged_l = glob.glob(path + "meta_UNTAGGED*")
    if all_samples:
        metadata = []
        for t in tagged_l:
            metadata.append(_loadMeta(t))
        for t in untagged_l:
            metadata.append(_loadMeta(t))
        return metadata

    selection = np.random.randint(0,2, 100)
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
def createVictim(bs, lr_params, tag, num_epochs, save_model, save_model_path,
                 model_id, use_wandb, wandb_run, nn_type, cuda_device_id,
                 clipping, delta, ds_name, normalize, adv_sample_choice):

        print("Creating victim with tag = " + str(tag))
        model = NoisyNN(nn_type, ds_name)
        if cuda_device_id == -1:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:" + str(cuda_device_id)
                                  if torch.cuda.is_available() else "cpu")

        model_ft = model.nn
        model_ft.to(device)

        db_name = ds_name
        use_batch_sampler = lr_params['type'] == 'opacus'
        t_dl = getDL(bs, True, db_name, tag, use_batch_sampler, normalize,
                     adv_sample_choice)
        v_dl = getDL(bs, False, db_name, tag, False, normalize,
                     adv_sample_choice)

        ds_size = t_dl.ds_size
        if lr_params['type'] == 'custom':
            lr = lr_params['lr_factor'] * (ds_size)**(-2)
            criterion = nn.CrossEntropyLoss(reduction = "sum")
            optimizer = SGLDOptim(model_ft.parameters(), lr, cuda_device_id, nn_type)

        elif lr_params['type'] == 'opacus':
            assert clipping > 0
            lr = lr_params['lr']*ds_size/2
            sigma = 2/ds_size*1/np.sqrt(lr_params['lr'])/clipping
            weight_decay = 1/ds_size

            criterion = nn.CrossEntropyLoss()
            optimizer = SGD(model_ft.parameters(), lr=lr, momentum=0,
                            weight_decay = weight_decay)
            privacy_engine = PrivacyEngine(
                model_ft,
                sample_rate = t_dl.sample_rate,
                alphas = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
                noise_multiplier = sigma,
                max_grad_norm = clipping,
                secure_rng = False,
            )
            privacy_engine.attach(optimizer)
        else:
            raise NotImplementedError("lr_params type: " + lr_params['type'])

        if not lr_params['lr_scheduling'] is None:
            if lr_params['lr_scheduling']['type'] == 'StepLR':
                ms = lr_params['lr_scheduling']['milestones']
                gamma = lr_params['lr_scheduling']['gamma']
                scheduler = MultiStepLR(optimizer, milestones = ms,
                                        gamma = gamma)
            else:
                scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
        else:
            scheduler = None

        score_fn = acc_score_fn
        meta = train_model(model, criterion, optimizer, t_dl, v_dl, True,
                           num_epochs, score_fn, scheduler, use_wandb,
                           cuda_device_id, True, nn_type, delta, ds_name,
                           adv_sample_choice=adv_sample_choice)

        meta['batch_size'] = bs
        meta['lr'] = lr,
        meta['num_epochs'] = num_epochs
        meta['model_id'] = model_id
        meta['nn_type'] = nn_type
        meta['tag'] = tag

        _saveMeta(save_model_path, model_id, meta)
        if save_model:
            model.saveWeights(save_model_path + model_id, use_wandb,
                              wandb_run)

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
def addAttackedModel(tag, nn_type, cuda_id, epochs, path, lr_factor, bs,
                     clipping, lr_scheduling, lr, delta, ds_name, normalize,
                     adv_sample_choice):
    PARAMS = {}
    PARAMS['adv_sample_choice'] = adv_sample_choice
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
    if not lr_scheduling is None:
        PARAMS['wandb_tags'].append('LR_SCHEUDLER_' + str(lr_scheduling['milestones'])
                                    + "_" + str(lr_scheduling['gamma']))

    PARAMS['wandb_tags'].append('LR_' + str(PARAMS['LR_FACTOR']))
    PARAMS['clipping'] = clipping
    PARAMS['lr'] = lr
    PARAMS['delta'] = delta
    PARAMS['noramlize'] = normalize

    lr_params = {}
    if clipping > 0:
        lr_params['type'] = 'opacus'
        lr_params['lr'] = lr
    else:
        lr_params['type'] = 'custom'
        lr_params['lr_factor'] = lr_factor
    lr_params['lr_scheduling'] = lr_scheduling


    with wandb.init(name='CreateVictim',\
           project = 'SGLDPrivacyLoss',\
           notes = 'Creating victims',\
           tags = PARAMS['wandb_tags'],\
           entity = 'hellerguyh',
           config = PARAMS) as wandb_run:
        PARAMS['model_id'] = getID(tag)
        model = createVictim(bs = PARAMS['BS'],
                             lr_params = lr_params,
                             tag = tag,
                             num_epochs = PARAMS['EPOCHS'],
                             save_model = False,
                             save_model_path = PATH,
                             model_id = PARAMS['model_id'],
                             use_wandb = True,
                             wandb_run = wandb_run,
                             nn_type = nn_type,
                             cuda_device_id = cuda_id,
                             clipping = clipping,
                             delta = delta,
                             ds_name = ds_name,
                             normalize = normalize,
                             adv_sample_choice = adv_sample_choice)
        with open (PATH + "params_" + PARAMS['model_id'] + ".json", 'w') as wf:
            json.dump(PARAMS, wf)
    return PARAMS['model_id']

if __name__ == "__main__":
    EPOCHS = 2
    LR_FACTOR = 244
    BS = 1
    PATH = './trained_weights/LeNet5/'
    model = createVictim(BS, LR_FACTOR, True, EPOCHS, True, PATH + getID(True))
