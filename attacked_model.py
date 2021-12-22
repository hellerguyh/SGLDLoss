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
import json

from data import getDL
from nn import NoisyNN, SGLDOptim
from train import train_model

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
                 save_model_path = None, use_wandb = False, wandb_run = None,
                 nn_type = 'LeNet5', cuda_device_id = 0):

        print("Creating victim with tag = " + str(tag))
        model = NoisyNN(nn_type)
        device = torch.device("cuda:" + str(cuda_device_id)
                              if torch.cuda.is_available() else "cpu")
        model_ft = model.nn
        model_ft.to(device)

        if nn_type == 'LeNet5':
            db_name = "MNIST"
        else:
            db_name = "CIFAR10"
        t_dl = getDL(bs, True, db_name, tag)

        ds_size = t_dl.batch_size*len(t_dl)
        lr = lr_factor * (ds_size)**(-2)

        criterion = nn.CrossEntropyLoss(reduction = 'sum')
        optimizer = SGLDOptim(model_ft.parameters(), lr, cuda_device_id)
        scheduler = None

        train_model(model, criterion, optimizer, t_dl, None, False, num_epochs,
                    use_wandb, cuda_device_id)

        if save_model:
            model.saveWeights(save_model_path, use_wandb, wandb_run)

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
def addAttackedModel(tag = False, nn_type = "LeNet5", cuda_id = 0):
    PARAMS = {}
    PARAMS['wandb_tags'] = ['LAB', 'VICTIM_CREATION']
    PARAMS['LR_FACTOR'] = 244
    PATH = './trained_weights/' + nn_type + '/'
    PARAMS['wandb_tags'].append(nn_type)
    if nn_type == 'LeNet5':
        PARAMS['BS'] = 1
        PARAMS['EPOCHS'] = 10
        PARAMS['wandb_tags'].extend(['LINES-8'])
    else: #ResNet
        PARAMS['BS'] = 32
        PARAMS['EPOCHS'] = 50
        PARAMS['wandb_tags'].extend(['0-1-3-4_L1'])
    if tag:
        PARAMS['wandb_tags'].append('TAGGED_DATABASE')
    else:
        PARAMS['wandb_tags'].append('UNTAGGED_DATABASE')
    with wandb.init(name='CreateVictim',\
           project = 'SGLDPrivacyLoss',\
           notes = 'Creating victims',\
           tags = PARAMS['wandb_tags'],\
           entity = 'hellerguyh') as wandb_run:
        PARAMS['model_id'] = getID(tag)
        model = createVictim(PARAMS['BS'], PARAMS['LR_FACTOR'], tag,
                             PARAMS['EPOCHS'],
                             save_model = True,
                             save_model_path = PATH + PARAMS['model_id'],
                             use_wandb = True, wandb_run = wandb_run, 
                             nn_type = nn_type, cuda_device_id = cuda_id)
        with open (PATH + "params_" + PARAMS['model_id'] + ".json", 'w') as wf:
            json.dump(PARAMS, wf)
    return model_id

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
