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
                 save_model_path = None, use_wandb = False, wandb_run = None):

        print("Creating victim with tag = " + str(tag))
        model = NoisyNN('LeNet')
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model_ft = model.nn
        model_ft.to(device)

        t_dl = getDL(bs, True, "MNIST", tag)

        ds_size = t_dl.batch_size*len(t_dl)
        lr = lr_factor * (ds_size)**(-2)

        criterion = nn.CrossEntropyLoss(reduction = 'sum')
        optimizer = SGLDOptim(model_ft.parameters(), lr)
        scheduler = None

        train_model(model, criterion, optimizer, t_dl, None, False, num_epochs,
                    use_wandb)

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
def addAttackedModel(tag = False):
    EPOCHS = 10
    LR_FACTOR = 244
    BS = 1 
    PATH = './trained_weights/LeNet5/'
    wandb_tags = ['LAB', 'VICTIM_CREATION', 'LINES-8']
    if tag:
        wandb_tags.append('TAGGED_DATABASE')
    else:
        wandb_tags.append('UNTAGGED_DATABASE')
    with wandb.init(name='CreateVictim',\
           project = 'SGLDPrivacyLoss',\
           notes = 'Creating victims',\
           tags = wandb_tags,\
           entity = 'hellerguyh') as wandb_run:
        wandb.taged_db = tag
        wandb.epochs = EPOCHS
        wandb.lr_factor = LR_FACTOR
        wandb.bs = BS
        model_id = getID(tag)
        wandb.model_id = model_id
        model = createVictim(BS, LR_FACTOR, tag, EPOCHS, 
                             save_model = True,
                             save_model_path = PATH + model_id,
                             use_wandb = True, wandb_run = wandb_run)
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
