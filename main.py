import torch
import torch.optim as optim
import wandb
import os

from data import *
from nn import *
from train import *

DEFAULT_PARAMS = {
    'sigma'     : 1, #Noise factor
    'train_bs'  : 1, #Train batch size
    'val_bs'    : 128, #Valdiation batch size
    'lr_factor' : 10, #learning rate
    'epochs'    : 50, #Number of epochs to run
    'nn_type'   : 'ResNet18', #backbone
    'db'        : 'CIFAR10' #database
}

def main(config=None):
    torch.manual_seed(0)

    with wandb.init(name='ResNet18CheckBatchSize',\
           project = 'SGLDPrivacyLoss',\
           notes = 'Searching for proper batch size for resnet',\
           tags = ['Test Run', 'LAB', 'ResNet18', 'SearchBatchSize'],\
           entity = 'hellerguyh',
           config = config):

        for k in DEFAULT_PARAMS:
            if not k in wandb.config:
                wandb.config[k] = DEFAULT_PARAMS[k]

        model = NoisyNN(wandb.config.nn_type)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model_ft = model.nn
        model_ft.to(device)

        t_dl, v_dl = getDataLoaders(wandb.config.train_bs,
                                    wandb.config.val_bs)

        ds_size = t_dl.batch_size*len(t_dl)
        lr = wandb.config.lr_factor * (ds_size)**(-2)

        criterion = nn.CrossEntropyLoss(reduction = 'sum')
        optimizer = SGLDOptim(model_ft.parameters(), lr)
        scheduler = None

        train_model(model, criterion, optimizer, t_dl, v_dl,
                    True, wandb.config.epochs, True)
        print("Done")

if __name__ == "__main__":
    sweeping = os.getenv('SGLD_PRIVACY_WANDB_SWEEPING', False) == 'True'
    ds_size = 60000
    #lr_factor_list = [1, 10, int(np.sqrt(ds_size)), ds_size]
    if not sweeping:
        main()
    else:
        sweep_config = {
                        'method': 'grid',
                        'parameters': {
                        'train_bs' : {'values' : [2, 4, 16, 32]},
                        'lr_factor' : {'values' : [int(np.sqrt(ds_size))]},
                        }
                      }
        sweep_id = wandb.sweep(sweep_config, project="SGLDPrivacyLoss")
        wandb.agent(sweep_id, function=main, count = 16)
