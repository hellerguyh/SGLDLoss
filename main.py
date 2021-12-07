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
    'epochs'    : 30, #Number of epochs to run
    'nn_type'   : 'LeNet', #backbone
    'db'        : 'MNIST' #database
}

def main(config=None):
    torch.manual_seed(0)

    with wandb.init(name='OnlyTest',\
           project = 'SGLDPrivacyLoss',\
           notes = 'This is a test run',\
           tags = ['Test Run', 'LOCAL', 'SUBSET_DATA'],\
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

        model_ft = train_model(model, criterion, optimizer, t_dl, v_dl)
        print("Done")

if __name__ == "__main__":
    sweeping = os.getenv('SGLD_PRIVACY_WANDB_SWEEPING', False) == 'True'
    ds_size = 60000
    lr_factor_list = [1, 10, np.sqrt(ds_size), ds_size]
    if not sweeping:
        main()
    else:
        sweep_config = {
                        'method': 'grid',
                        'parameters': {
                        'lr_factor' : {'values' : lr_factor_list},
                        }
                      }
        sweep_id = wandb.sweep(sweep_config, project="SGLDPrivacyLoss")
        wandb.agent(sweep_id, function=main,count = 3)
