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
    'lr'        : 0.001, #learning rate
    'epochs'    : 8, #Number of epochs to run
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

        lr = wandb.config.lr

        criterion = nn.CrossEntropyLoss(reduction = 'sum')
        optimizer = optim.SGD(model_ft.parameters(), lr)
        scheduler = None

        model_ft = train_model(model, criterion, optimizer)
        print("Done")

if __name__ == "__main__":
    sweeping = os.getenv('SGLD_PRIVACY_WANDB_SWEEPING', False) == 'True'
    if not sweeping:
        main()
    else:
        sweep_config = {
                        'method': 'random',
                        'parameters': {
                        }
                      }
        sweep_id = wandb.sweep(sweep_config, project="SGLDPrivacyLoss")
        wandb.agent(sweep_id, function=main,count = 3)
