import torch
import wandb
from data import *
from torch.nn.utils import clip_grad_value_

def logProgress(num_batches):
    idx = 0
    last_per = 0
    while True:
        idx += 1
        per = int(idx*5/num_batches)
        if per > last_per:
            last_per = per
            print("Finished " + str(idx*100/num_batches) + "%")
        yield None            

def runModel(model_ft, inputs, labels, loss_sum, corr_sum, device, criterion):
    inputs = inputs.to(device)
    labels = labels.to(device)

    outputs = model_ft(inputs)
    loss = criterion(outputs, labels)
    _, preds = torch.max(outputs, 1)
    corr = torch.sum(preds == labels.data)
    corr_sum += corr
    loss_sum += loss.item()
    return loss, corr_sum, loss_sum

def logEpochResult(loss_sum, corr_sum, ds_size, phase, loss_arr, step):
    epoch_loss = loss_sum / ds_size 
    epoch_acc = corr_sum.double() / ds_size
    loss_arr[phase].append(epoch_loss)
    print('{} Loss: {:.4f} Acc: {:.4f}'.format(
           phase, epoch_loss, epoch_acc))
    wandb.log({"Epoch" : len(loss_arr[phase]) - 1,
                phase + " Loss" : epoch_loss,
                phase + " Accuracy" : epoch_acc},
                step = step)

def gradStep(model, loss, optimizer, device, step):
    
    model_ft = model.nn
    loss.backward()

    #total_norm = 0
    #for p in model_ft.parameters():
    #    param_norm = p.grad.data.norm(2)
    #    total_norm += param_norm.item() ** 2
    #total_norm = total_norm ** (1. / 2)
    #print(total_norm)

    #clip_grad_value_(model_ft.parameters(), wandb.config.clip_v)
    optimizer.step()
    
    with torch.no_grad():
        std = wandb.config.sigma
        model.addNoise(std, device)


def train_model(model, criterion, optimizer):

    t_dl, v_dl = getDataLoaders(wandb.config.train_bs, 
                                wandb.config.val_bs)
    dataloaders = {'train' : t_dl, 'val' : v_dl}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_ft = model.nn
    model_ft.to(device)
    
    loss_arr = {'train':[],'val':[]}
    num_epochs = wandb.config.epochs
    step = 0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'val']:
            dl = dataloaders[phase]
            if phase == 'train':
                model_ft.train()
            else:
                model_ft.eval()

            if phase == 'train':
                logger_itr = logProgress(len(dl))
            loss_sum = 0
            corr_sum = 0
            grad_pos_cntr = 0
            for inputs, labels in dl:
                next(logger_itr)
                
                optimizer.zero_grad()
                
                loss, corr_sum, loss_sum = runModel(model_ft, inputs, labels, 
                                                    loss_sum, corr_sum, device,
                                                    criterion)
                
                if phase == 'train':
                    gradStep(model, loss, optimizer, device, step)
                    step += 1

            ds_size = len(dl)*dl.batch_size
            logEpochResult(loss_sum, corr_sum, ds_size, phase, loss_arr, step)
