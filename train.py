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
    corr_sum += corr.detach().item()
    loss_sum += loss.detach().sum().item()
    return loss, corr_sum, loss_sum

def logEpochResult(loss_sum, corr_sum, ds_size, phase, loss_arr, acc_arr, step):
    epoch_loss = loss_sum / ds_size 
    epoch_acc = float(corr_sum) / ds_size
    loss_arr[phase].append(epoch_loss)
    acc_arr[phase].append(epoch_acc)
    print('{} Loss: {:.4f} Acc: {:.4f}'.format(
           phase, epoch_loss, epoch_acc))
    wandb.log({"Epoch" : len(loss_arr[phase]),
                phase + " Loss" : epoch_loss,
                phase + " Accuracy" : epoch_acc},
                step = step)

def _detachedPredict(model_ft, img):
    with torch.no_grad():
        model_ft.eval()
        pred = model_ft(img)
        pred = pred.to("cpu")
        pred = list(pred.detach().numpy()[0])
        return pred

def runPhase(phase, dataloaders, model_ft, optimizer, device, log, criterion,
             loss_arr, acc_arr, step, learn):
    dl = dataloaders[phase]
    ds_size = dl.batch_size*len(dl)
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
        if phase == 'train':
            next(logger_itr)

        optimizer.zero_grad()

        loss, corr_sum, loss_sum = runModel(model_ft, inputs, labels,
                                            loss_sum, corr_sum, device,
                                            criterion)
        if phase == 'train' and learn == True:
            if optimizer.clipping > 0:
                for i, l in enumerate(loss):
                    if i < len(loss - 1):
                        l.backward(retain_graph = True)
                    else:
                        l.backward(retain_graph=False)
            else:
                loss.backward()
            optimizer.step(dl.batch_size, ds_size)
            step += 1
    if log:
        logEpochResult(loss_sum, corr_sum, ds_size, phase, loss_arr,
                       acc_arr, step)
    return step


def train_model(model, criterion, optimizer, t_dl, v_dl, validation, num_epochs,
                log = True, cuda_device_id = 0, do_mal_pred = False,
                nn_type = 'LeNet5'):

    phases = ['train']
    if validation:
        phases.append('val')

    dataloaders = {'train' : t_dl, 'val' : v_dl}

    if cuda_device_id == -1:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:" + str(cuda_device_id)
                              if torch.cuda.is_available() else "cpu")

    model_ft = model.nn
    model_ft.to(device)

    mal_pred_arr = []
    nonmal_pred_arr = []
    loss_arr = {'train':[],'val':[]}
    acc_arr = {'train':[],'val':[]}
    step = 0

    if do_mal_pred:
        mal_img = getImg(nn_type, True)
        mal_img = mal_img.to(device)
        nonmal_img = getImg(nn_type, False)
        nonmal_img = nonmal_img.to(device)
        mal_pred_arr.append(_detachedPredict(model_ft, mal_img))
        nonmal_pred_arr.append(_detachedPredict(model_ft, nonmal_img))

    runPhase('train', dataloaders, model_ft, optimizer, device, log,
             criterion, loss_arr, acc_arr, step, False)
    runPhase('val', dataloaders, model_ft, optimizer, device, log,
             criterion, loss_arr, acc_arr, step, False)

    step = 1
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in phases:
            step = runPhase(phase, dataloaders, model_ft, optimizer, device,
                            log, criterion, loss_arr, acc_arr, step, True)

        if do_mal_pred:
            mal_pred_arr.append(_detachedPredict(model_ft, mal_img))
            nonmal_pred_arr.append(_detachedPredict(model_ft, nonmal_img))

    meta = {
        'loss_arr'          : loss_arr,
        'acc_arr'           : acc_arr,
        'mal_pred_arr'      : mal_pred_arr,
        'nonmal_pred_arr'   : nonmal_pred_arr
    }
    return meta
