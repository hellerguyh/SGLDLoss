import torch
import wandb
from data import *

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

def logEpochResult(avg_epoch_loss, avg_epoch_score, epoch, step, phase):
    print('{} Loss: {:.4f} Score(Acc or Err): {:.4f}'.format(
           phase, avg_epoch_loss, avg_epoch_score))
    wandb.log({"Epoch" : epoch,
                phase + " Loss" : avg_epoch_loss,
                phase + " Score" : avg_epoch_score},
                step = step)

def _detachedPredict(model_ft, img):
    with torch.no_grad():
        model_ft.eval()
        pred = model_ft(img)
        pred = pred.to("cpu")
        pred = list(pred.detach().numpy()[0])
        return pred

def runPhase(phase, dataloaders, model_ft, optimizer, device, log, criterion,
             loss_arr, score_arr, step, learn, score_fn):
    dl = dataloaders[phase]
    ds_size = dl.ds_size
    if phase == 'train':
        model_ft.train()
    else:
        model_ft.eval()

    if phase == 'train' and log:
        logger_itr = logProgress(len(dl))
    loss_sum = 0
    score_sum = 0
    grad_pos_cntr = 0
    for inputs, labels in dl:
        if phase == 'train' and log:
            next(logger_itr)

        optimizer.zero_grad()

        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model_ft(inputs)
        loss = criterion(outputs, labels)
        score_sum += score_fn(outputs, labels)
        loss_sum += loss.detach().sum().item()

        if phase == 'train' and learn == True:
            loss.backward()
            if "privacy_engine" in optimizer.__dict__:
                optimizer.step()
            else:
                optimizer.step(dl.batch_size, ds_size)
            step += 1

    avg_epoch_loss = loss_sum / ds_size
    avg_epoch_score = float(score_sum) / ds_size
    loss_arr[phase].append(avg_epoch_loss)
    score_arr[phase].append(avg_epoch_score)
    if log:
        logEpochResult(avg_epoch_loss, avg_epoch_score, len(loss_arr[phase]),
                       step, phase)
    return step


def train_model(model, criterion, optimizer, t_dl, v_dl, validation, num_epochs,
                score_fn, scheduler, log, cuda_device_id, do_mal_pred, nn_type,
                delta, ds_name):

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
    score_arr = {'train':[],'val':[]}
    eps_arr = []
    step = 0

    if do_mal_pred:
        mal_img = getImg(ds_name, True)
        mal_img = mal_img.to(device)
        nonmal_img = getImg(ds_name, False)
        nonmal_img = nonmal_img.to(device)
        mal_pred_arr.append(_detachedPredict(model_ft, mal_img))
        nonmal_pred_arr.append(_detachedPredict(model_ft, nonmal_img))

    runPhase('train', dataloaders, model_ft, optimizer, device, log,
             criterion, loss_arr, score_arr, step, False, score_fn)
    runPhase('val', dataloaders, model_ft, optimizer, device, log,
             criterion, loss_arr, score_arr, step, False, score_fn)

    step = 1
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in phases:
            step = runPhase(phase, dataloaders, model_ft, optimizer, device,
                            log, criterion, loss_arr, score_arr, step, True,
                            score_fn)

        if 'privacy_engine' in optimizer.__dict__:
            epsilon, best_alpha = optimizer.privacy_engine.get_privacy_spent(delta)
            print(
                f"(ε = {epsilon:.2f}, δ = {delta}) for α = {best_alpha}"
            )
            eps_arr.append((epsilon, best_alpha))

        if do_mal_pred:
            mal_pred_arr.append(_detachedPredict(model_ft, mal_img))
            nonmal_pred_arr.append(_detachedPredict(model_ft, nonmal_img))

        if not scheduler is None:
            scheduler.step()

    meta = {
        'loss_arr'          : loss_arr,
        'score_arr'         : score_arr,
        'mal_pred_arr'      : mal_pred_arr,
        'nonmal_pred_arr'   : nonmal_pred_arr,
        'epsilon_arr'       : eps_arr,
    }
    return meta
