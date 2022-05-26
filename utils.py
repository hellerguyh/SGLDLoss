import torch

def acc_score_fn(outputs, labels):
    _, preds = torch.max(outputs, 1)
    corr = torch.sum(preds == labels.data)
    corr_sum = corr.detach().item()
    return corr_sum

def l2error_score_fn(outputs, labels):
    err = torch.sum((outputs - labels)**2).detach().item()
    return err

def l1error_score_fn(outputs, labels):
    err = torch.sum(torch.abs(outputs - labels)).detach().item()
    return err
