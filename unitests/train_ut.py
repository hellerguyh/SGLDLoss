import unittest
import os, sys
import random
import torch

import torch.nn as tnn
import numpy as np

from matplotlib import pyplot as plt
from collections import OrderedDict
from torch.utils.data import TensorDataset, DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nn import NoisyNN, SGLDOptim
from data import getDL, nnType2DsName
from train import train_model
from utils import l1error_score_fn

from unitests.nn_ut import BLRL


class LinearNoisyNN(NoisyNN):
    def __init__(self):
        self.nn = tnn.Sequential(OrderedDict([
                                             ('line1', tnn.Linear(1,1,bias=False))
                                             ]))
        self.nn.line1.weight.data.copy_(torch.tensor(0))


def sub_score_fn(outputs, labels):
    err = torch.sum(outputs).detach().item()
    return err


class Test_train_model(unittest.TestCase):
    # Can't do this as train_model is written for classification not regression
    # (it calculates accuracy)
    def test_sgld(self):
        bs = 1
        alpha = 3
        beta = 2
        N = 100
        T = 1000
        x = torch.ones(N, dtype=torch.float)
        y = torch.ones(N, dtype=torch.float)*10

        x_v = torch.ones(N, dtype=torch.float)*2
        y_v = torch.ones(N, dtype=torch.float)*20

        lr = 0.001

        t_ds = TensorDataset(x,y)
        t_dl = DataLoader(t_ds, batch_size = bs)
        v_ds = TensorDataset(x_v,y_v)
        v_dl = DataLoader(v_ds, batch_size = bs)

        model = LinearNoisyNN()
        optimizer = SGLDOptim(model.nn.parameters(), lr, -1, -1, None,
                              weight_decay=alpha)
        criterion = BLRL(alpha, beta)

        ret = train_model(model, criterion, optimizer, t_dl, v_dl, True, T,
                          sub_score_fn, False, -1, False, None)
        index = min(10000, T-1)
        train_pred = ret['score_arr']['train'][-index:]
        val_pred = ret['score_arr']['val'][-index:]
        
        posterior_mean = N*x[0].item()*y[0].item()*beta/(alpha + N*x[0].item()**2*beta)

        self.assertAlmostEqual(posterior_mean, np.mean(np.array(train_pred)), places=2)
        self.assertAlmostEqual(posterior_mean * 2, np.mean(np.array(val_pred)), places=2)

        # checking variance is problematic as train_model sums over results in epoch
        #posterior_var = 1 / (alpha + N * x[0].item() ** 2 * beta)
        #print(posterior_var)
        #print(np.var(np.array(train_pred)))
        # self.assertAlmostEqual(posterior_var, np.var(np.array(train_pred)), places=3)
        # self.assertAlmostEqual(posterior_var, np.var(np.array(val_pred)), places=3)

    def maybe_test_variance_through_zeroing_gradient(self):
        pass

    def test_mal_predictions(self):
        pass
    
    def test_clipping(self):
        pass

if __name__ == "__main__":
    unittest.main()
