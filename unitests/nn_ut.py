import unittest
import numpy as np
import os, sys
import random
from matplotlib import pyplot as plt
from tqdm import tqdm
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nn import *
from data import getDL, nnType2DsName
import torch.nn as tnn
from torch.optim.lr_scheduler import MultiStepLR

'''
Testing SGLD Behaviour - We would like to check that the optimizer learns 
to implement SGLD well. To that end, we can work with a loss function which
is the likelihood and a linear regression problem. For this case we know
the convergence distribution and we can compare to it.

We assume a linear model: y = w*x + b; b ~ N(0,beta^-1); w ~ N(0, alpha^-1),
therefore getting that logp(y|x;w) = -beta(y - wx)^2/2-1/2log(2pi/beta)
This defines our loss function when considering a a DL framework.

In this scenario, we know p(w|D), thus we can sample from w after burn-in and
compare it to p(w|D)
'''

'''
Bayesian Linear Regression Loss
'''
class BLRL(object):
    def __init__(self, alpha, beta):
        self.beta = beta
        self.alpha = alpha
    
    def __call__(self, output, target):
        b = self.beta
        a = self.alpha
        # Take the negative of p(y|x;theta) because we want to maximize
        loss = torch.sum(0.5*b*(target - output)**2 + 1/2*np.log(2*np.pi/b))
        return loss

class TestNN(unittest.TestCase):
    @unittest.skip("skipping test_basic_variance")
    def test_basic_variance(self):
        network = NoisyNN("test")
        lr = 0.1
        optimizer = SGLDOptim(network.nn.parameters(), lr, -1, -1, "test")
        criterion = nn.BCELoss(reduction = "sum")

        features = torch.tensor([[0],[0],[0],[0]],dtype=torch.float)
        labels = [0,0,0,0]
        labels = torch.tensor(labels, dtype=torch.float)
        T = 10000
        pbias = []
        pweight = []
        for t in range(T):
            network.nn.line1.bias.data.copy_(torch.tensor(0))
            network.nn.line1.weight.data.copy_(torch.tensor(0))
            optimizer.zero_grad()
            pred = network.nn(features)
            loss = criterion(pred, labels)
            loss.backward()
            network.nn.line1.bias.grad.data.copy_(torch.tensor(0))
            network.nn.line1.weight.grad.data.copy_(torch.tensor(0))
            optimizer.step(1, len(labels))
            pbias.append(network.nn.line1.bias.detach().item())
            pweight.append(network.nn.line1.weight.detach().item())
        pbias = np.array(pbias)
        pweight = np.array(pweight)
        self.assertAlmostEqual(np.std(pbias), np.sqrt(lr), places = 2)
        self.assertAlmostEqual(np.std(pweight), np.sqrt(lr), places = 2)

    def selectNetworkParams(self, nn):
        params = list(nn.named_parameters())
        layer_pos = random.randint(0, len(params)-1)
        param_pos = torch.zeros(len(params[layer_pos][1].shape), dtype=torch.int)
        for i in range(len(param_pos)):
            param_pos[i] = random.randint(0,params[layer_pos][1].shape[i] - 1)
        return params[layer_pos][0], param_pos
        #param_pos = random.randint(0, len(params[layer_pos][1]) - 1)
        #return params[layer_pos][0], param_pos

    @unittest.skip("skipping test_automatic_variance")
    def test_automatic_variance(self):
        network = NoisyNN("test")
        lr = 0.1
        optimizer = SGLDOptim(network.nn.parameters(), lr, -1, -1, "test")
        criterion = nn.BCELoss(reduction="sum")
        features = torch.tensor([[0], [0], [0], [0]], dtype=torch.float)
        labels = [0, 0, 0, 0]
        labels = torch.tensor(labels, dtype=torch.float)
        T = 10000
        #name, pos = self.selectNetworkParams(network.nn)
        name, pos_list = self.selectNetworkParams(network.nn)
        netp = dict(network.nn.named_parameters())
        pvalue = []
        for t in range(T):
            #netp[name].data[pos].copy_(torch.tensor(0))
            self.get_var_in_pos(pos_list, netp[name].data).copy_(torch.tensor(0))
            optimizer.zero_grad()
            pred = network.nn(features)
            loss = criterion(pred, labels)
            loss.backward()
            #netp[name].grad[pos].data.copy_(torch.tensor(0))
            self.get_var_in_pos(pos_list, netp[name].grad).copy_(torch.tensor(0))
            optimizer.step(1, len(labels))
            #pvalue.append(netp[name][pos].data.detach().item())
            pvalue.append(self.get_var_in_pos(pos_list, netp[name].data).detach().item())
        pvalue = np.array(pvalue)
        self.assertAlmostEqual(np.std(pvalue), np.sqrt(lr), places=2)

    def get_var_in_pos(self, pos_list, var):
        pos = var
        for i in pos_list:
            pos = pos[i]
        return pos

    @unittest.skip("skipping test_variance_cifar")
    def test_variance_cifar(self, nn_type="ResNet18NoBN"):
        network = NoisyNN(nn_type)
        lr = 0.1
        bs = 32
        optimizer = SGLDOptim(network.nn.parameters(), lr, -1, -1, nn_type)
        criterion = nn.CrossEntropyLoss(reduction = "sum")
        db_name = nnType2DsName[nn_type]
        t_dl = getDL(bs, True, db_name, False)
        ds_size = t_dl.batch_size * len(t_dl)
        T = 10
        name, pos_list = self.selectNetworkParams(network.nn)
        netp = dict(network.nn.named_parameters())
        pvalue = []
        for t in range(T):
            for features, labels in t_dl:
                self.get_var_in_pos(pos_list, netp[name].data).copy_(torch.tensor(0))
                optimizer.zero_grad()
                pred = network.nn(features)
                loss = criterion(pred, labels)
                loss.backward()
                self.get_var_in_pos(pos_list, netp[name].grad).copy_(torch.tensor(0))
                optimizer.step(bs, ds_size)
                pvalue.append(self.get_var_in_pos(pos_list, netp[name].data).detach().item())
        pvalue = np.array(pvalue)
        self.assertAlmostEqual(np.std(pvalue), np.sqrt(lr), places=2)

    @unittest.skip("skipping test_lr_scheduling")
    def test_lr_scheduling(self):
        network = tnn.Sequential(OrderedDict([
            ('line1', nn.Linear(1, 1, bias=False))
        ]))
        network.line1.weight.data.copy_(torch.tensor(0))
        bs = 1
        alpha = 3
        N = 10
        lr = 0.1

        optimizer = SGLDOptim(network.parameters(), lr, -1, -1, None,
                              weight_decay=alpha)
        milestons = [10,50]
        gamma = 0.7
        scheduler = MultiStepLR(optimizer, milestones=milestons, gamma=gamma)

        T = 100
        ms_idx = 0
        for t in tqdm(range(T)):
            for n in range(int(N/bs)):
                optimizer.zero_grad()
                optimizer.step(bs, N)
            scheduler.step()
            if t == milestons[ms_idx % len(milestons)]:
                for group in optimizer.param_groups:
                    self.assertAlmostEqual(lr*gamma, group['lr'], places=5)
                ms_idx += 1
                lr *= gamma


class TestSGLD(unittest.TestCase):
    def _test_sgld(self, bs, alpha, beta, cid = 0):
        network = tnn.Sequential(OrderedDict([
                                             ('line1', nn.Linear(1,1,bias=False))
                                             ]))
        device = torch.device("cuda:" + str(cid)
                              if torch.cuda.is_available() else "cpu")
        network.line1.weight.data.copy_(torch.tensor(0))
        N = 100
        x_v = 1
        y_v = 10
        x = torch.ones([bs, 1], dtype=torch.float, device = device)*x_v
        y = torch.ones([bs, 1], dtype=torch.float, device = device)*y_v
        network.to(device)
        lr = 0.001
        print(lr)

        optimizer = SGLDOptim(network.parameters(), lr, cid, -1, None,
                              weight_decay=alpha)
        criterion = BLRL(alpha, beta)

        T = 1000000
        w_arr = []
        for t in tqdm(range(T)):
            optimizer.zero_grad()
            pred = network(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step(bs, N)
            if T > 800000:
                w_arr.append(network.line1.weight.data.detach().cpu().item())
        w_arr = np.array(w_arr)
        posterior_mean = N*x_v*y_v*beta/(alpha + N*x_v**2*beta)
        posterior_var = 1/(alpha + N*x_v**2*beta)
        with self.subTest(msg='Mean check'):
            self.assertAlmostEqual(np.mean(w_arr), posterior_mean, places=2)
        with self.subTest(msg='Var check'):
            self.assertAlmostEqual(np.var(w_arr), posterior_var, places=3)
        print("Mean: Approximated ", np.mean(w_arr), " Posterior ", posterior_mean)
        print("Variance: Approximated ", np.var(w_arr), " Posterior ", posterior_var)

    @unittest.skip("test_sgld_simple")
    def test_sgld_simple(self):
        bs = 1
        alpha = 3
        beta = 2
        self._test_sgld(bs, alpha, beta)

    #@unittest.skip("test_sgld_bs")
    def test_sgld_bs(self):
        bs = 32
        alpha = 3
        beta = 2
        self._test_sgld(bs, alpha, beta)


if __name__ == '__main__':
    unittest.main()
