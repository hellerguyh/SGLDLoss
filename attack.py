'''
Performs the attack:
1. Creates database of 100 models
2. Create database of models prediction on plantted sample
3. Train 100 different classifiers? train 1 classifier based on 10 sampels?
4. Get classifier predictions
'''
import torch
import pickle
import glob
import math
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt

from attacked_model import addAttackedModel, collectMeta
from nn import NoisyNN
from data import getImg, getMalLabels
from utils_dp import get_emp_eps, get_eps_lower_bound, plotEpsLB, getStats

'''
weightsToPredictions() - translate the models weights into predictions
@weights_path - path to existing models weights
@dst_path - path to save dictionary

dictionary is in the form: ID, PREDICTION
'''
def weightsToPredictions(weights_path, dst_path, nn_type, ds_name):
    tagged_l = glob.glob(weights_path + "TAGGED*")
    untagged_l = glob.glob(weights_path + "UNTAGGED*")

    img = getImg(ds_name, True)
    model = NoisyNN(nn_type, ds_name)

    pred_d = {}
    with torch.no_grad():
        for dtype, l, dtype_int  in\
        [("TAGGED", tagged_l, 1), ("UNTAGGED", untagged_l, 0)]:
            for w_path in l:
                model.loadWeights(w_path)
                model.nn.eval()
                pred = model.nn(img)
                pred = list(pred.detach().numpy()[0])
                sample_id = w_path[w_path.find(dtype):]
                pred_d[sample_id] = (dtype_int, pred)
    with open(dst_path, 'wb') as wf:
        pickle.dump(pred_d, wf)

def getAdvExample(weights_path,  nn_type, ds_name):
    untagged_l = glob.glob(weights_path + "UNTAGGED*")

    img = getImg(ds_name, True)
    labels = getMalLabels(ds_name)
    model = NoisyNN(nn_type, ds_name)

    sum_p = 0
    with torch.no_grad():
        for i, w_path in enumerate(untagged_l):
             model.loadWeights(w_path)
             model.nn.eval()
             pred = model.nn(img)
             pred = list(pred.detach().numpy()[0])
             sum_p += np.array(pred)
    avg = sum_p / len(untagged_l)
    print("Malicious labels predictions:")
    print(labels)
    for l in labels:
        print(str(l), str(avg[int(l)]))

'''
predictions2Dataset() - translate predictions to a classifier dataset
@path: path to predictions dictionary
@data_indexes: which predictions to take for each sample

Function translates the dictionary into a matrix. It takes only the labels
selected by the data_indexes.

Returns: dataset
'''
def predictions2Dataset(path, data_indexes = [8]):
    with open(path, 'rb') as rf:
        data_dict = pickle.load(rf)
    X = np.zeros([len(data_dict), len(data_indexes)])
    Y = np.zeros(len(data_dict))
    for i, sample_id in enumerate(data_dict.keys()):
        for j, index in enumerate(data_indexes):
            X[i, j] = data_dict[sample_id][1][index]
            Y[i] = int(data_dict[sample_id][0])
    return X, Y

'''
MetaDS - Dataset used to handle meta-data

Should be suitable for working with sklearn - i.e. provide the whole data
at once)
'''
class MetaDS(object):
    def __init__(self, path):
        meta = collectMeta(path)
        self.meta_train, self.meta_test = train_test_split(meta,
                                                           test_size = 0.9,
                                                           random_state = 0)

    def _getMalPred(self, ds, epoch, data_index):
        ds_size = len(ds)
        X = np.zeros((ds_size, 2*len(data_index)))
        Y = np.zeros(ds_size)
        softmax = torch.nn.Softmax(dim=0)
        for i, sample in enumerate(ds):
            x_prop1 = softmax(torch.tensor(sample['mal_pred_arr'][epoch])).detach().numpy()
            x_prop2 = softmax(torch.tensor(sample['nonmal_pred_arr'][epoch])).detach().numpy()
            for k, j in enumerate(data_index):
                X[i][k] = x_prop1[j]
                X[i][k+len(data_index)] = x_prop2[j]
            # Y[i] = ds['tag'] == True
            if 'UNTAGGED' in sample['model_id']:
                Y[i] = False
            else:
                Y[i] = True

        return X, Y

    def getField(self, f_name, epoch, ds_type = 'val'):
        if ds_type == 'val':
            ds = self.meta_test
        elif ds_type == 'train':
            ds = self.meta_train
        ds_size = len(ds)
        X = np.zeros(ds_size)
        for i, sample in enumerate(ds):
            if len(sample[f_name][ds_type]) == 51 and f_name == 'score_arr':
                X[i] = sample[f_name][ds_type][epoch + 1]
            else:
                X[i] = sample[f_name][ds_type][epoch]
        return X

    def getMalPred(self, epoch, data_index = [8]):
        X_train, Y_train = self._getMalPred(self.meta_train, epoch, data_index)
        X_test, Y_test = self._getMalPred(self.meta_test, epoch, data_index)

        return X_train, X_test, Y_train, Y_test

def calcEpsGraph(path, delta, label, epochs, nn):
    ds = MetaDS(path)
    emp_eps_arr = [0]
    eps_lb_arr = [0]
    score_arr = [0.1]

    for epoch in range(1, epochs+1):
        X_train, X_test, Y_train, Y_test = ds.getMalPred(epoch, label)
        clf = make_pipeline(StandardScaler(),
                            SGDClassifier(max_iter=1000, tol=1e-3))
        clf.fit(X_train, Y_train)
        P = clf.predict(X_test)
        FN_rate, FP_rate, FN, FP, pos, negs = getStats(P, Y_test)

        emp_eps = get_emp_eps(FN_rate, FP_rate, delta)
        emp_eps_arr.append(emp_eps)

        eps_lb = get_eps_lower_bound(FN, FP, pos, negs, delta)
        eps_lb_arr.append(eps_lb)

        acc = ds.getField('score_arr', epoch - 1)
        acc = np.sum(acc)/len(acc)
        score_arr.append(acc)

    plt.plot(emp_eps_arr, label = 'emp_eps')
    plt.plot(eps_lb_arr, label = 'lower bound')
    plt.legend()
    plt.savefig(path + 'empirical_epsilon_' + str(nn) + '.png')
    plotEpsLB(eps_lb_arr, score_arr, path, nn)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description = "Create Victims")
    parser.add_argument("--tag", action = "store_true")
    parser.add_argument("--train_model", action = "store_true")
    parser.add_argument("--pred_mal_labels", action = "store_true")
    parser.add_argument("--nn", choices = ['LeNet5','ResNet18', 'ResNet18-100', 'ResNet18NoBN'])
    parser.add_argument("--dataset", choices = ['MNIST','CIFAR10', 'CIFAR100'])
    parser.add_argument("--cuda_id", type = int)
    parser.add_argument("--eps_graph", action = "store_true")
    parser.add_argument("--repeat", type = int, default = 1)
    parser.add_argument("--epochs", type = int, default = -1)
    parser.add_argument("--path", type = str, default = None)
    parser.add_argument("--lr_factor", type = float, default = -1, help =
                        "SGLD lr will be lr_factor*|dataset|**(-2), only available"\
                        "when clipping is not active")
    parser.add_argument("--lr", type = float, default = -1, help = "learning rate"\
                        "when clipping is active")
    parser.add_argument("--bs", type = int, default = -1)
    parser.add_argument("--clipping", type = float, default = -1)
    parser.add_argument("--lr_sched_type", type=str, default=None, choices = ['StepLR', 'Cosine'])
    parser.add_argument("--lr_sched_milestones", nargs="+", type=float, default=None)
    parser.add_argument("--lr_sched_gamma", type=float, default=-1)
    parser.add_argument("--delta", type=float, default=10**(-5))
    args = parser.parse_args()
    if args.path == None:
        path = './trained_weights/' + args.nn + '/'
    else:
        if args.path[-1] != "/":
            raise Exception("Path is not a folder!: " + str(args.path))
        path = args.path

    if args.train_model:
        assert (args.clipping > -1 and args.lr > -1) or (args.clipping == -1 and
                args.lr == -1), "Either clipping and lr both set or unset"
        assert (args.clipping > -1 and args.lr_factor == -1) or (args.clipping == -1 and
                args.lr_factor > -1), "Clipping and lr_factor are mutually exclusive"

        if args.clipping > -1 and not torch.cuda.is_available():
            torch.multiprocessing.set_sharing_strategy('file_system')
        for i in range(args.repeat):
            print("Starting Attack " + str(i))
            if args.lr_sched_type:
                if args.lr_sched_type == 'StepLR':
                    assert (not args.lr_sched_milestons is None) and (args.lr_sched_gamma != -1)
                lr_scheduling = {'type' : args.lr_sched_type,
                                 'milestones' : args.lr_sched_milestones,
                                 'gamma' : args.lr_sched_gamma}
            else:
                lr_scheduling = None
            addAttackedModel(args.tag, args.nn, args.cuda_id, args.epochs,
                             path, args.lr_factor, args.bs, args.clipping,
                             lr_scheduling, args.lr, args.delta, args.dataset)

    if args.eps_graph:
        if args.epochs == -1:
            raise Exception("need argument epochs")
        if args.nn == 'LeNet5':
            label = [8]
        else:
            label = list(range(10))#1
        calcEpsGraph(path, 10**(-5), label, args.epochs, args.nn)

    if args.pred_mal_labels:
        getAdvExample(path, args.dataset)
