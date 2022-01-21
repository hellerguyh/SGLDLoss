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
from scipy import stats

from attacked_model import addAttackedModel, collectMeta
from nn import NoisyNN
from data import getImg

MAX_EPS = 6

'''
weightsToPredictions() - translate the models weights into predictions
@weights_path - path to existing models weights
@dst_path - path to save dictionary

dictionary is in the form: ID, PREDICTION
'''
def weightsToPredictions(weights_path, dst_path, nn_type = 'LeNet5'):
    tagged_l = glob.glob(weights_path + "TAGGED*")
    untagged_l = glob.glob(weights_path + "UNTAGGED*")

    img = getImg(nn_type, True)
    model = NoisyNN(nn_type)

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

def getAdvExample(weights_path,  nn_type = 'LeNet5'):
    untagged_l = glob.glob(weights_path + "UNTAGGED*")

    img = getImg(nn_type, True)
    labels = getMalLabels(nn_type)
    model = NoisyNN(nn_type)

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

def getStats(P, Y):
    # False Negatives: P == 0, Y_TEST = 1
    # False Positives: P == 1, Y_TEST = 0
    P = np.array(P, dtype = int)
    Y = np.array(Y, dtype = int)

    neg = len(np.where(Y == 0)[0])
    pos = len(Y) - neg

    FN_mask = (P == 0) & (Y == 1)
    FP_mask = (P == 1) & (Y == 0)

    if pos > 0:
        FN = len(np.where(FN_mask == True)[0])
        FN_rate = FN/pos
    else:
        FN = 0
        FN_rate = 0
    if neg > 0:
        FP = len(np.where(FP_mask == True)[0])
        FP_rate = FP/neg
    else:
        FP = 0
        FP_rate = 0
    return FN_rate, FP_rate, FN, FP, pos, neg

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
        X = np.zeros((ds_size, 1))
        Y = np.zeros(ds_size)
        softmax = torch.nn.Softmax(dim=0)
        for i, sample in enumerate(ds):
            x_prop = softmax(torch.tensor(sample['mal_pred_arr'][epoch]))
            X[i][0] = x_prop[data_index].detach().numpy()
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
            if len(sample[f_name][ds_type]) == 51 and f_name == 'acc_arr':
                X[i] = sample[f_name][ds_type][epoch + 1]
            else:
                X[i] = sample[f_name][ds_type][epoch]
        return X

    def getMalPred(self, epoch, data_index = [8]):
        X_train, Y_train = self._getMalPred(self.meta_train, epoch, data_index)
        X_test, Y_test = self._getMalPred(self.meta_test, epoch, data_index)

        return X_train, X_test, Y_train, Y_test

def clopper_pearson_interval(x, N, alpha = 0.05):
    if x == 0:
        lb = 0
        up = 1 - (alpha/2.0)**(1/N)
    elif x == N:
        lb = (alpha/2.0)**(1/N)
        up = 1
    else:
        lb = stats.beta.ppf(alpha/2, x, N - x + 1)
        up = stats.beta.isf(alpha/2, x + 1, N - x)
        if math.isnan(lb) or math.isnan(up):
            raise Exception()
    return lb, up

def get_eps_lower_bound(FN, FP, pos, negs, delta):
    FN_cp = clopper_pearson_interval(FN, pos)[1]
    FP_cp = clopper_pearson_interval(FP, negs)[1]

    if FN_cp < 1 - delta:
        v1 = np.log((1 - delta - FN_cp)/FP_cp)
    else:
        v1 = 0
    if FP_cp < 1 - delta:
        v2 = np.log((1 - delta - FP_cp)/FN_cp)
    else:
        v2 = 0

    return np.max([v1, v2, 0])

def get_emp_eps(FN_rate, FP_rate, delta):
    if (FN_rate == 0 and FP_rate == 1) or (FP_rate == 0 and FN_rate == 1):
        emp_eps = 0
    elif FN_rate == 0 or FP_rate == 0:
        emp_eps = MAX_EPS
    else:
        emp_eps = np.log(np.max([(1 - delta - FN_rate)/FP_rate,
                                (1 - delta - FP_rate)/FN_rate]))
    return np.max([emp_eps, 0])

def plotEpsLB(eps_lb_arr, acc_arr, path, nn):
    plt.clf()
    plt.rcParams["font.family"] = "serif"
    fig, ax = plt.subplots()
    pt_1 = ax.plot(eps_lb_arr, color = "Green", label = '$\epsilon_{lb}^{emp}$')
    ax.set_ylabel('$\epsilon_{lb}^{emp}$', color = "Green", fontsize = 9)
    ax.set_xlabel("Epoch", fontsize = 9)

    ax2 = ax.twinx()
    pt_2 = ax2.plot(acc_arr, color = "Black", label = 'accuracy')
    ax2.set_ylabel("Accuracy", color = "Black", fontsize = 9)

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc=0, fontsize = 9)

    if len(acc_arr) > 20:
        step = 5
    else:
        step = 2

    plt.xticks(np.arange(0, len(acc_arr), step=step))
    plt.grid()
    plt.savefig(path + 'eps_lb_prop_based_' + str(nn) + '.png', dpi=300)

def calcEpsGraph(path, delta, label, epochs, nn):
    ds = MetaDS(path)
    emp_eps_arr = [0]
    eps_lb_arr = [0]
    acc_arr = [0.1]

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

        acc = ds.getField('acc_arr', epoch - 1)
        acc = np.sum(acc)/len(acc)
        acc_arr.append(acc)

    plt.plot(emp_eps_arr, label = 'emp_eps')
    plt.plot(eps_lb_arr, label = 'lower bound')
    plt.legend()
    plt.savefig(path + 'empirical_epsilon_' + str(nn) + '.png')
    plotEpsLB(eps_lb_arr, acc_arr, path, nn)


'''
calcEps() - Calculate empirical epsilon on predictions dataset
@path: path to the prediction dataset
'''
def calcEps(path, delta, label):
    X, Y = predictions2Dataset(path, [label])
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.9,
                                                       random_state = 0)
    clf = make_pipeline(StandardScaler(),
                        SGDClassifier(max_iter=1000, tol=1e-3))
    clf.fit(X_train, Y_train)
    P = clf.predict(X_test)
    FN_rate, FP_rate = getStats(P, Y_test)
    print(FN_rate)
    print(FP_rate)
    emp_eps = np.max([(1 - delta - FN_rate)/FP_rate, (
                       1 - delta - FP_rate)/FN_rate])
    print(emp_eps)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description = "Create Victims")
    parser.add_argument("--tag", action = "store_true")
    parser.add_argument("--train_model", action = "store_true")
    parser.add_argument("--calc_eps", action = "store_true")
    parser.add_argument("--pred_mal_labels", action = "store_true")
    parser.add_argument("--nn", choices = ['LeNet5','ResNet18', 'ResNet18-100'])
    parser.add_argument("--cuda_id", type = int)
    parser.add_argument("--eps_graph", action = "store_true")
    parser.add_argument("--repeat", type = int, default = 1)
    parser.add_argument("--epochs", type = int, default = -1)
    parser.add_argument("--path", type = str, default = None)
    parser.add_argument("--lr_factor", type = int, default = -1)
    parser.add_argument("--bs", type = int, default = -1)
    args = parser.parse_args()
    if args.path == None:
        path = './trained_weights/' + args.nn + '/'
    else:
        if args.path[-1] != "/":
            raise Exception("Path is not a folder!: " + str(args.path))
        path = args.path

    if args.train_model:
        for i in range(args.repeat):
            print("Starting Attack " + str(i))
            addAttackedModel(args.tag, args.nn, args.cuda_id, args.epochs,
                             path, args.lr_factor, args.bs)

    if args.eps_graph:
        if args.epochs == -1:
            raise Exception("need argument epochs")
        if args.nn == 'LeNet5':
            label = 8
        else:
            label = 1
        calcEpsGraph(path, 10**(-5), label, args.epochs, args.nn)

    if args.calc_eps:
        pred_path = args.nn + "_Dictionary.pkl"
        weightsToPredictions(path, pred_path, args.nn)
        if args.nn == 'LeNet5':
            label = 8
        else:
            label = 1
        calcEps(pred_path, 0.01, label)

    if args.pred_mal_labels:
        getAdvExample(path, args.nn)
