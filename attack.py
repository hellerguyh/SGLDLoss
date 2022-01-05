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
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt

from attacked_model import addAttackedModel, collectMeta
from nn import NoisyNN
from data import getImg

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
        FN = np.where(FN_mask == True)[0]
        FN_rate = len(FN)/pos
    else:
        FN_rate = 0
    if neg > 0:
        FP = np.where(FP_mask == True)[0]
        FP_rate = len(FP)/neg
    else:
        FP_rate = 0
    return FN_rate, FP_rate

'''
MetaDS - Dataset used to handle meta-data

Should be suitable for working with sklearn - i.e. provide the whole data
at once)
'''
class MetaDS(object):
    def __init__(self, path):
        meta = collectMeta(path)
        self.meta_train, self.meta_test = train_test_split(meta,
                                                           test_size = 0.8,
                                                           random_state = 0)

    def _getMalPred(self, ds, epoch, data_index):
        ds_size = len(ds)
        X = np.zeros((ds_size, 1))
        Y = np.zeros(ds_size)
        for i, sample in enumerate(ds):
            X[i][0] = sample['mal_pred_arr'][epoch][data_index]
            # Y[i] = ds['tag'] == True
            if 'UNTAGGED' in sample['model_id']:
                Y[i] = False
            else:
                Y[i] = True

        return X, Y

    def getField(self, f_name, epoch):
        ds_size = len(ds)
        X = np.zeros(ds_size)
        for i in range(ds_size):
            X[i] = ds[f_name][epoch]

        return X

    def getMalPred(self, epoch, data_index = [8]):
        X_train, Y_train = self._getMalPred(self.meta_train, epoch, data_index)
        X_test, Y_test = self._getMalPred(self.meta_test, epoch, data_index)

        return X_train, X_test, Y_train, Y_test

def calcEpsGraph(path, delta, label, epochs):
    ds = MetaDS(path)
    emp_eps_arr = []
    for epoch in range(epochs+1):
        X_train, X_test, Y_train, Y_test = ds.getMalPred(epoch, label)
        clf = make_pipeline(StandardScaler(),
                            SGDClassifier(max_iter=1000, tol=1e-3))
        clf.fit(X_train, Y_train)
        P = clf.predict(X_test)
        FN_rate, FP_rate = getStats(P, Y_test)
        if (FN_rate == 0 and FP_rate == 1) or (FP_rate == 0 and FN_rate == 1):
            emp_eps = 0
        elif FN_rate == 0 or FP_rate == 0:
            emp_eps = 10
        else:
            emp_eps = np.log(np.max([(1 - delta - FN_rate)/FP_rate,
                                    (1 - delta - FP_rate)/FN_rate]))
        emp_eps_arr.append(emp_eps)

    plt.plot(emp_eps_arr)
    plt.savefig('emp_eps_arr.png')

'''
calcEps() - Calculate empirical epsilon on predictions dataset
@path: path to the prediction dataset
'''
def calcEps(path, delta, label):
    X, Y = predictions2Dataset(path, [label])
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.8,
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
    parser.add_argument("--nn", choices = ['LeNet5','ResNet18'])
    parser.add_argument("--cuda_id", type = int)
    parser.add_argument("--eps_graph", action = "store_true")
    parser.add_argument("--repeat", type = int, default = 1)
    args = parser.parse_args()
    if args.train_model:
        for i in range(args.repeat):
            addAttackedModel(args.tag, args.nn, args.cuda_id)
    if args.eps_graph:
        path = './trained_weights/' + args.nn + '/'
        if args.nn == 'LeNet5':
            label = 8
            epochs = 10
        else:
            label = 1
            epochs = 50
        calcEpsGraph(path, 10**(-5), label, epochs)

    if args.calc_eps:
        PATH = './trained_weights/' + args.nn + '/'
        pred_path = args.nn + "_Dictionary.pkl"
        weightsToPredictions(PATH, pred_path, args.nn)
        if args.nn == 'LeNet5':
            label = 8
        else:
            label = 1
        calcEps(pred_path, 0.01, label)
