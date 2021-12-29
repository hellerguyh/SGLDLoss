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

from attacked_model import addAttackedModel
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
        FN_rate = 999
    if neg > 0:
        FP = np.where(FP_mask == True)[0]
        FP_rate = len(FP)/neg
    else:
        FP_rate = 999
    return FN_rate, FP_rate

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
    parser.add_argument("--nn", nargs = 1, choices = ['LeNet5','ResNet18'])
    parser.add_argument("--cuda_id", type = int)
    args = parser.parse_args()
    if args.train_model:
        addAttackedModel(args.tag, args.nn[0], args.cuda_id)
    if args.calc_eps:
        PATH = './trained_weights/' + args.nn[0] + '/'
        pred_path = args.nn[0] + "_Dictionary.pkl"
        weightsToPredictions(PATH, pred_path, args.nn[0])
        if args.nn[0] == 'LeNet5':
            label = 8
        else:
            label = 1
        calcEps(pred_path, 0.01, label)
