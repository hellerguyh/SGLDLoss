import numpy as np
import math
import matplotlib.pyplot as plt
from attack import *

def testGetStats():
    Y = [0,0,0,0,0,1,1,1,1,1]
    P = [0,1,0,1,0,0,1,0,1,0]
    neg = 5
    pos = 5
    fn = 3
    fp = 2
    fp_rate = fp/neg
    fn_rate = fn/pos
    FN, FP = getStats(P, Y)
    if FN != fn_rate:
        raise Exception("testGetStats Failed!: FN = " + str(FN) + " fn_rate = "
                        + str(fn_rate))
    if FP != fp_rate:
        raise Exception("testGetStats Failed!: FP = " + str(FP) + " fp_rate = "
                        + str(fp_rate))

'''
showHistogram() - translate a prediction dataset into histogram
'''
def showHistogram(path):
    X, Y = predictions2Dataset(path)
    Y = np.array(Y)
    X = [x[0] for x in X]
    mask = np.ma.masked_where(Y == 1,X)
    tagged_predictions = np.ma.compressed(mask)
    mask = np.ma.masked_where(Y == 0,X)
    untagged_predictions = np.ma.compressed(mask)
    minimum = math.floor(min([min(tagged_predictions), min(untagged_predictions)]))
    maximum = math.ceil(max([max(tagged_predictions), max(untagged_predictions)])) + 1
    bins = np.linspace(minimum, maximum, 50) # fixed number of bins
    plt.xlim([minimum-2, maximum+2])
    plt.hist([tagged_predictions, untagged_predictions], bins=bins)
    plt.savefig("histogram.png")

def smallTest():
    loader = getDL(1, True, "MNIST", True)
    for img, label in loader:
        ml_sample = img
        break

    model = NoisyNN()

    PATH = './trained_weights/LeNet5/'
    tagged_l = glob.glob(PATH + "TAGGED*")
    print("Printing Tagged List")
    for w_path in tagged_l:
        model.loadWeights(w_path)
        model.nn.eval()
        print(model.nn(ml_sample))
    tagged_l = glob.glob(PATH + "UNTAGGED*")
    print("Printing UnTagged List")
    for w_path in tagged_l:
        model.loadWeights(w_path)
        model.nn.eval()
        print(model.nn(ml_sample))

if __name__ == "__main__":
    testGetStats()
    showHistogram("Dictionary.pkl")
