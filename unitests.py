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
def showHistogram(path, label):
    X, Y = predictions2Dataset(path, label)
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
    loader = getDL(1, True, "MNIST", True, False)
    for img, label in loader:
        ml_sample = img
        break

    model = NoisyNN(nn_type='LeNet5', ds_name='MNIST')

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

def showScatterPlot(path, epoch):
    from mpl_toolkits.mplot3d import axes3d
    import matplotlib.pyplot as plt
    from matplotlib import style
    style.use('ggplot')
    ds = MetaDS(path, True)
    xx = 1 
    yy = 9
    zz = 0
    for j in [0]:
        for i in range(epoch):
            X, X_T, Y, Y_T = ds.getMalPred(i, [xx, yy, zz])
            fig = plt.figure()
            X = np.concatenate([X, X_T])
            Y = np.concatenate([Y, Y_T])
            #ax1 = fig.add_subplot(111, projection='3d')
            tagged_predictions = X[Y == 1]
            untagged_predictions = X[Y == 0]
            plt.scatter(tagged_predictions.T[0 + j], tagged_predictions.T[1 + j], c='g', marker='o', label="tagged")
            plt.scatter(untagged_predictions.T[0 + j], untagged_predictions.T[1 + j], c='b', marker='o', label="untagged")
            plt.xlabel(str(xx))
            plt.ylabel(str(yy))
            #ax1.scatter(tagged_predictions.T[0], tagged_predictions.T[1], tagged_predictions.T[2], c='g', marker='o', label="tagged")
            #ax1.scatter(untagged_predictions.T[0], untagged_predictions.T[1], untagged_predictions.T[2], c='b', marker='o', label="untagged")
            #ax1.set_xlabel(str(xx))
            #ax1.set_ylabel(str(yy))
            #ax1.set_zlabel(str(zz))
            plt.legend()
            if j == 0:
                plt.savefig("example_learning/tagged/" + str(i) + ".png")
            else:
                plt.savefig("example_learning/untagged/" + str(i) + ".png")
            plt.close(fig)
            #plt.show()

def showHistogramFromPredictions(path, epoch):
    ds = MetaDS(path)
    X, _, Y, _ = ds.getMalPred(epoch, [1, 9])
    Y = np.array(Y)
    #mask = np.ma.masked_where(Y == 1, X)
    #tagged_predictions = np.ma.compressed(mask)
    tagged_predictions = X[Y == 1]
    tagged_predictions = tagged_predictions
    #tagged_predictions = tagged_predictions.reshape(tagged_predictions.shape[0])
    #mask = np.ma.masked_where(Y == 0, X)
    #untagged_predictions = np.ma.compressed(mask)
    untagged_predictions = X[Y == 0]
    untagged_predictions = untagged_predictions
    #untagged_predictions = untagged_predictions.reshape(untagged_predictions.shape[0])
    plt.scatter(tagged_predictions, untagged_predictions)
    #print(untagged_predictions)
    #print(tagged_predictions)
    #minimum = math.floor(min([min(tagged_predictions), min(untagged_predictions)]))
    #maximum = max([max(tagged_predictions), max(untagged_predictions)])
    #bins = np.linspace(minimum, maximum, 100)  # fixed number of bins
    #plt.xlim([minimum, maximum])
    #plt.hist([tagged_predictions, untagged_predictions], label=['tagged','untagged'], bins=bins)
    #plt.legend()
    #plt.show()


if __name__ == "__main__":
    #showHistogram("ResNet18_Dictionary.pkl", [1])
    import argparse
    parser = argparse.ArgumentParser(description="Small Tests")
    parser.add_argument("--path", type=str, default=None)
    parser.add_argument("--epoch", type=int, default=-1)
    args = parser.parse_args()

    showScatterPlot(args.path, args.epoch)
    #showHistogramFromPredictions(args.path, args.epoch)
