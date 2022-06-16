from scipy import stats
import numpy as np
import math
import matplotlib.pyplot as plt

MAX_EPS = 6

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
    if (FN_rate == 0 and FP_rate >= 0.7) or (FP_rate == 0 and FN_rate >= 0.7):
        emp_eps = 0
    elif FN_rate == 0 or FP_rate == 0:
        emp_eps = MAX_EPS
    else:
        emp_eps = np.log(np.max([(1 - delta - FN_rate)/FP_rate,
                                (1 - delta - FP_rate)/FN_rate]))
    return np.max([emp_eps, 0])


def plotEpsLB(eps_lb_arr, score_arr, path, nn):
    plt.clf()
    plt.rcParams["font.family"] = "serif"
    fig, ax = plt.subplots()
    pt_1 = ax.plot(eps_lb_arr, color = "Green", label = '$\epsilon_{lb}^{emp}$')
    ax.set_ylabel('$\epsilon_{lb}^{emp}$', color = "Green", fontsize = 9)
    ax.set_xlabel("Epoch", fontsize = 9)

    ax2 = ax.twinx()
    pt_2 = ax2.plot(score_arr, color = "Black", label = 'accuracy')
    ax2.set_ylabel("Accuracy", color = "Black", fontsize = 9)

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc=0, fontsize = 9)

    if len(score_arr) > 20:
        step = 5
    else:
        step = 2

    plt.xticks(np.arange(0, len(score_arr), step=step))
    plt.grid()
    plt.savefig(path + 'eps_lb_prop_based_' + str(nn) + '.png', dpi=300)

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