import glob
from tqdm import tqdm
from deepfool import deepfool, does_nets_agree

from nn import NoisyNN
from data import getDS, getTransforms

def create_adv_sample(weights_folder, nn_type, ds_name, normalize,
                      max_num_models = -1):
    # Get basic image
    db = getDS(ds_name, False)
    data = db(root = './dataset/',
              train = False, download = True,
              transform = getTransforms(normalize, ds_name),
              adv_sample_choice = None)

    # Load weights from path
    weights_paths = glob.glob(weights_folder + "UNTAGGED*")
    nets = []
    for i, path in enumerate(weights_paths):
        if i == max_num_models:
            break
        model = NoisyNN(nn_type, ds_name)
        model.loadWeights(path)
        model.nn.eval()
        nets.append(model.nn)

    for im in tqdm(data):
        idx, agree = does_nets_agree(im[0], nets)
        if agree:
            break
    if not agree:
        raise Exception("Couldn't find an image")
    else:
        print("Image index = " + str(idx))

    im = im[0]

    # Get pertubated image
    r_total, _, label_orig, label_pert, pert_image, found, idx_in_ds = deepfool(im, nets)
    if not found:
        raise Exception("Failed to find pertubated image")

    return label_orig, label_pert, pert_image, im, r_total, idx_in_ds


if __name__ == "__main__":
    import argparse
    import json
    import torch
    import numpy as np
    from matplotlib import pyplot as plt

    parser = argparse.ArgumentParser(description="scatter plot")
    parser.add_argument("--pkl_path", type=str, default=None)
    parser.add_argument("--json_path", type=str, default=None)
    parser.add_argument("--weights_folder", type=str, default=None)
    parser.add_argument("--tagged", action= "store_true")
    args = parser.parse_args()
    if args.weights_folder [-1] != "/":
        args.weights_folder += "/"
    if args.tagged:
        postfix = "TAG*"
    else:
        postfix = "UNTA*"
    weights_paths = glob.glob(args.weights_folder + postfix)
    print(len(weights_paths))
    adv_image = torch.load(args.pkl_path)
    adv_image = adv_image.reshape([1, *adv_image.shape])
    with open(args.json_path, 'r') as rf:
        jf = json.load(rf)
        label_idx = int(jf['orig_label'])
    pred_arr = np.zeros(len(weights_paths))
    softmax = torch.nn.Softmax(dim=0)
    for i, path in enumerate(weights_paths):
        model = NoisyNN("LeNet5", "CIFAR10")
        model.loadWeights(path)
        model.nn.eval()
        pred = model.nn(adv_image).detach()[0]
        pred = softmax(pred)
        pred_arr[i] = pred.numpy()[label_idx]

    plt.scatter(range(len(pred_arr)), pred_arr)
    plt.show()

#--create_adv_sample --path trained_weights/lenet5/cifar10/lr500e400bs32normalized/createAdvExample/ --nn LeNet5 --dataset CIFAR10 --normalize
#--train_model --path trained_weights/variable/ --nn LeNet5 --dataset CIFAR10 --normalize --cuda_id -1 --lr_factor 500
