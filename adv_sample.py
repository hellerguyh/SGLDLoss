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
              transform = getTransforms(normalize, ds_name))

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

#--create_adv_sample --path trained_weights/lenet5/cifar10/lr500e400bs32normalized/createAdvExample/ --nn LeNet5 --dataset CIFAR10 --normalize
#--train_model --path trained_weights/variable/ --nn LeNet5 --dataset CIFAR10 --normalize --cuda_id -1 --lr_factor 500