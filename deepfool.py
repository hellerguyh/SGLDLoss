import numpy as np
from torch.autograd import Variable
import torch as torch
import copy

def zero_gradients(x):
    if isinstance(x, torch.Tensor):
        if x.grad is not None:
            x.grad.detach_()
            x.grad.zero_()
    elif isinstance(x, collections.abc.Iterable):
        for elem in x:
            zero_gradients(elem)

def does_nets_agree(image, nets, num_classes = 10):
    is_cuda = torch.cuda.is_available()
    N = len(nets)

    if is_cuda:
        print("Using GPU")
        image = image.cuda()
        for net in nets:
            net = net.cuda()
    else:
        print("Using CPU")

    label = None
    all_equal = True
    for n in range(N):
        f_image = nets[n].forward(Variable(image[None, :, :, :], requires_grad=True)).data.cpu().numpy().flatten()
        I = (np.array(f_image)).flatten().argsort()[::-1]

        I = I[0:num_classes]
        if label is None:
            label = I[0]
        elif label != I[0]:
            all_equal = False
    return n, all_equal

def deepfool(image, nets, num_classes=10, overshoot=0.02, max_iter=200):

    """
       :param image: Image of size HxWx3
       :param net: network (input: images, output: values of activation **BEFORE** softmax).
       :param num_classes: num_classes (limits the number of classes to test against, by default = 10)
       :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
       :param max_iter: maximum number of iterations for deepfool (default = 50)
       :return: minimal perturbation that fools the classifier, number of iterations that it required, new estimated_label and perturbed image
    """
    is_cuda = torch.cuda.is_available()
    N = len(nets)

    if is_cuda:
        print("Using GPU")
        image = image.cuda()
        for net in nets:
            net = net.cuda()
    else:
        print("Using CPU")

    #Finding the class of the label, assuming all nets tag it the same
    label = None
    for n in range(N):
        f_image = nets[n].forward(Variable(image[None, :, :, :], requires_grad=True)).data.cpu().numpy().flatten()
        I = (np.array(f_image)).flatten().argsort()[::-1]

        I = I[0:num_classes]
        if label is None:
            label = I[0]
        elif label != I[0]:
            raise Exception("Labels are not equal!: " + str(label) + "," + str(I[0]))

    input_shape = image.cpu().numpy().shape
    pert_image = copy.deepcopy(image)
    r_tot = np.zeros(input_shape)

    loop_i = 0

    x = Variable(pert_image[None, :], requires_grad=True)
    fs = [net.forward(x) for net in nets]
    found = False

    while (not found) and loop_i < max_iter:
        print(loop_i)
        pert_arr = torch.zeros(num_classes-1)
        w_arr = [torch.zeros(list(x.shape)) for k in range(num_classes-1)]
        for n in range(N):
            zero_gradients(x)
            fs[n][0, I[0]].backward(retain_graph=True)
            grad_orig = x.grad.data.cpu().numpy().copy()

            for k in range(1, num_classes):
                zero_gradients(x)

                fs[n][0, I[k]].backward(retain_graph=True)
                cur_grad = x.grad.data.cpu().numpy().copy()

                # set new w_k and new f_k
                w_k = cur_grad - grad_orig
                f_k = (fs[n][0, I[k]] - fs[n][0, I[0]]).data.cpu().numpy()
                if np.sum(np.abs(w_k)) > 0:
                    pert_arr[k-1] += abs(f_k)/np.linalg.norm(w_k.flatten())
                w_arr[k-1] += w_k

        l = torch.argmin(pert_arr)
        pert = pert_arr[l]/N
        w = w_arr[l]/N
        w = w.data.cpu().numpy()
        pert = pert.data.cpu().numpy()

        # compute r_i and r_tot
        # Added 1e-4 for numerical stability
        r_i =  (pert+1e-4) * w / np.linalg.norm(w)
        r_tot = np.float32(r_tot + r_i)

        if is_cuda:
            pert_image = image + (1+overshoot)*torch.from_numpy(r_tot).cuda()
        else:
            pert_image = image + (1+overshoot)*torch.from_numpy(r_tot)

        x = Variable(pert_image, requires_grad=True)
        fs = [net.forward(x) for net in nets]
        k_i = np.array([np.argmax(fs[n].data.cpu().numpy().flatten()) for n in range(N)])
        found = all(k_i != label)

        loop_i += 1

    r_tot = (1+overshoot)*r_tot

    return r_tot, loop_i, label, k_i[0], pert_image, found, loop_i - 1
