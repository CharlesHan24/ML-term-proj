import numpy as np
import torch

def pert_single(image, f, num_classes=10, overshoot=0.02, max_iter=10):

    """
       :param image: Image of size HxWx3
       :param f: feedforward network (input: images, output: values of activation BEFORE softmax).
       :param num_classes: num_classes (limits the number of classes to test against, by default = 10)
       :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
       :param max_iter: maximum number of iterations for deepfool (default = 10)
       :return: minimal perturbation that fools the classifier, number of iterations that it required, new estimated_label and perturbed image
    """
    loop_i = 0
    
    input_shape = image.shape
    w = 0
    r_tot = 0
    
    pert_image = image.clone().reshape(1, 3, 32, 32)
    pert_image = torch.autograd.Variable(pert_image, requires_grad=True)
    label = None
    k_i = None
    
    while loop_i < max_iter:
        out = f(pert_image).view(-1)
        val,arg = out.sort(descending = True)
        val = val[:num_classes] - val[0]
        arg = arg[:num_classes]
        k_i = int(arg[0])
        
        if loop_i == 0:
            label = int(arg[0])
        
        if k_i != label:
            break
        
        gradients = []
        for k in range(1,num_classes):
            pert_image.grad.zero_()
            f.zero_grad()
            val[k].backward(retain_graph=True)
            gradients.append(pert_image.grad.clone().detach())
        
        pert = np.inf
        for k in range(1, num_classes):
            pert_k = abs(val[k])/np.linalg.norm(gradients[k])

            # determine which k to use in gradients
            if pert_k < pert:
                pert = pert_k
                w = gradients[k]
        
        r_i =  pert * w / np.linalg.norm(w)
        r_tot = (r_tot + r_i)*(1+overshoot)
        
        pert_image = image + r_tot
        loop_i += 1 

    return r_tot, loop_i, k_i, pert_image
