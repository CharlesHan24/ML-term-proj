import numpy as np
import torch
from pert_single import pert_single

def proj_lp(v, xi, p):

    # Project on the lp ball centered at 0 and of radius xi

    # SUPPORTS only p = 2 and p = Inf for now
    if p == 2:
        v = v * min(1, xi/float(v.norm()))
        # v = v / np.linalg.norm(v.flatten(1)) * xi
    elif p == np.inf:
        v = v.sign() * torch.min(v.abs(), xi)
    else:
         raise ValueError('Values of p different from 2 and Inf are currently not supported...')

    return v

def pert_universal(trainloader, device, f, delta=0.2, epochs = 100, xi=10, p=np.inf, num_classes=10, overshoot=0.02, max_iter_df=10):
    """
    what is fooling ?
    1. prediction(with perturbation) != ground truth
    2. prediction(with perturbation) != original prediction
    use 2. here, 1. should be easier
    
    :param delta: controls the desired fooling rate (default = 80% fooling rate)
    :param epochs: optional other termination criterion (maximum number of iteration, default = 100)
    :param xi: controls the l_p magnitude of the perturbation (default = 10)
    :param p: norm to be used (FOR NOW, ONLY p = 2, and p = np.inf ARE ACCEPTED!) (default = np.inf)
    :param num_classes: num_classes (limits the number of classes to test against, by default = 10)
    :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
    :param max_iter_df: maximum number of iterations for deepfool (default = 10)
    :return: the universal perturbation.
    """
    v = 0
    fooling_rate = 0.0
    for ep in range(epochs):
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            for i in range(inputs.shape[0]):
                
                with torch.no_grad():
                    orig_pred = f(inputs[i:i+1]).view(-1).argmax()
                    pert_pred = f(inputs[i:i+1]+v).view(-1).argmax()
                
                # fooling vec v is not sufficient
                if orig_pred == pert_pred:
                    dr,iters,_,_ = pert_single(inputs[i] + v, f, num_classes=num_classes, overshoot=overshoot, max_iter=max_iter_df)
                    
                    # Make sure it converged...
                    if iters < max_iter_df-1:
                        v = v + dr

                        # Project on l_p ball
                        v = proj_lp(v, xi, p)
            
        # fooling rate
        fooled = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(trainloader):
                inputs, targets = inputs.to(device), targets.to(device)
                orig_preds = f(inputs).argmax(1).to("cpu").numpy()
                pert_preds = f(inputs+v).argmax(1).to("cpu").numpy()
                fooled += np.sum(orig_preds != pert_preds) 
                total += orig_preds.shape[0]
            
            fooling_rate = float(fooled)/total
            print("epoch %d, batch %d, FOOLING RATE = %f" %(ep,batch_idx,fooling_rate))
        
        if fooling_rate >= 1-delta:
            break
    return v
