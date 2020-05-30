import torch
import numpy as np

def boundary(f, pert_list, device, testloader, tan=0, samples = 20, average=False):
    '''
    input
    f: the network
    
    pert_list: Nxlen(pert), N is the number of different values of norm(pert); one universal perturbation for each choice of norms
    
    tan: tangent of angle between the universal perturbation and the random perturbation vector
         for tan=0, the universal perturbation is used
         
    samples: the number of random purturbation vectors to use
         
    average: if True, the percentage of successful attack
    
    ###########################################################################
    
    output
    percent: N x samples, each entry is the percentage of successful attack, i.e. perturbed prediction != original prediction; 
                     each row corresponds to a norm, each column corresponds to a random perturbation with respect to tan
             if average = True, return shape is (N,)
    '''
    
    N = len(pert_list)
    percent = np.zeros((N,samples))
    
    f.eval()
    for i in range(N):
        uni_pert = pert_list[i]
        norm = uni_pert.norm()
        scale = norm*tan
        for j in range(samples):
            print(j)
            # project a random vector onto an orthogonal direction of universal perturbation
            add = torch.randn(uni_pert.shape).to(device)
            add = add - ((uni_pert * add).sum()/norm)*uni_pert/norm
            add = add*scale/add.norm() + uni_pert
            add = add*norm/add.norm()
            
            fooled = 0
            total = 0
            
            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(testloader):
                    inputs, targets = inputs.to(device), targets.to(device)
                    orig_preds = f(inputs).argmax(1).to("cpu").numpy()
                    pert_preds = f(inputs+add).argmax(1).to("cpu").numpy()
                    fooled += np.sum(orig_preds != pert_preds) 
                    total += orig_preds.shape[0]
                    if batch_idx >= 3:
                        break
            
            percent[i][j]=fooled/total
    
    if average:
        return np.mean(percent,axis=1)
    return percent

def boundary2(f, g, pert_list, device, testloader, tan=0, samples = 20, average=False):
    '''
    input
    f: the network
    
    pert_list: Nxlen(pert), N is the number of different values of norm(pert); one universal perturbation for each choice of norms
    
    tan: tangent of angle between the universal perturbation and the random perturbation vector
         for tan=0, the universal perturbation is used
         
    samples: the number of random purturbation vectors to use
         
    average: if True, the percentage of successful attack
    
    ###########################################################################
    
    output
    percent: N x samples, each entry is the percentage of successful attack, i.e. perturbed prediction != original prediction; 
                     each row corresponds to a norm, each column corresponds to a random perturbation with respect to tan
             if average = True, return shape is (N,)
    '''
    
    N = len(pert_list)
    percent = np.zeros((N,samples, 3))
    
    f.eval()
    for i in range(N):
        uni_pert = pert_list[i]
        norm = uni_pert.norm()
        scale = norm*tan
        for j in range(samples):
            print(j)
            # project a random vector onto an orthogonal direction of universal perturbation
            add = torch.randn(uni_pert.shape).to(device)
            add = add - ((uni_pert * add).sum()/norm)*uni_pert/norm
            add = add*scale/add.norm() + uni_pert
            add = add*norm/add.norm()
            
            fooled1 = 0
            fooled2 = 0
            acc = 0
            total = 0
            
            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(testloader):
                    inputs, targets = inputs.to(device), targets.to(device)
                    orig_preds = f(inputs).argmax(1).to("cpu").numpy()
                    pert_preds = f(inputs+add).argmax(1).to("cpu").numpy()
                    fooled1 += np.sum(orig_preds != pert_preds) 
                    
                    inputs, targets = inputs.to(device), targets.to(device)
                    orig_preds2 = g(inputs).argmax(1).to("cpu").numpy()
                    pert_preds2 = g(inputs+add).argmax(1).to("cpu").numpy()
                    fooled2 += np.sum(orig_preds2 != pert_preds2) 
                    
                    acc += np.sum(pert_preds != pert_preds2)
                    total += orig_preds.shape[0]
                    if batch_idx >= 3:
                        break
            
            percent[i][j][0]=fooled1/total
            percent[i][j][1] = fooled2 / total
            percent[i][j][2] = acc / total
    
    if average:
        return np.mean(percent,axis=1)
    return percent