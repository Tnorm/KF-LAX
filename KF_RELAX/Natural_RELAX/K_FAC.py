import torch
import scipy
import numpy as np
from torch.autograd import Variable

## Taken from pytorch forum
def kronecker_product(t1, t2):
    """
    Computes the Kronecker product between two tensors.
    See https://en.wikipedia.org/wiki/Kronecker_product
    """
    t1_height, t1_width = t1.size()
    t2_height, t2_width = t2.size()
    out_height = t1_height * t2_height
    out_width = t1_width * t2_width

    tiled_t2 = t2.repeat(t1_height, t1_width)
    expanded_t1 = (
        t1.unsqueeze(2)
          .unsqueeze(3)
          .repeat(1, t2_height, t2_width, 1)
          .view(out_height, out_width)
    )

    return expanded_t1 * tiled_t2


## Taken from https://github.com/yaroslavvb/kfac_pytorch/blob/master/kfac_pytorch.py
def regularized_inverse(mat, lambda_=3e-10, inverse_method='cpu',
                        use_cuda=False):
    assert mat.shape[0] == mat.shape[1]
    ii = torch.eye(mat.shape[0])
    if use_cuda:
        ii = ii.cuda()
    regmat = mat + lambda_*ii

    if inverse_method == 'numpy':
        result = torch.from_numpy(scipy.linalg.inv(regmat.cpu().numpy()))
        if use_cuda:
            result = result.cuda()
    elif inverse_method == 'gpu':
        assert use_cuda
        result = torch.inverse(regmat).cuda()
    elif inverse_method == 'cpu':
        result = torch.inverse(regmat)
    else:
        assert False, 'unknown inverse_method ' + str(inverse_method)
    return result


### k_fac_update for a neural net layer parameters.
# Refer to "Optimizing neural networks with kronecker-factored approximate curvature" paper.
def k_fac_update(forward_activations, backward_saved_grads, layer_grad):
    fw_activation = torch.bmm(forward_activations.view(-1, forward_activations.size()[1], 1),
                              forward_activations.view(-1, 1, forward_activations.size()[1]))
    bw_grad = torch.bmm(backward_saved_grads.view(-1, backward_saved_grads.size()[1], 1),
                        backward_saved_grads.view(-1, 1, backward_saved_grads.size()[1]))
    A = torch.mean(fw_activation,0)
    S = torch.mean(bw_grad,0)
    A_inv = regularized_inverse(A)
    S_inv = regularized_inverse(S)
    if A.shape[0]*S.shape[1] > A.shape[1]*S.shape[0]:
        kfac_update = torch.mm(A_inv, torch.mm(layer_grad, S_inv))
    else:
        kfac_update = torch.mm(torch.mm(A_inv, layer_grad), S_inv)

    return kfac_update



def KFAC(forward_activation, backward_saved_grad, layer_grad):
    A_inv = regularized_inverse(forward_activation)
    S_inv = regularized_inverse(backward_saved_grad)
    if A_inv.shape[0]*S_inv.shape[1] > A_inv.shape[1]*S_inv.shape[0]:
        kfac_update = torch.mm(A_inv, torch.mm(layer_grad, S_inv))
    else:
        kfac_update = torch.mm(torch.mm(A_inv, layer_grad), S_inv)

    return kfac_update
