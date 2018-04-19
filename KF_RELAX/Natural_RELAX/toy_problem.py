import torch
from torch.autograd import Variable

toy_t = Variable(torch.FloatTensor([0.499]))
toy_theta = Variable(torch.FloatTensor([0.5]), requires_grad = True)


def toy_func(b):
    return ((b - toy_t) ** 2)



def toy_true_grad(t=toy_t):
    return (1-t) ** 2 - t ** 2


def toy_expected_error(param, t=toy_t):
    return (1-param) * (t**2) + param * ((1-t) ** 2)