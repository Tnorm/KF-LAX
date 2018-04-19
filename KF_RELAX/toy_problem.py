import torch
from torch.autograd import Variable

toy_t = Variable(torch.FloatTensor([0.45]))
toy_theta = Variable(torch.FloatTensor([0.5]), requires_grad = True)


def toy_func(b):
    return ((b - toy_t) ** 2)