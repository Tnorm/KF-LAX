

import torch


def var_comp(grad_samples):
    return torch.sum(grad_samples ** 2) - torch.mean(grad_samples) ** 2




