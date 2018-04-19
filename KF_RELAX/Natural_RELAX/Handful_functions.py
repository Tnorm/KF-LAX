


import torch





def heaviside(z):
    return z>=0


def softmax(z, log_temperature=1):
    temperature = torch.exp(log_temperature)
    return torch.nn.Softmax(z / temperature)


def uniform(sample_num):
    return torch.rand(sample_num, 1)


def standard_gaussian_noise_sampler_func(num):
    return torch.randn(num)


def bernoulli_density(samples, param):
    return param * samples + (1-param) * (1-samples)

pass
