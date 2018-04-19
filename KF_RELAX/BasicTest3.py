


import torch
from torch.autograd import Variable
from Handful_functions import heaviside, uniform
from Neural_net_architecture import Net


import matplotlib.pyplot as plt
import numpy as np

import time

import copy



#torch.backends.cudnn.enabled = False

REINFORCE_samples_num = 1
samples_num = 1

ITERATIONS = 3000


theta = torch.FloatTensor([0.5])


toy_t = 0.49


lr = torch.Tensor([0.005])
lr1 = torch.Tensor([0.005])
lr2 = torch.Tensor([0.005])

def toy_func(b):
    return (b - toy_t) ** 2


def bernoulli_density(samples, param):
    return param * samples + (1-param) * (1-samples)


def noise_sampler_func_toy(noise, param):
    return torch.log(param/(1-param)) + torch.log(noise/(1-noise))


def conditional_noise_sampler_func_toy(noise, param, input):
    param_term = torch.log(param/(1-param))
    vp = noise*param*input + noise*(1-param)*(1-input)
    #vp = noise*param *  if input ==0 else noise*(1-param) + param
    noise_term = torch.log(vp/(1-vp))
    return param_term + noise_term


## reinforce without reparametrization

def REINFORCE(parameter, function, density_func):
    param = parameter
    S = torch.Tensor(REINFORCE_samples_num,1).uniform_(0, 1) + param - 0.5
    samples = torch.clamp(S, 0, 1)
    samples = Variable(torch.bernoulli(samples), requires_grad = False)
    func_values = function(samples)
    param = Variable(param, requires_grad = True)
    #print torch.log(density_func(samples, param.data))
    log_prob = torch.log(density_func(samples, param))
    #print func_values, log_prob
    grad_approximator = torch.dot(func_values, log_prob)
    grad_approximator.backward()
    param_collector(param.data)
    return param.grad.data


## RELAX with reparametrization trick

PARAM_HIST = []

def param_collector(param):
    #PARAM_HIST.append(param + 0)
    temp = param + 0
    PARAM_HIST.append(toy_func(0)* (1-temp) + toy_func(1)*temp)



def RELAX_Toy(parameter, function, approx_net, density_func, H_mapping,
              noise_init_sampler_func, noise_sampler_func, conditional_noise_sampler_func, iterations):
    param = Variable(parameter, requires_grad = True)
    for iter in range(iterations):
        param.requires_grad = True
        init_noise = Variable(noise_init_sampler_func(samples_num), requires_grad = False)
        init_noise_for_conditional = Variable(noise_init_sampler_func(samples_num), requires_grad = False)
        noise = noise_sampler_func(init_noise, param)
        input_sample = H_mapping(noise)
        input_sample = input_sample.type(torch.FloatTensor)
        input_sample.detach_()
        conditional_noise = conditional_noise_sampler_func(init_noise_for_conditional, param, input_sample)
        f = function(input_sample)

        #print approx_net.w.weight

        approx_net_cond_noise = approx_net(conditional_noise)
        approx_net_cond_noise_copy = copy.copy(approx_net_cond_noise)
        approx_net_cond_noise_copy.detach_()
        approx_net_noise = approx_net(noise)
        log_prob = torch.log(density_func(input_sample, param))

        #log_prob_grad = torch.autograd.grad(log_prob, param, create_graph=True)

        param_grad_approximator = torch.sum((f - approx_net_cond_noise_copy)*log_prob + \
                                    approx_net_noise - \
                                    approx_net_cond_noise)

        param_grad_approximator_func = torch.autograd.grad(param_grad_approximator, param, create_graph = True)
        #approx_net_cond_noise_param_grad_func = param.grad


        param.requires_grad = False

        var_grad_approx = 0
        for grad in param_grad_approximator_func:
            var_grad_approx += grad

        var_grad_approx -= torch.dot(approx_net_cond_noise,log_prob)

        var_grad_approx.backward()

        #approx_net.w.data -= lr2 * 2 * param_grad_approximator_func[0].data * approx_net.w.grad.data

        for par in approx_net.parameters():
            if par.grad is not None:
                par.data -= lr2*(2* param_grad_approximator_func[0].data * par.grad.data/samples_num)


        param.data -= lr1 * param_grad_approximator_func[0].data/samples_num
        param.data = torch.clamp(param.data, 0.001, 0.999)

        #approx_net_noise_param_grad = param.grad.data
        #approx_net_noise_grad_func = param.grad
        #param.grad.data.zero_()

        #log_prob.backward(retain_graph = True)
        #log_prob_param_grad = param.grad

        #print approx_net_cond_noise_param_grad, approx_net_noise_param_grad, log_prob_param_grad


        #print DELTA_PARAM, param
        #param_grad_approximator_func = param.grad

        #start = time.time()
        param_collector(param.data)
        approx_net.zero_grad()
        #param_grad_approximator_func.backward()
        #stop = time.time()

        #for par in approx_net.parameters():
        #    var_grad_approx = 2 * param_grad_approximator * par.grad.data
        #    par.data -= lr2 * var_grad_approx
        #param.grad.data.zero_()


        #print param
        # for par in approx_net.parameters():
        #    par.requires_grad = True

        # grad_approximator_as_function = param.grad
        # grad_approximator_as_function.backward(retain_graph=True)
        #param.grad.data.zero_()


    return param, torch.mean(f)


    ##TODO IMPLEMENT A SIMPLE PYTORCH NET WITH TWO INPUT PARAMETERS
    #param -= lr1*



if __name__ == "__main__":
    print "start"

    function_approx_net = Net()

    parameter = theta
    #X = RELAX_Toy(parameter, toy_func, function_approx_net, bernoulli_density, heaviside,
    #                       uniform, noise_sampler_func_toy,
    #                       conditional_noise_sampler_func_toy, iterations=ITERATIONS)
        #print theta

    ### REINFORCE STARTS ####

    for iterations in range(ITERATIONS):
        gradient = REINFORCE(parameter, toy_func, bernoulli_density)
        parameter = torch.clamp(parameter - lr * gradient, 0.001, 0.999)

    ### REINFORCE ENDS ###

        #theta.grad.data.zero_()
        #print theta.data

    plt.plot(range(ITERATIONS), PARAM_HIST)
    plt.show()


    print X