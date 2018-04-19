##### THE CODE PROVIDES AN IMPLEMENTATION OF REINFORCE, RELAX and REBAR algorithms
# REBAR is implemented as a particular case of RELAX algorithm
# More comments and hints to use will be provided soon...

import torch
from torch.autograd import Variable
from Handful_functions import heaviside, uniform, softmax
from Neural_net_architecture import Net, REBAR_Net
import matplotlib.pyplot as plt
import copy


## Number of mini-batch samples in the REINFORCE algorithm and RELAX/REBAR algorithms
# Warning: Currently, there are some issues with samples_num > 1 for RELAX/REBAR
REINFORCE_samples_num = 1
samples_num = 1

## initial temperature and scale parameters for the REBAR algorithm
init_temperature = 1
scale = 0.5

## Number of iterations for the algorithms
ITERATIONS = 2000


## Initial parameters for optimization
theta = torch.FloatTensor([0.5])

## Used to collect bias of the estimator
bias_sum = torch.FloatTensor([0.0])


## Input space, The input domain for variable b that is introduced in the Backpropagating through the void paper
# The input space for toy problem is {0, 1}
input_space = torch.FloatTensor([0, 1]).view(2,1)


## Toy problem target value
toy_t = 0.501


## Learning rates initialization
lr = torch.Tensor([0.005])
lr1 = torch.Tensor([0.01])
lr2 = torch.Tensor([0.0005])

## The function in the toy problem used to take expectation w.r.t.
def toy_func(b):
    return (b - toy_t) ** 2


## Bernoulli density function
def bernoulli_density(samples, param):
    return param * samples + (1-param) * (1-samples)

## this function is the (z|initial_noise,theta) sampler function. (refer to the Backpropagating through the void paper)
def noise_sampler_func_toy(noise, param):
    return torch.log(param/(1-param)) + torch.log(noise/(1-noise))


def noise_sampler_func_toy_2(noise, param):
    noise2= Variable(uniform(samples_num), requires_grad = False)
    o1 = torch.sigmoid(torch.log(param/(1-param)) + torch.log(noise/(1-noise)))
    return (torch.log(o1/(1-o1)) + torch.log(noise2/(1-noise2)))

## This function is the (z_tilde|initial_noise,theta,b) sampler function.
# (refer to the Backpropagating through the void paper Appendix A)
# The function is for toy problem and similar structure could be used for general implementations too
# The function is named S in Appendix A of the paper (4th line in the loop of the algorithm):
def conditional_noise_sampler_func_toy(noise, param, input):
    param_term = torch.log(param/(1-param))
    vp = noise*param*(1-input) + noise*(1-param)*input
    noise_term = torch.log(vp/(1-vp))
    return param_term + noise_term




PARAM_HIST_RELAX = []
PARAM_HIST_REINFORCE = []


## This function stores the parameters(theta) moves during the optimization for plot usage
def param_collector(param, collector):
    temp = param + 0
    collector.append(toy_func(0)* (1-temp) + toy_func(1)*temp)





## typical REINFORCE algorithm
# The following function is the implementation of REINFORCE algorithm
# inputs are: parameters(theta), function that used to take expectation,\
# density function of the variable that we take expectation w.r.t., number of iterations to loop on the algorithm
# Samples are taken for the particular toy problem. So, it will be needed minor changes to
def REINFORCE(parameter, function, density_func, iterations):
    param = Variable(parameter, requires_grad=True)
    for iter in range(iterations):
        S = torch.Tensor(REINFORCE_samples_num,1).uniform_(0, 1) + param.data - 0.5
        samples = torch.clamp(S, 0, 1)
        samples = Variable(torch.bernoulli(samples), requires_grad = False)
        func_values = function(samples)
        log_prob = torch.log(density_func(samples, param))
        grad_approximator = torch.dot(func_values, log_prob)
        grad_approximator.backward()
        param.data = torch.clamp(param.data - lr * param.grad.data/samples_num, 0.001, 0.999)

        true_grad = true_grad_of_expectedval(parameter, toy_func, bernoulli_density, input_space)
        bias_compute(param.grad.data/samples_num, true_grad)

        param.grad.data.zero_()
        param_collector(param.data, PARAM_HIST_REINFORCE)
    return param, torch.mean(func_values)


## Implementation of RELAX with reparametrization tricks,
# Inputs of the algorithm are: (refer to the Appendix A algorithm in the paper Backpropagating through the void)
# parameters: initial parameters theta
# function: function f in the algorithm
# approx_net: approximator neural network in the RELAX algorithm
# density_func: density function of variable b
# H_mapping: function H in the algorithm (this function probably would be heaviside in practice)
# noise_init_sampler_func: initial noise sampler function (sampler function for epsilons in the algorithm)
# initial_noise_sampler_function: (refer to first line in the loop of the algorithm in the paper)
# noise_sampler_func: reparametrization trick function to sample noise conditioned on epsilon and theta
# conditional_noise_sampler_func: reparametrization trick function to sample noise conditioned on epsilon and theta and b
# TODO: This function should be divided parts
# implementation is straightforward, using pytorch package to take derivatives, etc.
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

        approx_net_cond_noise = approx_net(conditional_noise)
        approx_net_cond_noise_copy = copy.copy(approx_net_cond_noise)
        approx_net_cond_noise_copy.detach_()
        approx_net_noise = approx_net(noise)
        log_prob = torch.log(density_func(input_sample, param))
        log_prob_grad = torch.autograd.grad(log_prob, param, create_graph=True)
        approx_net_dif_grad = torch.autograd.grad(approx_net_noise - approx_net_cond_noise, param, create_graph=True)

        param_grad_approximator_func = torch.sum((f - approx_net_cond_noise)*log_prob_grad[0] + approx_net_dif_grad[0])

        param.requires_grad = False
        var_grad_approx = 0
        for grad in param_grad_approximator_func:
            var_grad_approx += grad

        var_grad_approx -= torch.dot(approx_net_cond_noise,log_prob)

        var_grad_approx.backward()

        for par in approx_net.parameters():
            if par.grad is not None:
                par.data -= lr2*(2* param_grad_approximator_func[0].data * par.grad.data/samples_num)

        true_grad = true_grad_of_expectedval(parameter, toy_func, bernoulli_density, input_space)
        bias_compute(param_grad_approximator_func[0].data/samples_num, true_grad)

        param.data -= lr1 * param_grad_approximator_func[0].data/samples_num
        param.data = torch.clamp(param.data, 0.001, 0.999)

        param_collector(param.data, PARAM_HIST_RELAX)
        approx_net.zero_grad()

    return param, torch.mean(f)


## Computing the true gradient of E[f(b)] w.r.t. parameters
def true_grad_of_expectedval(parameter, function, density_func, inp_space_values):
    param = Variable(parameter, requires_grad=True)
    input_variable = Variable(inp_space_values, requires_grad = False)
    out = torch.dot(function(input_variable), density_func(input_variable, param))
    out.backward()
    return param.grad.data


## Implementation of REBAR as particular case of RELAX with additional parameters scale_param
## and init_temperature (refer to the REBAR cost function)
# The relaxations of REBAR algorithm is implemented using the approx_net in the RELAX algorithm
def REBAR_Toy(parameter, function, density_func, H_mapping, init_temperature, noise_init_sampler_func,
              noise_sampler_func, conditional_noise_sampler_func, scale_param, iterations):
    approx_net = REBAR_Net(init_temperature, function, scale_param)
    return RELAX_Toy(parameter, function, approx_net, density_func, H_mapping, noise_init_sampler_func,
              noise_sampler_func, conditional_noise_sampler_func, iterations)




## Calculating the difference between true gradient and approximate gradient
def bias_compute(approx_grad, true_grad):
    global bias_sum
    diff = approx_grad - true_grad
    bias_sum += diff
    return diff



## Returns the bias_estimation value, probably for one multi-iteration loop experiment
def get_bias_estimation():
    return bias_sum

## It restarts the bias estimator to the zero value
def reset_bias_estimator():
    global bias_sum
    bias_sum = 0




if __name__ == "__main__":
    print "start"

    parameter = theta
    totsum = 0
    S = []
    for i in range(500):
        function_approx_net = Net()
        X = RELAX_Toy(parameter, toy_func, function_approx_net, bernoulli_density, heaviside,
                           uniform, noise_sampler_func_toy,
                           conditional_noise_sampler_func_toy,iterations=ITERATIONS)
        print(get_bias_estimation())
        if i >= 100:
            S.append(get_bias_estimation())
            totsum += get_bias_estimation()
        reset_bias_estimator()

    print(totsum/400)
    plt.plot(range(400), S)

    #Y = RELAX_Toy(parameter, toy_func, function_approx_net, bernoulli_density, heaviside,
    #                       uniform, noise_sampler_func_toy_2,
    #                       conditional_noise_sampler_func_toy,iterations=ITERATIONS)


    #Y = REBAR_Toy(parameter, toy_func, bernoulli_density, heaviside, init_temperature,
    #                       uniform, noise_sampler_func_toy,
    #                       conditional_noise_sampler_func_toy, scale, iterations=ITERATIONS)
    #print(get_bias_estimation())

    #Z = REINFORCE(parameter, toy_func, bernoulli_density, iterations=ITERATIONS)


    #plt.plot(range(ITERATIONS), PARAM_HIST_RELAX[0:ITERATIONS], color='r')
    #plt.plot(range(ITERATIONS), PARAM_HIST_RELAX[ITERATIONS:2*ITERATIONS], color='g')
    #plt.plot(range(ITERATIONS), PARAM_HIST_REINFORCE, color='b')
    plt.show()