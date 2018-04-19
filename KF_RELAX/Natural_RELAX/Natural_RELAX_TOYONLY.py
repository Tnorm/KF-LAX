import torch
from torch.autograd import Variable
from Handful_functions import heaviside
from toy_problem import toy_func, toy_theta, toy_expected_error, toy_true_grad
from neural_net_architecture import RELAX_Net, REBAR_Net
from torch.distributions import Bernoulli
import matplotlib.pyplot as plt
from var_analysis import var_comp
from K_FAC import k_fac_update
import pickle



## typical REINFORCE gradient estimator
# The following function is the implementation of REINFORCE gradient estimator
# inputs are: mini batch function values, input samples,\
# density function of the variable that we take expectation w.r.t., parameters of the density function
def REINFORCE(func_vals, samples, density_func, parameters):
    log_probs = density_func.log_prob(samples)
    grad_approx = torch.autograd.grad(torch.dot(func_vals, log_probs), parameters, create_graph=True)
    return grad_approx


# Relaxing the inputs (see Concrete relaxation/Gumbel-Softmax trick)
def relaxed_input(noise, param):
    return torch.log(param+1e-16) - torch.log(1-param+1e-16) + torch.log(noise+1e-16) - torch.log(1-noise+1e-16)

# Conditional relaxation of the inputs
# Currently only for Bernoulli random variable
def cond_relaxed_input(noise, params, input):
    param_term = torch.log(params+1e-16) - torch.log(1-params+1e-16)
    vp = noise*params*(1-input) + (noise*(1-params)+params)*input
    noise_term = torch.log(vp+1e-16) - torch.log(1-vp+1e-16)
    return param_term + noise_term


# Function that outputs the RELAX gradient estimation
def RELAX_grad_est(func_vals, approx_net, density_func, parameters, samples, relaxed_samples, cond_relaxed_samples):
    func_REINFORCE_grad = REINFORCE(func_vals, samples, density_func, parameters)
    approx_net_REINFORCE_grad = REINFORCE(approx_net(cond_relaxed_samples).squeeze(), samples, density_func, parameters)

    approx_net_REPARAM_grad_rs = torch.autograd.grad(torch.sum(approx_net(relaxed_samples)),
                                                     parameters, create_graph=True) # rs = relaxed samples
    approx_net_REPARAM_grad_crs = torch.autograd.grad(torch.sum(approx_net(cond_relaxed_samples)),
                                                     parameters, create_graph=True) # crs = conditional relaxed samples

    RELAX_grad = func_REINFORCE_grad[0] - approx_net_REINFORCE_grad[0] + approx_net_REPARAM_grad_rs[0] - \
                 approx_net_REPARAM_grad_crs[0]

    return RELAX_grad

# Estimation of gradient of variance of the parameters
# The gradient is with respect to the approximation function (the neural net) parameters
def var_reduction(grad_estimation):
    #grad = torch.autograd.grad(torch.sum(grad_estimation), approx_net.parameters(), create_graph=True)
    (torch.sum(2*grad_estimation.detach()*grad_estimation)).backward()
    return


# Sampling process, including relaxation, conditional relaxation, and samples to feed the function
def sampling_process(parameters):
    #TODO: the way of choosing uniforms u and v should be fixed for higher parameter space dimensions

    u, v = Variable(torch.rand(parameters.shape)), Variable(torch.rand(parameters.shape))
    relaxed_samples = relaxed_input(u, parameters)
    samples = heaviside(relaxed_samples).detach_()
    samples = samples.type(torch.FloatTensor)
    cond_relaxed_samples = cond_relaxed_input(v, parameters, samples)
    return samples, relaxed_samples, cond_relaxed_samples


# Evaluating the function values
def function_eval(function, samples):
    return function(samples)

# The function returns the RELAX gradient estimation and the gradient of the variance of the gradient estimation
def RELAX(function, approx_net, density_func, parameters):
    samples, relaxed_samples, cond_relaxed_samples = sampling_process(parameters)
    func_vals = function_eval(function, samples)
    grad_estimation = RELAX_grad_est(func_vals, approx_net, density_func, parameters,
                                     samples, relaxed_samples, cond_relaxed_samples)
    var_reduction(grad_estimation)
    return grad_estimation


## K_FAC
k_fac = 1

forward_hooker = []
backward_hooker = []

def fw_hook(self, input, output):
    forward_hooker.append(input[0])

def bw_hook(self, grad_input, grad_output):
    backward_hooker.append(grad_output[0])



####

# REBAR as a particular case of RELAX
def REBAR(function, REBAR_net, density_func, parameters):
    return RELAX(function, REBAR_net, density_func, parameters)


######### LEARNING PARAMETERS, SCALE PARAMETER(REBAR), INITIAL TEMPERATURE(REBAR), iterations and batch size
#lr = torch.Tensor([0.005])
lr1 = torch.Tensor([0.05])
lr2 = torch.Tensor([0.002])

init_temperature = 1
scale_param = 1

iterations = 10000
batch_size = 30

#########


#check
rebar = 0
relax = 1
reinforce = 0


torch.manual_seed(7)
if __name__ == "__main__":
    approx_net = RELAX_Net()
    parameters = toy_theta.repeat(batch_size)
    if rebar == 1:
        rebar_net = REBAR_Net(init_temperature, toy_func, scale_param)
        approx_net = rebar_net


    ### Add hooking!
    if k_fac == 1:
        approx_net.w.register_backward_hook(bw_hook)
        approx_net.w.register_forward_hook(fw_hook)
        #for parameter in approx_net.parameters():
        #    print(parameter.register_backward_hook)
        #    parameter.register_backward_hook(bw_hook)
        #    parameter.register_forward_hook(fw_hook)
    ###


    optimizer = torch.optim.SGD(approx_net.parameters(), lr=lr1)
    S = []
    for i in range(iterations):
        if relax == 1:
            parameters_grad = RELAX(toy_func, approx_net, Bernoulli(parameters), parameters)
        #if reinforce == 1:
        #parameters_grad = REINFORCE(toy_func, approx_net, Bernoulli(parameters), parameters)


        if k_fac == 1:
            kfac_delta = k_fac_update(forward_hooker[0].data, backward_hooker[0].data, approx_net.w.weight.grad.data)
            approx_net.w.weight.data -= lr1 * kfac_delta / batch_size
            approx_net.w.bias.data -= lr1 * approx_net.w.bias.grad.data / batch_size
        elif relax == 1 and rebar == 0:
            approx_net.w.weight.data -= lr1 * approx_net.w.weight.grad.data / batch_size
            approx_net.w.bias.data -= lr1 * approx_net.w.bias.grad.data / batch_size
        else:
            for parameter in approx_net.parameters():
                if parameter.grad is not None:
                    parameter.data -= lr1 * parameter.grad.data / batch_size
        approx_net.zero_grad()


        ## Updating parameters
        parameters.data = (parameters.data - lr2 * torch.mean(parameters_grad.data)).clamp(0,1)
        #parameters.data =  (parameters.data - toy_true_grad().data).clamp(0,1)
        #print(var_comp(parameters_grad.data))
        S.append(toy_expected_error(parameters.data[0]).data[0])
        #S.append(var_comp(parameters_grad.data))



    #print parameters
    plt.plot(range(iterations), S, color='r')
    plt.show()
    with open('kf_rbr', 'wb') as fp:
        pickle.dump(S, fp)
