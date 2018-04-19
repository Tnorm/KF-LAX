import torch
from Handful_functions import uniform, heaviside
from torch.autograd import Variable
import matplotlib.pyplot as plt


### In this code we assumed lambda = 1, eta = 1

samples = 1


def g_func(u, theta):
    return torch.log(theta / (1 - theta)) + torch.log(u / (1 - u))


def g_tilde_func(v, theta, b):
    if(b == 1):
        return torch.log(v/(1-v)*1/(1-theta) + 1)
    elif(b == 0):
        return -torch.log(v/(1-v)*1/theta + 1)

def bernoulli(b, theta):
    return theta * b + (1-theta) * (1-b)


def toy_func(b):
    return (b - 0.3) ** 2

theta = Variable(torch.FloatTensor([0.6]), requires_grad = True)



iters = 5000
accumulated_grad_estimate = 0
#bias_estimator_antithetic = 0
accumulated_grad_estimate_true = 0
b1 = []
b2 = []



###### Monte Carlo gradient estimation of the second and forth terms
###### at the right side of the equation 4 of the REBAR paper
###### assuming u, v sampled independently
for i in range(iters):
    u, v = Variable(uniform(1), requires_grad = False), Variable(uniform(1), requires_grad = False)
    z = g_func(u, theta)
    b = heaviside(z)
    b = b.type(torch.FloatTensor)
    b.detach_()
    z_tilde = g_tilde_func(v, theta, b)
    log_prob_grad = torch.autograd.grad(torch.log(bernoulli(b, theta)), theta, create_graph=True)
    f_z_tilde_grad = torch.autograd.grad(toy_func(torch.sigmoid(z_tilde)), theta, create_graph=True)
    f_z_grad = torch.autograd.grad(toy_func(torch.sigmoid(z)), theta, create_graph=True)
    accumulated_grad_estimate += (toy_func(b) - toy_func(torch.sigmoid(z_tilde)))*log_prob_grad[0] \
                                 + f_z_grad[0] - f_z_tilde_grad[0]
    b1.append(accumulated_grad_estimate.data + 0)

print(accumulated_grad_estimate/iters)


###### Monte Carlo gradient estimation of the second and forth terms
###### at the right side of the equation 4 of the REBAR paper
###### assuming sampled u = v
for j in range(iters):
    u = Variable(uniform(1), requires_grad = False)
    z = g_func(u, theta)
    b = heaviside(z)
    b = b.type(torch.FloatTensor)
    b.detach_()
    z_tilde = g_tilde_func(u, theta, b)
    log_prob_grad = torch.autograd.grad(torch.log(bernoulli(b, theta)), theta, create_graph=True)
    f_z_tilde_grad = torch.autograd.grad(toy_func(torch.sigmoid(z_tilde)), theta, create_graph=True)
    f_z_grad = torch.autograd.grad(toy_func(torch.sigmoid(z)), theta, create_graph=True)
    accumulated_grad_estimate_true += (toy_func(b) - toy_func(torch.sigmoid(z_tilde)))*log_prob_grad[0] \
                                      + f_z_grad[0] - f_z_tilde_grad[0]
    b2.append(accumulated_grad_estimate_true.data + 0)

print(accumulated_grad_estimate_true/iters)


print((accumulated_grad_estimate_true - accumulated_grad_estimate)/iters)


plt.plot(range(iters), b1, color='r')
plt.plot(range(iters), b2, color='g')
plt.ylabel('acummulated 1 sample Monte Carlo gradient estimate')
plt.xlabel('samples number')

plt.show()