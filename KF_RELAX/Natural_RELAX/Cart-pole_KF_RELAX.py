"""
CARTPOLE
"""
import gym
import torch.optim as optim
import numpy as np

resume = False
render = False


import torch.nn as nn
import torch
from torch.autograd import Variable
from torch.distributions import Bernoulli
from natural_RELAX import RELAX, REINFORCE, sampling_process
from approx_net import RELAX_Net


class Agent(nn.Module):
    def __init__(self, input_dim):
        super(Agent, self).__init__()
        self.linear = nn.Linear(input_dim, 2)
        self.Softmax= nn.Softmax()
    def forward(self, input):
        out = self.Softmax(self.linear(input))
        return out


env = gym.make('CartPole-v0')



#### PARAMETERS
episodes = 5000
learning_rate = 0.001
learning_rate_rlxnet = 0.001
gamma = 0.99
####

if resume:
    agent = torch.load('save.pt')
else:
    agent = Agent(4)


total_reward = []


########## SELECT METHOD


# The method could be REINFORCE, RELAX, REBAR, BASELINE,
method = "REINFORCE"




relax_net = None
if method == "RELAX":
    relax_net = RELAX_Net(1)

# Set K-FAC = 1, if you would use Kronecker-factored approximation for the LAX/RELAX estimator,
k_fac = 0
if k_fac == 1 and method != "LAX" and method != "RELAX":
    raise KeyError("K-FAC is implemented for lax or relax")



forward_hooker = []
backward_hooker = []

def fw_hook(self, input, output):
    forward_hooker.append(input[0])

def bw_hook(self, grad_input, grad_output):
    backward_hooker.append(grad_output[0])



#### Currently, K-FAC just supports the fully-connected layered neural nets.
if k_fac == 1:
    for parameter in relax_net.modules():
        if isinstance(parameter, nn.Linear):
            parameter.register_backward_hook(bw_hook)
            parameter.register_forward_hook(fw_hook)
#relax_net(Variable(torch.FloatTensor([1])))


########## END SELECT METHOD

########## SET UP OPTIMIZERS
optimizer_agent = optim.SGD(agent.parameters(), lr=learning_rate)
if method == "RELAX":
    optimizer_approx_net = optim.SGD(relax_net.parameters(), lr=learning_rate_rlxnet)
########## END OPTIMIZERS


def finish_ep(ep_log_probs, ep_actions, ep_rewards, discounts):
    R = np.flip(np.cumsum(ep_rewards), 0)
    for log_prob, action, r_, discount in zip(ep_log_probs, ep_actions, R, discounts):
        f_val = (r_ / discount)
        if method == "REINFORCE":
            (-log_prob * f_val).backward()
        if method == "RELAX":
            rlx1 = relax_net(action[1])
            rlx2 = relax_net(action[2])

            log_prob_grad = torch.autograd.grad(-log_prob, agent.parameters(), create_graph = True)
            for grad in log_prob_grad:
                grad *= (f_val - rlx2).squeeze(0)
            gradients = torch.autograd.grad(-(rlx1 - rlx2), agent.parameters(), create_graph = True)

            for param, grad1, grad2 in zip(agent.parameters(), gradients, log_prob_grad):
                grad1 += grad2
                param.data -= learning_rate * grad1.data


    if method == "RELAX":
        relax_net.zero_grad()
        G = 0
        for gradient in gradients:
            G += torch.sum(gradient * gradient)
            #print(param.data)
            #print(G)
        G.backward()
    else:
        optimizer_agent.step()
    return


def get_action(observation, bernoulli_obj):
    action, relaxed_action, cond_relaxed_action = None, None, None
    if method == "REINFORCE":
        action = bernoulli_obj.sample()
    elif method == "RELAX" or method == "LAX" or method == "REBAR":
        action, relaxed_action, cond_relaxed_action = sampling_process(agent(observation)[0, 0])
    elif method == "baseline":
        raise KeyError("please use OpenAI baselines!")

    #print(action)
    return action, relaxed_action, cond_relaxed_action





if __name__ == "__main__":

    for i in range(episodes):
        ep_actions = []
        ep_log_probs = []
        ep_rewards = []
        discounts = []
        done = False
        reward = 0
        t = 0
        observation = Variable(torch.FloatTensor(env.reset())).resize(1,4)
        decay = 1
        running_reward = 0

        ####
        ###
        ####
        while not done:
            if render:
                env.render()
            #ep_states.append(obs)
            t += 1

            bernoulli_obj = Bernoulli(agent(observation)[0, 0])
            action, relaxed_action, cond_relaxed_action = get_action(observation, bernoulli_obj)

            observation, reward, done, info = env.step(int(action.data[0]))
            observation = Variable(torch.FloatTensor(observation)).resize(1, 4)

            ep_log_probs.append(bernoulli_obj.log_prob(action))
            ep_actions.append([action, relaxed_action, cond_relaxed_action])
            ep_rewards.append(Variable(torch.FloatTensor([reward * decay])))
            discounts.append(Variable(torch.FloatTensor([decay])))

            decay *= gamma

            running_reward += reward

        finish_ep(ep_log_probs, ep_actions, ep_rewards, discounts)
        #loss.backward()

        #optimizer_agent.step()
        if method == "RELAX":
            optimizer_approx_net.step()
        optimizer_agent.zero_grad()

        total_reward.append(running_reward)

        if i%10 == 0:
            print("episode:", i, "episode reward:", running_reward, "episode duration:", t)

        if i%500 == 0:
            print("average of 500 latest episode rewards:", sum(total_reward[-500:])/500)
            torch.save(agent, 'save.pt')

        if i%2000 == 0:
            print("average of 2000 latest episode rewards:", sum(total_reward[-2000:])/2000)









