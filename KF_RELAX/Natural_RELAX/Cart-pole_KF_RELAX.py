"""
CARTPOLE
"""
import gym
import torch.optim as optim
import numpy as np
import sys
import matplotlib.pyplot as plt

resume = False
render = False


import torch.nn as nn
import torch
from torch.autograd import Variable
from torch.distributions import Bernoulli
from natural_RELAX import RELAX, REINFORCE, sampling_process
from approx_net import RELAX_Net
from K_FAC import KFAC
from Handful_functions import saver


class Agent(nn.Module):
    def __init__(self, input_dim):
        super(Agent, self).__init__()
        self.linear = nn.Linear(input_dim, 2)
        self.Softmax= nn.Softmax(dim=1)
    def forward(self, input):
        out = self.Softmax(self.linear(input))
        return out


env = gym.make('CartPole-v0')



#### PARAMETERS
episodes = 3000
learning_rate_agent = 0.001
learning_rate_rlxnet = 0.001
gamma = 0.99
####

if resume:
    agent = torch.load('save.pt')
else:
    agent = Agent(4)


total_reward = []
collector = []


########## SELECT METHOD


# The method could be REINFORCE, RELAX, REBAR, BASELINE,
method = "RELAX"


torch.manual_seed(18)


relax_net = None
if method == "RELAX":
    relax_net = RELAX_Net(1)

# Set K-FAC = 1, if you would use Kronecker-factored approximation for the LAX/RELAX estimator,
k_fac = 1
if k_fac == 1 and method != "LAX" and method != "RELAX":
    raise KeyError("K-FAC is implemented for lax or relax")

forward_hooker = []
fws = []
backward_hooker = []
bws = []

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
optimizer_agent = optim.SGD(agent.parameters(), lr=learning_rate_agent)
if method == "RELAX":
    optimizer_approx_net = optim.SGD(relax_net.parameters(), lr=learning_rate_rlxnet)
########## END OPTIMIZERS


def finish_ep(ep_log_probs, ep_actions, ep_rewards, discounts):
    R = np.flip(np.cumsum(ep_rewards), 0)
    ## normalizing rewards?
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
                param.data -= learning_rate_agent * grad1.data

    if method == "RELAX":
        if k_fac == 1:
            global backward_hooker, fws, bws
            fws = forward_hooker_mean_aat()
        G = 0
        for gradient in gradients:
            G += torch.sum(gradient * gradient)
        backward_hooker = []
        relax_net.zero_grad()
        G.backward()
        print(backward_hooker)
        if k_fac == 1:
            bws = backward_hooker_mean_ggt()
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


def forward_hooker_mean_aat():
    tmp_fw_hook = []
    for hook in forward_hooker[0]:
        tmp_fw_hook.append(Variable(torch.zeros(hook.size()[0], hook.size()[0])))
    for forward_hook in forward_hooker:
        for tmp, hook in zip(tmp_fw_hook, forward_hook):
            tmp += torch.mm(hook.view(hook.size()[0], 1), hook.view(1, hook.size()[0]))

    tmp_fw_hook = [x / len(forward_hooker) for x in tmp_fw_hook]

    return tmp_fw_hook

def backward_hooker_mean_ggt():
    tmp_bw_hook = []
    for hook in backward_hooker[0]:
        tmp_bw_hook.append(Variable(torch.zeros(hook.size()[0], hook.size()[0])))
    for backward_hook in backward_hooker:
        for tmp, hook in zip(tmp_bw_hook, backward_hook):
            tmp += torch.mm(hook.view(hook.size()[0], 1), hook.view(1, hook.size()[0]))

    tmp_bw_hook = [x / len(backward_hooker) for x in tmp_bw_hook]

    return tmp_bw_hook


if __name__ == "__main__":
    for i in range(episodes):
        ep_actions = []
        ep_log_probs = []
        ep_rewards = []
        discounts = []
        forward_hooker = []
        backward_hooker = []
        all_fws = []
        all_bckws = []
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
            try:
                bernoulli_obj.sample()
            except:
                print(agent(observation)[0, 0], observation)
                sys.exit()
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
            if k_fac == 0:
                optimizer_approx_net.step()
            elif k_fac == 1:
                for fw, bw, parameter in zip(fws, bws, relax_net.parameters()): # or reverse(bws) ??
                    if parameter.grad is not None:
                        if k_fac == 1:
                            kfac_delta = torch.t(KFAC(fw.data, bw.data,
                                                  torch.t(parameter.grad.data)))
                            parameter.data -= learning_rate_rlxnet * kfac_delta

            #optimizer_approx_net.step()
        optimizer_agent.zero_grad()
        total_reward.append(running_reward)

        if i%10 == 0:
            print("episode:", i, "average of 500 latest episode rewards:", sum(total_reward[-100:])/100)
            collector.append(sum(total_reward[-500:])/500)
            np.save('plotfigs/' + method + str(k_fac) + '.npy', {'method': method, 'res': collector})
            #plt.plot(range(len(total_reward)), total_reward)
            #plt.pause(0.5)

        if i%500 == 0:
            print("average of 500 latest episode rewards:", sum(total_reward[-500:])/500)
            torch.save(agent, 'save.pt')

        if i%2000 == 0:
            print("average of 2000 latest episode rewards:", sum(total_reward[-2000:])/2000)






