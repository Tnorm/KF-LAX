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
import matplotlib.pyplot as plt
import torch.nn.functional as F


class Agent(nn.Module):
    def __init__(self, input_dim):
        super(Agent, self).__init__()
        self.linear = nn.Linear(input_dim, 2)
        self.Rlinear = nn.Linear(input_dim, 2)
        self.Softmax= nn.Softmax()

    def forward(self, input):
        out = F.softmax(self.linear(input))
        Rout = F.softmax(self.Rlinear(input))
        return out, Rout

env = gym.make('CartPole-v0')



#### PARAMETERS
episodes = 5000
learning_rate = 0.0001
learning_rate2 = 0.001
gamma = 0.99
gamma2 = 0.99
####

if resume:
    agent = torch.load('save.pt')
else:
    agent = Agent(4)


optimizer = optim.SGD(agent.parameters(), lr=learning_rate)

keep_skip = 1

total_reward = []

collector = []



def finish_ep(ep_log_probs, ep_actions, ep_rewards, discounts,
              ep_log_probs2, ep_actions2, ep_rewards2, discounts2):
    R = np.flip(np.cumsum(ep_rewards), 0)
    ## Noramalizing Rewards?
    R2 = np.flip(np.cumsum(ep_rewards2), 0)
    ## Noramalizing Rewards?
    for log_prob, action, r_, discount in zip(ep_log_probs, ep_actions, R, discounts):
        f_val = (r_ / discount)
        (-log_prob * f_val).backward()
    if keep_skip == 1:
        for log_prob2, action2, r_2, discount2 in zip(ep_log_probs2, ep_actions2, R2, discounts2):
            f_val2 = (r_2 / discount2)
            (-log_prob2 * f_val2).backward()

    optimizer.step()
    optimizer.zero_grad()

#torch.manual_seed(1)
for i in range(episodes):
    done = False
    reward = 0
    t = 0
    observation = Variable(torch.FloatTensor(env.reset())).resize(1,4)
    decay = 1
    running_reward = 0
    running_reward2 = 0
    ep_actions = []
    ep_actions2 = []
    ep_log_probs = []
    ep_log_probs2 = []
    ep_rewards = []
    ep_rewards2 = []
    discounts = []
    discounts2 = []

    while not done:
        if render:
            env.render()
        #ep_states.append(obs)
        t += 1
        try:
            bernoulli_obj = Bernoulli((agent(observation)[0][0, 0]).clamp(0,1))
            action = bernoulli_obj.sample()
        except:
            print(agent(observation)[0][0, 0])
            break

        observation, reward, done, info = env.step(int(action.data[0]))
        observation = Variable(torch.FloatTensor(observation)).resize(1, 4)

        if keep_skip == 1:
            try:
                bernoulli_obj2 = Bernoulli((agent(observation)[1][0, 0]).clamp(0,1))
            except:
                print(agent(observation)[1][0, 0])
                break
            action2 = bernoulli_obj2.sample()
            if action2 == 1:
                running_reward += reward
                ep_rewards.append(Variable(torch.FloatTensor([reward * decay])))
                discounts.append(Variable(torch.FloatTensor([decay])))
                ep_log_probs.append(bernoulli_obj.log_prob(action))
                ep_actions.append(action)
            running_reward2 += reward
            ep_rewards2.append(Variable(torch.FloatTensor([reward * decay])))
            discounts2.append(Variable(torch.FloatTensor([decay])))
            ep_log_probs2.append(bernoulli_obj2.log_prob(action2))
            ep_actions2.append(action2)
        else:
            running_reward += reward
            ep_rewards.append(Variable(torch.FloatTensor([reward * decay])))
            discounts.append(Variable(torch.FloatTensor([decay])))
            ep_log_probs.append(bernoulli_obj.log_prob(action))
            ep_actions.append(action)

        decay *= gamma

    finish_ep(ep_log_probs, ep_actions, ep_rewards, discounts,
              ep_log_probs2, ep_actions2, ep_rewards2, discounts2)



    if keep_skip:
        total_reward.append(running_reward2)
    else:
        total_reward.append(running_reward)

    #print("episode:", i, "episode reward:", running_reward, "episode duration:", t)

    if i%50 == 0:
        print("average of 50 latest episode rewards:", sum(total_reward[-50:]) / 50)
        collector.append(sum(total_reward[-50:])/50)

    if i%500 == 0:
        print("average of 500 latest episode rewards:", sum(total_reward[-500:])/500)
        torch.save(agent, 'save.pt')

    if i%2000 == 0:
        print("average of 2000 latest episode rewards:", sum(total_reward[-2000:])/2000)



plt.plot(range(episodes/50), collector, color='r')
if keep_skip == 1:
    plt.savefig('keep.png')
else:
    plt.savefig('nokeep.png')




