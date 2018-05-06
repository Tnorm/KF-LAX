## Gradient as VAE!

import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

from torch.autograd import Variable
import sys

resume = False
torch.manual_seed(2)
batch_size = 100
epoch = 1000
input_dim = 784
h_dim = 64
z_dim = 8
device = "cpu"
train = 1

def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))



class bpnet(torch.nn.Module):

    def __init__(self, input_dim):
        super(bpnet, self).__init__()
        # an affine operation: y = Wx + b
        self.w = torch.nn.Linear(input_dim, h_dim, bias=True)
        #self.w2 = torch.nn.Linear(h_dim, h_dim2, bias=True)
        self.wz = torch.nn.Linear(h_dim, z_dim)
        self.z_mean = torch.FloatTensor()
        self.z_std = torch.FloatTensor()
        #self.bn1 = torch.nn.BatchNorm1d(z_dim, affine=False)

    def encode(self, x):
        tmp_enc = self.wz(F.sigmoid(self.w(x)))
        return tmp_enc

    def forward(self, inp):
        out = self.encode(inp)
        return out

    def features_num(self, inp):
        size = inp.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

if resume == True:
    repnet = torch.load('repnet.pt')
else:
    repnet = bpnet(input_dim).to(device)
optimizer = optim.Adam(repnet.parameters(), lr=5e-4)

transform = transforms.Compose(
        [transforms.ToTensor()])


mnist_train = torchvision.datasets.MNIST('KF_RELAX/data/', train=True, download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size,
                                         shuffle=True, num_workers=4)



criterion = nn.MSELoss()
l = None
alpha = 200 #3 not bad
#beta = 1 #1 not bad
#sys.exit()
I = Variable(torch.eye(z_dim))
if train:
    for ep in range(epoch):
        for i, data in enumerate(trainloader, 0):
            inputs, classes = data
            inputs = Variable(inputs.resize_(batch_size, input_dim), requires_grad = True)
            inputs_ = inputs#torch.nn.BatchNorm1d(input_dim)(inputs)
            dec = repnet(inputs_)
            grads = torch.autograd.grad(dec.sum(), inputs_, create_graph=True)
            #regul = dec.grad(2,1).sum()
            #regul = (dec - g_noise2).norm(2,1)
            n_dec = dec/(dec.norm(2,0)+ 1e-16)
            regul = torch.max(torch.abs(torch.matmul(torch.t(n_dec),n_dec)) - I)
            #regul = (dec - torch.rand(dec.size())) ** 2
            #regul = grads[0].norm(2, 1).sum()
            loss = ((inputs_[:,0:input_dim] - grads[0][:,0:input_dim])
                    .norm(2, 1).sum() + alpha * regul)/batch_size

            #for param in repnet.parameters():
            #    loss += beta * param.norm(2)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            l = loss.data[0]
            if i%100 == 0:
                print(i, l, alpha * regul/batch_size)
            if i == 500 and ep % 10 == 0:
                grad_show = grads[0][:,0:input_dim].data.clamp(0,1).view(-1, 1, 28, 28)
                input_show = inputs.data.view(-1, 1, 28, 28)

                imshow(torchvision.utils.make_grid(input_show))
                plt.savefig("outputs/" + str(ep) + "_" + str(i) + "d.png")

                imshow(torchvision.utils.make_grid(grad_show))
                plt.savefig("outputs/" + str(ep) + "_" + str(i) + "g.png")
        print(ep, l)

        torch.save(repnet, "repnet.pt")
