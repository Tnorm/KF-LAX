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
batch_size = 50
epoch = 1000
input_dim = 784
h_dim = 64
h_dim2 = 16
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
        self.w2 = torch.nn.Linear(h_dim, h_dim2, bias=True)
        self._enc_mu = torch.nn.Linear(h_dim2, 2)
        self._enc_log_sigma = torch.nn.Linear(h_dim2, 2)
        self.z_mean = torch.FloatTensor()
        self.z_std = torch.FloatTensor()

    def encode(self, x):
        return F.relu(self.w2(F.relu(self.w(x))))

    def reparameterize(self, enc):
        mu = self._enc_mu(enc)
        logvar = self._enc_log_sigma(enc)
        std = logvar.exp_()
        self.z_mean = mu
        self.z_std = std
        eps = torch.randn_like(std)

        return eps.mul(std).add_(mu)

    def forward(self, inp):
        enc = self.encode(inp)
        out = self.reparameterize(enc)
        return out

    def features_num(self, inp):
        size = inp.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

def latent_loss(z_mean, z_stddev):
    mean_sq = z_mean * z_mean
    stddev_sq = z_stddev * z_stddev
    return 0.5 * torch.mean(mean_sq + stddev_sq - torch.log(stddev_sq) - 1)

if resume == True:
    repnet = torch.load('repnet.pt')
else:
    repnet = bpnet(input_dim).to(device)
optimizer = optim.Adam(repnet.parameters(), lr=1e-3)

transform = transforms.Compose(
        [transforms.ToTensor()])


mnist_train = torchvision.datasets.MNIST('KF_RELAX/data/', train=True, download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size,
                                         shuffle=True, num_workers=4)



criterion = nn.MSELoss()
l = None
alpha = 1
#sys.exit()
if train:
    for ep in range(epoch):
        for i, data in enumerate(trainloader, 0):
            inputs, classes = data
            inputs = Variable(inputs.resize_(batch_size, input_dim), requires_grad = True)
            dec = repnet(inputs)
            ll = latent_loss(repnet.z_mean, repnet.z_std)
            grads = torch.autograd.grad(dec.sum(), inputs, create_graph=True)
            loss = alpha * ((inputs - grads[0]).norm(2, 1).sum()/batch_size) + ll
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            l = loss.data[0]
            if i%100 == 0:
                print(i, l,ll)
            if i == 1000:
                grad_show = grads[0].data.clamp(0,1).view(-1, 1, 28, 28)
                input_show = inputs.data.view(-1, 1, 28, 28)

                imshow(torchvision.utils.make_grid(input_show))
                plt.savefig("outputs/" + str(ep) + "_" + str(i) + "d.png")

                imshow(torchvision.utils.make_grid(grad_show))
                plt.savefig("outputs/" + str(ep) + "_" + str(i) + "g.png")
        print(ep, l)

        torch.save(repnet, "repnet.pt")
