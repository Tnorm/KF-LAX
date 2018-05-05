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
batch_size = 10
epoch = 60
h_dim = 64
device = "cpu"
train = 1

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))



class bpnet(torch.nn.Module):

    def __init__(self):
        super(bpnet, self).__init__()
        # an affine operation: y = Wx + b
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(16)

        self._enc_mu = torch.nn.Linear(8 * 8 * 16, 8)
        self._enc_log_sigma = torch.nn.Linear(8 * 8 * 16, 8)
        self.z_mean = torch.FloatTensor()
        self.z_std = torch.FloatTensor()

    def encode(self, x):
        conv1 = F.sigmoid(self.bn1(self.conv1(x)))
        conv2 = F.sigmoid(self.bn2(self.conv2(conv1)))
        conv3 = F.sigmoid(self.bn3(self.conv3(conv2)))
        conv4 = F.sigmoid(self.bn4(self.conv4(conv3))).view(-1, 8 * 8 * 16)
        return conv4

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
    repnet = bpnet().to(device)
optimizer = optim.Adam(repnet.parameters(), lr=1e-3)


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

cifar_train = torchvision.datasets.CIFAR10('data/', train=True, download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(cifar_train, batch_size=batch_size,
                                         shuffle=True, num_workers=4)


criterion = nn.MSELoss()
l = None
alpha = 0.02
#sys.exit()
if train:
    for ep in range(epoch):
        for i, data in enumerate(trainloader, 0):
            inputs, classes = data
            inputs = Variable(inputs, requires_grad = True)
            dec = repnet(inputs)
            ll = latent_loss(repnet.z_mean, repnet.z_std)
            grads = torch.autograd.grad(dec.sum(), inputs, create_graph=True)
            loss = alpha * (inputs - grads[0]).norm(2, 1).sum()/batch_size + ll
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            l = loss.data[0]
            if i%100 == 0:
                print(i, l, ll)
            if i == 4900:
                grad_show = grads[0].data.clamp(-1,1).view(-1, 3, 32, 32)
                input_show = inputs.data.view(-1, 3, 32, 32)
                imshow(torchvision.utils.make_grid(input_show))
                plt.savefig("outputs/" + str(ep) + "_" + str(i) + "d.png")

                imshow(torchvision.utils.make_grid(grad_show))
                plt.savefig("outputs/" + str(ep) + "_" + str(i) + "g.png")
        print(ep, l)

        torch.save(repnet, "repnet.pt")
