import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
import torchvision
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable


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



transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 32

cifar_test = torchvision.datasets.CIFAR10('data/', train=False, download=True, transform=transform)

dataloader = torch.utils.data.DataLoader(cifar_test, batch_size=batch_size,
                                         shuffle=True, num_workers=4)

def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

dataiter = iter(dataloader)
images, labels = dataiter.next()
noisy_images = images + torch.randn(images.size())/2.5
all_images = torch.cat((images, noisy_images), 0).clamp(-1,1)
imshow(torchvision.utils.make_grid(all_images))
plt.savefig("data.png")
repnet = torch.load('repnet.pt')

inputs = Variable(all_images, requires_grad = True)
dec = repnet(inputs)
grads = torch.autograd.grad(dec.sum(), inputs)

grad_show = grads[0].view(-1, 3, 32, 32)

imshow(torchvision.utils.make_grid((grad_show.data).clamp(-1,1)))
plt.savefig("grad.png")
