import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
import torchvision
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable



Mus = None
Std = None
h_dim = 3
h_dim2 = 16


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
        global Mus
        Mus = mu
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
        [transforms.ToTensor()])

batch_size = 500

mnist_test = torchvision.datasets.MNIST('KF_RELAX/data/', train=False, download=True, transform=transform)

dataloader = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size,
                                         shuffle=True, num_workers=4)

def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

dataiter = iter(dataloader)
images, labels = dataiter.next()
noisy_images = images + torch.randn(images.size())/2.5
all_images = torch.cat((images, noisy_images), 0).clamp(0,1)
#imshow(torchvision.utils.make_grid(all_images))
#plt.savefig("data.png")
repnet = torch.load('repnet.pt')

inputs = Variable(all_images.view(-1,784), requires_grad = True)
dec = repnet(inputs)
grads = torch.autograd.grad(dec.sum(), inputs)

grad_show = grads[0].view(-1, 1, 28, 28)

#imshow(torchvision.utils.make_grid(grad_show.data.clamp(0,1)))
#plt.savefig("grad.png")


#Data = torch.normal(Mus, Std)
Data = Mus
Data2 = Data
D_mean = torch.mean(Data, 0)
Data2 = Data2 - D_mean.expand_as(Data2)
U,S,V = torch.svd(torch.t(Data2))
PCA = torch.mm(Data2, U[:, :2])
print(PCA)

print(Data)
fig, ax = plt.subplots()
for label in np.unique(labels[0:3]):
    ix = np.where(labels == label)
    ax.scatter(Data.data[ix,0], Data.data[ix,1], label = str(label), s=15)
    #ax.scatter(PCA.data[ix, 0], PCA.data[ix, 1], label=str(label), s=50)

ax.legend()
plt.show()