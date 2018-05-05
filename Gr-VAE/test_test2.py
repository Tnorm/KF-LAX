import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
import torchvision
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable


Std = None
h_dim = 64
z_dim = 2
h_dim2 = 16

class bpnet(torch.nn.Module):

    def __init__(self, input_dim):
        super(bpnet, self).__init__()
        # an affine operation: y = Wx + b
        self.w = torch.nn.Linear(input_dim, h_dim, bias=True)
        #self.w2 = torch.nn.Linear(h_dim, h_dim2, bias=True)
        self.wz = torch.nn.Linear(h_dim, z_dim)
        self.z_mean = torch.FloatTensor()
        self.z_std = torch.FloatTensor()

    def encode(self, x):
        return self.wz(F.sigmoid(self.w(x)))

    def forward(self, inp):
        out = self.encode(inp)
        return out

    def features_num(self, inp):
        size = inp.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features



transform = transforms.Compose(
        [transforms.ToTensor()])

batch_size = 4096

mnist_test = torchvision.datasets.MNIST('KF_RELAX/data/', train=False, download=True, transform=transform)

dataloader = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size,
                                         shuffle=True, num_workers=4)

def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

dataiter = iter(dataloader)
images, labels = dataiter.next()
noisy_images = images + torch.randn(images.size())/2.5
all_images = images
#all_images = torch.cat((images, noisy_images), 0).clamp(0,1)
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
Data = dec.data
Data2 = Data
D_mean = torch.mean(Data, 0)
Data2 = Data2 - D_mean.expand_as(Data2)
U,S,V = torch.svd(torch.t(Data2))
PCA = torch.mm(Data2, U[:, :2])
print(PCA)

fig, ax = plt.subplots()
for label in np.unique(labels[:3]):
    ix = np.where(labels == label)
    ax.scatter(Data.data[ix,0], Data.data[ix,1], label = str(label), s=15)
    #ax.scatter(PCA.data[ix, 0], PCA.data[ix, 1], label=str(label), s=20)

ax.legend()
plt.show()