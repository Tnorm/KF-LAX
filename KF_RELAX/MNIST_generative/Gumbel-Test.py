import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.cuda
from torch.distributions import Bernoulli

import torchvision.transforms as transforms
import torchvision

batch_size = 100
input_layer_size = 784
init_temperature = Variable(torch.FloatTensor([5]))
temperature = init_temperature
epochs= 10

anneal_rate= Variable(torch.FloatTensor([0.5]))
min_temperature= Variable(torch.FloatTensor([0.5]))

classes_num = 10
categorical_dists_num = 30

if torch.cuda.is_available():
    use_cuda = True
else:
    use_cuda = False


resume = False



def gumbel(shape):
    g = Variable(torch.rand(shape))
    return -torch.log(-torch.log(g+1e-20) + 1e-20)


def gumbel_softmax(logits, temperature, hard=False):
    if hard:
        return None
        #pass
    sftmax = torch.nn.Softmax(dim=2)
    return sftmax((gumbel(logits.size()) + logits)/temperature)

def data_load(name):
    if name == "MNIST":
        kwargs = {'num_workers': 2, 'pin_memory': True} if use_cuda else {}
        train_loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST('../data', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST('../data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])),
            batch_size=batch_size, shuffle=True, **kwargs)

        return train_loader, test_loader
    elif name == "Omniglot":
        return None

class VAE(torch.nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.ehl1 = torch.nn.Linear(input_layer_size, 512)
        self.ehl2 = torch.nn.Linear(512, 256)
        self.elogit_hl = torch.nn.Linear(256, categorical_dists_num*classes_num)

        self.dhl1 = torch.nn.Linear(categorical_dists_num*classes_num, 256)
        self.dhl2 = torch.nn.Linear(256, 512)
        self.dlogit_hl = torch.nn.Linear(512, input_layer_size)


        self.sftmax = torch.nn.Softmax(dim=2)
        self.drop_p = 0.5
        self.drpout = torch.nn.Dropout(self.drop_p)
        # an affine operation: y = Wx + b


    def encoder(self, x):
        hl1 = F.relu(self.ehl1(x))  ## batch*hidden_layer_size
        hl2 = F.relu(self.ehl2(hl1))  ## batch*hidden_layer_size
        logits = (self.elogit_hl(hl2)).view(-1, categorical_dists_num, classes_num)
        q_z = self.sftmax(logits)
        log_q_z = torch.log(q_z + 1e-20)
        return logits, q_z, log_q_z

    def decoder(self, z):
        #hl1 = F.relu(self.dhl1(self.drpout(
        #    z.view(-1, categorical_dists_num * classes_num))))  ## batch*hidden_layer_size
        hl1 = F.relu(self.dhl1(
            z.view(-1, categorical_dists_num * classes_num)))  ## batch*hidden_layer_size
        hl2 = F.relu(self.dhl2(hl1))  ## batch*hidden_layer_size
        p_x = torch.sigmoid(self.dlogit_hl(hl2)).clamp(0.0, 1.0)
        B = Bernoulli(p_x)
        x_hat = B.sample()
        return x_hat, B.log_prob(x_hat)

    def forward(self, x):
        z, q_z, log_q_z = self.encoder(x)
        gumbels = gumbel_softmax(z, temperature)  # I prefer to name it gumbelZ like bumbelZ
        x_h, log_p_x = self.decoder(gumbels)
        return x_h, log_p_x, z, q_z, log_q_z


if __name__ == "__main__":


    trainloader, testloader = data_load("MNIST")

    if resume:
        vae = torch.load('gumbel_vae.pt')
    else:
        vae = VAE()

    if use_cuda:
        vae.cuda()
    optimizer = torch.optim.Adam(vae.parameters(), lr=0.001)
    for i in range(epochs):
        for j, data in enumerate(trainloader, 0):
            inputs, labels = data
            if use_cuda:
                inputs, labels = inputs.cuda(), labels.cuda()
            inputs = inputs.view(-1, 784)
            #x = Variable(torch.rand([batch_size, input_layer_size]))
            x_h, log_p_x, z, q_z, log_q_z = vae(Variable(inputs))

            #kl_loss = (q_z * (log_q_z - torch.log(1.0/vae.drop_p/classes_num*Variable(torch.ones(log_q_z.size()))))).sum(2).sum(1)
            kl_loss = (
            q_z * (log_q_z - torch.log(1.0/classes_num * Variable(torch.ones(log_q_z.size()))))).sum(2).sum(1)
            log_prob_loss = log_p_x.sum(1)
            neg_elbo = torch.mean((-kl_loss + log_prob_loss)*-1)

            neg_elbo.backward()
            optimizer.step()
            optimizer.zero_grad()

            if j%100 == 0:
                print(j, "I'm still alive. And, right now, elbo is:", -neg_elbo.data[0])

            if(j % 1000 == 0):
                torch.save(vae, 'gumbel_vae.pt')
            #    temperature = torch.max(init_temperature*torch.exp(-(j+1)*anneal_rate), min_temperature)
        print("epoch {} elbo: {}".format(i, -neg_elbo.data[0]))