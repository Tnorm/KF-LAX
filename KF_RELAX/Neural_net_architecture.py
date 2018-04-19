import torch
from torch.autograd import Variable

## Neural network that is used in RELAX algorithm is defined here
class RELAX_Net(torch.nn.Module):

    def __init__(self):
        super(RELAX_Net, self).__init__()
        # an affine operation: y = Wx + b
        self.w = torch.nn.Linear(1, 1)
        self.w2 = torch.nn.Linear(1, 1)
        #self.m = torch.nn.Softmax()
        self.a = torch.nn.Parameter(torch.FloatTensor(1))
        self.b = torch.nn.Parameter(torch.FloatTensor(1))
        self.c = torch.nn.Parameter(torch.FloatTensor(1))
        #self.b = torch.nn.Parameter(torch.FloatTensor(1))

    def forward(self, inp):
        #out = self.m*inp
        #out = (self.m - self.rnd) * inp + self.b
        #out = self.m * inp + self.b
        inputs = inp.unsqueeze(1)
        out = torch.nn.Sigmoid()(self.w(inputs))
        #out = torch.nn.Sigmoid()(self.a * (inputs ** 1) + self.b * inputs + self.c)
        #out = self.w2(out)
        #return self.w* inp
        return out

    def features_num(self, inp):
        size = inp.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def turn_off_grad(self):
        for param in self.parameters():
            param.requires_grad = False

    def turn_on_grad(self):
        for param in self.parameters():
            param.requires_grad = True


## Neural network that is used in REBAR algorithm is defined here
# Since the REBAR algorithm is implemented as particular case of RELAX we used a particular neural network \
# for relaxations in REBAR algorithm.
# The neural network architecture must not be changed if we would use REBAR algorithm precisely.
class REBAR_Net(torch.nn.Module):

    def __init__(self, temp, function, scale_param):
        super(REBAR_Net, self).__init__()
        # an affine operation: y = Wx + b
        self.sigmoid = torch.nn.Sigmoid()
        self.function = function
        self.temp = torch.nn.Parameter(torch.FloatTensor([temp]), requires_grad = True)
        self.scale = torch.nn.Parameter(torch.FloatTensor([scale_param]), requires_grad = True)
        #self.temp2 = temp

    def forward(self, inp):
        inputs = inp.unsqueeze(1)
        out = self.function(self.sigmoid(inputs/self.temp))*self.scale
        #out = self.function(self.sigmoid(inputs / self.temp2)) * self.scale
        return out

    def features_num(self, inp):
        size = inp.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features





