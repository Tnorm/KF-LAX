import torch
from torch.autograd import Variable


### Cart-pole RELAX_net
class RELAX_Net(torch.nn.Module):

    def __init__(self, input_dim):
        super(RELAX_Net, self).__init__()
        # an affine operation: y = Wx + b
        self.w = torch.nn.Linear(input_dim+1, 1, bias=False)
        #self.w2 = torch.nn.Linear(1, 1)
        #self.m = torch.nn.Softmax()
        #self.a = torch.nn.Parameter(torch.FloatTensor(1))
        #self.b = torch.nn.Parameter(torch.FloatTensor(1))
        #self.c = torch.nn.Parameter(torch.FloatTensor(1))
        #self.b = torch.nn.Parameter(torch.FloatTensor(1))

    def forward(self, inp):
        #out = self.m*inp
        #out = (self.m - self.rnd) * inp + self.b
        #out = self.m * inp + self.b

        inp_u = inp.unsqueeze(1)
        input = torch.cat((inp_u, Variable(torch.ones(inp_u.shape[0],1))), 1)
        out = self.w(input)
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