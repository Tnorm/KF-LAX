import torch
from torch.autograd import Variable



class Net(torch.nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # an affine operation: y = Wx + b
        self.sigmoid = torch.nn.Sigmoid()
        self.sigmoid2 = torch.nn.Sigmoid()
        self.L = torch.nn.Linear(2,1)
        self.L2 = torch.nn.Linear(1, 5)

        #self.temp2 = temp

    def forward(self, inp):
        out = self.sigmoid2(self.L2(self.sigmoid(self.L(inp))))
        return out

    def features_num(self, inp):
        size = inp.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features



net = Net()



