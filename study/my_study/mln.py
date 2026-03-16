#多层感知机
import torch
from torch import nn
from torch.nn import functional as F
class MySequential(nn.Module):
    def __init__(self, *modules):
        super(MySequential, self).__init__()
        for block in modules:
            self._modules[block] = block
    def forward(self, X):
        for block in self._modules.values():
            X = block(X)
        return X
    
# net  = nn.Sequential(nn.Linear(20,256),nn.ReLU(),nn.Linear(256,10))
X=torch.rand(2,20)
# print(net(X))

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.hidden =nn.Linear(20,256)
        self.output = nn.Linear(256,10)
    def forward(self, X):
        return self.output(F.relu(self.hidden(X)))
net = MLP()
print(net(X))