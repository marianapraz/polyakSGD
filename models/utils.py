import torch.nn as nn
from stablenets import snn
import torch as th
from torch.nn import functional as F

# Return the flattened array
class View(nn.Module):
    def __init__(self,o):
        super(View, self).__init__()
        self.o = o
    def forward(self,x):
        return x.view(-1, self.o)

class Avg2d(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,x):
        sh = x.shape
        x = x.contiguous().view(sh[0], sh[1], -1)
        return x.mean(-1)

def num_parameters(model):
    return sum([w.numel() for w in model.parameters()])

class Expand(nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        self.out_channels = out_channels

    def forward(self,x):
        sh = list(x.shape)
        sh[1] =  self.out_channels
        return x.expand(*sh)

class Repeat(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.n = n

    def forward(self,x):
        sh = [1 for i in range(len(x.shape))]
        sh[1] = self.n

        return x.repeat(*sh).contiguous()

class Id(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,x):
        return x

class Activation(nn.Module):
    def __init__(self):
        super().__init__()
        self.nonlinear = True

    def forward(self, x):
        y = F.relu(x) if self.nonlinear else x
        return y

def norm_str_parse(s):
    l = s.split(',')
    if len(l)==1:
        return tuple([s+','+s]*3)
    elif len(l)==3:
        return (l[0]+','+l[1],
                l[1]+','+l[1],
                l[1]+','+l[2])
