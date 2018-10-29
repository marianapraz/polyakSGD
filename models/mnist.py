import torch.nn as nn
import torch as th
from warnings import warn
from numpy import inf

from .utils import View, num_parameters, Id, norm_str_parse, Activation
from stablenets import  snn
from stablenets.snn.simplexproj import SimplexProj



class LeNet(nn.Module):
    def __init__(self, bn=False, null_pseudocount=0.,
             classes=10, dropout=0., 
             sigmoid = False,  simplexproj=False,  eps=0.,
             lipschitz_cutoff=inf, **kwargs):
        """Implementation of LeNet [1].

        [1] LeCun Y, Bottou L, Bengio Y, Haffner P. Gradient-based learning applied to
               document recognition. Proceedings of the IEEE. 1998 Nov;86(11):2278-324."""
        super().__init__()
        self.nl = 4 if not bn else 7

        assert sigmoid in [True, False, 'all', 'final']
        sigmoidall = sigmoid=='all'

        try:
            norms=kwargs['norms']
        except KeyError as e:
            norms='inf'
        norms = norm_str_parse(norms)

        def convbn(ci,co,ksz,psz,p,norm_type='inf,inf',bn=False):
            s = norm_type.split(',')
            conv = snn.Conv2d(ci,co,ksz, norm_type=norm_type, bias=True,
                    lipschitz_cutoff=lipschitz_cutoff)

            if not bn:
                bn = Id()
            else:
                bn = snn.BatchNorm2d(co,norm_type=s[1]+','+s[1], 
                        affine=False, lipschitz_cutoff=lipschitz_cutoff)

            if not sigmoidall:
                m = nn.Sequential(
                    conv,
                    bn,
                    Activation(),
                    nn.MaxPool2d(psz,stride=psz),
                    nn.Dropout(p))
            else:
                sbn = nn.BatchNorm2d(co, affine=False)
                m = nn.Sequential(
                        conv,
                        bn,
                        Activation(),
                        nn.MaxPool2d(psz,stride=psz),
                        sbn,
                        snn.Sigmoid(),
                        nn.Dropout(p))
            return m



        lin = snn.Linear(50*2*2, 500, norm_type=norms[1], bias=True,
                lipschitz_cutoff=lipschitz_cutoff)
        if not bn:
            lbn = Id()
        else:
            lbn = snn.BatchNorm1d(500, norm_type=norms[1], affine=False,
                    lipschitz_cutoff=lipschitz_cutoff)

        if sigmoidall:
            lin = nn.Sequential(lin, lbn, Activation(), 
                                nn.BatchNorm1d(500, affine=False),
                                snn.Sigmoid(),
                                nn.Dropout())
        else:
            lin = nn.Sequential(lin, lbn, Activation(), 
                                nn.Dropout())


        self.m = nn.Sequential(
            convbn(1,20,5,3,dropout, norm_type=norms[0],bn=bn),
            convbn(20,50,5,2,dropout, norm_type=norms[1],bn=bn),
            View(50*2*2),
            lin,
            snn.Linear(500,classes, norm_type=norms[2], bias=True,
                       lipschitz_cutoff=lipschitz_cutoff))


        self.sigmoid = sigmoid
        if sigmoid:
            self.bn = nn.BatchNorm1d(classes, affine=False)

        self.simplexproj = simplexproj
        self._eps = eps
        self.classes=classes
        if simplexproj:
            self.proj = SimplexProj(z=1-eps,dim=-1)
            self.t = nn.Parameter(th.tensor(1.))
        self.nonlinear=True



    @property
    def name(self):
        return self.__class__.__name__

    def forward(self, x):
        x = self.m(x)
        if self.sigmoid:
            x = self.bn(x)
            if self.nonlinear:
                x = th.tanh(x/self.t.abs())
                x = self.t.abs() * x

        if self.simplexproj and self.nonlinear:
            x, xc = self.proj(x)
            y = x + self._eps/self.classes
        elif self.nonlinear:
            y = x.softmax(dim=-1)
            xc = y-x
        else:
            y = x
            xc = 0
        
        return y, xc
