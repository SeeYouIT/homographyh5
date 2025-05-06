import torch
import torch.nn as nn

import torch.nn.functional as F


class MyLossFunction(nn.Module):
    ##1,10,0.01,10,1,0.1,10,10,1
    def __init__(self, t0, t1, t2, t3,t4, t5,t6, t7,t8):
        super(MyLossFunction, self).__init__()

        self.t0 = t0
        self.t1 = t1
        self.t2 = t2
        self.t3 = t3
        self.t4 = t4
        self.t5 = t5
        self.t6 = t6
        self.t7 = t7
        self.t8 = t8
        return

    def forward(self, htest, Hbe):
        lossr00 = F.l1_loss(htest[:,0,0] , Hbe[:,0,0])
        lossr01 = F.l1_loss(htest[:,0,1] , Hbe[:,0,1])
        lossr02 =  F.l1_loss(htest[:,0,2], Hbe[:,0,2])
        lossr10 =  F.l1_loss(htest[:,1,0] ,Hbe[:,1,0])
        lossr11 =  F.l1_loss(htest[:,1,1] ,Hbe[:,1,1])
        lossr12 =  F.l1_loss(htest[:,1,2] ,Hbe[:,1,2])
        lossr20 =  F.l1_loss(htest[:,2,0], Hbe[:,2,0])
        lossr21 =  F.l1_loss(htest[:,2,1] ,Hbe[:,2,1])
        lossr22 =  F.l1_loss(htest[:,2,2] , Hbe[:,2,2])

        return (self.t0*lossr00+self.t1*lossr01+self.t2*lossr02+self.t3*lossr10+self.t4*lossr11+self.t5*lossr12+self.t6*lossr20+self.t7*lossr21+self.t8*lossr22)/2
