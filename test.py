import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

x = torch.rand(6,4,4)
print("x.size:",x.size())
print("x:",x)
x=x.topk(2, dim=2, largest=True, sorted=False)[0]
# y=x.topk(2,dim=2,largest=False,sorted=False)[0]
print("x.size:",x.size())
print("x=:",x)
# print("y.size:",y.size())
# print("y=:",y)

a,b= torch.chunk(x,2,dim=2)
d = x.view(-1, 8)
# print("a.size:",a.size())#(6,4,1)
# print("a=:",a)
# print("b.size:",b.size())#(6,4,1)
# print("b=:",b)
c = torch.cat([a,b],dim=1)#(6,8,1)
print("c.size:",c.size())
print("c=",c)
c=c.view(-1,8)
print("c.view.size:",c.size())
print("c.view=",c)

print("d.size:",d.size())
print("d:",d)

""" values0=x.topk(k=1, dim=0, sorted=False)[0]
print("values0.size:",values0.size())
print("values0=:",values0)

values1=x.topk(k=1, dim=1, sorted=False)[0]
print("values1.size:",values1.size())
print("values1=:",values1)

values2=x.topk(k=1, dim=2, sorted=False)[0]
print("values2.size:",values2.size())
print("values2=:",values2) """